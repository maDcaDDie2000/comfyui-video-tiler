"""
Post-merge color alignment using a low-res / pre-upscale reference.

Splits merged output into low / high frequency (Gaussian). Low-frequency RGB is
pulled toward the resized reference; highs stay from the merge (detail).
Optional per-pixel luminance rescale keeps Rec.709 luma matched to the merged
image so reference brightness does not flatten the upscale.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _scalar_float(v, default: float = 0.0) -> float:
    while isinstance(v, (list, tuple)):
        if len(v) == 0:
            return default
        v = v[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _scalar_bool(v, default: bool = False) -> bool:
    while isinstance(v, (list, tuple)):
        if len(v) == 0:
            return default
        v = v[0]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return default


def _resize_mode(s: str) -> str:
    s = str(s).strip().lower()
    if s in ("area", "nearest"):
        return s
    if s == "bilinear":
        return "bilinear"
    return "bicubic"


def _gaussian_blur_nchw(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Depthwise separable Gaussian blur. x: N,C,H,W."""
    sigma = float(max(sigma, 1e-3))
    radius = max(1, int(math.ceil(3 * sigma)))
    device, dtype = x.device, x.dtype
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k1d = torch.exp(-0.5 * (xs / sigma) ** 2)
    k1d = k1d / k1d.sum()
    k_len = k1d.numel()

    _, c, _, _ = x.shape
    # F.pad on NCHW: (pad_left, pad_right, pad_top, pad_bottom) for last two dims (W, H).
    k_vert = k1d.view(1, 1, k_len, 1).expand(c, 1, k_len, 1)
    x = F.pad(x, (0, 0, radius, radius), mode="reflect")
    x = F.conv2d(x, k_vert, groups=c)
    k_hor = k1d.view(1, 1, 1, k_len).expand(c, 1, 1, k_len)
    x = F.pad(x, (radius, radius, 0, 0), mode="reflect")
    x = F.conv2d(x, k_hor, groups=c)
    return x


def _rec709_luma(rgb: torch.Tensor) -> torch.Tensor:
    """rgb last dim = 3, linear-ish RGB in [0,1]."""
    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]
    return (
        0.21263900587151027 * r
        + 0.715168678767756 * g
        + 0.07219231536073371 * b
    )


class VideoTileReferenceColorMatch:
    """
    After Video Tile Merge: gently pull large-area color toward a reference clip
    (original LR or pre-upscaled) while keeping merged fine detail.

    Defaults favour fixing overall cast / seam tint without borrowing LR texture.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merged": ("IMAGE",),
                "reference": ("IMAGE",),
            },
            "optional": {
                "low_frequency_sigma": (
                    "FLOAT",
                    {"default": 14.0, "min": 0.0, "max": 256.0, "step": 0.5},
                ),
                "color_pull": (
                    "FLOAT",
                    {"default": 0.58, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "detail_mix": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "preserve_merged_luminance": (
                    "BOOLEAN",
                    {"default": True},
                ),
                "luma_scale_clamp": (
                    "FLOAT",
                    {"default": 4.0, "min": 1.05, "max": 10.0, "step": 0.05},
                ),
                "reference_resize": (
                    ["bicubic", "bilinear", "area"],
                    {"default": "bicubic"},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "match"
    CATEGORY = "Video Tiler"

    def match(
        self,
        merged: torch.Tensor,
        reference: torch.Tensor,
        low_frequency_sigma=14.0,
        color_pull=0.58,
        detail_mix=1.0,
        preserve_merged_luminance=True,
        luma_scale_clamp=4.0,
        reference_resize="bicubic",
    ):
        sigma = _scalar_float(low_frequency_sigma, 14.0)
        pull = max(0.0, min(1.0, _scalar_float(color_pull, 0.58)))
        dmix = max(0.0, min(1.0, _scalar_float(detail_mix, 1.0)))
        lock_luma = _scalar_bool(preserve_merged_luminance, True)
        clamp_hi = max(1.05, _scalar_float(luma_scale_clamp, 4.0))
        clamp_lo = 1.0 / clamp_hi
        mode = _resize_mode(reference_resize)

        if merged.dim() != 4:
            raise ValueError("merged IMAGE must be [B, H, W, C]")
        if reference.dim() != 4:
            raise ValueError("reference IMAGE must be [B, H, W, C]")

        device = merged.device
        dtype = merged.dtype
        b, h, w, c = merged.shape
        if c != 3:
            raise ValueError("Video Tile Reference Color Match expects RGB (C=3).")

        ref = reference.to(device=device, dtype=dtype)
        rb = ref.shape[0]
        if rb == 1 and b > 1:
            ref = ref.expand(b, -1, -1, -1)
        elif rb != b:
            raise ValueError(
                f"reference batch {rb} must equal merged batch {b}, or reference batch must be 1."
            )

        # NCHW for interpolate + blur
        m = merged.permute(0, 3, 1, 2).contiguous()
        r = ref.permute(0, 3, 1, 2).contiguous()
        if r.shape[2] != h or r.shape[3] != w:
            if mode == "area":
                r = F.interpolate(r, size=(h, w), mode="area")
            else:
                r = F.interpolate(r, size=(h, w), mode=mode, align_corners=False)

        work = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        m32 = m.to(work)
        r32 = r.to(work)

        if sigma <= 0.05:
            low_m = m32
            low_r = r32
        else:
            low_m = _gaussian_blur_nchw(m32, sigma)
            low_r = _gaussian_blur_nchw(r32, sigma)
        high_m = m32 - low_m

        low_out = (1.0 - pull) * low_m + pull * low_r
        out = low_out + dmix * high_m
        out = out.to(dtype=dtype)

        if lock_luma:
            # Per-pixel uniform RGB scale to match merged Rec.709 luma (preserves hue).
            merged_hwc = merged
            out_hwc = out.permute(0, 2, 3, 1)
            y_src = _rec709_luma(merged_hwc)
            y_dst = _rec709_luma(out_hwc)
            scale = y_src / (y_dst + 1e-6)
            scale = scale.clamp(clamp_lo, clamp_hi).unsqueeze(-1)
            out_hwc = (out_hwc * scale).clamp(0.0, 1.0)
            out = out_hwc.permute(0, 3, 1, 2)

        out = out.permute(0, 2, 3, 1).contiguous()
        return (out,)

