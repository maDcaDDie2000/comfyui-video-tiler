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

# F.pad / grouped conv2d on CUDA use 32-bit indexing; huge [B,H,W,C] batches exceed it.
_MAX_ELEMENTS_PER_OP = (1 << 30)  # stay below ~2^31 total elements per tensor


def _batch_chunk_size(height: int, width: int, channels: int = 3) -> int:
    """Max IMAGE batch slice length so NCHW tensors stay under PyTorch index limits."""
    per_frame = max(1, channels * height * width)
    return max(1, _MAX_ELEMENTS_PER_OP // per_frame)


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
    _, c, h, w = x.shape
    pad_mode = "reflect" if radius < h and radius < w else "replicate"
    xs = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k1d = torch.exp(-0.5 * (xs / sigma) ** 2)
    k1d = k1d / k1d.sum()
    k_len = k1d.numel()

    # F.pad on NCHW: (pad_left, pad_right, pad_top, pad_bottom) for last two dims (W, H).
    k_vert = k1d.view(1, 1, k_len, 1).expand(c, 1, k_len, 1)
    x = F.pad(x, (0, 0, radius, radius), mode=pad_mode)
    x = F.conv2d(x, k_vert, groups=c)
    k_hor = k1d.view(1, 1, 1, k_len).expand(c, 1, 1, k_len)
    x = F.pad(x, (radius, radius, 0, 0), mode=pad_mode)
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


def _match_one_chunk(
    merged_chunk: torch.Tensor,
    ref_chunk: torch.Tensor,
    sigma: float,
    pull: float,
    dmix: float,
    lock_luma: bool,
    clamp_lo: float,
    clamp_hi: float,
    mode: str,
    dtype: torch.dtype,
    work_dtype: torch.dtype,
) -> torch.Tensor:
    """merged_chunk / ref_chunk: [Nb, H, W, 3]. Returns same shape / dtype as merged_chunk."""
    # NCHW for interpolate + blur
    m = merged_chunk.permute(0, 3, 1, 2).contiguous()
    r = ref_chunk.permute(0, 3, 1, 2).contiguous()
    h, w = m.shape[2], m.shape[3]
    if r.shape[2] != h or r.shape[3] != w:
        if mode == "area":
            r = F.interpolate(r, size=(h, w), mode="area")
        else:
            r = F.interpolate(r, size=(h, w), mode=mode, align_corners=False)

    m32 = m.to(work_dtype)
    r32 = r.to(work_dtype)

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
        merged_hwc = merged_chunk
        out_hwc = out.permute(0, 2, 3, 1)
        y_src = _rec709_luma(merged_hwc)
        y_dst = _rec709_luma(out_hwc)
        scale = y_src / (y_dst + 1e-6)
        scale = scale.clamp(clamp_lo, clamp_hi).unsqueeze(-1)
        out_hwc = (out_hwc * scale).clamp(0.0, 1.0)
        out = out_hwc.permute(0, 3, 1, 2)

    return out.permute(0, 2, 3, 1).contiguous()


class VideoTileReferenceColorMatch:
    """
    After Video Tile Merge: gently pull large-area color toward a reference clip
    (original LR or pre-upscaled) while keeping merged fine detail.

    Defaults favour fixing overall cast / seam tint without borrowing LR texture.
    """

    DESCRIPTION = (
        "Post-merge: match large-scale color to a reference clip (LR/pre-upscale) while preserving merged detail. "
        "Uses Gaussian low/high split; long batches are chunked for PyTorch limits."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "merged": (
                    "IMAGE",
                    {"tooltip": "Video Tile Merge output [B,H,W,3]."},
                ),
                "reference": (
                    "IMAGE",
                    {
                        "tooltip": "Same content at LR or pre-upscale resolution; resized internally. Batch 1 broadcasts to merged batch.",
                    },
                ),
            },
            "optional": {
                "low_frequency_sigma": (
                    "FLOAT",
                    {
                        "default": 14.0,
                        "min": 0.0,
                        "max": 256.0,
                        "step": 0.5,
                        "tooltip": "Gaussian σ (px) for LF/HF split. Higher = only broader color moves. ~0 skips blur (strong reference imprint).",
                    },
                ),
                "color_pull": (
                    "FLOAT",
                    {
                        "default": 0.58,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "How much low-frequency RGB follows the reference vs merged LF.",
                    },
                ),
                "detail_mix": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Gain on high-frequency residual from merged (1 = keep full upscale detail).",
                    },
                ),
                "preserve_merged_luminance": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "If true, rescale RGB after pull so Rec.709 luma matches merged (reference brightness won’t flatten scene).",
                    },
                ),
                "luma_scale_clamp": (
                    "FLOAT",
                    {
                        "default": 4.0,
                        "min": 1.05,
                        "max": 10.0,
                        "step": 0.05,
                        "tooltip": "Max/min multiplier when locking luma (prevents extreme corrections).",
                    },
                ),
                "reference_resize": (
                    ["bicubic", "bilinear", "area"],
                    {
                        "default": "bicubic",
                        "tooltip": "Interpolation when upsampling reference to merged width/height.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Color-matched IMAGE; same shape/dtype as merged input.",)
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

        work = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
        chunk = _batch_chunk_size(h, w, c)
        parts: list[torch.Tensor] = []
        for s in range(0, b, chunk):
            e = min(s + chunk, b)
            m_slice = merged[s:e]
            if rb == 1:
                r_slice = ref[0:1].expand(e - s, -1, -1, -1)
            else:
                r_slice = ref[s:e]
            parts.append(
                _match_one_chunk(
                    m_slice,
                    r_slice,
                    sigma,
                    pull,
                    dmix,
                    lock_luma,
                    clamp_lo,
                    clamp_hi,
                    mode,
                    dtype,
                    work,
                )
            )
        out = torch.cat(parts, dim=0)
        return (out,)

