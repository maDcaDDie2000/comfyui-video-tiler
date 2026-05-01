"""
Video Tile Merge node - reconstructs video from processed tiles.
Uses shared output tensor. Feather masks gate alpha-over compositing where an earlier tile already
wrote the pixel (`covered`). Elsewhere the current tile is pasted at full opacity.

Geometric ramps are linear per-side (same footprint); optional **feather_curve** remaps opacity
pointwise before blending. Multiply-add runs in float32, then casts back to IMAGE dtype.

Single **feather** (not in TILE_CONFIG), same rule everywhere:
  **0–0.5** = fraction of this tile’s **width** for horizontal ramps and **height** for vertical ramps
  (oblong seam tiles get a wider ramp in px on the long side). Capped at **50%** per axis.

  - **Grid** (v1/v2/v4): seam tiles composited **over** normals with coverage-gated feather.
  - **Fixed** (v3/v5): traversal order + coverage; strip px from overlap × feather fraction.
"""

import torch

from .fixed_layout import compute_fixed_feather_strips
from .tile_config import parse_tile_config


def _scalar_float(v) -> float:
    """ComfyUI may wrap FLOAT widgets in list/tuple (e.g. list-iteration / batch execution)."""
    while isinstance(v, (list, tuple)):
        if len(v) == 0:
            return 0.0
        v = v[0]
    return float(v)


def _strip_fraction(feather: float) -> float:
    """Feather knob: fraction of local tile width/height for ramp extent, max 50%."""
    return max(0.0, min(0.5, _scalar_float(feather)))


def _feather_curve_mode(v) -> str:
    """Normalize widget / list-wrapped combo value to a curve id."""
    if v is None:
        return "linear"
    while isinstance(v, (list, tuple)):
        if len(v) == 0:
            return "linear"
        v = v[0]
    s = str(v).strip().lower().replace("-", "_").replace("+", "_")
    aliases = {
        "easein": "ease_in",
        "ease_out": "ease_out",
        "easeout": "ease_out",
        "easeinout": "ease_in_out",
        "ease_in_out": "ease_in_out",
    }
    if s in aliases:
        return aliases[s]
    if s in ("linear", "ease_in", "ease_out", "ease_in_out"):
        return s
    return "linear"


def _blend_mode_keyword(v) -> str:
    """alpha_over (default) vs symmetric weighted_average using geometric masks only."""
    if v is None:
        return "alpha_over"
    while isinstance(v, (list, tuple)):
        if len(v) == 0:
            return "alpha_over"
        v = v[0]
    s = str(v).strip().lower().replace(" ", "_").replace("-", "_")
    if s in ("weighted_average", "weighted", "average", "mean"):
        return "weighted_average"
    return "alpha_over"


def _apply_feather_curve(alpha: torch.Tensor, mode: str) -> torch.Tensor:
    """
    Pointwise remap of geometric alpha in [0, 1]. Interior remains ~1; edges reshape transition speed.
    ease_in_out uses smoothstep (3t² − 2t³).
    """
    if mode == "linear":
        return alpha
    t = alpha.clamp(0.0, 1.0)
    if mode == "ease_in":
        return t * t
    if mode == "ease_out":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if mode == "ease_in_out":
        return t * t * (3.0 - 2.0 * t)
    return alpha


def _feather_mask(
    w: int, h: int, feather: float,
    x: int, y: int, img_width: int, img_height: int,
) -> torch.Tensor:
    """
    Top-layer alpha for grid **seam** tiles: 1 in the interior, linear ramps at internal edges.
    Horizontal ramp width = w * strip_fraction; vertical = h * strip_fraction (oblong ⇒ different px).
    """
    frac = _strip_fraction(feather)
    fw = w * frac
    fh = h * frac
    if fw <= 1e-6 and fh <= 1e-6:
        return torch.ones((h, w), dtype=torch.float32)

    yy = torch.linspace(0, 1, h)
    xx = torch.linspace(0, 1, w)
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")

    eps = 1e-6
    left = torch.ones((h, w), dtype=torch.float32) if x == 0 else (
        torch.clamp(xx * (w / max(fw, eps)), 0, 1) if fw > eps else torch.ones((h, w), dtype=torch.float32)
    )
    right = torch.ones((h, w), dtype=torch.float32) if x + w == img_width else (
        torch.clamp((1 - xx) * (w / max(fw, eps)), 0, 1) if fw > eps else torch.ones((h, w), dtype=torch.float32)
    )
    top = torch.ones((h, w), dtype=torch.float32) if y == 0 else (
        torch.clamp(yy * (h / max(fh, eps)), 0, 1) if fh > eps else torch.ones((h, w), dtype=torch.float32)
    )
    bottom = torch.ones((h, w), dtype=torch.float32) if y + h == img_height else (
        torch.clamp((1 - yy) * (h / max(fh, eps)), 0, 1) if fh > eps else torch.ones((h, w), dtype=torch.float32)
    )

    return torch.minimum(torch.minimum(left, right), torch.minimum(top, bottom))


def _blend_tile_with_coverage(
    output: torch.Tensor,
    covered: torch.Tensor,
    tile: torch.Tensor,
    y: int,
    x: int,
    h: int,
    w: int,
    alpha_geom: torch.Tensor,
    feather_curve: str = "linear",
) -> None:
    """
    Composite tile over output[y:y+h,x:x+w]. Use geometric feather only where covered[b] is True;
    uncovered pixels get alpha=1 (full top opacity). Then mark entire tile rect covered.
    Feather remap + multiply-add use float32; write back in output dtype.
    """
    B = output.shape[0]
    dev = output.device
    dt = output.dtype
    cov = covered[:, y : y + h, x : x + w]
    a_g = alpha_geom.to(device=dev, dtype=torch.float32)
    a_g = _apply_feather_curve(a_g, feather_curve)
    a_g = a_g.unsqueeze(0).expand(B, h, w)
    ones = torch.ones((B, h, w), dtype=torch.float32, device=dev)
    a = torch.where(cov, a_g, ones)
    a4 = a.unsqueeze(-1)
    region = output[:, y : y + h, x : x + w, :]
    region_f = region.float()
    tile_f = tile.float()
    blended = region_f * (1.0 - a4) + tile_f * a4
    output[:, y : y + h, x : x + w, :] = blended.to(dtype=dt)
    covered[:, y : y + h, x : x + w] = True


def _weighted_accum_region(
    num: torch.Tensor,
    den: torch.Tensor,
    tile: torch.Tensor,
    y: int,
    x: int,
    h: int,
    w: int,
    weight_hw: torch.Tensor,
) -> None:
    """Accumulate sum(weight × tile) / sum(weight) merge; weight_hw (h,w) float32, geometry-only."""
    B, _, _, C = num.shape
    dev = num.device
    wt = weight_hw.to(device=dev, dtype=torch.float32).clamp(0.0, 1.0)
    wt4 = wt.unsqueeze(0).unsqueeze(-1).expand(B, h, w, C)
    tf = tile.float()
    num[:, y : y + h, x : x + w, :] += tf * wt4
    den[:, y : y + h, x : x + w] += wt.unsqueeze(0).expand(B, h, w)


def _normalize_tile(t):
    """Unwrap (tensor,) from node output format; ensure 4D (B,H,W,C)."""
    if isinstance(t, (list, tuple)) and len(t) > 0:
        t = t[0]
    if not isinstance(t, torch.Tensor):
        return t
    s = t.shape
    if len(s) == 2:
        t = t.unsqueeze(0).unsqueeze(-1)
    elif len(s) == 3:
        t = t.unsqueeze(0)
    elif len(s) == 4 and s[-1] not in (3, 4) and s[1] in (3, 4):
        t = t.permute(0, 2, 3, 1)
    return t


def _fixed_tile_top_alpha(
    w: int, h: int, x: int, y: int,
    img_w: int, img_h: int,
    feather_x: int, feather_y: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Alpha of the **current** (top) fixed tile: 1 inward from internal edges, linear ramp to 0 at the
    tile border where it meets already-painted pixels (same geometry as former weight mask, but used
    for `dst = dst * (1 - a) + src * a` so lower layers show through only in the ramp).
    """
    ly = torch.arange(h, device=device, dtype=torch.float32).view(-1, 1).expand(h, w)
    lx = torch.arange(w, device=device, dtype=torch.float32).view(1, -1).expand(h, w)
    m = torch.ones((h, w), device=device, dtype=torch.float32)
    if x > 0 and feather_x > 0:
        m *= torch.where(lx < feather_x, lx / feather_x, torch.ones_like(m))
    if y > 0 and feather_y > 0:
        m *= torch.where(ly < feather_y, ly / feather_y, torch.ones_like(m))
    if x + w < img_w and feather_x > 0:
        d = (w - 1) - lx
        m *= torch.where(d < feather_x, d / feather_x, torch.ones_like(m))
    if y + h < img_h and feather_y > 0:
        d = (h - 1) - ly
        m *= torch.where(d < feather_y, d / feather_y, torch.ones_like(m))
    return m


def merge_fixed_grid_tiles(
    tiles: list[torch.Tensor],
    config_tuple: tuple,
    feather: float,
    device: torch.device | None = None,
    feather_curve: str = "linear",
    blend_mode: str = "alpha_over",
) -> torch.Tensor:
    """Fixed layout: alpha_over (painter + coverage) or weighted_average (geometry weights only)."""
    v = config_tuple[0]
    if v == 5:
        (
            _ver,
            width,
            height,
            tw,
            th,
            overlap_x,
            overlap_y,
            mult,
            _pattern_id,
            _tiles_data,
        ) = config_tuple
        bx, by = compute_fixed_feather_strips(tw, th, overlap_x, overlap_y, mult, _strip_fraction(feather))
    elif v == 3:
        (
            _ver,
            width,
            height,
            _tw,
            _th,
            overlap_x,
            overlap_y,
            bx,
            by,
            _pattern_id,
            _tiles_data,
        ) = config_tuple
    else:
        raise ValueError(f"merge_fixed_grid_tiles: unsupported TILE_CONFIG version {v}")

    tiles_list = [_normalize_tile(t) for t in tiles]
    if not tiles_list:
        raise ValueError("No tiles to merge")
    first = tiles_list[0]
    if len(first.shape) < 4:
        raise ValueError(f"Tile shape {first.shape} - expected (B,H,W,C)")
    B, C = first.shape[0], first.shape[3]
    if first.is_cuda:
        dev = first.device
    elif torch.cuda.is_available():
        dev = torch.device("cuda:0")
    else:
        dev = first.device

    _, _, _, tile_specs = parse_tile_config(config_tuple)
    bx_u, by_u = max(0, int(bx)), max(0, int(by))

    sorted_pairs = sorted(enumerate(tile_specs), key=lambda it: it[1].order)

    if blend_mode == "weighted_average":
        num = torch.zeros((B, height, width, C), dtype=torch.float32, device=dev)
        den = torch.zeros((B, height, width), dtype=torch.float32, device=dev)
        for i, (orig_idx, spec) in enumerate(sorted_pairs):
            if orig_idx >= len(tiles_list):
                break
            tile = tiles_list[orig_idx].to(dev)
            x, y, w, h = spec.x, spec.y, spec.w, spec.h
            if i == 0:
                wmask = torch.ones((h, w), dtype=torch.float32, device=dev)
            else:
                wmask = _fixed_tile_top_alpha(w, h, x, y, width, height, bx_u, by_u, dev, first.dtype)
            wmask = _apply_feather_curve(wmask, feather_curve)
            _weighted_accum_region(num, den, tile, y, x, h, w, wmask)
        out = num / den.unsqueeze(-1).clamp(min=1e-8)
        return out.to(dtype=first.dtype)

    output = torch.zeros((B, height, width, C), dtype=first.dtype, device=dev)
    covered = torch.zeros((B, height, width), dtype=torch.bool, device=dev)
    for i, (orig_idx, spec) in enumerate(sorted_pairs):
        if orig_idx >= len(tiles_list):
            break
        tile = tiles_list[orig_idx].to(dev)
        x, y, w, h = spec.x, spec.y, spec.w, spec.h
        if i == 0:
            output[:, y : y + h, x : x + w, :] = tile
            covered[:, y : y + h, x : x + w] = True
            continue
        alpha = _fixed_tile_top_alpha(w, h, x, y, width, height, bx_u, by_u, dev, first.dtype)
        _blend_tile_with_coverage(
            output, covered, tile, y, x, h, w, alpha, feather_curve=feather_curve
        )

    return output


def merge_tiles(
    tiles: list[torch.Tensor],
    config_tuple: tuple,
    feather: float,
    device: torch.device | None = None,
    feather_curve: str = "linear",
    blend_mode: str = "alpha_over",
) -> torch.Tensor:
    """Merge: grid or fixed; alpha_over vs weighted_average (symmetric geometry weights)."""
    ver = config_tuple[0]
    if ver in (3, 5):
        return merge_fixed_grid_tiles(
            tiles,
            config_tuple,
            feather,
            device=device,
            feather_curve=feather_curve,
            blend_mode=blend_mode,
        )
    width, height, _multiple, tile_specs = parse_tile_config(config_tuple)

    tiles_list = [_normalize_tile(t) for t in tiles]
    if not tiles_list:
        raise ValueError("No tiles to merge - Tile Loop produced empty list")
    first = tiles_list[0]
    if len(first.shape) < 4:
        raise ValueError(f"Tile shape {first.shape} - expected (B,H,W,C)")
    B, C = first.shape[0], first.shape[3]

    if first.is_cuda:
        device = first.device
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = first.device

    if blend_mode == "weighted_average":
        num = torch.zeros((B, height, width, C), dtype=torch.float32, device=device)
        den = torch.zeros((B, height, width), dtype=torch.float32, device=device)
        for orig_idx, spec in sorted(enumerate(tile_specs), key=lambda it: it[1].order):
            if orig_idx >= len(tiles_list):
                break
            tile = tiles_list[orig_idx].to(device)
            x, y, w, h = spec.x, spec.y, spec.w, spec.h
            if spec.type == "normal":
                wmask = torch.ones((h, w), dtype=torch.float32, device=device)
            else:
                wmask = _feather_mask(w, h, feather, x, y, width, height).to(
                    device=device, dtype=torch.float32
                )
            wmask = _apply_feather_curve(wmask, feather_curve)
            _weighted_accum_region(num, den, tile, y, x, h, w, wmask)
        out = num / den.unsqueeze(-1).clamp(min=1e-8)
        return out.to(dtype=first.dtype)

    output = torch.zeros((B, height, width, C), dtype=first.dtype, device=device)
    covered = torch.zeros((B, height, width), dtype=torch.bool, device=device)

    for orig_idx, spec in sorted(enumerate(tile_specs), key=lambda it: it[1].order):
        if orig_idx >= len(tiles_list):
            break
        tile = tiles_list[orig_idx].to(device)
        x, y, w, h = spec.x, spec.y, spec.w, spec.h

        if spec.type == "normal":
            output[:, y : y + h, x : x + w, :] = tile
            covered[:, y : y + h, x : x + w] = True
        else:
            alpha = _feather_mask(w, h, feather, x, y, width, height).to(
                device=device, dtype=torch.float32
            )
            _blend_tile_with_coverage(
                output, covered, tile, y, x, h, w, alpha, feather_curve=feather_curve
            )

    return output


class VideoTileMerge:
    """
    **feather**: 0–0.5 = fraction of tile **width** (H ramps) and **height** (V ramps); max 50% per axis.
    **Optional** — **feather_curve**, **blend_mode** (defaults keep legacy behavior if widgets absent).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_config": ("TILE_CONFIG", {"forceInput": True}),
                "tiles": ("IMAGE",),
                "feather": (
                    "FLOAT",
                    {
                        "default": 0.125,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.005,
                    },
                ),
            },
            "optional": {
                "feather_curve": (
                    ["linear", "ease_in", "ease_out", "ease_in_out"],
                    {"default": "linear"},
                ),
                "blend_mode": (
                    ["alpha_over", "weighted_average"],
                    {"default": "alpha_over"},
                ),
            },
        }

    INPUT_IS_LIST = (False, True)
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "merge"
    CATEGORY = "Video Tiler"

    def merge(self, tile_config, tiles, feather, feather_curve=None, blend_mode=None, **kwargs):
        if isinstance(tile_config, (list, tuple)) and len(tile_config) > 0:
            tile_config = tile_config[0]
        feather = _scalar_float(feather)
        curve = _feather_curve_mode(feather_curve)
        mode = _blend_mode_keyword(blend_mode)
        if isinstance(tiles, (list, tuple)):
            tiles_list = list(tiles)
        else:
            tiles_list = [tiles]
        print(f"[Video Tiler] Merging {len(tiles_list)} tiles → single IMAGE batch ({mode})")
        result = merge_tiles(
            tiles_list,
            tile_config,
            feather,
            feather_curve=curve,
            blend_mode=mode,
        )
        return (result,)
