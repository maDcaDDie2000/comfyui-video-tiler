"""
Video Tile Merge node - reconstructs video from processed tiles.
Uses shared output tensor. Feathering is **linear alpha** on the **top** layer only: opaque (1)
in the tile interior, ramping toward 0 at internal edges so the **underlying** composite shows through.
Not symmetric “blur” / averaging in the overlap core.

Single **feather** (not in TILE_CONFIG), same rule everywhere:
  **0–0.5** = fraction of this tile’s **width** for horizontal ramps and **height** for vertical ramps
  (oblong seam tiles get a wider ramp in px on the long side). Capped at **50%** per axis. No pixel ring.

  - **Grid** (v1/v2/v4): seam tiles composited **over** normals with that alpha model.
  - **Fixed** (v3/v5): same at internal edges; strip px = fraction × tile_w / tile_h, then min(overlap),
    snapped to multiple. **v3**: baked tuple strips; merge **feather** ignored.
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
) -> torch.Tensor:
    """Fixed layout: paint tiles in traversal order, each composited over with linear alpha (feather toward below)."""
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
    output = torch.zeros((B, height, width, C), dtype=first.dtype, device=dev)
    bx_u, by_u = max(0, int(bx)), max(0, int(by))

    sorted_pairs = sorted(enumerate(tile_specs), key=lambda it: it[1].order)
    for i, (orig_idx, spec) in enumerate(sorted_pairs):
        if orig_idx >= len(tiles_list):
            break
        tile = tiles_list[orig_idx].to(dev)
        x, y, w, h = spec.x, spec.y, spec.w, spec.h
        if i == 0:
            output[:, y : y + h, x : x + w, :] = tile
            continue
        alpha = _fixed_tile_top_alpha(w, h, x, y, width, height, bx_u, by_u, dev, first.dtype)
        a = alpha.reshape(1, h, w, 1)
        region = output[:, y : y + h, x : x + w, :]
        output[:, y : y + h, x : x + w, :] = region * (1.0 - a) + tile * a

    return output


def merge_tiles(
    tiles: list[torch.Tensor],
    config_tuple: tuple,
    feather: float,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Merge: grid uses over + tile-fraction alpha on seam tiles; fixed uses traversal order + same fraction model."""
    ver = config_tuple[0]
    if ver in (3, 5):
        return merge_fixed_grid_tiles(tiles, config_tuple, feather, device=device)
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

    output = torch.zeros((B, height, width, C), dtype=first.dtype, device=device)

    for orig_idx, spec in sorted(enumerate(tile_specs), key=lambda it: it[1].order):
        if orig_idx >= len(tiles_list):
            break
        tile = tiles_list[orig_idx].to(device)
        x, y, w, h = spec.x, spec.y, spec.w, spec.h

        if spec.type == "normal":
            output[:, y : y + h, x : x + w, :] = tile
        else:
            alpha = _feather_mask(w, h, feather, x, y, width, height).to(tile.device)
            a = alpha.reshape(1, h, w, 1)
            region = output[:, y : y + h, x : x + w, :]
            output[:, y : y + h, x : x + w, :] = region * (1.0 - a) + tile * a

    return output


class VideoTileMerge:
    """
    **feather**: 0–0.5 = fraction of tile **width** (H ramps) and **height** (V ramps); max 50% per axis.
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
        }

    INPUT_IS_LIST = (False, True)
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "merge"
    CATEGORY = "Video Tiler"

    def merge(self, tile_config, tiles, feather):
        if isinstance(tile_config, (list, tuple)) and len(tile_config) > 0:
            tile_config = tile_config[0]
        feather = _scalar_float(feather)
        if isinstance(tiles, (list, tuple)):
            tiles_list = list(tiles)
        else:
            tiles_list = [tiles]
        print(f"[Video Tiler] Merging {len(tiles_list)} tiles → single IMAGE batch")
        result = merge_tiles(tiles_list, tile_config, feather)
        return (result,)
