"""
Symmetric overlap merge: sum(w × color) / sum(w) with geometry-only weights (same masks as
**Video Tile Merge** fixed/grid feather helpers — nearest-edge cosine ramps).
"""

from __future__ import annotations

import torch

from .fixed_layout import compute_fixed_feather_strips
from .merge_node import (
    _feather_mask,
    _fixed_tile_top_alpha,
    _normalize_tile,
    _scalar_float,
    _strip_fraction,
)
from .tile_config import parse_tile_config


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
    B, _, _, C = num.shape
    dev = num.device
    wt = weight_hw.to(device=dev, dtype=torch.float32).clamp(min=1e-6, max=1.0)
    wt4 = wt.unsqueeze(0).unsqueeze(-1).expand(B, h, w, C)
    tf = tile.float()
    num[:, y : y + h, x : x + w, :] += tf * wt4
    den[:, y : y + h, x : x + w] += wt.unsqueeze(0).expand(B, h, w)


def _merge_fixed_soft(
    tiles: list[torch.Tensor],
    config_tuple: tuple,
    feather: float,
) -> torch.Tensor:
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
        raise ValueError(f"Video Tile Merge (overlap soft): unsupported TILE_CONFIG version {v}")

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
    num = torch.zeros((B, height, width, C), dtype=torch.float32, device=dev)
    den = torch.zeros((B, height, width), dtype=torch.float32, device=dev)
    bx_u, by_u = max(0, int(bx)), max(0, int(by))

    sorted_pairs = sorted(enumerate(tile_specs), key=lambda it: it[1].order)
    for i, (orig_idx, spec) in enumerate(sorted_pairs):
        if orig_idx >= len(tiles_list):
            break
        tile = tiles_list[orig_idx].to(dev)
        x, y, w, h = spec.x, spec.y, spec.w, spec.h
        if i == 0:
            wmask = torch.ones((h, w), dtype=torch.float32, device=dev)
        else:
            wmask = _fixed_tile_top_alpha(w, h, x, y, width, height, bx_u, by_u, dev, first.dtype)
        _weighted_accum_region(num, den, tile, y, x, h, w, wmask)

    out = num / den.unsqueeze(-1).clamp(min=1e-8)
    return out.to(dtype=first.dtype)


def _merge_grid_soft(
    tiles: list[torch.Tensor],
    config_tuple: tuple,
    feather: float,
) -> torch.Tensor:
    width, height, _multiple, tile_specs = parse_tile_config(config_tuple)
    tiles_list = [_normalize_tile(t) for t in tiles]
    if not tiles_list:
        raise ValueError("No tiles to merge")
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
        _weighted_accum_region(num, den, tile, y, x, h, w, wmask)

    out = num / den.unsqueeze(-1).clamp(min=1e-8)
    return out.to(dtype=first.dtype)


def merge_soft_overlap(
    tiles: list[torch.Tensor],
    config_tuple: tuple,
    feather: float,
) -> torch.Tensor:
    ver = config_tuple[0]
    if ver in (3, 5):
        return _merge_fixed_soft(tiles, config_tuple, feather)
    return _merge_grid_soft(tiles, config_tuple, feather)


class VideoTileMergeOverlapSoft:
    """
    Same wiring as **Video Tile Merge**; symmetric weighted sums using shared feather geometry.
    """

    DESCRIPTION = (
        "Alternative merge: always uses weighted sum of geometry feather weights (symmetric overlap blend). "
        "Same inputs as Video Tile Merge; good when you want normalized averaging instead of alpha-over paint order."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_config": (
                    "TILE_CONFIG",
                    {
                        "forceInput": True,
                        "tooltip": "TILE_CONFIG from the matching slicer.",
                    },
                ),
                "tiles": (
                    "IMAGE",
                    {"tooltip": "Processed tiles as list (INPUT_IS_LIST)."},
                ),
                "feather": (
                    "FLOAT",
                    {
                        "default": 0.125,
                        "min": 0.0,
                        "max": 0.5,
                        "step": 0.005,
                        "tooltip": "Feather fraction (same interpretation as Video Tile Merge). Drives soft overlap weights.",
                    },
                ),
            },
        }

    INPUT_IS_LIST = (False, True)
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("Weighted-average merged IMAGE.",)
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
        print(f"[Video Tiler] Overlap-soft merging {len(tiles_list)} tiles → single IMAGE batch")
        result = merge_soft_overlap(tiles_list, tile_config, feather)
        return (result,)
