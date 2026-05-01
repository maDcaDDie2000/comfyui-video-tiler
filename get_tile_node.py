"""
Get Tile node - extract a single tile by index for individual processing.

`images` may be either:
  - The full frame IMAGE [B,H,W,C] (crop using tile_config), or
  - The slicer's `tiles` output: a list of tile tensors (OUTPUT_IS_LIST) — pick index directly.

Outputs: tile (single IMAGE), tile_index (INT), tile_config.
"""

from __future__ import annotations

import torch

from .tile_config import parse_tile_config


def _normalize_tile(t):
    """Ensure 4D (B,H,W,C) Comfy IMAGE."""
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


def _unwrap_outer_lists(obj):
    """Strip [[...]] wrappers until tensor or non-single-element list."""
    while isinstance(obj, (list, tuple)) and len(obj) == 1:
        inner = obj[0]
        if isinstance(inner, torch.Tensor):
            return inner
        obj = inner
    return obj


def get_tile_by_index(
    images: torch.Tensor,
    config_tuple: tuple,
    tile_index: int,
) -> torch.Tensor:
    """Extract tile at index from full frame. Clamped to image bounds; contiguous for PIL."""
    _, _, _, tile_specs = parse_tile_config(config_tuple)
    tile_index = max(0, min(tile_index, len(tile_specs) - 1))
    spec = tile_specs[tile_index]
    _, H, W, _ = images.shape
    x1 = max(0, min(spec.x, W - 1))
    y1 = max(0, min(spec.y, H - 1))
    x2 = min(spec.x + spec.w, W)
    y2 = min(spec.y + spec.h, H)
    if x2 <= x1 or y2 <= y1:
        return images[:, :1, :1, :].clone()
    tile = images[:, y1:y2, x1:x2, :].contiguous()
    return tile


class GetTile:
    """
    One tile by index. Wire either the **full** IMAGE into `images`, or the slicer **`tiles`** list.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tile_config": ("TILE_CONFIG", {"forceInput": True}),
                "tile_index": ("INT", {"default": 0, "min": 0, "max": 999}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "TILE_CONFIG")
    RETURN_NAMES = ("tile", "tile_index", "tile_config")
    INPUT_IS_LIST = (True, False, False)
    FUNCTION = "get_tile"
    CATEGORY = "Video Tiler"

    def get_tile(self, images, tile_config: tuple, tile_index: int):
        if isinstance(tile_config, (list, tuple)) and len(tile_config) > 0:
            tile_config = tile_config[0]
        ti = int(tile_index)

        width, height, _, tile_specs = parse_tile_config(tile_config)
        n = len(tile_specs)
        ti = max(0, min(ti, n - 1))

        raw = _unwrap_outer_lists(images)

        if isinstance(raw, (list, tuple)):
            items = [x for x in raw if isinstance(x, torch.Tensor)]
            if not items:
                raise ValueError(
                    "Get Tile: `images` must be a full IMAGE tensor or a non-empty list of tile tensors "
                    "(connect Video Tile Slicer **tiles** output)."
                )
            if len(items) > 1:
                ti = min(ti, len(items) - 1)
                tile = _normalize_tile(items[ti]).contiguous()
                return (tile, ti, tile_config)

            only = _normalize_tile(items[0])
            _, H, W, _ = only.shape
            if H == height and W == width:
                tile = get_tile_by_index(only, tile_config, ti).contiguous()
                return (tile, ti, tile_config)
            if ti != 0:
                raise ValueError(
                    "Get Tile: received a single tensor not matching frame size — likely only one tile in "
                    "the list but tile_index > 0. Connect the slicer's full **tiles** list."
                )
            tile = only.contiguous()
            return (tile, ti, tile_config)

        images_tensor = _normalize_tile(raw)
        tile = get_tile_by_index(images_tensor, tile_config, ti).contiguous()
        return (tile, ti, tile_config)
