"""
Get Tile node - extract a single tile by index for individual processing.
Outputs tile (view), tile_index (INT for steering), and tile_config.
"""

import torch

from .tile_config import parse_tile_config


def get_tile_by_index(
    images: torch.Tensor,
    config_tuple: tuple,
    tile_index: int,
) -> torch.Tensor:
    """Extract tile at index. Clamped to image bounds; contiguous for PIL compatibility."""
    _, _, _, _, tile_specs = parse_tile_config(config_tuple)
    tile_index = max(0, min(tile_index, len(tile_specs) - 1))
    spec = tile_specs[tile_index]
    _, H, W, _ = images.shape
    x1 = max(0, min(spec.x, W - 1))
    y1 = max(0, min(spec.y, H - 1))
    x2 = min(spec.x + spec.w, W)
    y2 = min(spec.y + spec.h, H)
    if x2 <= x1 or y2 <= y1:
        return images[:, :1, :1, :].clone()  # fallback: 1x1 tile if bounds invalid
    tile = images[:, y1:y2, x1:x2, :].contiguous()
    return tile


class GetTile:
    """
    Get a single tile by index for individual processing.
    Outputs: tile (view), tile_index (INT for steering), tile_config.
    Use with loop nodes or manual index for one-at-a-time processing.
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
    FUNCTION = "get_tile"
    CATEGORY = "Video Tiler"

    def get_tile(self, images: torch.Tensor, tile_config: tuple, tile_index: int):
        _, _, _, _, tile_specs = parse_tile_config(tile_config)
        n = len(tile_specs)
        tile_index = max(0, min(tile_index, n - 1))
        tile = get_tile_by_index(images, tile_config, tile_index)
        return (tile, tile_index, tile_config)


