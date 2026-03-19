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
    """Extract tile at index as view - no copy."""
    _, _, _, _, tile_specs = parse_tile_config(config_tuple)
    tile_index = max(0, min(tile_index, len(tile_specs) - 1))
    spec = tile_specs[tile_index]
    return images[:, spec.y : spec.y + spec.h, spec.x : spec.x + spec.w, :]


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


class GetTileCount:
    """
    Get the number of tiles from tile_config.
    Use this to set loop iteration count (e.g. ForLoopOpen remaining).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_config": ("TILE_CONFIG", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("INT", "TILE_CONFIG")
    RETURN_NAMES = ("count", "tile_config")
    FUNCTION = "get_count"
    CATEGORY = "Video Tiler"

    def get_count(self, tile_config: tuple):
        _, _, _, _, tile_specs = parse_tile_config(tile_config)
        return (len(tile_specs), tile_config)
