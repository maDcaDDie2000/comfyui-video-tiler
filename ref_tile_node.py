"""
Reference Tile Slice node - cuts matching tiles from a reference image.
Output order is synchronous with Video Tile Slice so each ref tile matches its video tile.
"""

from __future__ import annotations

import torch

from .layout import TileSpec
from .tile_config import parse_tile_config


def slice_ref_tiles(reference: torch.Tensor, tile_specs: list[TileSpec]) -> list[torch.Tensor]:
    """Extract tiles from reference image as views - same order as Video Tile Slice."""
    result = []
    for spec in tile_specs:
        tile = reference[:, spec.y : spec.y + spec.h, spec.x : spec.x + spec.w, :]
        result.append(tile)
    return result


class ReferenceTileSlice:
    """
    Slice a reference image into tiles matching the Video Tile Slice layout.
    Connect tile_config from Video Tile Slice. Output list is synchronous with
    the video tile list so ref_tile[i] matches video_tile[i] during processing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "tile_config": ("TILE_CONFIG", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("ref_tiles",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "slice_ref"
    CATEGORY = "Video Tiler"

    def slice_ref(self, reference_image: torch.Tensor, tile_config):
        if isinstance(tile_config, (list, tuple)) and len(tile_config) > 0:
            tile_config = tile_config[0]
        _, _, _, _, tile_specs = parse_tile_config(tile_config)
        ref_tiles = slice_ref_tiles(reference_image, tile_specs)
        print(f"[Video Tiler] Ref slice: {len(ref_tiles)} tiles (sync with video tiles)")
        return (ref_tiles,)
