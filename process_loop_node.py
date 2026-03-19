"""
Video Tile Process Loop - self-contained sequential tile processing.
Processes all tiles one-by-one in a Python loop, no external node packs required.
"""

import torch

from .get_tile_node import get_tile_by_index
from .merge_node import merge_tiles
from .tile_config import parse_tile_config


class VideoTileProcessLoop:
    """
    Process all tiles sequentially in a single node.
    Loops through each tile, applies processing (or passthrough), merges.
    No external dependencies - fully self-contained.
    
    For custom processing: connect this node's output to your processing chain,
    then use the parallel workflow (Slice -> process each tile -> Merge).
    This node provides the sequential loop for when you need to process
    tiles one-at-a-time with minimal RAM.
    
    Processing modes:
    - passthrough: merge tiles as-is (no processing, for testing)
    - clip: clamp values to 0-1 before merge
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tile_config": ("TILE_CONFIG", {"forceInput": True}),
                "processing_mode": (["passthrough", "clip"], {"default": "passthrough"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("IMAGE", "tiles_processed")
    FUNCTION = "process_loop"
    CATEGORY = "Video Tiler"

    def process_loop(self, images: torch.Tensor, tile_config: tuple, processing_mode: str):
        _, _, _, _, tile_specs = parse_tile_config(tile_config)
        processed_tiles = []

        for i in range(len(tile_specs)):
            tile = get_tile_by_index(images, tile_config, i)
            if processing_mode == "clip":
                tile = torch.clamp(tile, 0.0, 1.0)
            processed_tiles.append(tile)

        result = merge_tiles(processed_tiles, tile_config)
        return (result, len(processed_tiles))
