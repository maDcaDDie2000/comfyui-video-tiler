"""
Video Tile Slice node - splits video into tiles using views (no copy).
"""

import torch

from .layout import TileSpec
from .tile_config import create_tile_config


def slice_tiles_as_views(images: torch.Tensor, tiles: list[TileSpec]) -> list[torch.Tensor]:
    """Extract each tile as a view - no memory copy."""
    result = []
    for t in tiles:
        tile = images[:, t.y : t.y + t.h, t.x : t.x + t.w, :]
        result.append(tile)
    return result


def create_visualization(
    width: int,
    height: int,
    tiles: list[TileSpec],
    feather: float,
    batch_size: int = 1,
) -> torch.Tensor:
    """
    Create single image outlining tiles and feather regions.
    Same resolution as input. RGB, values 0-1.
    """
    import numpy as np

    # Create canvas (white background)
    img = np.ones((height, width, 3), dtype=np.float32) * 0.95

    # Draw normal tiles - green outline
    for t in tiles:
        if t.type == "normal":
            x1, y1 = t.x, t.y
            x2, y2 = t.x + t.w, t.y + t.h
            img[y1:y2, x1:x1 + 2, :] = [0, 0.8, 0]  # left
            img[y1:y2, x2 - 2 : x2, :] = [0, 0.8, 0]  # right
            img[y1 : y1 + 2, x1:x2, :] = [0, 0.8, 0]  # top
            img[y2 - 2 : y2, x1:x2, :] = [0, 0.8, 0]  # bottom

    # Draw overlap tiles - blue outline
    for t in tiles:
        if t.type != "normal":
            x1, y1 = t.x, t.y
            x2, y2 = t.x + t.w, t.y + t.h
            img[y1:y2, x1:x1 + 2, :] = [0, 0.4, 1.0]
            img[y1:y2, x2 - 2 : x2, :] = [0, 0.4, 1.0]
            img[y1 : y1 + 2, x1:x2, :] = [0, 0.4, 1.0]
            img[y2 - 2 : y2, x1:x2, :] = [0, 0.4, 1.0]

    # Draw feather region (semi-transparent overlay at tile edges)
    if feather > 0:
        for t in tiles:
            if t.type != "normal":
                x1, y1 = t.x, t.y
                x2, y2 = t.x + t.w, t.y + t.h
                fe = int(feather)
                if fe > 0:
                    # Feather zone - light red tint
                    overlay = np.array([1.0, 0.7, 0.7], dtype=np.float32)
                    for i in range(min(fe, t.w // 2)):
                        alpha = 0.15 * (1 - i / max(fe, 1))
                        img[y1:y2, x1 + i, :] = img[y1:y2, x1 + i, :] * (1 - alpha) + overlay * alpha
                        img[y1:y2, x2 - 1 - i, :] = img[y1:y2, x2 - 1 - i, :] * (1 - alpha) + overlay * alpha
                    for i in range(min(fe, t.h // 2)):
                        alpha = 0.15 * (1 - i / max(fe, 1))
                        img[y1 + i, x1:x2, :] = img[y1 + i, x1:x2, :] * (1 - alpha) + overlay * alpha
                        img[y2 - 1 - i, x1:x2, :] = img[y2 - 1 - i, x1:x2, :] * (1 - alpha) + overlay * alpha

    # Batch dimension
    img_batch = np.tile(img[np.newaxis, ...], (batch_size, 1, 1, 1))
    return torch.from_numpy(img_batch)


class VideoTileSlice:
    """Slice video/image batch into tiles with gaps and overlaps. Outputs a single tiles list (no copy)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tiles_x": ("INT", {"default": 2, "min": 1, "max": 5}),
                "tiles_y": ("INT", {"default": 2, "min": 1, "max": 5}),
                "multiple": ("INT", {"default": 16, "min": 1, "max": 64}),
                "overlap_extension": ("INT", {"default": 32, "min": 0, "max": 256}),
                "feather": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 64.0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "TILE_CONFIG", "IMAGE", "INT")
    RETURN_NAMES = ("tiles", "tile_config", "visualization", "tile_count")
    OUTPUT_IS_LIST = (True, False, False, False)
    FUNCTION = "slice"
    CATEGORY = "Video Tiler"

    def slice(
        self,
        images: torch.Tensor,
        tiles_x: int,
        tiles_y: int,
        multiple: int,
        overlap_extension: int,
        feather: float,
    ):
        B, H, W, C = images.shape
        tiles, config_tuple = create_tile_config(
            width=W,
            height=H,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            multiple=multiple,
            overlap_extension=overlap_extension,
            feather=feather,
        )

        tile_tensors = slice_tiles_as_views(images, tiles)
        viz = create_visualization(W, H, tiles, feather, B)

        return (tile_tensors, config_tuple, viz, len(tile_tensors))
