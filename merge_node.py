"""
Video Tile Merge node - reconstructs video from processed tiles.
Uses shared output tensor, feather blending for overlaps. No duplicate storage.
"""

import torch

from .tile_config import parse_tile_config


def _feather_mask(w: int, h: int, feather: float) -> torch.Tensor:
    """Create feather mask: 1 in center, falloff at edges. Shape (h, w)."""
    if feather <= 0:
        return torch.ones((h, w), dtype=torch.float32)

    y = torch.linspace(0, 1, h)
    x = torch.linspace(0, 1, w)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    # Distance from each edge (0 at edge, 1 at center)
    left = xx * (w / max(feather, 1))
    right = (1 - xx) * (w / max(feather, 1))
    top = yy * (h / max(feather, 1))
    bottom = (1 - yy) * (h / max(feather, 1))

    # Clamp and min across edges
    left = torch.clamp(left, 0, 1)
    right = torch.clamp(right, 0, 1)
    top = torch.clamp(top, 0, 1)
    bottom = torch.clamp(bottom, 0, 1)
    mask = torch.minimum(torch.minimum(left, right), torch.minimum(top, bottom))
    return mask


def merge_tiles(
    tiles: list[torch.Tensor],
    config_tuple: tuple,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Merge processed tiles into output. Writes into preallocated tensor.
    Overlap tiles are feathered on top of normal tiles.
    """
    width, height, multiple, feather, tile_specs = parse_tile_config(config_tuple)

    # Use first tile to get batch size and channels
    B = tiles[0].shape[0]
    C = tiles[0].shape[3]

    # Allocate output (single allocation - "before" state)
    output = torch.zeros((B, height, width, C), dtype=tiles[0].dtype, device=tiles[0].device)

    # Sort by order: normal first, then overlaps
    sorted_specs = sorted(tile_specs, key=lambda t: t.order)

    for idx, spec in enumerate(sorted_specs):
        if idx >= len(tiles):
            break
        tile = tiles[idx]
        x, y, w, h = spec.x, spec.y, spec.w, spec.h

        if spec.type == "normal":
            # Normal tile: direct copy (no feather)
            output[:, y : y + h, x : x + w, :] = tile
        else:
            # Overlap tile: feather blend on top
            mask = _feather_mask(w, h, feather).to(tile.device)
            mask = mask.reshape(1, h, w, 1)
            existing = output[:, y : y + h, x : x + w, :]
            blended = existing * (1 - mask) + tile * mask
            output[:, y : y + h, x : x + w, :] = blended

    return output


class VideoTileMerge:
    """Merge processed tiles back into video. Uses tile_config from Slice - no separate settings."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_config": ("TILE_CONFIG", {"forceInput": True}),
                "tiles": ("IMAGE", {"multiple": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "merge"
    CATEGORY = "Video Tiler"

    def merge(self, tile_config, tiles):
        if isinstance(tiles, (list, tuple)):
            tiles_list = list(tiles)
        else:
            tiles_list = [tiles]
        result = merge_tiles(tiles_list, tile_config)
        return (result,)
