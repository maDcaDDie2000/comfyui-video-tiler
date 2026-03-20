"""
Video Tile Merge node - reconstructs video from processed tiles.
Uses shared output tensor, feather blending for overlaps. No duplicate storage.
"""

import torch

from .tile_config import parse_tile_config


def _feather_mask(
    w: int, h: int, feather: float,
    x: int, y: int, img_width: int, img_height: int,
) -> torch.Tensor:
    """Create feather mask: 1 in center, falloff at edges. No feather on edges touching image border. Shape (h, w)."""
    if feather <= 0:
        return torch.ones((h, w), dtype=torch.float32)

    yy = torch.linspace(0, 1, h)
    xx = torch.linspace(0, 1, w)
    yy, xx = torch.meshgrid(yy, xx, indexing="ij")

    # Distance from each edge (0 at edge, 1 at center). Use 1 (no feather) if edge touches image border.
    feather_w = max(feather, 1)
    feather_h = max(feather, 1)
    left = torch.ones((h, w), dtype=torch.float32) if x == 0 else torch.clamp(xx * (w / feather_w), 0, 1)
    right = torch.ones((h, w), dtype=torch.float32) if x + w == img_width else torch.clamp((1 - xx) * (w / feather_w), 0, 1)
    top = torch.ones((h, w), dtype=torch.float32) if y == 0 else torch.clamp(yy * (h / feather_h), 0, 1)
    bottom = torch.ones((h, w), dtype=torch.float32) if y + h == img_height else torch.clamp((1 - yy) * (h / feather_h), 0, 1)

    # Geometric mean: gradient flows smoothly away from all edges toward center (no spiky corners)
    mask = (left * right * top * bottom) ** 0.25
    return mask


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

    tiles_list = [_normalize_tile(t) for t in tiles]
    if not tiles_list:
        raise ValueError("No tiles to merge - Tile Loop produced empty list")
    first = tiles_list[0]
    if len(first.shape) < 4:
        raise ValueError(f"Tile shape {first.shape} - expected (B,H,W,C)")
    B, C = first.shape[0], first.shape[3]

    # Prefer CUDA for merge to avoid exhausting CPU RAM on large outputs
    if first.is_cuda:
        device = first.device
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = first.device

    output = torch.zeros((B, height, width, C), dtype=first.dtype, device=device)

    # Sort by order: normal first, then overlaps
    sorted_specs = sorted(tile_specs, key=lambda t: t.order)

    for idx, spec in enumerate(sorted_specs):
        if idx >= len(tiles_list):
            break
        tile = tiles_list[idx].to(device)
        x, y, w, h = spec.x, spec.y, spec.w, spec.h

        if spec.type == "normal":
            # Normal tile: direct copy (no feather)
            output[:, y : y + h, x : x + w, :] = tile
        else:
            # Overlap tile: feather blend on top (no feather on edges touching image border)
            mask = _feather_mask(w, h, feather, x, y, width, height).to(tile.device)
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
                "tiles": ("IMAGE",),
            },
        }

    INPUT_IS_LIST = (False, True)  # tiles: list from Sequential Batcher–style iteration
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "merge"
    CATEGORY = "Video Tiler"

    def merge(self, tile_config, tiles):
        # ComfyUI may pass tile_config as list (one per iteration) when in list context
        if isinstance(tile_config, (list, tuple)) and len(tile_config) > 0:
            tile_config = tile_config[0]
        if isinstance(tiles, (list, tuple)):
            tiles_list = list(tiles)
        else:
            tiles_list = [tiles]
        print(f"[Video Tiler] Merging {len(tiles_list)} tiles → single IMAGE batch")
        result = merge_tiles(tiles_list, tile_config)
        return (result,)
