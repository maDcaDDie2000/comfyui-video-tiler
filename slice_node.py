"""
Video Tile Slice node - splits video into tiles using views (no copy).
Feathering is applied in Video Tile Merge — TILE_CONFIG is geometry only (cache-friendly).
"""

import numpy as np
import torch

from .fixed_layout import order_gradient_rgb
from .layout import TileSpec
from .tile_config import create_tile_config

BYTES_PER_PIXEL = 16  # float32, 4 channels (IMAGE)


def estimate_peak_memory(
    width: int, height: int, tiles: list[TileSpec], batch_size: int = 1,
) -> str:
    """
    Rough peak memory hints. VAE stages are usually VRAM on GPU; merge can be VRAM (CUDA) or RAM (CPU).
    """
    B = max(1, batch_size)
    total_tiles_pixels = sum(t.w * t.h for t in tiles)
    largest_tile_pixels = max(t.w * t.h for t in tiles) if tiles else 0

    merge_output = B * width * height * BYTES_PER_PIXEL
    merge_tiles = total_tiles_pixels * B * BYTES_PER_PIXEL
    merge_peak = merge_output + merge_tiles

    latent_scale = 8
    vae_encode_peak = B * largest_tile_pixels * BYTES_PER_PIXEL * 2

    vae_decode_peak = (
        B * (largest_tile_pixels // (latent_scale * latent_scale)) * BYTES_PER_PIXEL
        + B * largest_tile_pixels * BYTES_PER_PIXEL
    )

    def _fmt(b: float) -> str:
        if b >= 1e9:
            return f"{b / 1e9:.1f} GB"
        return f"{b / 1e6:.0f} MB"

    return (
        f"Memory estimates (approx.):\n"
        f"  [typically VRAM] VAE encode / tile: {_fmt(vae_encode_peak)}\n"
        f"  [typically VRAM] VAE decode / tile: {_fmt(vae_decode_peak)}\n"
        f"  [VRAM if merge on CUDA, else RAM] Merge: {_fmt(merge_peak)} (output + {len(tiles)} tiles)"
    )


def combine_layout_and_memory(layout_text: str, memory_text: str) -> str:
    return f"{layout_text}\n\n---\n{memory_text}"


def slice_tiles_as_views(images: torch.Tensor, tiles: list[TileSpec]) -> list[torch.Tensor]:
    """Extract each tile as a view - no memory copy."""
    result = []
    for t in tiles:
        tile = images[:, t.y : t.y + t.h, t.x : t.x + t.w, :]
        result.append(tile)
    return result


def _get_font(draw, width: int):
    import os
    from PIL import ImageFont
    font_size = max(10, min(14, width // 60))
    candidates = [
        os.path.join(os.environ.get("WINDIR", ""), "Fonts", "arial.ttf"),
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "arial.ttf",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                pass
    return ImageFont.load_default()


def _tile_rank_by_traversal(tiles: list[TileSpec]) -> dict[int, int]:
    indices = sorted(range(len(tiles)), key=lambda i: (tiles[i].order, tiles[i].y, tiles[i].x))
    return {idx: rank for rank, idx in enumerate(indices)}


def create_grid_traversal_viz_labels(
    width: int,
    height: int,
    tiles: list[TileSpec],
) -> np.ndarray:
    """Borders + indices; fill color follows traversal order (same gradient idea as fixed slicer)."""
    from PIL import Image, ImageDraw

    n = len(tiles)
    rank_of = _tile_rank_by_traversal(tiles)
    img = np.ones((height, width, 3), dtype=np.float32) * 0.95
    for ti, t in enumerate(tiles):
        x1, y1 = t.x, t.y
        x2, y2 = t.x + t.w, t.y + t.h
        rk = rank_of[ti]
        color = list(order_gradient_rgb(rk, max(n, 1)))
        img[y1:y2, x1:x1 + 2, :] = color
        img[y1:y2, x2 - 2 : x2, :] = color
        img[y1 : y1 + 2, x1:x2, :] = color
        img[y2 - 2 : y2, x1:x2, :] = color

    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(draw, width)
    for ti, t in enumerate(tiles):
        x1, y1 = t.x, t.y
        x2, y2 = t.x + t.w, t.y + t.h
        rk = rank_of[ti]
        label = f"{rk:02d}"
        rgb = tuple(int(c * 255) for c in order_gradient_rgb(rk, max(n, 1)))
        pad = 4
        for (lx, ly) in [(x1 + pad, y1 + pad), (x2 - 24, y1 + pad), (x1 + pad, y2 - 16), (x2 - 24, y2 - 16)]:
            draw.text((lx, ly), label, fill=rgb, font=font)
    return np.array(pil_img).astype(np.float32) / 255.0


def create_grid_traversal_viz_fill(
    width: int,
    height: int,
    tiles: list[TileSpec],
) -> np.ndarray:
    """Solid fill per tile: later traversal rank paints over overlaps (gradient hues)."""
    n = len(tiles)
    if n == 0:
        return np.ones((height, width, 3), dtype=np.float32) * 0.95
    rank_of = _tile_rank_by_traversal(tiles)
    img = np.ones((height, width, 3), dtype=np.float32) * 0.95
    order_paint = sorted(range(n), key=lambda ti: rank_of[ti])
    for ti in order_paint:
        t = tiles[ti]
        x1, y1 = t.x, t.y
        x2, y2 = t.x + t.w, t.y + t.h
        rk = rank_of[ti]
        color = np.array(order_gradient_rgb(rk, max(n, 1)), dtype=np.float32)
        img[y1:y2, x1:x2, :] = color
    return img


def tile_layout_label_string(
    width: int,
    height: int,
    tiles: list[TileSpec],
    tiles_x: int,
    tiles_y: int,
    multiple: int,
    overlap_extension_x: int,
    overlap_extension_y: int,
) -> str:
    """Multiline label: tile sizes and layout (same content as viz info box)."""
    normals = [t for t in tiles if t.type == "normal"]
    overlaps = [t for t in tiles if t.type != "normal"]
    tile_w = normals[0].w if normals else 0
    tile_h = normals[0].h if normals else 0
    gap_x = int(normals[1].x - normals[0].x - normals[0].w) if tiles_x > 1 and len(normals) > 1 else 0
    gap_y = int(normals[tiles_x].y - normals[0].y - normals[0].h) if tiles_y > 1 and len(normals) > tiles_x else 0
    ov_h = next((t for t in overlaps if t.type == "overlap_h"), None)
    ov_v = next((t for t in overlaps if t.type == "overlap_v"), None)
    ov_c = next((t for t in overlaps if t.type == "overlap_corner"), None)
    lines = [
        f"Output: {width}x{height}",
        f"Grid: {tiles_x}x{tiles_y}",
        f"Normal tile: {tile_w}x{tile_h}",
        f"Gap: {gap_x}x{gap_y}",
        f"Overlap ext: {overlap_extension_x}x{overlap_extension_y}",
        f"Multiple: {multiple}",
        f"Total tiles: {len(tiles)}",
        f"Feather: Video Tile Merge — fraction of tile W/H (max 50%)",
    ]
    if ov_h:
        lines.append(f"Overlap H: {ov_h.w}x{ov_h.h}")
    if ov_v:
        lines.append(f"Overlap V: {ov_v.w}x{ov_v.h}")
    if ov_c:
        lines.append(f"Overlap corner: {ov_c.w}x{ov_c.h}")
    return "\n".join(lines)


def create_visualization(
    width: int,
    height: int,
    tiles: list[TileSpec],
    tiles_x: int,
    tiles_y: int,
    multiple: int,
    overlap_extension_x: int,
    overlap_extension_y: int,
) -> torch.Tensor:
    """
    Two visualization images as batch:
    [0] Borders + traversal-order gradient (labels)
    [1] Filled regions, traversal gradient (later order on top)
    """
    from PIL import Image, ImageDraw

    img1_arr = create_grid_traversal_viz_labels(width, height, tiles)
    img2_arr = create_grid_traversal_viz_fill(width, height, tiles)

    label = tile_layout_label_string(
        width, height, tiles,
        tiles_x, tiles_y, multiple,
        overlap_extension_x, overlap_extension_y,
    )
    lines = label.split("\n")

    pil1 = Image.fromarray((img1_arr * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil1)
    font = _get_font(draw, width)
    pad = 6
    try:
        bbox = draw.textbbox((0, 0), "Mg", font=font)
        line_h = bbox[3] - bbox[1] + 4
    except Exception:
        line_h = 14
    max_w = 0
    for line in lines:
        try:
            bb = draw.textbbox((0, 0), line, font=font)
            max_w = max(max_w, bb[2] - bb[0])
        except Exception:
            max_w = max(max_w, len(line) * 8)
    box_x, box_y = 8, 8
    box_w = max(140, int(max_w) + pad * 2)
    box_h = len(lines) * line_h + pad * 2
    draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(30, 30, 30), outline=(100, 100, 100))
    for i, line in enumerate(lines):
        draw.text((box_x + pad, box_y + pad + i * line_h), line, fill=(255, 255, 255), font=font)
    img1 = np.array(pil1).astype(np.float32) / 255.0

    stacked = np.stack([img1, img2_arr], axis=0).astype(np.float32)
    return torch.from_numpy(stacked)


class VideoTileSlice:
    """Slice video/image batch into tiles with gaps and overlaps. Outputs a single tiles list (no copy)."""

    DESCRIPTION = (
        "Splits each frame into a tiles_x × tiles_y grid with gaps and overlap seam tiles. "
        "Tiles are tensor views (memory-efficient). Feather blending is done in Video Tile Merge."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "Video or image batch [B,H,W,C]. Same frame size for all batches."},
                ),
                "tiles_x": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 5,
                        "tooltip": "Horizontal grid cells (1–5). Defines how many columns of normal tiles.",
                    },
                ),
                "tiles_y": (
                    "INT",
                    {
                        "default": 2,
                        "min": 1,
                        "max": 5,
                        "tooltip": "Vertical grid cells (1–5). Defines how many rows of normal tiles.",
                    },
                ),
                "multiple": (
                    "INT",
                    {
                        "default": 32,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Snap tile sizes and overlap geometry to this pixel multiple (e.g. 8, 16, 32).",
                    },
                ),
                "overlap_extension_x": (
                    "INT",
                    {
                        "default": 128,
                        "min": 0,
                        "max": 4096,
                        "tooltip": "Horizontal overlap seam thickness (pixels), snapped to multiple. Seam tiles blend in Merge.",
                    },
                ),
                "overlap_extension_y": (
                    "INT",
                    {
                        "default": 128,
                        "min": 0,
                        "max": 4096,
                        "tooltip": "Vertical overlap seam thickness (pixels), snapped to multiple.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "TILE_CONFIG", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("tiles", "tile_config", "visualization", "tile_count", "layout_label")
    OUTPUT_TOOLTIPS = (
        "List of tile crops (OUTPUT_IS_LIST). Wire to processing, then Video Tile Merge tiles input.",
        "Opaque layout tuple for Merge / Get Tile / Reference Tile Slice. Geometry only; changing Merge feather does not require re-slicing.",
        "Two preview images: bordered layout + traversal gradient (not for final encode).",
        "Number of tiles in the list.",
        "Human-readable layout summary and rough memory hints.",
    )
    OUTPUT_IS_LIST = (True, False, False, False, False)
    FUNCTION = "slice"
    CATEGORY = "Video Tiler"

    def slice(
        self,
        images: torch.Tensor,
        tiles_x: int,
        tiles_y: int,
        multiple: int,
        overlap_extension_x: int,
        overlap_extension_y: int,
    ):
        B, H, W, C = images.shape
        batch_size = int(B)
        tiles, config_tuple = create_tile_config(
            width=W,
            height=H,
            tiles_x=tiles_x,
            tiles_y=tiles_y,
            multiple=multiple,
            overlap_extension_x=overlap_extension_x,
            overlap_extension_y=overlap_extension_y,
        )

        tile_tensors = slice_tiles_as_views(images, tiles)
        viz = create_visualization(
            W, H, tiles,
            tiles_x=tiles_x, tiles_y=tiles_y,
            multiple=multiple,
            overlap_extension_x=overlap_extension_x,
            overlap_extension_y=overlap_extension_y,
        )

        mem_est = estimate_peak_memory(W, H, tiles, batch_size)
        layout_core = tile_layout_label_string(
            W, H, tiles,
            tiles_x, tiles_y, multiple,
            overlap_extension_x, overlap_extension_y,
        )
        layout_full = combine_layout_and_memory(layout_core, mem_est)
        print(f"[Video Tiler] Slice: {W}x{H} → {len(tiles)} tiles ({tiles_x}x{tiles_y} grid)")
        print(mem_est)

        return (tile_tensors, config_tuple, viz, len(tile_tensors), layout_full)
