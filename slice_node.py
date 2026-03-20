"""
Video Tile Slice node - splits video into tiles using views (no copy).
"""

import numpy as np
import torch

from .layout import TileSpec
from .tile_config import create_tile_config


BYTES_PER_PIXEL = 16  # float32, 4 channels (IMAGE)


def estimate_peak_memory(
    width: int, height: int, tiles: list[TileSpec], batch_size: int = 1,
) -> str:
    """
    Rough estimate of peak RAM/VRAM for VAE encode (per tile), VAE decode (per tile), and merge.
    Assumes float32, 4 channels. VAE latent: 8x downscale, 4 channels.
    """
    B = max(1, batch_size)
    total_tiles_pixels = sum(t.w * t.h for t in tiles)
    largest_tile_pixels = max(t.w * t.h for t in tiles) if tiles else 0

    # Merge: output buffer + all tiles held (ComfyUI collects before merge)
    merge_output = B * width * height * BYTES_PER_PIXEL
    merge_tiles = total_tiles_pixels * B * BYTES_PER_PIXEL
    merge_peak = merge_output + merge_tiles

    # VAE encode: input image + latent output + intermediate. ~2x input + latent for rough estimate.
    latent_scale = 8  # typical 8x downscale
    vae_encode_peak = B * largest_tile_pixels * BYTES_PER_PIXEL * 2  # input + output/intermediate

    # VAE decode: latent input + image output
    vae_decode_peak = (
        B * (largest_tile_pixels // (latent_scale * latent_scale)) * BYTES_PER_PIXEL
        + B * largest_tile_pixels * BYTES_PER_PIXEL
    )

    def _fmt(b: float) -> str:
        if b >= 1e9:
            return f"{b / 1e9:.1f} GB"
        return f"{b / 1e6:.0f} MB"

    return (
        f"Est. peak:\n"
        f"  VAE encode/tile: {_fmt(vae_encode_peak)}\n"
        f"  VAE decode/tile: {_fmt(vae_decode_peak)}\n"
        f"  Merge: {_fmt(merge_peak)} (output + {len(tiles)} tiles)"
    )


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


def _tile_color(t: TileSpec) -> tuple[float, float, float]:
    """RGB 0-1: green normal, orange H/V, magenta corner."""
    if t.type == "normal":
        return (0, 0.8, 0)
    if t.type == "overlap_h" or t.type == "overlap_v":
        return (1.0, 0.5, 0)
    return (1.0, 0, 0.8)


def create_visualization_labels(
    width: int, height: int, tiles: list[TileSpec],
) -> np.ndarray:
    """Viz 1: Borders only, tile number in all four corners. Green/orange/magenta by type."""
    from PIL import Image, ImageDraw, ImageFont

    img = np.ones((height, width, 3), dtype=np.float32) * 0.95

    for idx, t in enumerate(tiles):
        x1, y1 = t.x, t.y
        x2, y2 = t.x + t.w, t.y + t.h
        color = list(_tile_color(t))
        # Borders only
        img[y1:y2, x1:x1 + 2, :] = color
        img[y1:y2, x2 - 2 : x2, :] = color
        img[y1 : y1 + 2, x1:x2, :] = color
        img[y2 - 2 : y2, x1:x2, :] = color

    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(draw, width)

    for idx, t in enumerate(tiles):
        x1, y1 = t.x, t.y
        x2, y2 = t.x + t.w, t.y + t.h
        label = f"{idx:02d}"
        rgb = tuple(int(c * 255) for c in _tile_color(t))
        pad = 4
        for (lx, ly) in [(x1 + pad, y1 + pad), (x2 - 24, y1 + pad), (x1 + pad, y2 - 16), (x2 - 24, y2 - 16)]:
            draw.text((lx, ly), label, fill=rgb, font=font)

    return np.array(pil_img).astype(np.float32) / 255.0


def create_visualization_types(
    width: int, height: int, tiles: list[TileSpec], feather: float,
) -> np.ndarray:
    """Viz 2: Tile type as topmost color (green/orange/magenta). Feather regions show gradient."""
    # Sort by order so topmost wins
    sorted_tiles = sorted(tiles, key=lambda t: t.order)
    img = np.ones((height, width, 3), dtype=np.float32) * 0.95

    fe = max(1, int(feather))

    for t in sorted_tiles:
        x1, y1 = t.x, t.y
        x2, y2 = t.x + t.w, t.y + t.h
        color = np.array(_tile_color(t), dtype=np.float32)

        if t.type == "normal":
            img[y1:y2, x1:x2, :] = color
        else:
            # Overlap: solid in center, gradient in feather toward adjacent tile color
            # H/V overlap borders green → gradient orange↔green
            # Corner overlaps border orange → gradient magenta↔orange
            h, w = y2 - y1, x2 - x1
            yy = np.linspace(0, 1, h)
            xx = np.linspace(0, 1, w)
            yy, xx = np.meshgrid(yy, xx, indexing="ij")
            left = np.clip(xx * (w / fe), 0, 1) if x1 > 0 else np.ones((h, w))
            right = np.clip((1 - xx) * (w / fe), 0, 1) if x2 < width else np.ones((h, w))
            top = np.clip(yy * (h / fe), 0, 1) if y1 > 0 else np.ones((h, w))
            bottom = np.clip((1 - yy) * (h / fe), 0, 1) if y2 < height else np.ones((h, w))
            # Linear falloff (no ease-in/ease-out) to avoid visible seams
            mask = np.minimum(np.minimum(left, right), np.minimum(top, bottom))
            mask = np.expand_dims(mask, axis=-1)
            if t.type == "overlap_corner":
                edge_color = np.array([1.0, 0.5, 0.0], dtype=np.float32)  # orange
            else:
                edge_color = np.array([0, 0.8, 0], dtype=np.float32)  # green
            blended = color * mask + edge_color * (1 - mask)
            img[y1:y2, x1:x2, :] = blended

    return img


def create_visualization(
    width: int,
    height: int,
    tiles: list[TileSpec],
    feather: float,
    tiles_x: int,
    tiles_y: int,
    multiple: int,
    overlap_extension_x: int,
    overlap_extension_y: int,
) -> torch.Tensor:
    """
    Create two visualization images as batch:
    [0] Borders only, tile labels in 4 corners (green/orange/magenta by type)
    [1] Tile type colors (topmost) with feather gradient
    """
    from PIL import Image, ImageDraw, ImageFont

    img1 = create_visualization_labels(width, height, tiles)
    img2 = create_visualization_types(width, height, tiles, feather)

    # Add info box to first image
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
        f"Normal: {tile_w}x{tile_h}",
        f"Gap: {gap_x}x{gap_y}",
        f"Feather: {feather}",
        f"Tiles: {len(tiles)}",
    ]
    if ov_h:
        lines.append(f"Overlap H: {ov_h.w}x{ov_h.h}")
    if ov_c:
        lines.append(f"Overlap corner: {ov_c.w}x{ov_c.h}")

    pil1 = Image.fromarray((img1 * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil1)
    font = _get_font(draw, width)
    pad = 6
    try:
        bbox = draw.textbbox((0, 0), "Mg", font=font)
        line_h = bbox[3] - bbox[1] + 4
    except Exception:
        line_h = 14
    box_x, box_y = 8, 8
    box_w, box_h = 140, len(lines) * line_h + pad * 2
    draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(30, 30, 30), outline=(100, 100, 100))
    for i, line in enumerate(lines):
        draw.text((box_x + pad, box_y + pad + i * line_h), line, fill=(255, 255, 255), font=font)
    img1 = np.array(pil1).astype(np.float32) / 255.0

    stacked = np.stack([img1, img2], axis=0).astype(np.float32)
    return torch.from_numpy(stacked)


class VideoTileSlice:
    """Slice video/image batch into tiles with gaps and overlaps. Outputs a single tiles list (no copy)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "tiles_x": ("INT", {"default": 2, "min": 1, "max": 5}),
                "tiles_y": ("INT", {"default": 2, "min": 1, "max": 5}),
                "multiple": ("INT", {"default": 32, "min": 1, "max": 64}),
                "overlap_extension_x": ("INT", {"default": 128, "min": 0, "max": 4096}),
                "overlap_extension_y": ("INT", {"default": 128, "min": 0, "max": 4096}),
                "feather": ("FLOAT", {"default": 64.0, "min": 0.0, "max": 128.0}),
            },
        }

    RETURN_TYPES = ("IMAGE", "TILE_CONFIG", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("tiles", "tile_config", "visualization", "tile_count", "memory_estimate")
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
        feather: float,
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
            feather=feather,
        )

        tile_tensors = slice_tiles_as_views(images, tiles)
        viz = create_visualization(
            W, H, tiles, feather,
            tiles_x=tiles_x, tiles_y=tiles_y,
            multiple=multiple,
            overlap_extension_x=overlap_extension_x,
            overlap_extension_y=overlap_extension_y,
        )

        mem_est = estimate_peak_memory(W, H, tiles, batch_size)
        print(f"[Video Tiler] Slice: {W}x{H} → {len(tiles)} tiles ({tiles_x}x{tiles_y} grid)")
        print(mem_est)

        return (tile_tensors, config_tuple, viz, len(tile_tensors), mem_est)
