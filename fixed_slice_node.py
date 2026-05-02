"""
Fixed tile-size slicer with fractional overlap and traversal patterns.
TILE_CONFIG v5 — edge blur is set on Video Tile Merge only.
"""

from __future__ import annotations

import numpy as np
import torch

from .fixed_layout import (
    compute_fixed_layout,
    create_fixed_traversal_viz_fill,
    create_fixed_traversal_viz_labels,
    fixed_layout_label_string,
)
from .layout import TileSpec
from .slice_node import combine_layout_and_memory, estimate_peak_memory, slice_tiles_as_views, _get_font


def _build_fixed_visualization(
    width: int,
    height: int,
    tiles: list[TileSpec],
    info_lines: list[str],
) -> torch.Tensor:
    from PIL import Image, ImageDraw

    img1 = create_fixed_traversal_viz_labels(width, height, tiles)
    img2 = create_fixed_traversal_viz_fill(width, height, tiles, 0.0)
    pil1 = Image.fromarray((img1 * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil1)
    font = _get_font(draw, width)
    pad = 6
    try:
        bbox = draw.textbbox((0, 0), "Mg", font=font)
        line_h = bbox[3] - bbox[1] + 4
    except Exception:
        line_h = 14
    max_w = 0
    for line in info_lines:
        try:
            bb = draw.textbbox((0, 0), line, font=font)
            max_w = max(max_w, bb[2] - bb[0])
        except Exception:
            max_w = max(max_w, len(line) * 8)
    box_x, box_y = 8, 8
    box_w = max(140, int(max_w) + pad * 2)
    box_h = len(info_lines) * line_h + pad * 2
    draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(30, 30, 30), outline=(100, 100, 100))
    for i, line in enumerate(info_lines):
        draw.text((box_x + pad, box_y + pad + i * line_h), line, fill=(255, 255, 255), font=font)
    img1 = np.array(pil1).astype(np.float32) / 255.0
    stacked = np.stack([img1, img2], axis=0).astype(np.float32)
    return torch.from_numpy(stacked)


def _parse_overlap(label: str) -> tuple[int, int]:
    lut = {"1/8": (1, 8), "1/4": (1, 4), "3/8": (3, 8), "1/2": (1, 2)}
    return lut.get(label, (1, 4))


class VideoTileSliceFixed:
    """Fixed tile size, fractional overlap, row/column/spiral/double-spiral order."""

    DESCRIPTION = (
        "Fixed-width/height tiles with fractional minimum overlap and traversal pattern. "
        "Emits the same sockets as the variable grid slicer; TILE_CONFIG is v5 (Merge applies feather)."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {"tooltip": "Video or image batch [B,H,W,C]."},
                ),
                "tile_width": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "tooltip": "Tile width in pixels (step 8).",
                    },
                ),
                "tile_height": (
                    "INT",
                    {
                        "default": 512,
                        "min": 64,
                        "max": 8192,
                        "step": 8,
                        "tooltip": "Tile height in pixels (step 8).",
                    },
                ),
                "multiple": (
                    "INT",
                    {
                        "default": 32,
                        "min": 1,
                        "max": 64,
                        "tooltip": "Stride/overlap positions snap to this pixel grid.",
                    },
                ),
                "overlap": (
                    ["1/8", "1/4", "3/8", "1/2"],
                    {
                        "default": "1/4",
                        "tooltip": "Minimum neighbour overlap as a fraction of tile width/height per axis (layout chooses fewest tiles that satisfy it).",
                    },
                ),
                "pattern": (
                    ["row", "column", "spiral", "double_spiral"],
                    {
                        "default": "row",
                        "tooltip": "Tile visitation order (affects traversal viz and merge paint order for overlaps).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "TILE_CONFIG", "IMAGE", "INT", "STRING")
    RETURN_NAMES = ("tiles", "tile_config", "visualization", "tile_count", "layout_label")
    OUTPUT_TOOLTIPS = (
        "Tile list for list-mode processing → Video Tile Merge.",
        "TILE_CONFIG v5 for Merge / Get Tile / Reference Tile Slice.",
        "Preview: labels + traversal gradient (not final output).",
        "Tile count.",
        "Layout string + memory hints.",
    )
    OUTPUT_IS_LIST = (True, False, False, False, False)
    FUNCTION = "slice_fixed"
    CATEGORY = "Video Tiler"

    def slice_fixed(
        self,
        images: torch.Tensor,
        tile_width: int,
        tile_height: int,
        multiple: int,
        overlap: str,
        pattern: str,
    ):
        B, H, W, C = images.shape
        batch_size = int(B)
        on, od = _parse_overlap(overlap)
        tiles, config_tuple = compute_fixed_layout(
            W,
            H,
            tile_width,
            tile_height,
            multiple,
            on,
            od,
            pattern,  # type: ignore[arg-type]
        )
        tw = config_tuple[3]
        th = config_tuple[4]
        ox = config_tuple[5]
        oy = config_tuple[6]

        tile_tensors = slice_tiles_as_views(images, tiles)
        mem_est = estimate_peak_memory(W, H, tiles, batch_size)
        layout_core = fixed_layout_label_string(
            W, H, tw, th, ox, oy, pattern, len(tiles),
        )
        layout_full = combine_layout_and_memory(layout_core, mem_est)
        info_lines = layout_core.split("\n")
        viz = _build_fixed_visualization(W, H, tiles, info_lines)

        print(f"[Video Tiler] Fixed slice: {W}x{H} → {len(tiles)} tiles ({pattern})")
        print(mem_est)

        return (tile_tensors, config_tuple, viz, len(tile_tensors), layout_full)
