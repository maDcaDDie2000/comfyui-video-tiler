"""
TILE_CONFIG - ComfyUI-friendly serializable config for tile layout.

Passed from Slice node to Merge node. Contains only layout metadata (positions),
no tensor data - minimal memory.
"""

from __future__ import annotations

from typing import Any

from .layout import TileSpec, compute_layout


def create_tile_config(
    width: int,
    height: int,
    tiles_x: int,
    tiles_y: int,
    multiple: int,
    overlap_extension_x: int,
    overlap_extension_y: int,
    feather: float,
) -> tuple[list[TileSpec], tuple[Any, ...]]:
    """
    Create tile specs and serializable config tuple for ComfyUI.

    Returns: (tile_specs, config_tuple)
    config_tuple is picklable and can be passed between nodes.
    """
    tiles = compute_layout(
        width=width,
        height=height,
        tiles_x=tiles_x,
        tiles_y=tiles_y,
        multiple=multiple,
        overlap_extension_x=overlap_extension_x,
        overlap_extension_y=overlap_extension_y,
    )

    # Serialize to tuple of primitives (ComfyUI workflow serialization)
    config_tuple = (
        2,  # version
        width,
        height,
        tiles_x,
        tiles_y,
        multiple,
        overlap_extension_x,
        overlap_extension_y,
        feather,
        tuple(
            (t.type, t.x, t.y, t.w, t.h, t.col, t.row, t.order)
            for t in tiles
        ),
    )
    return tiles, config_tuple


def parse_tile_config(config_tuple: tuple[Any, ...]) -> tuple[int, int, int, float, list[TileSpec]]:
    """
    Parse config tuple back to (width, height, multiple, feather, tile_specs).
    Supports version 1 (single overlap) and version 2 (overlap_x, overlap_y).
    """
    version = config_tuple[0]
    if version >= 2:
        (
            _version,
            width,
            height,
            tiles_x,
            tiles_y,
            multiple,
            _overlap_x,
            _overlap_y,
            feather,
            tiles_data,
        ) = config_tuple
    else:
        (
            _version,
            width,
            height,
            tiles_x,
            tiles_y,
            multiple,
            _overlap_ext,
            feather,
            tiles_data,
        ) = config_tuple

    tiles = [
        TileSpec(
            type=t[0],
            x=t[1], y=t[2], w=t[3], h=t[4],
            col=t[5], row=t[6], order=t[7],
        )
        for t in tiles_data
    ]
    return width, height, multiple, feather, tiles
