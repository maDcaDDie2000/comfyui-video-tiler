"""
TILE_CONFIG - ComfyUI-friendly serializable config for tile layout.

Passed from Slice node to Merge node. Contains only layout metadata (positions),
no tensor data — no feather/blur (those are merge inputs so upstream stays cached).
"""

from __future__ import annotations

from typing import Any

from .layout import TileSpec, compute_layout


TILE_CONFIG_VERSIONS = {1, 2, 3, 4, 5}


def is_tile_config_tuple(value: Any) -> bool:
    """Return True for the actual TILE_CONFIG payload, not a list wrapper around it."""
    return (
        isinstance(value, (list, tuple))
        and len(value) > 0
        and isinstance(value[0], int)
        and value[0] in TILE_CONFIG_VERSIONS
    )


def unwrap_tile_config(value: Any) -> tuple[Any, ...]:
    """Accept direct TILE_CONFIG tuples and ComfyUI single-item list wrappers."""
    while isinstance(value, (list, tuple)) and len(value) == 1 and not is_tile_config_tuple(value):
        value = value[0]
    if not is_tile_config_tuple(value):
        raise ValueError(
            "tile_config must be TILE_CONFIG from Video Tile Slice / Slice Fixed "
            "(version 1-5 tuple)."
        )
    return tuple(value)


def create_tile_config(
    width: int,
    height: int,
    tiles_x: int,
    tiles_y: int,
    multiple: int,
    overlap_extension_x: int,
    overlap_extension_y: int,
) -> tuple[list[TileSpec], tuple[Any, ...]]:
    """
    Create tile specs and serializable config tuple for ComfyUI.

    Returns: (tile_specs, config_tuple)
    Grid layout version 4 — feather is set on Video Tile Merge only.
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

    config_tuple = (
        4,  # version: grid seams, no feather in tuple
        width,
        height,
        tiles_x,
        tiles_y,
        multiple,
        overlap_extension_x,
        overlap_extension_y,
        tuple(
            (t.type, t.x, t.y, t.w, t.h, t.col, t.row, t.order)
            for t in tiles
        ),
    )
    return tiles, config_tuple


def _tiles_from_data(tiles_data: tuple) -> list[TileSpec]:
    return [
        TileSpec(
            type=t[0],
            x=t[1], y=t[2], w=t[3], h=t[4],
            col=t[5], row=t[6], order=t[7],
        )
        for t in tiles_data
    ]


def parse_tile_config(config_tuple: tuple[Any, ...]) -> tuple[int, int, int, list[TileSpec]]:
    """
    Parse config tuple to (width, height, multiple, tile_specs).

    v4: grid seams. v5: fixed tiles (blur from merge). v3: fixed with blur in tuple (legacy).
    v1–v2: grid with legacy feather slot in tuple (ignored — use merge feather).
    """
    config_tuple = unwrap_tile_config(config_tuple)
    version = config_tuple[0]
    if version == 5:
        (
            _v,
            width,
            height,
            _tile_w,
            _tile_h,
            _ox,
            _oy,
            multiple,
            _pattern_id,
            tiles_data,
        ) = config_tuple
        return width, height, int(multiple), _tiles_from_data(tiles_data)
    if version == 4:
        (
            _v,
            width,
            height,
            _tiles_x,
            _tiles_y,
            multiple,
            _oex,
            _oey,
            tiles_data,
        ) = config_tuple
        return width, height, int(multiple), _tiles_from_data(tiles_data)
    if version == 3:
        (
            _v,
            width,
            height,
            tile_w,
            _tile_h,
            _ox,
            _oy,
            _blur_x,
            _blur_y,
            _pattern_id,
            tiles_data,
        ) = config_tuple
        tiles = _tiles_from_data(tiles_data)
        return width, height, int(tile_w), tiles
    if version >= 2:
        (
            _version,
            width,
            height,
            _tiles_x,
            _tiles_y,
            multiple,
            _overlap_x,
            _overlap_y,
            _feather_legacy,
            tiles_data,
        ) = config_tuple
    else:
        (
            _version,
            width,
            height,
            _tiles_x,
            _tiles_y,
            multiple,
            _overlap_ext,
            _feather_legacy,
            tiles_data,
        ) = config_tuple

    return width, height, int(multiple), _tiles_from_data(tiles_data)
