"""
Tile layout algorithm for video tiling with gaps and overlaps.

- Normal tiles: same size, multiple of X
- Gaps: only between tiles (never on outside)
- Overlap tiles: centered on gaps, extend into adjacent tiles
- All dimensions follow multiple-of-X rule
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class TileSpec:
    """Single tile specification."""
    type: Literal["normal", "overlap_h", "overlap_v", "overlap_corner"]
    x: int
    y: int
    w: int
    h: int
    col: int  # grid column (for normal)
    row: int  # grid row (for normal)
    order: int  # blend order: normal first (0), then overlaps (1+)


def _snap_to_multiple(val: int, multiple: int) -> int:
    """Round down to nearest multiple."""
    return (val // multiple) * multiple


def _compute_axis_layout(
    size: int,
    tile_count: int,
    multiple: int,
) -> tuple[int, float, list[tuple[int, int]]]:
    """
    Compute tile positions along one axis.
    Returns: (tile_size, gap_size, [(tile_start, tile_end), ...])
    """
    if tile_count <= 0:
        return 0, 0.0, []

    if tile_count == 1:
        tile_size = _snap_to_multiple(size, multiple)
        return tile_size, 0.0, [(0, tile_size)]

    # tile_count >= 2: we have gaps between tiles
    # size = tile_count * tile_size + (tile_count - 1) * gap
    # Maximize tile_size (multiple of X) such that we fit
    # tile_size = (size - (tile_count-1)*gap) / tile_count
    # For gap >= 0: tile_size <= size / tile_count
    max_tile = size // tile_count
    tile_size = _snap_to_multiple(max_tile, multiple)
    if tile_size < multiple:
        tile_size = multiple

    gap_total = size - tile_count * tile_size
    gap_count = tile_count - 1
    gap = gap_total / gap_count if gap_count > 0 else 0.0

    positions = []
    for i in range(tile_count):
        start = int(i * (tile_size + gap))
        end = start + tile_size
        positions.append((start, end))

    return tile_size, gap, positions


def compute_layout(
    width: int,
    height: int,
    tiles_x: int,
    tiles_y: int,
    multiple: int,
    overlap_extension_x: int,
    overlap_extension_y: int,
) -> list[TileSpec]:
    """
    Compute full 2D tile layout with normal and overlap tiles.

    overlap_extension_x: horizontal overlap (pixels to extend into adjacent tiles)
    overlap_extension_y: vertical overlap
    """
    tiles_x = max(1, min(5, tiles_x))
    tiles_y = max(1, min(5, tiles_y))
    overlap_x = _snap_to_multiple(overlap_extension_x, multiple)
    overlap_y = _snap_to_multiple(overlap_extension_y, multiple)
    if overlap_x < multiple:
        overlap_x = multiple
    if overlap_y < multiple:
        overlap_y = multiple

    tile_w, gap_x, x_positions = _compute_axis_layout(width, tiles_x, multiple)
    tile_h, gap_y, y_positions = _compute_axis_layout(height, tiles_y, multiple)

    overlap_x = min(overlap_x, max(tile_w // 2, multiple))
    overlap_y = min(overlap_y, max(tile_h // 2, multiple))

    tiles: list[TileSpec] = []
    order = 0

    # 1. Normal tiles (order 0)
    for row in range(tiles_y):
        for col in range(tiles_x):
            x1, x2 = x_positions[col]
            y1, y2 = y_positions[row]
            tiles.append(TileSpec(
                type="normal",
                x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                col=col, row=row,
                order=order,
            ))
            order += 1

    # 2. Overlap tiles (horizontal seams - between rows) - order 1
    if tiles_y > 1 and overlap_y >= multiple:
        for row in range(tiles_y - 1):
            y_gap_center = y_positions[row][1] + (gap_y / 2 if gap_y > 0 else 0)
            overlap_h = gap_y + 2 * overlap_y
            overlap_h = _snap_to_multiple(int(overlap_h), multiple)
            if overlap_h < multiple:
                overlap_h = multiple
            y1 = max(0, int(y_gap_center - overlap_h / 2))
            y2 = min(height, y1 + overlap_h)

            for col in range(tiles_x):
                x1, x2 = x_positions[col]
                tiles.append(TileSpec(
                    type="overlap_v",
                    x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                    col=col, row=row,
                    order=order,
                ))
                order += 1

    # 3. Overlap tiles (vertical seams - between cols) - order 2
    if tiles_x > 1 and overlap_x >= multiple:
        for col in range(tiles_x - 1):
            x_gap_center = x_positions[col][1] + (gap_x / 2 if gap_x > 0 else 0)
            overlap_w = gap_x + 2 * overlap_x
            overlap_w = _snap_to_multiple(int(overlap_w), multiple)
            if overlap_w < multiple:
                overlap_w = multiple
            x1 = max(0, int(x_gap_center - overlap_w / 2))
            x2 = min(width, x1 + overlap_w)

            for row in range(tiles_y):
                y1, y2 = y_positions[row]
                tiles.append(TileSpec(
                    type="overlap_h",
                    x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                    col=col, row=row,
                    order=order,
                ))
                order += 1

    # 4. Overlap tiles (corners - on top of seam overlaps) - order 10000 so always topmost
    if tiles_x > 1 and tiles_y > 1 and overlap_x >= multiple and overlap_y >= multiple:
        for col in range(tiles_x - 1):
            for row in range(tiles_y - 1):
                x_gap_center = x_positions[col][1] + (gap_x / 2 if gap_x > 0 else 0)
                y_gap_center = y_positions[row][1] + (gap_y / 2 if gap_y > 0 else 0)
                overlap_w = gap_x + 2 * overlap_x
                overlap_h = gap_y + 2 * overlap_y
                overlap_w = _snap_to_multiple(int(overlap_w), multiple)
                overlap_h = _snap_to_multiple(int(overlap_h), multiple)
                if overlap_w < multiple:
                    overlap_w = multiple
                if overlap_h < multiple:
                    overlap_h = multiple
                x1 = max(0, int(x_gap_center - overlap_w / 2))
                y1 = max(0, int(y_gap_center - overlap_h / 2))
                x2 = min(width, x1 + overlap_w)
                y2 = min(height, y1 + overlap_h)

                tiles.append(TileSpec(
                    type="overlap_corner",
                    x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                    col=col, row=row,
                    order=10000 + order,
                ))
                order += 1

    return tiles
