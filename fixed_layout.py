"""
Fixed-size tile layout with fractional overlap and traversal patterns (row, column, spiral, double_spiral).
"""

from __future__ import annotations

import colorsys
from typing import Literal

import numpy as np

from .layout import TileSpec

Pattern = Literal["row", "column", "spiral", "double_spiral"]


def _snap_to_multiple(val: int, multiple: int) -> int:
    if val <= 0:
        return 0
    return max(multiple, (val // multiple) * multiple)


def _axis_positions(size: int, tile: int, step: int) -> list[int]:
    """Legacy: start positions with fixed step until the right edge (uneven last gap possible)."""
    tile = max(1, tile)
    if tile >= size:
        return [0]
    positions: list[int] = [0]
    while True:
        nxt = positions[-1] + step
        if nxt + tile > size:
            back = size - tile
            if back > positions[-1] and back not in positions:
                positions.append(back)
            break
        if nxt <= positions[-1]:
            break
        positions.append(nxt)
    return sorted(set(positions))


def _even_axis_positions_equal_overlap(
    size: int, tile: int, overlap_min: int, m: int,
) -> tuple[list[int], int] | None:
    """
    Tile starts 0, L, 2L, …, (n-1)L with (n-1)*L = size - tile so the last tile ends at `size`.
    Neighbour overlap is always tile - L (constant). Pick the largest valid L (fewest tiles)
    with overlap >= overlap_min. L is a multiple of m.

    Returns (x_starts, actual_overlap) or None if no exact stride fits.
    """
    tile = max(1, tile)
    size = max(1, size)
    if tile >= size:
        return [0], 0
    rem = size - tile
    if rem <= 0:
        return [0], 0
    overlap_min = max(0, min(overlap_min, tile // 2))
    L_max = tile - overlap_min
    if L_max < m:
        return None
    L_start = (L_max // m) * m
    for L in range(L_start, m - 1, -m):
        if L <= 0:
            break
        if rem % L != 0:
            continue
        O = tile - L
        if O < overlap_min:
            continue
        k = rem // L
        if k < 1:
            continue
        xs = [i * L for i in range(k + 1)]
        return xs, O
    return None


def _even_axis_positions_approximate(
    size: int, tile: int, overlap_min: int, m: int,
) -> tuple[list[int], int]:
    """When no exact equal stride divides (size - tile), fall back to legacy stepping (uneven last gap)."""
    tile = max(1, tile)
    if tile >= size:
        return [0], 0
    overlap_min = max(0, min(overlap_min, tile // 2))
    step = max(1, tile - overlap_min)
    xs = _axis_positions(size, tile, step)
    O = tile - (xs[1] - xs[0]) if len(xs) > 1 else 0
    return xs, max(0, O)


def spiral_indices(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    """Clockwise spiral from top-left (row, col)."""
    if n_rows <= 0 or n_cols <= 0:
        return []
    result: list[tuple[int, int]] = []
    top, bottom = 0, n_rows - 1
    left, right = 0, n_cols - 1
    while top <= bottom and left <= right:
        for c in range(left, right + 1):
            result.append((top, c))
        top += 1
        for r in range(top, bottom + 1):
            result.append((r, right))
        right -= 1
        if top <= bottom:
            for c in range(right, left - 1, -1):
                result.append((bottom, c))
            bottom -= 1
        if left <= right:
            for r in range(bottom, top - 1, -1):
                result.append((r, left))
            left += 1
    return result


def double_spiral_indices(n_rows: int, n_cols: int) -> list[tuple[int, int]]:
    """Interleave clockwise spirals from top-left and bottom-right."""
    a = spiral_indices(n_rows, n_cols)
    b = [(n_rows - 1 - r, n_cols - 1 - c) for r, c in spiral_indices(n_rows, n_cols)]
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i in range(max(len(a), len(b))):
        for src in (a, b):
            if i < len(src):
                p = src[i]
                if p not in seen:
                    seen.add(p)
                    out.append(p)
    return out


def order_grid_cells(
    pattern: Pattern,
    n_rows: int,
    n_cols: int,
) -> list[tuple[int, int]]:
    if pattern == "row":
        return [(r, c) for r in range(n_rows) for c in range(n_cols)]
    if pattern == "column":
        return [(r, c) for c in range(n_cols) for r in range(n_rows)]
    if pattern == "spiral":
        return spiral_indices(n_rows, n_cols)
    return double_spiral_indices(n_rows, n_cols)


def compute_fixed_feather_strips(
    tile_w: int,
    tile_h: int,
    overlap_x: int,
    overlap_y: int,
    multiple: int,
    feather_fraction: float,
) -> tuple[int, int]:
    """
    Pixel width of feather on internal edges for fixed layout.
    feather_fraction is a fraction of tile_w / tile_h per axis (oblong ⇒ different px on H vs V),
    at most 0.5 (50%) per axis, then capped by stride overlap, snapped to multiple.
    """
    m = max(1, min(64, multiple))
    f = max(0.0, min(0.5, feather_fraction))
    raw_x = int(tile_w * f)
    raw_y = int(tile_h * f)
    bx = min(overlap_x, _snap_to_multiple(raw_x, m)) if overlap_x > 0 else 0
    by = min(overlap_y, _snap_to_multiple(raw_y, m)) if overlap_y > 0 else 0
    return bx, by


def compute_fixed_layout(
    width: int,
    height: int,
    tile_w_in: int,
    tile_h_in: int,
    multiple: int,
    overlap_numer: int,
    overlap_denom: int,
    pattern: Pattern,
) -> tuple[list[TileSpec], tuple]:
    """
    overlap = minimum neighbour overlap (fraction of tile per axis), capped at half tile.
    Tiles are spaced with **equal** stride along each axis so every interior pair shares the same
    overlap (>= that minimum); last tile is flush to the image edge. Feather is on Video Tile Merge.
    """
    m = max(1, min(64, multiple))
    tile_w = _snap_to_multiple(max(m, tile_w_in), m)
    tile_h = _snap_to_multiple(max(m, tile_h_in), m)

    max_ox = tile_w // 2
    max_oy = tile_h // 2
    raw_ox = int((tile_w * overlap_numer) / max(overlap_denom, 1))
    raw_oy = int((tile_h * overlap_numer) / max(overlap_denom, 1))
    overlap_x_min = min(max_ox, _snap_to_multiple(raw_ox, m))
    overlap_y_min = min(max_oy, _snap_to_multiple(raw_oy, m))
    if overlap_x_min < 0:
        overlap_x_min = 0
    if overlap_y_min < 0:
        overlap_y_min = 0

    even_x = _even_axis_positions_equal_overlap(width, tile_w, overlap_x_min, m)
    if even_x is not None:
        xs, overlap_x = even_x
    else:
        xs, overlap_x = _even_axis_positions_approximate(width, tile_w, overlap_x_min, m)

    even_y = _even_axis_positions_equal_overlap(height, tile_h, overlap_y_min, m)
    if even_y is not None:
        ys, overlap_y = even_y
    else:
        ys, overlap_y = _even_axis_positions_approximate(height, tile_h, overlap_y_min, m)
    n_cols = len(xs)
    n_rows = len(ys)

    cell_order = order_grid_cells(pattern, n_rows, n_cols)
    pos_to_rc = {(ys[r], xs[c]): (r, c) for r in range(n_rows) for c in range(n_cols)}
    # Filter valid (some patterns might duplicate if n_rows/n_cols wrong - shouldn't)
    tiles_ordered: list[tuple[int, int, int, int]] = []
    for r, c in cell_order:
        if 0 <= r < n_rows and 0 <= c < n_cols:
            x, y = xs[c], ys[r]
            w = min(tile_w, width - x)
            h = min(tile_h, height - y)
            if w > 0 and h > 0:
                tiles_ordered.append((x, y, w, h))

    tile_specs: list[TileSpec] = []
    for i, (x, y, w, h) in enumerate(tiles_ordered):
        rc = pos_to_rc.get((y, x))
        col = rc[1] if rc else 0
        row = rc[0] if rc else 0
        tile_specs.append(
            TileSpec(
                type="normal",
                x=x,
                y=y,
                w=w,
                h=h,
                col=col,
                row=row,
                order=i,
            )
        )

    pattern_id = {"row": 0, "column": 1, "spiral": 2, "double_spiral": 3}[pattern]
    config_tuple = (
        5,
        width,
        height,
        tile_w,
        tile_h,
        overlap_x,
        overlap_y,
        m,
        pattern_id,
        tuple(
            (t.type, t.x, t.y, t.w, t.h, t.col, t.row, t.order)
            for t in tile_specs
        ),
    )
    return tile_specs, config_tuple


def fixed_layout_label_string(
    width: int,
    height: int,
    tile_w: int,
    tile_h: int,
    overlap_x: int,
    overlap_y: int,
    pattern: str,
    n_tiles: int,
) -> str:
    lines = [
        f"Output: {width}x{height}",
        f"Mode: fixed tile",
        f"Tile: {tile_w}x{tile_h}",
        f"Overlap (equal between neighbours, ≥ min): {overlap_x}x{overlap_y}",
        f"Pattern: {pattern}",
        f"Total tiles: {n_tiles}",
        f"Feather: set on Video Tile Merge (fraction of tile W/H)",
    ]
    return "\n".join(lines)


def order_gradient_rgb(order: int, n_tiles: int) -> tuple[float, float, float]:
    """
    RGB 0–1 for traversal order: first tile cool (blue–cyan), last warm (red–orange).
    """
    if n_tiles <= 1:
        r, g, b = colorsys.hsv_to_rgb(0.56, 0.82, 0.96)
        return (r, g, b)
    t = order / (n_tiles - 1)
    h = 0.58 * (1.0 - t) + 0.02 * t
    s = 0.78 + 0.12 * t
    v = 0.90 + 0.08 * (1.0 - t)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (float(r), float(g), float(b))


def create_fixed_traversal_viz_labels(
    width: int,
    height: int,
    tiles: list[TileSpec],
) -> np.ndarray:
    """Borders + index labels; colors follow traversal order (start→end gradient)."""
    from PIL import Image, ImageDraw

    from .slice_node import _get_font

    n = len(tiles)
    img = np.ones((height, width, 3), dtype=np.float32) * 0.95
    for t in tiles:
        x1, y1 = t.x, t.y
        x2, y2 = t.x + t.w, t.y + t.h
        color = list(order_gradient_rgb(t.order, n))
        img[y1:y2, x1:x1 + 2, :] = color
        img[y1:y2, x2 - 2 : x2, :] = color
        img[y1 : y1 + 2, x1:x2, :] = color
        img[y2 - 2 : y2, x1:x2, :] = color

    pil_img = Image.fromarray((img * 255).astype(np.uint8))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(draw, width)
    for t in tiles:
        x1, y1 = t.x, t.y
        x2, y2 = t.x + t.w, t.y + t.h
        label = f"{t.order:02d}"
        rgb = tuple(int(c * 255) for c in order_gradient_rgb(t.order, n))
        pad = 4
        for (lx, ly) in [(x1 + pad, y1 + pad), (x2 - 24, y1 + pad), (x1 + pad, y2 - 16), (x2 - 24, y2 - 16)]:
            draw.text((lx, ly), label, fill=rgb, font=font)
    return np.array(pil_img).astype(np.float32) / 255.0


def create_fixed_traversal_viz_fill(
    width: int,
    height: int,
    tiles: list[TileSpec],
    _blur_max: float,
) -> np.ndarray:
    """
    Solid fill per tile in traversal gradient color; later order paints over overlaps
    so overlapping regions show the later tile hue.
    """
    n = len(tiles)
    img = np.ones((height, width, 3), dtype=np.float32) * 0.95
    for t in sorted(tiles, key=lambda x: x.order):
        x1, y1 = t.x, t.y
        x2, y2 = t.x + t.w, t.y + t.h
        color = np.array(order_gradient_rgb(t.order, n), dtype=np.float32)
        img[y1:y2, x1:x2, :] = color
    return img
