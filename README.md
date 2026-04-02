# ComfyUI Video Tiler

Memory-efficient video/image tiling for ComfyUI with overlap tiles, gaps, and feather blending. **No duplicate storage**—slice uses tensor views, merge uses a single output buffer.

Two layout modes: a **grid / gap / seam** slicer (variable tile sizes) and a **fixed tile size** slicer with fractional overlap and traversal patterns. **Video Tile Merge** reconstructs the frame from either slicer using the same `tile_config` you got at slice time.

This pack is **intended for use with** and has been **tested on** **LTX 2.3** workflows. Other models or node stacks may work, but they are not the main focus.

Tiling reduces *per-step* memory versus full-frame processing, but end-to-end workflows can still be **heavy on RAM and VRAM**, strongly depending on **how long the clip is** (batch / frame count) and **how large each frame is** (resolution and channel layout). The slicers’ `layout_label` string ends with a rough memory section (lines tag **typically VRAM** vs **VRAM if merge on CUDA, else RAM**); treat it as a guide, not a guarantee.

This was vibe coded for personal use. It is not actively maintained, and issues or pull requests may not be addressed. Feel free to use it, but please do so at your own discretion.

## Features

- **Video Tile Slicer (var. size)** – Grid layout: explicit tile counts, gaps, seam tiles, per-axis overlap extension (feather/blend is on **Video Tile Merge**)
- **Video Tile Slicer (fixed size)** – Constant tile size, fractional overlap, traversal patterns; **feather** (as fraction of tile size) is on **Video Tile Merge**
- **Dual visualization** – **2-frame** IMAGE batch: (1) bordered overview + info box, (2) traversal **start→end color gradient** (filled). Same gradient style for both slicers; no feather preview (adjust merge instead)
- **`layout_label`** – One string: human-readable layout **plus** combined memory estimates (each line notes **VRAM** vs **RAM** where relevant)
- **Multiple-of-X** – Dimensions snapped to your multiple (e.g. 16, 32 for video models)
- **Low RAM** – Slice outputs views (no copy where possible); merge writes into one shared tensor

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/maDcaDDie2000/comfyui-video-tiler
```

Restart ComfyUI. Nodes appear under category **Video Tiler** with these display names:

| Internal class       | Display name        |
|---------------------|---------------------|
| `VideoTileSlice`    | **Video Tile Slicer (var. size)** |
| `VideoTileSliceFixed` | **Video Tile Slicer (fixed size)** |
| `VideoTileMerge`    | **Video Tile Merge** |
| `GetTile`           | **Get Tile** |
| `ReferenceTileSlice`| **Reference Tile Slice** |

## Nodes

### Video Tile Slicer (var. size)

Splits a video/image batch into a list of tiles using a **tiles_x × tiles_y** grid, gaps, and overlap/seam tiles.

| Input | Description |
|-------|-------------|
| `images` | IMAGE `[B, H, W, C]` |
| `tiles_x` | 1–5 horizontal cells |
| `tiles_y` | 1–5 vertical cells |
| `multiple` | Tile size multiple (e.g. 8, 16, 32) |
| `overlap_extension_x` / `overlap_extension_y` | Pixels seams extend into adjacent tiles (centered on gaps) |

**Outputs:** `tiles` (list of IMAGE views, `OUTPUT_IS_LIST`), `tile_config`, `visualization` (2 images), `tile_count`, `layout_label` (layout + memory block)

`tile_config` is **geometry only** (**v4**): changing **Video Tile Merge** `feather` does not require re-slicing, so upstream nodes can stay cached.

### Video Tile Slicer (fixed size)

Same output **socket types** as **Video Tile Slicer (var. size)**, but layout is driven by tile dimensions and stride (fractional overlap), not a small fixed grid count.

| Input | Description |
|-------|-------------|
| `images` | IMAGE |
| `tile_width` / `tile_height` | Tile size (step 8) |
| `multiple` | Snaps positions/sizes to this multiple |
| `overlap` | `1/8` … `1/2` (fraction of shorter tile side) |
| `pattern` | `row`, `column`, `spiral`, `double_spiral` |

**Outputs:** Same sockets as **Video Tile Slicer (var. size)**. `tile_config` is **v5** (geometry + overlap stride only). Changing **Video Tile Merge** `feather` does not require re-slicing.

Saved workflows with an older **v3** `tile_config` still merge: legacy blur baked in the tuple is used; new slices emit **v5**.

### Video Tile Merge

Reconstructs the full image/video from processed tiles. **`tile_config`** is geometry only; a single **`feather`** on the merge node controls blending after expensive steps (keeps slicer cache stable).

| Input | Description |
|-------|-------------|
| `tile_config` | From either slicer (**v4** grid or **v5** fixed; legacy **v3** still supported) |
| `tiles` | Processed tiles (same list order as slice) |
| `feather` | **Grid (v4):** feather in **pixels** on seam tiles (`overlap_h` / `overlap_v` / `overlap_corner`). **Fixed (v5):** **0–1 fraction of tile width/height** per axis for the internal-edge ramp (capped by stride overlap). Values **> 1** are treated as `min(1, feather/64)` for compatibility (e.g. **32 ≈ 0.5**). **v3** fixed: feather strip is still read from the old tuple (merge `feather` ignored). |

**Output:** merged `IMAGE`. Both modes use **painter-style “over”** with a **linear alpha** on the **top** layer only: the upper layer is **fully opaque (100%)** in its interior; at internal edges the alpha ramps toward **0** so you see the **layer below** (not a symmetric blur between two equal weights). **Grid:** normals are drawn first (order), then seam tiles on top with pixel **`feather`** controlling ramp width. **Fixed:** tiles are drawn in **traversal order** (slicer `order`); later tiles are on top; **`feather`** sets ramp width (fraction of tile size on v5).

**Wiring:** Slicer `tiles` → your processing → Merge `tiles`. Same slicer `tile_config` → Merge `tile_config`. Tune **`feather`** on the merge node.

### Reference Tile Slice

Cuts the **same** spatial tiles from a reference image as your video slicer, so `ref_tiles[i]` aligns with `video_tiles[i]`. Requires `tile_config` from either slicer.

### Get Tile

Returns **one** tile by index for manual wiring or external loops.

| Input | Description |
|-------|-------------|
| `images` | IMAGE |
| `tile_config` | From a slicer |
| `tile_index` | 0-based index |

**Outputs:** `tile` (clamped to frame bounds, contiguous for tools like PIL), `tile_index`, `tile_config` (for steering downstream).

## Workflows

### Option A: Sequential (recommended) – list iteration

Slice nodes emit `tiles` as a **list**. ComfyUI can run downstream nodes **once per tile** when wired in list context.

**Wiring:** Slicer `tiles` → your processing → Merge `tiles`. Same slicer `tile_config` → Merge `tile_config`. Tune **`feather`** on the merge node (`pixels` vs `fraction`; see merge table above).

No custom loop nodes required; processing runs one tile at a time (VRAM-friendly).

### Option B: Reference aligned with video tiles

**Video Tile Slicer (var. size)** or **Video Tile Slicer (fixed size)** → `tile_config` → **Reference Tile Slice** (`reference_image` + `tile_config`). Use the reference tile list in parallel with the video tile list (same indices).

### Option C: Direct (no processing)

Slicer → Merge (connect `tiles` and `tile_config`).

### Option D: Manual / parallel branches

1. Slicer → `tile_config`. Feed `images` + `tile_config` into **Get Tile** with indices 0, 1, 2, …
2. Process each branch; connect results to Merge `tiles` **in the same order** as tile indices.
3. Connect slicer `tile_config` to Merge `tile_config`.

## Compatibility

- **LTX 2.3** — primary target; development and testing assume this stack
- Standard ComfyUI **IMAGE** tensors `[B, H, W, C]`
- Compatible with VideoHelperSuite (VHS)
