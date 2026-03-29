# ComfyUI Video Tiler

Memory-efficient video/image tiling for ComfyUI with overlap tiles, gaps, and feather blending. **No duplicate storage**—slice uses tensor views, merge uses a single output buffer.

Two layout modes: a **grid / gap / seam** slicer (variable tile sizes) and a **fixed tile size** slicer with fractional overlap and traversal patterns. **Video Tile Merge** reconstructs the frame from either slicer using the same `tile_config` you got at slice time.

This was vibe coded for personal use. It is not actively maintained, and issues or pull requests may not be addressed. Feel free to use it, but please do so at your own discretion.

## Features

- **Variable grid (Variable Tile Size)** – Explicit horizontal/vertical tile counts, gaps between tiles, seam tiles that extend into neighbors, per-axis overlap extension and feather
- **Fixed grid (Fixed Tile Size)** – Constant `tile_width` × `tile_height`, overlap as a fraction of the smaller side (⅛–½), row/column/spiral/double-spiral order, `blur_fraction` for merge weight falloff
- **Dual visualization** – `visualization` is a **2-frame** IMAGE batch: (1) layout/outline and labels, (2) type or traversal gradient. Variable slicer: borders + tile indices; second frame shows tile types and feather regions. Fixed slicer: traversal order as a **start→end color gradient**
- **Diagnostics** – `memory_estimate` (STRING): rough VAE encode/decode and merge peaks; `layout_label` (STRING): human-readable layout summary
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
| `VideoTileSlice`    | **Variable Tile Size** |
| `VideoTileSliceFixed` | **Fixed Tile Size** |
| `VideoTileMerge`    | **Video Tile Merge** |
| `GetTile`           | **Get Tile** |
| `ReferenceTileSlice`| **Reference Tile Slice** |

## Nodes

### Variable Tile Size

Splits a video/image batch into a list of tiles using a **tiles_x × tiles_y** grid, gaps, and overlap/seam tiles.

| Input | Description |
|-------|-------------|
| `images` | IMAGE `[B, H, W, C]` |
| `tiles_x` | 1–5 horizontal cells |
| `tiles_y` | 1–5 vertical cells |
| `multiple` | Tile size multiple (e.g. 8, 16, 32) |
| `overlap_extension_x` / `overlap_extension_y` | Pixels seams extend into adjacent tiles (centered on gaps) |
| `feather` | Feather for overlap blending at merge |

**Outputs:** `tiles` (list of IMAGE views, `OUTPUT_IS_LIST`), `tile_config`, `visualization` (2 images), `tile_count`, `memory_estimate`, `layout_label`

### Fixed Tile Size

Same output **socket types** as Variable Tile Size, but layout is driven by tile dimensions and stride (fractional overlap), not a small fixed grid count.

| Input | Description |
|-------|-------------|
| `images` | IMAGE |
| `tile_width` / `tile_height` | Tile size (step 8) |
| `multiple` | Snaps positions/sizes to this multiple |
| `overlap` | `1/8` … `1/2` (fraction of shorter tile side) |
| `pattern` | `row`, `column`, `spiral`, `double_spiral` |
| `blur_fraction` | Weight falloff at tile edges for merge (0–1) |

**Outputs:** Same as Variable Tile Size. `tile_config` uses an internal **v3** format; merge detects it and uses weighted fixed-grid blending.

### Video Tile Merge

Reconstructs the full image/video from processed tiles. Connect the **`tile_config`** from the slicer you used—no need to re-enter layout parameters.

| Input | Description |
|-------|-------------|
| `tile_config` | From **Variable** or **Fixed Tile Size** |
| `tiles` | Processed tiles (same list order as slice) |

**Output:** merged `IMAGE`. Variable layouts use seam/corner feather rules; fixed **v3** layouts use per-tile weights with blending on internal seams only.

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

**Wiring:** Slicer `tiles` → your processing → Merge `tiles`. Same slicer `tile_config` → Merge `tile_config`.

No custom loop nodes required; processing runs one tile at a time (VRAM-friendly).

### Option B: Reference aligned with video tiles

**Variable / Fixed Tile Size** → `tile_config` → **Reference Tile Slice** (`reference_image` + `tile_config`). Use the reference tile list in parallel with the video tile list (same indices).

### Option C: Direct (no processing)

Slicer → Merge (connect `tiles` and `tile_config`).

### Option D: Manual / parallel branches

1. Slicer → `tile_config`. Feed `images` + `tile_config` into **Get Tile** with indices 0, 1, 2, …
2. Process each branch; connect results to Merge `tiles` **in the same order** as tile indices.
3. Connect slicer `tile_config` to Merge `tile_config`.

## Layout example (Variable Tile Size)

512×512, 3×2 tiles, multiple 32:

- Normal tiles: 160×256 each  
- Gaps: 16px between tiles  
- Overlap/seam tiles: cover gaps and extend into neighbors for blending  

## Compatibility

- Standard ComfyUI **IMAGE** tensors `[B, H, W, C]`
- Compatible with VideoHelperSuite (VHS)

## Reference

This pack follows the **Sequential Batcher** pattern (list iteration). Related packs:

| Pack | Use case | Repo |
|------|----------|------|
| **ControlFlowUtils** | Loop Open / Loop Close | [VykosX/ControlFlowUtils](https://github.com/VykosX/ControlFlowUtils) |
| **Inspire Pack** | Foreach / list iteration | [ltdrdata/ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) |
| **Sequential Batcher** | Video/batch loops (Image Batch To List, List To Batch) | [Meisoftcoltd/ComfyUI-Sequential-Batcher](https://github.com/Meisoftcoltd/ComfyUI-Sequential-Batcher) |
