# ComfyUI Video Tiler

Memory-efficient video/image tiling for ComfyUI with overlap tiles, gaps, and feather blending. **No duplicate storage**—slice uses tensor views, merge uses a single output buffer.

This was vibe coded for personal use. It is not actively maintained, and issues or pull requests may not be addressed. Feel free to use it, but please do so at your own discretion.

## Features

- **Overlap tiles** – Seam/overlap tiles cover gaps and blend over normal tiles
- **Multiple-of-X** – All tile dimensions follow your multiple (e.g. 16, 32 for video models)
- **Gaps between tiles** – When full tiles don't cover the image, gaps sit between tiles (never on outside)
- **Tiling visualization** – Single image showing tile outlines and feather regions
- **Low RAM** – Slice outputs views (no copy); merge writes into one shared tensor

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/maDcaDDie2000/comfyui-video-tiler
```

Restart ComfyUI.

## Nodes

### Video Tile Slice

Splits video/image batch into tiles.

| Input | Description |
|-------|-------------|
| `images` | IMAGE `[B, H, W, C]` |
| `tiles_x` | 1–5 horizontal tiles |
| `tiles_y` | 1–5 vertical tiles |
| `multiple` | Tile size multiple (e.g. 8, 16) |
| `overlap_extension` | Pixels to extend into adjacent tiles (centered on gaps) |
| `feather` | Feather amount for overlap blending |

**Outputs:** `tiles` (list of IMAGE views), `tile_config`, `visualization`, `tile_count`

### Video Tile Merge

Reconstructs video from processed tiles. Uses `tile_config` from Slice—no separate settings.

| Input | Description |
|-------|-------------|
| `tile_config` | From Video Tile Slice |
| `tiles` | Processed tiles (list from Slice) |

**Output:** `IMAGE` (merged video)

### Get Tile
Returns a single tile by index for manual extraction or parallel processing.

## Workflows

### Option A: Sequential (recommended) – Sequential Batcher pattern
Uses ComfyUI's native list iteration. Slice outputs `tiles` as a list; ComfyUI runs downstream **once per tile** automatically.

**Wiring:** Slice `tiles` → your processing → Merge `tiles`. Slice `tile_config` → Merge `tile_config`.

No loop nodes needed. Processing runs sequentially, one tile at a time (VRAM-friendly).

### Option B: Direct (no processing)
Slice → Merge (connect `tiles` and `tile_config`).

### Option C: Parallel (all tiles at once)
1. **Slice** → `tile_config`. Connect images to **Get Tile** × N (indices 0, 1, 2, …).
2. Process each Get Tile output, connect to Merge `tiles` input (same order).
3. Connect Slice `tile_config` to Merge `tile_config`.

## Layout Example

512×512, 3×2 tiles, multiple 32:

- Normal tiles: 160×256 each
- Gaps: 16px between tiles
- Overlap tiles: Cover gaps and extend into adjacent tiles for blending

## Compatibility

- Works with standard ComfyUI IMAGE type. Compatible with VideoHelperSuite (VHS).

## Reference

This pack follows the **Sequential Batcher** pattern (list iteration). Related packs:

| Pack | Use case | Repo |
|------|----------|------|
| **ControlFlowUtils** | Best overall for looping (Loop Open / Loop Close) | [VykosX/ControlFlowUtils](https://github.com/VykosX/ControlFlowUtils) |
| **Inspire Pack** | Best for foreach/list iteration | [ltdrdata/ComfyUI-Inspire-Pack](https://github.com/ltdrdata/ComfyUI-Inspire-Pack) |
| **Sequential Batcher** | Best for video/batch loops (Image Batch To List, List To Batch) | [Meisoftcoltd/ComfyUI-Sequential-Batcher](https://github.com/Meisoftcoltd/ComfyUI-Sequential-Batcher) |
