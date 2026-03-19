# ComfyUI Video Tiler

Memory-efficient video/image tiling for ComfyUI with overlap tiles, gaps, and feather blending. **No duplicate storage**—slice uses tensor views, merge uses a single output buffer.

*Vibe coded.*

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
| `tiles` | Processed tiles (list from Slice, or from Tile Loop Close) |

**Output:** `IMAGE` (merged video)

### Tile Loop Open / Tile Loop Close / Collect Tile
Sequential loop: process tiles one-by-one. **Tile Loop Open** outputs one tile per iteration; chain your processing → **Collect Tile** → **Tile Loop Close**.

**Why Collect Tile?** Merge expects a list of tiles. The loop produces one tile per iteration. Collect Tile builds that list as the loop runs—it's the bridge between the loop and Merge.

### Get Tile
Returns a single tile by index (used by the loop or for manual extraction).

## Workflows

### Option A: Direct (no processing)
Slice → Merge (connect `tiles` and `tile_config`).

### Option B: Parallel (all tiles at once)
1. **Slice** → `tile_config`. Connect images to **Get Tile** × N (indices 0, 1, 2, …).
2. Process each Get Tile output, connect to Merge’s `tiles` input (same order).
3. Connect Slice’s `tile_config` to Merge’s `tile_config`.

### Option C: Tile Loop (sequential, one-at-a-time)
Process tiles one-by-one. **No external node packs.** Requires ComfyUI with execution model (recent versions).

1. **Slice** → `tile_config`, `images`
2. **Tile Loop Open** (images, tile_config) → tile, tile_index, tile_config, flow_control, accumulation
3. tile → **your processing** → **Collect Tile** (to_add=processed, accumulation=accumulation)
4. **Tile Loop Close** (flow_control, initial_value1=accumulation) → tiles
5. **Merge** (tile_config, tiles) → output

Use `tile_index` to steer processing per tile. Runs sequentially, no extra input.

## Layout Example

512×512, 3×2 tiles, multiple 32:

- Normal tiles: 160×256 each
- Gaps: 16px between tiles
- Overlap tiles: Cover gaps and extend into adjacent tiles for blending

## Compatibility

- Works with standard ComfyUI IMAGE type. Compatible with VideoHelperSuite (VHS).
- **Tile Loop** requires ComfyUI with `comfy_execution` (execution model, recent versions). On older ComfyUI, use the parallel workflow (Slice → process each tile → Merge).
