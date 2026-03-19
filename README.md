# ComfyUI Video Tiler

Memory-efficient video/image tiling for ComfyUI with overlap tiles, gaps, and feather blending. **No duplicate storage**—slice uses tensor views, merge uses a single output buffer.

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

**Outputs:** `tile_0` … `tile_49` (IMAGE views), `tile_config`, `visualization`

### Video Tile Merge

Reconstructs video from processed tiles. Uses `tile_config` from Slice—no separate settings.

| Input | Description |
|-------|-------------|
| `tile_config` | From Video Tile Slice |
| `tiles` | Processed tiles (connect each slice tile output after processing) |

**Output:** `IMAGE` (merged video)

### Tile Loop Open / Tile Loop Close
Self-contained sequential loop. Tile Loop Open outputs one tile per iteration; chain your processing, then Accumulate, then Tile Loop Close. No external packs.

### Video Tile Process Loop
Fallback when Tile Loop isn't available: loops over tiles in one node (passthrough or clip mode).

### Get Tile / Get Tile Count / Remaining to Index
For manual index or advanced setups.

## Workflows

### Option A: Parallel (all tiles at once)
1. **Slice** → Connect `tile_config` and each `tile_N` to your processing.
2. Connect processed outputs to Merge’s `tiles` input (same order as slice outputs).
3. Connect Slice’s `tile_config` to Merge’s `tile_config`.

### Option B: Tile Loop (sequential, self-contained)
Process tiles one-by-one. **No external node packs.** Requires ComfyUI with execution model (recent versions).

1. **Slice** → `tile_config`, `images`
2. **Tile Loop Open** (images, tile_config) → tile, tile_index, tile_config, flow_control, accumulation
3. tile → **your processing** → **Accumulate** (to_add=processed, accumulation=accumulation)
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
- **Tile Loop** requires ComfyUI with `comfy_execution` (execution model inversion, PR #2666). Use **Video Tile Process Loop** on older ComfyUI.
