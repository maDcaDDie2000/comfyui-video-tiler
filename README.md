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
| `VideoTileMerge`            | **Video Tile Merge** |
| `VideoTileMergeOverlapSoft` | **Video Tile Merge (overlap soft)** |
| `GetTile`           | **Get Tile** |
| `ReferenceTileSlice`| **Reference Tile Slice** |
| `VideoTileReferenceColorMatch` | **Video Tile Reference Color Match** |
| `VideoTileAudioFFprobeLTX` | **Video Tile Audio Present** |

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

Same output **socket types** as **Video Tile Slicer (var. size)**, but layout uses **fixed tile size** and **overlap** along each axis.

| Input | Description |
|-------|-------------|
| `images` | IMAGE |
| `tile_width` / `tile_height` | Tile size (step 8) |
| `multiple` | Snaps positions/sizes to this multiple |
| `overlap` | `1/8` … `1/2` — **minimum** neighbour overlap (fraction of tile W / H per axis). The slicer uses the **fewest** tiles per axis so every neighbour stride stays ≤ `tile − overlap` (overlap never **below** your minimum). Strides are snapped to `multiple` with the spare pixels spread across gaps—so overlaps can differ slightly by one `multiple`, but you no longer get pathological layouts where the only valid “exact” stride was a few pixels. |
| `pattern` | `row`, `column`, `spiral`, `double_spiral` |

**Outputs:** Same sockets as **Video Tile Slicer (var. size)**. `tile_config` is **v5** (geometry + overlap stride only). Changing **Video Tile Merge** `feather` does not require re-slicing.

Saved workflows with an older **v3** `tile_config` still merge: legacy blur baked in the tuple is used; new slices emit **v5**.

### Video Tile Merge

Reconstructs the full image/video from processed tiles. **`tile_config`** is geometry only; **`feather`** plus optional **`feather_curve`** / **`blend_mode`** control blending after expensive steps (keeps slicer cache stable).

| Input | Description |
|-------|-------------|
| `tile_config` | From either slicer (**v4** grid or **v5** fixed; legacy **v3** still supported) |
| `tiles` | Processed tiles (same list order as slice) |
| `feather` | **0–0.5** for all layouts: fraction of the **local tile width** used for horizontal alpha ramps and of **tile height** for vertical ramps (oblong tiles ⇒ different ramp thickness in px on short vs long side). **Hard cap 50%** per axis. Then **fixed (v5)** also caps strip by stride **overlap** and snaps to `multiple`. **v3** fixed: strips come from the saved tuple (merge `feather` ignored). |
| `feather_curve` *(optional)* | **`linear`** (default), **`ease_in`** / **`ease_out`** (quadratic), **`ease_in_out`** (smoothstep). Pointwise remap of geometric weights only. |
| `blend_mode` *(optional)* | **`alpha_over`** (default): painter order + **`covered`** gate (same rules as before). **`weighted_average`**: normalized sum of geometry-weighted tile colors (`sum(w × pixel) / sum(w)`), **same geometry-derived weights only** (no RGB thresholds). |

**Output:** merged `IMAGE`. **`alpha_over`** uses coverage-gated compositing in **float32** then casts to IMAGE dtype. **`weighted_average`** accumulates weighted sums in float32. Feather geometry uses **nearest-internal-edge** normalized distance (then cosine) so overlap weights don’t multiply into corner “dips.”

**Wiring:** Slicer `tiles` → your processing → Merge `tiles`. Same slicer `tile_config` → Merge `tile_config`. Tune **`feather`** and optional **`feather_curve`** / **`blend_mode`**. In the graph UI, expand the merge node’s **optional** inputs if **`feather_curve`** / **`blend_mode`** are collapsed — older workflows without those sockets still validate (defaults apply).

**Video Tile Merge (overlap soft)** — same inputs as **Video Tile Merge** (no extra widgets). Normalized weighted sum merge using the **same feather geometry** as the main merge (nearest-edge cosine ramps).

### Reference Tile Slice

Cuts the **same** spatial tiles from a reference image as your video slicer, so `ref_tiles[i]` aligns with `video_tiles[i]`. Requires `tile_config` from either slicer.

### Video Tile Reference Color Match

Runs **after** **Video Tile Merge**. Uses a **reference** clip (LR or pre-upscaled RGB) resized to the merged frame: Gaussian **low-frequency** color is blended toward that reference; **high-frequency** detail stays from the merged output (so LR texture is not pasted in). Defaults bias gentle cast correction and seam tint without flattening contrast.

| Input / setting | Description |
|-----------------|-------------|
| `merged` | Merge node output (upscaled RGB). |
| `reference` | Same shot before / without tiled upscale; batch **1** broadcasts across merged batch. |
| `low_frequency_sigma` | Gaussian σ (pixels) for LF/HF split; **14** default; **0** ≈ no blur (full-res pull — stronger reference imprint). |
| `color_pull` | How much LF RGB follows reference (**0.58** default). |
| `detail_mix` | Scale of merged highs (**1** = keep merged detail). |
| `preserve_merged_luminance` | **On** (default): per-pixel RGB scale so Rec.709 luma matches **merged** after the pull (reference brightness won’t wash the upscale). |
| `luma_scale_clamp` | Max RGB multiplier when locking luma (**4** default; reciprocal min). |
| `reference_resize` | **`bicubic`** / **`bilinear`** / **`area`** when upsampling reference to merged resolution. |

Long IMAGE batches (many frames × large resolution) are processed in **chunks** so blur/pad stays under PyTorch’s **32-bit element limit** (~2³¹ elements per tensor).

### Video Tile Audio Present

One job only: **True** when the **`AUDIO`** bundle looks like real clip audio from **`Load Video`** (or similar), not an empty/silent stand‑in.

| Step | Behavior |
|------|----------|
| File path | If the dict exposes a resolvable path, **`ffprobe`** **≥ 1 audio stream** ⇒ passes stream gate; **0 streams** (video‑only) ⇒ **False**. If ffprobe is missing or errors ⇒ **warning**, gate skipped — decision falls back to waveform only (your MP3 is **not** rejected just because ffprobe failed). |
| Waveform | **Non‑silent:** **`min_peak` = 0** uses automatic peak/RMS floors; **`min_peak` > 0** applies your cutoff plus an RMS helper. |

The dict is scanned for path‑like strings ending in `.mp3`, `.wav`, `.mp4`, etc., not only fixed keys.

**Tensor layout:** `[B, C, T]` and obvious **`[B, T, C]`** (small channel dim) are both accepted.

| Input | Description |
|-------|-------------|
| `audio` | **`AUDIO`** from Load Video / Load Audio / VHS. |
| `min_peak` | **0** = auto silence detection; increase only if you need stricter silence rejection. |

**Output:** **`has_audio`**. Console shows **`peak`**, **`rms`**, and whether ffprobe confirmed streams.

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

**Wiring:** Slicer `tiles` → your processing → Merge `tiles`. Same slicer `tile_config` → Merge `tile_config`. Tune **`feather`** (0–0.5, fraction of tile W/H per axis; see merge table) and **`feather_curve`** if needed.

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
