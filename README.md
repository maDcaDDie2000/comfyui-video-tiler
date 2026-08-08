# ComfyUI Video Tiler

Memory-conscious video/image tiling for ComfyUI with overlap tiles, gaps, and feather blending.
The normal slicers use tensor views where possible, and the merge nodes write into a single output buffer.

This pack was built and tested primarily for **LTX 2.3** and **MiniMax H3** tiled video upscale workflows. Other models or node stacks may work, but they are not the main target.

> **Hardware disclaimer:** Tiled video upscaling is still a demanding workflow
> intended for systems with high hardware specifications. Tiling reduces
> per-step memory, but long clips and high resolutions can still require a lot
> of VRAM, system RAM, processing time, and disk space. If the in-memory path
> does not fit, use the disk-backed workflow to process and save one tile at a
> time instead of keeping every processed tile in memory.

> **Maintenance disclaimer:** This node pack was vibe-coded for personal use
> and may not be actively maintained. Issues, compatibility updates, or pull
> requests are not guaranteed to be addressed. Use it at your own discretion.

## What Changed Recently

- Added optional `merge_device` to merge nodes: `auto`, `cpu`, or `cuda`.
  - `auto` keeps the first tile's device.
  - `cpu` merges in system RAM and returns a CPU IMAGE.
  - `cuda` uses VRAM when CUDA is available.
- Added disk-backed tile nodes for workflows where processed tiles are saved one by one and merged later; Disk Merge can wait until all expected tile files exist.
- Fixed direct `TILE_CONFIG` handling for helper/merge nodes.
- Fixed small-frame Reference Color Match crashes by falling back to replicate padding when reflect padding is too large.
- Removed tracked Python bytecode and added `.gitignore` entries for cache/output folders.

## Features

- **Video Tile Slicer (var. size)**: grid layout with explicit tile counts, gaps, seam tiles, and overlap extension.
- **Video Tile Slicer (fixed size)**: constant tile size, fractional overlap, and traversal patterns.
- **Video Tile Merge**: reconstructs either layout with adjustable feather, feather curve, blend mode, and merge device.
- **Disk-backed workflow**: process one tile per run, save numbered `.pt` tile files, and stream-merge saved tiles later.
- **Reference tile alignment**: cuts matching reference tiles from the same `tile_config`.
- **Reference color match**: post-merge low-frequency color pull toward a reference clip.
- **Audio present check**: detects real audio streams/waveform energy from `AUDIO` bundles.
- **Sampler timing**: measures sampler wall-clock time and exposes averages as workflow FLOATs.

Tiling reduces per-step VRAM, but total workflow memory still depends heavily on clip length, resolution, channels, dtype, and how ComfyUI caches nodes. Disk-backed tile saving is the lowest-VRAM path for the expensive upscale branch because processed tiles do not all need to remain in graph memory.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/maDcaDDie2000/comfyui-video-tiler
```

Restart ComfyUI. The regular nodes, including the sampler timing nodes, appear
under **Video Tiler**. Disk-backed nodes appear under **Video Tiler/Disk**.

## Node List

| Internal class | Display name |
|---|---|
| `VideoTileSlice` | Video Tile Slicer (var. size) |
| `VideoTileSliceFixed` | Video Tile Slicer (fixed size) |
| `VideoTileMerge` | Video Tile Merge |
| `VideoTileMergeOverlapSoft` | Video Tile Merge (overlap soft) |
| `GetTile` | Get Tile |
| `ReferenceTileSlice` | Reference Tile Slice |
| `VideoTileReferenceColorMatch` | Video Tile Reference Color Match |
| `VideoTileAudioFFprobeLTX` | Video Tile Audio Present |
| `VideoTileSamplerTimerStart` | Sampler Timer Start |
| `VideoTileSamplerTimerResult` | Sampler Timing Result |
| `VideoTileDiskJob` | Video Tile Disk Job |
| `VideoTileDiskOpenJob` | Video Tile Disk Open Job |
| `VideoTileDiskIndexes` | Video Tile Disk Indexes |
| `VideoTileDiskGetTile` | Video Tile Disk Get Tile |
| `VideoTileDiskSaveTile` | Video Tile Disk Save Tile |
| `VideoTileDiskMerge` | Video Tile Disk Merge |
| `VideoTileDiskFolderMerge` | Video Tile Disk Folder Merge |
| `VideoTileDiskPreview` | Video Tile Disk Preview |

## Example Workflows

Both supported targets include a normal in-memory example and a disk-buffered example:

| Target | Normal workflow | Disk-buffered workflow |
|---|---|---|
| LTX 2.3 | [Video Tiler LTX 2.3 upscaling workflow](example_workflow/Video%20Tiler%20LTX%202.3%20upscaling%20workflow.json) | [Video Tiler LTX 2.3 disk buffered workflow](example_workflow/Video%20Tiler%20LTX%202.3%20disk%20buffered%20workflow.json) |
| MiniMax H3 | [Video Tiler MiniMax H3 upscaler workflow](example_workflow/Video%20Tiler%20Minimax%20H3%20upscaler%20workflow.json) | [Video Tiler MiniMax H3 disk buffered workflow](example_workflow/Video%20Tiler%20Minimax%20H3%20disk%20buffered%20workflow.json) |

The normal examples keep the processed tiles in memory. The disk-buffered variants save one indexed tile per queued run, then merge the saved tiles and export the video in a second pass. Each disk-buffered workflow contains notes describing when to increment the tile index and when to enable the merge pass.

## Sampler Timing

Use the two timing nodes around a standard **KSampler**, **KSampler Advanced**, or
custom sampler pipeline:

1. Connect the checkpoint/model loader's `MODEL` to **Sampler Timer Start**.
2. Connect its `model` output to the sampler's `model` input. For a custom
   sampling pipeline, connect it where the MODEL enters the guider/pipeline.
3. Connect the sampler's `LATENT` output to **Sampler Timing Result** `samples`.
4. Connect `sampler_timer` between the two timing nodes.

**Sampler Timing Result** passes the LATENT through unchanged and provides:

| Output | Meaning |
|---|---|
| `average_seconds` | Average wall-clock seconds per completed sampler call. This is the main output for tiled/list sampling. |
| `total_seconds` | Total sampler time in the current queued workflow execution. |
| `seconds_per_step` | Total time divided by the number of sampled sigma intervals. |
| `sampler_calls` | Number of sampler calls included in the average. |
| `total_steps` | Number of sigma intervals included in `seconds_per_step`. |

The FLOAT outputs can connect to any node that accepts a FLOAT. GPU work is
synchronized at the measurement boundaries for useful wall-clock results. Model
loading, VAE decode, and other nodes are not included. Timing data is reset for
every queued workflow execution.

## Standard In-Memory Workflow

Use this when the full processed tile list fits comfortably in memory.

1. **Video Tile Slicer (var. size)** or **Video Tile Slicer (fixed size)**
2. Process `tiles` with your upscale branch.
3. Connect processed `tiles` plus the original `tile_config` into **Video Tile Merge**.
4. Tune `feather`, `feather_curve`, `blend_mode`, and `merge_device`.

The slicer `tiles` output is a list. ComfyUI can run downstream nodes in list context, which is usually much lighter than processing the full frame at once. The final merge still needs the output image/video buffer, and in the in-memory workflow ComfyUI may also keep processed tile results cached.

## Disk-Backed Low-VRAM Workflow

Use this when the expensive upscale branch cannot keep all processed tiles cached, or when you want to run very long clips on smaller PCs.

### Pass 1: Save Processed Tiles

1. Run either slicer to get `tile_config`.
2. Connect `tile_config` to **Video Tile Disk Job**.
   - `job_name`: stable folder name for this run.
   - `output_folder`: defaults visibly to the folder used for disk tile jobs; change it if needed.
   - `audio` (optional): connect the source `AUDIO` once to store it with the job for independent export.
   - Output `tile_job` is the manifest path used by the other disk nodes.
3. Choose a tile index.
   - Manual/reliable path: set `tile_index` yourself and queue one run per tile.
   - Helper path: use **Video Tile Disk Indexes** to emit `0..tile_count-1` for list execution or queue automation.
4. Use **Video Tile Disk Get Tile** with original `images`, `tile_job`, and `tile_index`.
5. Send that single tile through your expensive upscale branch.
6. Save the processed result with **Video Tile Disk Save Tile**.

Saved files are exact PyTorch tensor files:

```text
manifest.json
audio.pt          # present when source AUDIO was connected to Disk Job
tile_00000.pt
tile_00001.pt
tile_00002.pt
...
```

The numbering matches the original tile order in `tile_config`. The manifest records frame size, tile geometry, tile count, and saved tile metadata.

### Pass 2: Independent Folder Merge and Video Export

Pass 2 can live in a completely separate ComfyUI workflow and does not need an active slicer, Disk Job node, source video, or tile-processing branch:

1. Add **Video Tile Disk Folder Merge**.
2. Paste the completed job folder or its `manifest.json` path into `job_folder`.
3. Set `feather`, `feather_curve`, `blend_mode`, and `merge_device` to the values you want for the final assembly.
4. Connect its `IMAGE` and `audio` outputs to your video encoder (for example, VideoHelperSuite Video Combine), then configure frame rate and format there.
5. Queue this output branch independently. The node rescans the folder on every run. It stops with the saved count and exact missing tile indices until every expected file is present, then streams the tiles into the final frame batch.

For jobs created before audio storage was added, connect the original audio loader to Folder Merge's optional `audio_override` input. The resulting `audio` output can still be wired to the encoder without regenerating any tiles. An override takes priority over stored `audio.pt`.

If an existing graph still uses a `TILE_JOB` connection, **Video Tile Disk Open Job** can open the folder and provide that connection without rerunning the slicer. The original **Video Tile Disk Merge** remains available for connected graphs.

This avoids loading every processed tile at once. The final merged IMAGE still exists as one tensor, so very long videos can still require a lot of system RAM if `merge_device=cpu` or VRAM if `merge_device=cuda`.

### Review Tiles While Pass 1 Is Running

Use **Video Tile Disk Preview** as a separate review branch or in a small review-only workflow:

1. Point `job_folder` at the same disk job folder.
2. Choose `tile_index` and `frame_index`. Set `frame_index=-1` to output every frame stored in that tile.
3. Connect `preview` to ComfyUI's **Preview Image** node.
4. Leave `missing_tile=nearest_available` to keep reviewing while the requested tile is pending, or choose `error` when you only want the exact tile.

The preview node reports the actual tile/frame shown and the current `saved_count / tile_count`. It rescans the folder each time it runs, so it does not need to be connected to or wait for the active tile branch.

## Disk Node Details

### Video Tile Disk Job

Creates or updates a disk job manifest from a slicer `tile_config`.

| Input | Description |
|---|---|
| `tile_config` | From either slicer. |
| `job_name` | Folder-safe name for the tile job. |
| `output_folder` | Folder where disk tile job folders are written. The widget shows the default path. |
| `audio` | Optional source AUDIO saved as `audio.pt` for the independent merge/export workflow. |

Outputs: `tile_job`, `manifest_path`, `tile_count`, `status`.

### Video Tile Disk Open Job

Opens a saved job independently from the slicer/processing workflow.

| Input | Description |
|---|---|
| `job_folder` | Job folder, `manifest.json`, or a parent folder containing exactly one job. |

Outputs: `tile_job`, `manifest_path`, `saved_count`, `tile_count`, `status`, and stored `audio` when available.

### Video Tile Disk Indexes

Outputs an INT list of tile indices for a job. This can help queue all tile runs or drive ComfyUI list execution where your graph supports it.

| Input | Description |
|---|---|
| `tile_job` | Manifest from Disk Job. |
| `start_index` | First tile index. |
| `end_index` | Last tile index; `-1` means final tile. |

### Video Tile Disk Get Tile

Loads the manifest and extracts exactly one tile from the original IMAGE batch.

| Input | Description |
|---|---|
| `images` | Original full IMAGE batch `[B,H,W,C]`. |
| `tile_job` | Manifest from Disk Job. |
| `tile_index` | Tile index to extract. |

Outputs: `tile`, resolved `tile_index`, and `tile_job` passthrough.

### Video Tile Disk Save Tile

Saves one processed tile as `tile_XXXXX.pt` and updates the manifest.

| Input | Description |
|---|---|
| `tile` | Processed tile IMAGE. |
| `tile_job` | Manifest from Disk Job. |
| `tile_index` | Tile index for numbering. |
| `overwrite` | Replace an existing tile file for the same index. |

### Video Tile Disk Merge

Loads saved tiles one by one from disk and merges them.

| Input | Description |
|---|---|
| `tile_job` | Manifest from Disk Job. |
| `feather` | Same meaning as Video Tile Merge. |
| `feather_curve` | Optional: `linear`, `ease_in`, `ease_out`, `ease_in_out`. |
| `blend_mode` | Optional: `alpha_over` or `weighted_average`. |
| `merge_device` | Optional: `cpu` default, `auto`, or `cuda`. |
| `require_all_tiles` | Optional: default `True`; stop before merging until every expected tile file exists. |
| `audio_override` | Optional AUDIO for older jobs; takes priority over stored `audio.pt`. |

Outputs: merged `IMAGE` and stored/override `audio`.

### Video Tile Disk Folder Merge

Standalone final assembly node. It takes a folder/path widget instead of a `TILE_JOB` connection, always requires the complete expected tile set, and can therefore be queued in a separate export workflow. Connect its reconstructed `IMAGE` frame batch and stored/override `audio` outputs to your preferred video encoder.

| Input | Description |
|---|---|
| `job_folder` | Job folder, `manifest.json`, or a parent folder containing exactly one job. |
| `feather` | Same meaning as Video Tile Merge. |
| `feather_curve` | Optional seam-alpha curve. |
| `blend_mode` | Optional: `alpha_over` or `weighted_average`. |
| `merge_device` | Optional: `cpu` default, `auto`, or `cuda`. |
| `audio_override` | Optional AUDIO for an older job without `audio.pt`; takes priority when connected. |

Outputs: merged `IMAGE`, status, tile count, and stored/override `audio`.

### Video Tile Disk Preview

Loads one saved tile for interactive review, including while the job is incomplete.

| Input | Description |
|---|---|
| `job_folder` | Existing job folder or manifest path. |
| `tile_index` | Tile to inspect. |
| `frame_index` | Frame to inspect; `-1` returns the complete saved frame batch. |
| `missing_tile` | Use the nearest available tile or require the exact requested tile. |

Outputs: preview `IMAGE`, actual tile/frame indices, saved/total counts, tile path, and status.

## Slicer Nodes

### Video Tile Slicer (var. size)

Splits a video/image batch into a grid with normal tiles, gaps, and overlap/seam tiles.

| Input | Description |
|---|---|
| `images` | IMAGE `[B,H,W,C]`. |
| `tiles_x` | 1-5 horizontal cells. |
| `tiles_y` | 1-5 vertical cells. |
| `multiple` | Tile size multiple, such as 8, 16, or 32. |
| `overlap_extension_x` / `overlap_extension_y` | Pixels seams extend into adjacent tiles, snapped to multiple. |

Outputs: `tiles`, `tile_config`, `visualization`, `tile_count`, `layout_label`.

`tile_config` is geometry-only v4. Changing merge feather does not require re-slicing.

### Video Tile Slicer (fixed size)

Uses fixed tile size and fractional overlap.

| Input | Description |
|---|---|
| `images` | IMAGE `[B,H,W,C]`. |
| `tile_width` / `tile_height` | Tile size. |
| `multiple` | Position/size snap multiple. |
| `overlap` | `1/8`, `1/4`, `3/8`, or `1/2` minimum neighbor overlap. |
| `pattern` | `row`, `column`, `spiral`, or `double_spiral`. |

Outputs match the variable-size slicer. New slices emit `tile_config` v5. Legacy v3 configs still merge.

## Merge Nodes

### Video Tile Merge

Reconstructs full IMAGE output from processed tiles.

| Input | Description |
|---|---|
| `tile_config` | From either slicer. |
| `tiles` | Processed tile list in slicer order. |
| `feather` | `0.0` to `0.5`, fraction of local tile width/height used for alpha ramps. |
| `feather_curve` | Optional alpha remap. |
| `blend_mode` | `alpha_over` or `weighted_average`. |
| `merge_device` | `auto`, `cpu`, or `cuda`. |

`alpha_over` uses painter order with a coverage gate. `weighted_average` computes normalized geometry-weighted color sums.

### Video Tile Merge (overlap soft)

Compatibility node for normalized weighted overlap merging. The main merge node can now do the same style with `blend_mode=weighted_average`.

## Helper Nodes

### Get Tile

Returns one tile by index from a full IMAGE or from a slicer tile list.

### Reference Tile Slice

Cuts matching reference tiles from a reference image using the same `tile_config`.

### Video Tile Reference Color Match

Post-merge color alignment. It resizes a reference clip, pulls low-frequency RGB toward it, and preserves merged high-frequency detail. Long batches are chunked to avoid PyTorch indexing limits.

### Video Tile Audio Present

Returns true when an `AUDIO` bundle appears to contain real audio. If a path is available and `ffprobe` works, it requires at least one audio stream; otherwise it falls back to waveform energy.

## Practical Notes

- Disk tiles are `.pt` tensor files, not PNG/video files. This is intentional: it is lossless and preserves exact batch tensors.
- Disk merge greatly reduces tile-cache memory, but the final merged IMAGE tensor still has to exist.
- For the lowest VRAM during merge, use **Video Tile Disk Merge** with `merge_device=cpu`.
- For fastest merge when VRAM is available, use `merge_device=cuda`.
- Existing workflows that use the original slicer and merge nodes should continue to load because the disk workflow is added as separate nodes.

## Compatibility

- Primary tested targets: LTX 2.3 and MiniMax H3 workflows.
- IMAGE tensors use ComfyUI layout `[B,H,W,C]`.
- Compatible with VideoHelperSuite-style IMAGE/AUDIO use.

## License

Licensed under the [Apache License 2.0](LICENSE).
