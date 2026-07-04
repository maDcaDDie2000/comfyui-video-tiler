
"""
Disk-backed tile workflow nodes.

These nodes intentionally sit beside the existing in-memory slicer/merge nodes. They reuse
TILE_CONFIG geometry and merge helpers, but store processed tiles on disk so expensive tile
branches can be run one tile at a time and merged later without keeping every tile cached.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import torch

from .fixed_layout import compute_fixed_feather_strips
from .get_tile_node import get_tile_by_index
from .merge_node import (
    _apply_feather_curve,
    _blend_mode_keyword,
    _blend_tile_with_coverage,
    _device_mode_keyword,
    _feather_curve_mode,
    _feather_mask,
    _fixed_tile_top_alpha,
    _normalize_tile,
    _scalar_float,
    _select_merge_device,
    _strip_fraction,
    _weighted_accum_region,
)
from .tile_config import parse_tile_config, unwrap_tile_config


_MANIFEST_VERSION = 1
_TILE_DIGITS = 5


def _scalar_int(v, default: int = 0) -> int:
    while isinstance(v, (list, tuple)):
        if len(v) == 0:
            return default
        v = v[0]
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _scalar_bool(v, default: bool = False) -> bool:
    while isinstance(v, (list, tuple)):
        if len(v) == 0:
            return default
        v = v[0]
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return default


def _scalar_str(v, default: str = "") -> str:
    while isinstance(v, (list, tuple)):
        if len(v) == 0:
            return default
        v = v[0]
    if v is None:
        return default
    return str(v)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(x) for x in value]
    if isinstance(value, list):
        return [_jsonable(x) for x in value]
    return value


def _tupleize(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_tupleize(x) for x in value)
    return value


def _safe_job_name(name: str) -> str:
    name = (name or "video_tile_job").strip()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._")
    return name or "video_tile_job"


def _default_root() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.get_output_directory()) / "video_tiler_tiles"
    except Exception:
        return Path.cwd() / "video_tiler_tiles"


def _job_dir(output_folder: str, job_name: str) -> Path:
    root = Path(output_folder).expanduser() if output_folder else _default_root()
    return root / _safe_job_name(job_name)


def _tile_path(job_dir: Path, tile_index: int) -> Path:
    return job_dir / f"tile_{tile_index:0{_TILE_DIGITS}d}.pt"


def _save_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _load_manifest(tile_job) -> tuple[Path, dict[str, Any]]:
    manifest_path = Path(_scalar_str(tile_job)).expanduser()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Tile job manifest not found: {manifest_path}")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(data.get("manifest_version", 0)) != _MANIFEST_VERSION:
        raise ValueError(f"Unsupported tile job manifest version: {data.get('manifest_version')}")
    return manifest_path, data


def _manifest_tile_config(data: dict[str, Any]) -> tuple[Any, ...]:
    return unwrap_tile_config(_tupleize(data["tile_config"]))


def _load_saved_tile(path: Path) -> torch.Tensor:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "tile" in payload:
        tile = payload["tile"]
    else:
        tile = payload
    tile = _normalize_tile(tile)
    if not isinstance(tile, torch.Tensor):
        raise ValueError(f"Saved tile is not a tensor: {path}")
    return tile


def _sorted_tile_specs(config_tuple: tuple[Any, ...]):
    _, _, _, tile_specs = parse_tile_config(config_tuple)
    return sorted(enumerate(tile_specs), key=lambda item: item[1].order)


class VideoTileDiskJob:
    """Create a manifest for a disk-backed tile job from an existing TILE_CONFIG."""

    DESCRIPTION = (
        "Creates a disk-backed tile job manifest from a slicer tile_config. "
        "Use this with Disk Tile by Index, Save Disk Tile, and Merge Disk Tiles."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_config": (
                    "TILE_CONFIG",
                    {"forceInput": True, "tooltip": "tile_config from either Video Tile Slicer."},
                ),
                "job_name": (
                    "STRING",
                    {"default": "video_tile_job", "tooltip": "Folder-safe job name used for manifest and tile files."},
                ),
                "output_folder": (
                    "STRING",
                    {"default": "", "tooltip": "Empty = ComfyUI output/video_tiler_tiles. Otherwise use this folder."},
                ),
            },
        }

    RETURN_TYPES = ("TILE_JOB", "STRING", "INT", "STRING")
    RETURN_NAMES = ("tile_job", "manifest_path", "tile_count", "status")
    FUNCTION = "plan"
    CATEGORY = "Video Tiler/Disk"

    def plan(self, tile_config, job_name: str, output_folder: str = ""):
        config_tuple = unwrap_tile_config(tile_config)
        width, height, multiple, tile_specs = parse_tile_config(config_tuple)
        job_dir = _job_dir(_scalar_str(output_folder), _scalar_str(job_name, "video_tile_job"))
        job_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = job_dir / "manifest.json"
        tile_count = len(tile_specs)
        existing_saved: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                existing_saved = json.loads(manifest_path.read_text(encoding="utf-8")).get("saved_tiles", {})
            except Exception:
                existing_saved = {}
        data = {
            "manifest_version": _MANIFEST_VERSION,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "job_name": _safe_job_name(_scalar_str(job_name, "video_tile_job")),
            "job_dir": str(job_dir.resolve()),
            "tile_pattern": f"tile_%0{_TILE_DIGITS}d.pt",
            "tile_config": _jsonable(config_tuple),
            "width": int(width),
            "height": int(height),
            "multiple": int(multiple),
            "tile_count": int(tile_count),
            "saved_tiles": existing_saved,
        }
        _save_manifest(manifest_path, data)
        status = f"Tile job ready: {tile_count} tiles -> {manifest_path}"
        print(f"[Video Tiler] {status}")
        return (str(manifest_path), str(manifest_path), tile_count, status)


class VideoTileDiskIndexes:
    """Emit tile indices for a disk job, useful for queueing or Comfy list execution."""

    DESCRIPTION = "Outputs the tile indices for a disk tile job. Connect to Disk Tile by Index or use for queue automation."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_job": ("TILE_JOB", {"forceInput": True}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "end_index": (
                    "INT",
                    {"default": -1, "min": -1, "max": 999999, "tooltip": "-1 means the final tile."},
                ),
            },
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("tile_indices", "tile_count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "indices"
    CATEGORY = "Video Tiler/Disk"

    def indices(self, tile_job, start_index: int = 0, end_index: int = -1):
        _manifest_path, data = _load_manifest(tile_job)
        tile_count = int(data["tile_count"])
        start = max(0, min(_scalar_int(start_index), max(0, tile_count - 1)))
        end = _scalar_int(end_index, -1)
        if end < 0:
            end = tile_count - 1
        end = max(start, min(end, tile_count - 1))
        return (list(range(start, end + 1)), tile_count)


class VideoTileDiskGetTile:
    """Extract exactly one tile from the original IMAGE batch using a disk job manifest."""

    DESCRIPTION = "Reads a disk tile job manifest and extracts one source tile by index for the expensive branch."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Original full IMAGE batch [B,H,W,C]."}),
                "tile_job": ("TILE_JOB", {"forceInput": True}),
                "tile_index": ("INT", {"default": 0, "min": 0, "max": 999999}),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "TILE_JOB")
    RETURN_NAMES = ("tile", "tile_index", "tile_job")
    INPUT_IS_LIST = (False, False, True)
    FUNCTION = "get_tile"
    CATEGORY = "Video Tiler/Disk"

    def get_tile(self, images: torch.Tensor, tile_job, tile_index: int):
        manifest_path, data = _load_manifest(tile_job)
        config_tuple = _manifest_tile_config(data)
        _, _, _, tile_specs = parse_tile_config(config_tuple)
        n = len(tile_specs)
        ti = max(0, min(_scalar_int(tile_index), n - 1))
        images = _normalize_tile(images)
        tile = get_tile_by_index(images, config_tuple, ti)
        return (tile, ti, str(manifest_path))


class VideoTileDiskSaveTile:
    """Save one processed tile tensor to the disk job folder."""

    DESCRIPTION = "Saves a processed tile as a numbered .pt tensor file and updates the tile job manifest."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile": ("IMAGE", {"tooltip": "Processed tile to save."}),
                "tile_job": ("TILE_JOB", {"forceInput": True}),
                "tile_index": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "overwrite": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("tile_path", "tile_index", "saved")
    INPUT_IS_LIST = (True, False, True, False)
    FUNCTION = "save_tile"
    CATEGORY = "Video Tiler/Disk"

    def save_tile(self, tile, tile_job, tile_index: int, overwrite=True):
        manifest_path, data = _load_manifest(tile_job)
        tile_count = int(data["tile_count"])
        ti = max(0, min(_scalar_int(tile_index), tile_count - 1))
        job_dir = Path(data.get("job_dir") or manifest_path.parent)
        path = _tile_path(job_dir, ti)
        if path.exists() and not _scalar_bool(overwrite, True):
            raise FileExistsError(f"Tile already exists and overwrite is off: {path}")
        tensor = _normalize_tile(tile)
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Save Disk Tile expected an IMAGE tensor")
        payload = {
            "tile": tensor.detach().cpu().contiguous(),
            "tile_index": ti,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        saved_tiles = data.setdefault("saved_tiles", {})
        saved_tiles[str(ti)] = {"path": path.name, "shape": payload["shape"], "dtype": payload["dtype"]}
        _save_manifest(manifest_path, data)
        print(f"[Video Tiler] Saved tile {ti:0{_TILE_DIGITS}d}/{tile_count - 1:0{_TILE_DIGITS}d}: {path}")
        return (str(path), ti, True)


class VideoTileDiskMerge:
    """Stream saved tiles from disk and merge them without loading all tiles at once."""

    DESCRIPTION = "Loads numbered .pt tiles one at a time from a disk tile job and merges them into one IMAGE."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tile_job": ("TILE_JOB", {"forceInput": True}),
                "feather": (
                    "FLOAT",
                    {"default": 0.125, "min": 0.0, "max": 0.5, "step": 0.005},
                ),
            },
            "optional": {
                "feather_curve": (["linear", "ease_in", "ease_out", "ease_in_out"], {"default": "linear"}),
                "blend_mode": (["alpha_over", "weighted_average"], {"default": "alpha_over"}),
                "merge_device": (["auto", "cpu", "cuda"], {"default": "cpu"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "merge"
    CATEGORY = "Video Tiler/Disk"

    def merge(self, tile_job, feather, feather_curve=None, blend_mode=None, merge_device=None):
        manifest_path, data = _load_manifest(tile_job)
        config_tuple = _manifest_tile_config(data)
        width, height, _multiple, tile_specs = parse_tile_config(config_tuple)
        saved_tiles = data.get("saved_tiles", {})
        missing = [i for i in range(len(tile_specs)) if str(i) not in saved_tiles and not _tile_path(manifest_path.parent, i).is_file()]
        if missing:
            preview = ", ".join(str(i) for i in missing[:12])
            more = "..." if len(missing) > 12 else ""
            raise FileNotFoundError(f"Missing saved tiles: {preview}{more}")

        mode = _blend_mode_keyword(blend_mode)
        curve = _feather_curve_mode(feather_curve)
        dev_mode = _device_mode_keyword(merge_device)
        frac = _strip_fraction(_scalar_float(feather))
        sorted_pairs = _sorted_tile_specs(config_tuple)
        first_idx, _first_spec = sorted_pairs[0]
        first_tile = _load_saved_tile(_tile_path(manifest_path.parent, first_idx))
        if len(first_tile.shape) < 4:
            raise ValueError(f"Tile shape {first_tile.shape} - expected (B,H,W,C)")
        B, C = first_tile.shape[0], first_tile.shape[3]
        device = _select_merge_device(first_tile, None, dev_mode)
        dtype = first_tile.dtype

        fixed_bx = fixed_by = 0
        ver = config_tuple[0]
        if ver == 5:
            (_v, _w, _h, tw, th, ox, oy, mult, _pattern_id, _tiles_data) = config_tuple
            fixed_bx, fixed_by = compute_fixed_feather_strips(tw, th, ox, oy, mult, frac)
        elif ver == 3:
            (_v, _w, _h, _tw, _th, _ox, _oy, fixed_bx, fixed_by, _pattern_id, _tiles_data) = config_tuple
        fixed_bx = max(0, int(fixed_bx))
        fixed_by = max(0, int(fixed_by))

        if mode == "weighted_average":
            num = torch.zeros((B, height, width, C), dtype=torch.float32, device=device)
            den = torch.zeros((B, height, width), dtype=torch.float32, device=device)
            for paint_i, (orig_idx, spec) in enumerate(sorted_pairs):
                tile = _load_saved_tile(_tile_path(manifest_path.parent, orig_idx))
                x, y, w, h = spec.x, spec.y, spec.w, spec.h
                if ver in (3, 5):
                    if paint_i == 0:
                        wmask = torch.ones((h, w), dtype=torch.float32, device=device)
                    else:
                        wmask = _fixed_tile_top_alpha(w, h, x, y, width, height, fixed_bx, fixed_by, device, dtype)
                elif spec.type == "normal":
                    wmask = torch.ones((h, w), dtype=torch.float32, device=device)
                else:
                    wmask = _feather_mask(w, h, frac, x, y, width, height).to(device=device)
                wmask = _apply_feather_curve(wmask, curve)
                _weighted_accum_region(num, den, tile, y, x, h, w, wmask)
                del tile
            out = num / den.unsqueeze(-1).clamp(min=1e-8)
            print(f"[Video Tiler] Disk merge complete: {len(sorted_pairs)} tiles -> IMAGE ({mode}, device={device})")
            return (out.to(dtype=dtype),)

        output = torch.zeros((B, height, width, C), dtype=dtype, device=device)
        covered = torch.zeros((B, height, width), dtype=torch.bool, device=device)
        for paint_i, (orig_idx, spec) in enumerate(sorted_pairs):
            tile = _load_saved_tile(_tile_path(manifest_path.parent, orig_idx)).to(device)
            x, y, w, h = spec.x, spec.y, spec.w, spec.h
            if ver in (3, 5):
                if paint_i == 0:
                    output[:, y : y + h, x : x + w, :] = tile
                    covered[:, y : y + h, x : x + w] = True
                else:
                    alpha = _fixed_tile_top_alpha(w, h, x, y, width, height, fixed_bx, fixed_by, device, dtype)
                    _blend_tile_with_coverage(output, covered, tile, y, x, h, w, alpha, feather_curve=curve)
            elif spec.type == "normal":
                output[:, y : y + h, x : x + w, :] = tile
                covered[:, y : y + h, x : x + w] = True
            else:
                alpha = _feather_mask(w, h, frac, x, y, width, height).to(device=device)
                _blend_tile_with_coverage(output, covered, tile, y, x, h, w, alpha, feather_curve=curve)
            del tile
        print(f"[Video Tiler] Disk merge complete: {len(sorted_pairs)} tiles -> IMAGE ({mode}, device={device})")
        return (output,)
