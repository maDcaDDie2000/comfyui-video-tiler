
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
_AUDIO_FILENAME = "audio.pt"


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


def _default_root_string() -> str:
    return str(_default_root())


def _job_dir(output_folder: str, job_name: str) -> Path:
    root = Path(output_folder).expanduser() if output_folder else _default_root()
    return root / _safe_job_name(job_name)


def _tile_path(job_dir: Path, tile_index: int) -> Path:
    return job_dir / f"tile_{tile_index:0{_TILE_DIGITS}d}.pt"


def _save_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temp_path.replace(path)


def _resolve_manifest_path(folder_or_manifest) -> Path:
    """Resolve a standalone job folder, manifest path, or single-job parent folder."""
    raw = _scalar_str(folder_or_manifest).strip()
    candidate = Path(raw).expanduser() if raw else _default_root()
    if candidate.is_file():
        return candidate
    if not candidate.exists():
        raise FileNotFoundError(f"Tile job folder or manifest not found: {candidate}")
    if not candidate.is_dir():
        raise FileNotFoundError(f"Tile job path is not a folder or manifest: {candidate}")

    direct = candidate / "manifest.json"
    if direct.is_file():
        return direct

    manifests = sorted(path for path in candidate.glob("*/manifest.json") if path.is_file())
    if len(manifests) == 1:
        return manifests[0]
    if not manifests:
        raise FileNotFoundError(f"No manifest.json found in tile job folder: {candidate}")
    raise ValueError(
        f"Multiple tile jobs found in {candidate}; choose a specific job folder or manifest.json"
    )


def _load_manifest(tile_job) -> tuple[Path, dict[str, Any]]:
    manifest_path = _resolve_manifest_path(tile_job)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(data.get("manifest_version", 0)) != _MANIFEST_VERSION:
        raise ValueError(f"Unsupported tile job manifest version: {data.get('manifest_version')}")
    if "tile_config" not in data or "tile_count" not in data:
        raise ValueError(f"Invalid tile job manifest (missing tile_config/tile_count): {manifest_path}")
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


def _unwrap_audio(audio):
    while isinstance(audio, (list, tuple)) and len(audio) == 1:
        audio = audio[0]
    return audio


def _cpu_serializable(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value, dict):
        return {key: _cpu_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_serializable(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_serializable(item) for item in value)
    return value


def _save_job_audio(job_dir: Path, audio) -> dict[str, Any] | None:
    audio = _unwrap_audio(audio)
    if audio is None:
        return None
    path = job_dir / _AUDIO_FILENAME
    temp_path = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "audio": _cpu_serializable(audio),
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    torch.save(payload, temp_path)
    temp_path.replace(path)
    metadata: dict[str, Any] = {"path": path.name, "saved_at": payload["saved_at"]}
    if isinstance(audio, dict) and audio.get("sample_rate") is not None:
        metadata["sample_rate"] = _scalar_int(audio["sample_rate"])
    return metadata


def _load_job_audio(manifest_path: Path, data: dict[str, Any]):
    metadata = data.get("audio")
    relative_path = metadata.get("path") if isinstance(metadata, dict) else None
    path = manifest_path.parent / (relative_path or _AUDIO_FILENAME)
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "audio" in payload:
        return payload["audio"]
    return payload


def _resolve_job_audio(manifest_path: Path, data: dict[str, Any], audio_override=None):
    override = _unwrap_audio(audio_override)
    if override is not None:
        return override
    return _load_job_audio(manifest_path, data)


def _sorted_tile_specs(config_tuple: tuple[Any, ...]):
    _, _, _, tile_specs = parse_tile_config(config_tuple)
    return sorted(enumerate(tile_specs), key=lambda item: item[1].order)


def _available_tile_indices(manifest_path: Path, data: dict[str, Any], tile_count: int) -> set[int]:
    # The folder containing manifest.json is authoritative. This keeps copied or moved
    # job folders usable even when an older absolute job_dir remains in the manifest.
    job_dir = manifest_path.parent
    available: set[int] = set()
    for raw_idx, meta in (data.get("saved_tiles") or {}).items():
        try:
            idx = int(raw_idx)
        except (TypeError, ValueError):
            continue
        rel = meta.get("path") if isinstance(meta, dict) else None
        path = job_dir / rel if rel else _tile_path(job_dir, idx)
        if 0 <= idx < tile_count and path.is_file():
            available.add(idx)
    for idx in range(tile_count):
        if _tile_path(job_dir, idx).is_file():
            available.add(idx)
    return available


def _format_missing_tiles(missing: list[int]) -> str:
    preview = ", ".join(str(i) for i in missing[:12])
    more = "..." if len(missing) > 12 else ""
    return f"{preview}{more}"


def _job_progress(folder_or_manifest) -> tuple[Path, dict[str, Any], list[int], list[int]]:
    manifest_path, data = _load_manifest(folder_or_manifest)
    tile_count = int(data["tile_count"])
    _width, _height, _multiple, tile_specs = parse_tile_config(_manifest_tile_config(data))
    if tile_count != len(tile_specs):
        raise ValueError(
            f"Tile job manifest count mismatch: tile_count={tile_count}, "
            f"tile_config has {len(tile_specs)} tiles ({manifest_path})"
        )
    available = sorted(_available_tile_indices(manifest_path, data, tile_count))
    available_set = set(available)
    missing = [i for i in range(tile_count) if i not in available_set]
    return manifest_path, data, available, missing


class VideoTileDiskJob:
    """Create a manifest for a disk-backed tile job from an existing TILE_CONFIG."""

    DESCRIPTION = (
        "Creates a disk-backed tile job manifest from a slicer tile_config. "
        "Optionally stores source audio for independent final export. Use this with Disk Tile by Index, "
        "Save Disk Tile, and Merge Disk Tiles."
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
                    {"default": _default_root_string(), "tooltip": "Folder where disk tile job folders are written."},
                ),
            },
            "optional": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Optional source audio to store once as audio.pt with the disk job. "
                            "Folder Merge will reload and output it."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("TILE_JOB", "STRING", "INT", "STRING")
    RETURN_NAMES = ("tile_job", "manifest_path", "tile_count", "status")
    FUNCTION = "plan"
    CATEGORY = "Video Tiler/Disk"

    def plan(self, tile_config, job_name: str, output_folder: str = "", audio=None):
        config_tuple = unwrap_tile_config(tile_config)
        width, height, multiple, tile_specs = parse_tile_config(config_tuple)
        job_dir = _job_dir(_scalar_str(output_folder), _scalar_str(job_name, "video_tile_job"))
        job_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = job_dir / "manifest.json"
        tile_count = len(tile_specs)
        existing_data: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                existing_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception:
                existing_data = {}
        existing_saved = existing_data.get("saved_tiles", {})
        audio_metadata = _save_job_audio(job_dir, audio)
        if audio_metadata is None and (job_dir / _AUDIO_FILENAME).is_file():
            audio_metadata = existing_data.get("audio") or {"path": _AUDIO_FILENAME}
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
        if audio_metadata is not None:
            data["audio"] = audio_metadata
        _save_manifest(manifest_path, data)
        audio_status = "audio stored" if audio_metadata is not None else "no stored audio"
        status = f"Tile job ready: {tile_count} tiles, {audio_status} -> {manifest_path}"
        print(f"[Video Tiler] {status}")
        return (str(manifest_path), str(manifest_path), tile_count, status)


class VideoTileDiskOpenJob:
    """Open an existing disk job without running a slicer or tile-processing branch."""

    DESCRIPTION = (
        "Opens an existing tile job folder or manifest independently of the tile-generation workflow. "
        "Use its tile_job output with Disk Merge, Disk Indexes, or other disk nodes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_folder": (
                    "STRING",
                    {
                        "default": _default_root_string(),
                        "tooltip": "Job folder, manifest.json, or a parent folder containing exactly one tile job.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("TILE_JOB", "STRING", "INT", "INT", "STRING", "AUDIO")
    RETURN_NAMES = ("tile_job", "manifest_path", "saved_count", "tile_count", "status", "audio")
    FUNCTION = "open_job"
    CATEGORY = "Video Tiler/Disk"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def open_job(self, job_folder: str):
        manifest_path, data, available, missing = _job_progress(job_folder)
        tile_count = int(data["tile_count"])
        audio = _load_job_audio(manifest_path, data)
        if missing:
            status = (
                f"Tile job incomplete: {len(available)}/{tile_count} saved; "
                f"missing {_format_missing_tiles(missing)}"
            )
        else:
            status = f"Tile job complete: {tile_count}/{tile_count} saved"
        status += "; stored audio available" if audio is not None else "; no stored audio"
        print(f"[Video Tiler] {status}: {manifest_path}")
        return (str(manifest_path), str(manifest_path), len(available), tile_count, status, audio)


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
        job_dir = manifest_path.parent
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
        temp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temp_path)
        temp_path.replace(path)
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
                "require_all_tiles": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "Stop before merging until every expected tile file is available."},
                ),
                "audio_override": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Optional audio for an older job without stored audio. Overrides audio.pt when connected."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("IMAGE", "audio")
    FUNCTION = "merge"
    CATEGORY = "Video Tiler/Disk"

    def merge(
        self,
        tile_job,
        feather,
        feather_curve=None,
        blend_mode=None,
        merge_device=None,
        require_all_tiles=True,
        audio_override=None,
    ):
        manifest_path, data = _load_manifest(tile_job)
        audio = _resolve_job_audio(manifest_path, data, audio_override)
        config_tuple = _manifest_tile_config(data)
        width, height, _multiple, tile_specs = parse_tile_config(config_tuple)
        tile_count = len(tile_specs)
        available = _available_tile_indices(manifest_path, data, tile_count)
        missing = [i for i in range(tile_count) if i not in available]
        if missing and _scalar_bool(require_all_tiles, True):
            raise RuntimeError(
                f"Disk merge waiting for tiles: saved {len(available)}/{tile_count}; "
                f"missing {_format_missing_tiles(missing)}"
            )
        if missing:
            print(
                f"[Video Tiler] Disk merge warning: saved {len(available)}/{tile_count}; "
                f"missing {_format_missing_tiles(missing)}"
            )

        mode = _blend_mode_keyword(blend_mode)
        curve = _feather_curve_mode(feather_curve)
        dev_mode = _device_mode_keyword(merge_device)
        frac = _strip_fraction(_scalar_float(feather))
        sorted_pairs = _sorted_tile_specs(config_tuple)
        if not available:
            raise RuntimeError(f"Disk merge found no saved tiles in: {manifest_path.parent}")
        first_idx, _first_spec = next(pair for pair in sorted_pairs if pair[0] in available)
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
                if orig_idx not in available:
                    continue
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
            print(f"[Video Tiler] Disk merge complete: {len(available)} tiles -> IMAGE ({mode}, device={device})")
            return (out.to(dtype=dtype), audio)

        output = torch.zeros((B, height, width, C), dtype=dtype, device=device)
        covered = torch.zeros((B, height, width), dtype=torch.bool, device=device)
        for paint_i, (orig_idx, spec) in enumerate(sorted_pairs):
            if orig_idx not in available:
                continue
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
        print(f"[Video Tiler] Disk merge complete: {len(available)} tiles -> IMAGE ({mode}, device={device})")
        return (output, audio)


class VideoTileDiskFolderMerge:
    """Standalone folder-driven merge for a separate final-export workflow."""

    DESCRIPTION = (
        "Independently scans an existing disk tile job folder and merges it only when every expected "
        "tile is available. Outputs stored or override audio with the IMAGE frame batch; no active slicer "
        "or tile run is required."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_folder": (
                    "STRING",
                    {
                        "default": _default_root_string(),
                        "tooltip": "Job folder, manifest.json, or a parent folder containing exactly one tile job.",
                    },
                ),
                "feather": (
                    "FLOAT",
                    {"default": 0.125, "min": 0.0, "max": 0.5, "step": 0.005},
                ),
            },
            "optional": {
                "feather_curve": (["linear", "ease_in", "ease_out", "ease_in_out"], {"default": "linear"}),
                "blend_mode": (["alpha_over", "weighted_average"], {"default": "alpha_over"}),
                "merge_device": (["auto", "cpu", "cuda"], {"default": "cpu"}),
                "audio_override": (
                    "AUDIO",
                    {
                        "tooltip": (
                            "Optional audio for an existing job without audio.pt. "
                            "When connected, this takes priority over stored audio."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "AUDIO")
    RETURN_NAMES = ("IMAGE", "status", "tile_count", "audio")
    FUNCTION = "merge_folder"
    CATEGORY = "Video Tiler/Disk"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # Folder contents can change while the workflow graph and widgets stay unchanged.
        return float("nan")

    def merge_folder(
        self,
        job_folder,
        feather,
        feather_curve=None,
        blend_mode=None,
        merge_device=None,
        audio_override=None,
    ):
        manifest_path, data, available, missing = _job_progress(job_folder)
        tile_count = int(data["tile_count"])
        if missing:
            raise RuntimeError(
                f"Standalone disk merge waiting for tiles: saved {len(available)}/{tile_count}; "
                f"missing {_format_missing_tiles(missing)}; folder {manifest_path.parent}"
            )
        image, audio = VideoTileDiskMerge().merge(
            str(manifest_path),
            feather,
            feather_curve=feather_curve,
            blend_mode=blend_mode,
            merge_device=merge_device,
            require_all_tiles=True,
            audio_override=audio_override,
        )
        audio_status = "audio ready" if audio is not None else "no audio stored or connected"
        status = (
            f"Standalone disk merge complete: {tile_count}/{tile_count} tiles, "
            f"{audio_status} from {manifest_path.parent}"
        )
        return (image, status, tile_count, audio)


class VideoTileDiskPreview:
    """Interactively inspect saved tiles from an incomplete or complete disk job."""

    DESCRIPTION = (
        "Loads one already-saved tile (and optionally one frame) from a disk job folder for review. "
        "Works while other tiles are still being generated and does not require an active tile run."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "job_folder": (
                    "STRING",
                    {
                        "default": _default_root_string(),
                        "tooltip": "Job folder, manifest.json, or a parent folder containing exactly one tile job.",
                    },
                ),
                "tile_index": ("INT", {"default": 0, "min": 0, "max": 999999}),
                "frame_index": (
                    "INT",
                    {
                        "default": 0,
                        "min": -1,
                        "max": 999999,
                        "tooltip": "Frame to preview. Use -1 to output the tile's complete frame batch.",
                    },
                ),
                "missing_tile": (
                    ["nearest_available", "error"],
                    {
                        "default": "nearest_available",
                        "tooltip": (
                            "Use the closest saved tile while the requested tile is still pending, "
                            "or stop with an error."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = (
        "preview",
        "actual_tile_index",
        "actual_frame_index",
        "saved_count",
        "tile_count",
        "tile_path",
        "status",
    )
    FUNCTION = "preview"
    CATEGORY = "Video Tiler/Disk"

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def preview(self, job_folder, tile_index=0, frame_index=0, missing_tile="nearest_available"):
        manifest_path, data, available, missing = _job_progress(job_folder)
        tile_count = int(data["tile_count"])
        if not available:
            raise RuntimeError(f"Tile preview found no saved tiles yet in: {manifest_path.parent}")

        requested = max(0, min(_scalar_int(tile_index), tile_count - 1))
        actual = requested
        used_fallback = False
        if actual not in available:
            if _scalar_str(missing_tile) == "error":
                raise RuntimeError(
                    f"Tile {actual} is not available yet; saved {len(available)}/{tile_count}; "
                    f"missing {_format_missing_tiles(missing)}"
                )
            actual = min(available, key=lambda idx: (abs(idx - requested), idx))
            used_fallback = True

        path = _tile_path(manifest_path.parent, actual)
        tile = _load_saved_tile(path)
        if tile.ndim != 4:
            raise ValueError(f"Saved tile shape {tuple(tile.shape)} - expected (B,H,W,C): {path}")
        requested_frame = _scalar_int(frame_index)
        if requested_frame < 0:
            preview = tile
            actual_frame = -1
        else:
            actual_frame = max(0, min(requested_frame, tile.shape[0] - 1))
            preview = tile[actual_frame : actual_frame + 1]

        fallback = f" (requested {requested}, nearest saved tile used)" if used_fallback else ""
        status = (
            f"Preview tile {actual}/{tile_count - 1}{fallback}; frame "
            f"{'all' if actual_frame < 0 else actual_frame}; saved {len(available)}/{tile_count}"
        )
        print(f"[Video Tiler] {status}: {path}")
        return (preview, actual, actual_frame, len(available), tile_count, str(path), status)
