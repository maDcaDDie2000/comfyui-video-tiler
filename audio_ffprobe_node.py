"""
Detect real audio vs silent placeholder from Load Video / Load Audio.

When a file path is known: ffprobe counts audio streams — zero streams ⇒ FALSE.
If ffprobe is missing or errors, falls back to waveform only (does not treat as FALSE).

Waveform: detects silence via peak or RMS; accepts common tensor layouts [B,C,T] vs [B,T,C].
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys

import torch

logger = logging.getLogger(__name__)

_MEDIA_EXTS = (
    ".mp3",
    ".wav",
    ".flac",
    ".ogg",
    ".m4a",
    ".aac",
    ".opus",
    ".wma",
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".avi",
)


def _unwrap_audio(audio):
    while isinstance(audio, (list, tuple)) and len(audio) == 1:
        audio = audio[0]
    return audio


def _resolve_filepath(raw: str | None) -> str | None:
    if not raw or not isinstance(raw, str):
        return None
    candidates = [raw.strip()]
    try:
        import folder_paths

        candidates.append(folder_paths.get_annotated_filepath(raw))
    except Exception:
        pass
    for p in candidates:
        if p and os.path.isfile(p):
            return os.path.abspath(os.path.normpath(p))
    return None


def _looks_media_path(s: str) -> bool:
    low = s.strip().lower()
    return any(low.endswith(ext) for ext in _MEDIA_EXTS)


def _path_from_audio_dict(d: dict) -> str | None:
    for key in (
        "filename",
        "path",
        "file",
        "filepath",
        "audio_file",
        "full_path",
        "loaded_path",
        "source",
        "audio_path",
    ):
        v = d.get(key)
        if isinstance(v, str):
            hit = _resolve_filepath(v)
            if hit:
                return hit
    for _k, v in d.items():
        if isinstance(v, str) and _looks_media_path(v):
            hit = _resolve_filepath(v)
            if hit:
                return hit
    return None


def _ffprobe_audio_stream_count(path: str) -> int | None:
    """Return number of audio streams, or None if ffprobe failed (caller may fallback)."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a",
        "-show_entries",
        "stream=index",
        "-of",
        "json",
        path,
    ]
    kwargs: dict = {
        "capture_output": True,
        "text": True,
        "timeout": 120,
    }
    if sys.platform == "win32":
        cnw = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        if cnw:
            kwargs["creationflags"] = cnw
    try:
        proc = subprocess.run(cmd, check=False, **kwargs)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    streams = data.get("streams") or []
    return len(streams)


def _normalize_bct(wf: torch.Tensor) -> torch.Tensor:
    """Force [B, C, T] when obvious; torchaudio/VHS sometimes differs."""
    if wf.dim() == 2:
        return wf.unsqueeze(0)
    if wf.dim() != 3:
        return wf
    _b, d1, d2 = wf.shape
    # Typical: C is 1–16, T is thousands+
    if d1 <= 16 and d2 > max(64, d1 * 32):
        return wf
    if d2 <= 16 and d1 > max(64, d2 * 32):
        return wf.permute(0, 2, 1)
    return wf


def _scalar_float(v, default: float) -> float:
    while isinstance(v, (list, tuple)):
        if len(v) == 0:
            return default
        v = v[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _waveform_has_energy(wf: torch.Tensor, min_peak: float) -> tuple[bool, float, float]:
    """peak and RMS in float32 over full tensor."""
    x = wf.detach().float().reshape(-1)
    if x.numel() == 0:
        return False, 0.0, 0.0
    peak = float(x.abs().max().cpu())
    rms = float(torch.sqrt(torch.mean(x * x)).cpu())
    if min_peak > 0:
        loud = peak >= min_peak or rms >= (min_peak * 0.01)
    else:
        # Auto: reject obvious flatlines; keep permissive for normal decoded audio
        loud = peak > 1e-14 or rms > 1e-16
    return loud, peak, rms


class VideoTileAudioFFprobeLTX:
    """
    True when media has an audio stream (if ffprobe can run) and waveform is not silent.
    """

    DESCRIPTION = (
        "True if clip has real audio: ffprobe reports ≥1 audio stream when file path is known "
        "(ffprobe errors fall back to waveform-only); waveform peak/RMS above silence floor."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": "Load Video / Load Audio output. Optional path in dict → ffprobe stream count; always checks waveform energy.",
                    },
                ),
            },
            "optional": {
                "min_peak": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 1e-12,
                        "tooltip": "Silence floor on |sample|. Use 0 for auto (~1e-12 effective via RMS branch).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("has_audio",)
    OUTPUT_TOOLTIPS = ("True if audio stream exists when probed and waveform is not silent.",)
    FUNCTION = "probe"
    CATEGORY = "Video Tiler"

    def probe(self, audio, min_peak=0.0):
        thresh = max(0.0, _scalar_float(min_peak, 0.0))

        def _fail(msg: str):
            print(f"[Video Tiler] Audio present check: FALSE — {msg}")
            return (False,)

        def _ok(msg: str):
            print(f"[Video Tiler] Audio present check: TRUE — {msg}")
            return (True,)

        audio = _unwrap_audio(audio)
        if not isinstance(audio, dict):
            return _fail("not an AUDIO dict")

        wf = audio.get("waveform")
        if wf is None or not isinstance(wf, torch.Tensor):
            return _fail("no waveform tensor")

        if wf.numel() == 0:
            return _fail("empty waveform")

        wf = _normalize_bct(wf)
        if wf.dim() != 3:
            return _fail(f"unsupported waveform rank {wf.dim()}, shape {tuple(wf.shape)}")

        loud, peak, rms = _waveform_has_energy(wf, thresh)

        path = _path_from_audio_dict(audio)
        stream_checked = False
        if path:
            n_audio = _ffprobe_audio_stream_count(path)
            if n_audio is None:
                print(
                    "[Video Tiler] Audio present check: ffprobe unavailable or failed — "
                    "using waveform-only (install FFmpeg on PATH for stream detection)"
                )
            else:
                stream_checked = True
                if n_audio < 1:
                    return _fail(f"no audio stream in file (ffprobe): {path}")

        if not loud:
            hint = f"min_peak={thresh:g}" if thresh > 0 else "auto floor"
            return _fail(f"silent / placeholder waveform (peak={peak:g}, rms={rms:g}; {hint})")

        detail = f"peak={peak:g}, rms={rms:g}"
        if stream_checked:
            return _ok(f"audio stream OK + {detail}")
        if path:
            return _ok(f"{detail} (ffprobe skipped; path={path})")
        return _ok(f"{detail} (no media path in dict — ffprobe stream check skipped)")
