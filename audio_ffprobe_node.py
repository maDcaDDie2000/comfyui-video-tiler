"""
Detect real audio vs silent placeholder from Load Video / AUDIO wiring.

If the AUDIO dict resolves to a media file path, ffprobe checks that an audio stream
exists. Always checks that the decoded waveform peak exceeds a tiny threshold so
silent filler tensors read as absent.

Requires **ffprobe** on PATH only when a filepath can be resolved from the dict.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys

import torch

logger = logging.getLogger(__name__)


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


def _path_from_audio_dict(d: dict) -> str | None:
    for key in ("filename", "path", "file", "filepath", "audio_file", "full_path"):
        v = d.get(key)
        if isinstance(v, str):
            hit = _resolve_filepath(v)
            if hit:
                return hit
    return None


def _ffprobe_audio_stream_count(path: str) -> int | None:
    """Return number of audio streams, or None if ffprobe failed."""
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
    except FileNotFoundError:
        logger.warning("[Video Tiler] ffprobe not found on PATH")
        return None
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    streams = data.get("streams") or []
    return len(streams)


def _scalar_float(v, default: float) -> float:
    while isinstance(v, (list, tuple)):
        if len(v) == 0:
            return default
        v = v[0]
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class VideoTileAudioFFprobeLTX:
    """
    True only when the clip appears to have real audio: optional ffprobe confirms an
    audio stream on the source file (when path known), and waveform peak is above threshold.
    """

    DESCRIPTION = (
        "Single-purpose: True if Load Video (or similar) actually carried audio — "
        "ffprobe finds an audio stream when a file path is present, and the waveform is not silent. "
        "False for video-without-audio placeholders or flatlined buffers."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (
                    "AUDIO",
                    {
                        "tooltip": "Output from Load Video / VHS. Checks file path (if any) for an audio stream, then non-silent waveform.",
                    },
                ),
            },
            "optional": {
                "min_peak": (
                    "FLOAT",
                    {
                        "default": 1e-6,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 1e-9,
                        "tooltip": "Peak |sample| below this counts as silent (Comfy audio is typically ~[-1, 1]).",
                    },
                ),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("has_audio",)
    OUTPUT_TOOLTIPS = ("True if an audio stream exists (when path known) and waveform is not silent.",)
    FUNCTION = "probe"
    CATEGORY = "Video Tiler"

    def probe(self, audio, min_peak=1e-6):
        thresh = max(0.0, _scalar_float(min_peak, 1e-6))

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

        if wf.dim() != 3:
            return _fail(f"expected waveform [B,C,T], got {tuple(wf.shape)}")

        path = _path_from_audio_dict(audio)
        if path:
            n_audio = _ffprobe_audio_stream_count(path)
            if n_audio is None:
                return _fail("ffprobe failed or missing from PATH (cannot verify stream on file)")
            if n_audio < 1:
                return _fail("no audio stream in media file (video-only or silent container as reported by ffprobe)")

        peak = float(wf.detach().abs().max().cpu())
        if peak < thresh:
            return _fail(f"waveform silent or placeholder (peak {peak:g} < min_peak {thresh:g})")

        if path:
            return _ok(f"audio stream present + peak {peak:g}")
        return _ok(f"non-silent waveform (peak {peak:g}); no file path — stream count not checked")
