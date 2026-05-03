"""
FFprobe-based sanity check for AUDIO wiring toward LTX 2.3 Audio VAE encode.

ComfyUI AUDIO is a dict with waveform [B,C,T] and sample_rate. If the dict carries a
resolvable file path, ffprobe runs on that file; otherwise a short-lived PCM WAV is
written from the tensor so ffprobe still validates the payload.

Requires **ffprobe** on PATH (ships with FFmpeg).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np
import torch

logger = logging.getLogger(__name__)


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


def _scalar_int(v, default: int) -> int:
    while isinstance(v, (list, tuple)):
        if len(v) == 0:
            return default
        v = v[0]
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


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


def _write_temp_wav_from_tensor(waveform_bct: torch.Tensor, sample_rate: int) -> str:
    """PCM16 mono/stereo WAV from first batch row [C,T]."""
    if waveform_bct.dim() != 3:
        raise ValueError(f"waveform must be [B,C,T], got shape {tuple(waveform_bct.shape)}")
    w = waveform_bct[0].detach().cpu().float().clamp(-1.0, 1.0)
    c, t = int(w.shape[0]), int(w.shape[1])
    if c not in (1, 2):
        raise ValueError(f"LTX-style probe expects 1 or 2 channels, got C={c}")
    if t < 1:
        raise ValueError("waveform has zero length")

    arr = w.numpy()
    if c == 1:
        pcm = (arr[0] * 32767.0).astype(np.int16)
        interleaved = pcm
    else:
        l = (arr[0] * 32767.0).astype(np.int16)
        r = (arr[1] * 32767.0).astype(np.int16)
        interleaved = np.empty(t * 2, dtype=np.int16)
        interleaved[0::2] = l
        interleaved[1::2] = r

    fd, path = tempfile.mkstemp(suffix=".wav", prefix="video_tiler_audio_probe_")
    os.close(fd)
    try:
        with wave.open(path, "wb") as wf:
            wf.setnchannels(c)
            wf.setsampwidth(2)
            wf.setframerate(int(sample_rate))
            wf.writeframes(interleaved.tobytes())
    except Exception:
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    return path


def _ffprobe_json(path: str) -> dict | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=codec_type,codec_name,sample_rate,channels",
        "-show_entries",
        "format=duration",
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
        logger.warning("[Video Tiler] ffprobe timed out")
        return None
    if proc.returncode != 0:
        logger.warning("[Video Tiler] ffprobe failed: %s", (proc.stderr or proc.stdout or "").strip())
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None


def _parse_positive_float(v) -> float | None:
    try:
        x = float(v)
        return x if x > 1e-9 else None
    except (TypeError, ValueError):
        return None


def _parse_int(v) -> int | None:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return None


class VideoTileAudioFFprobeLTX:
    """
    Runs ffprobe on the AUDIO clip (file path when present, else a temp PCM WAV from tensor).
    Checks properties aligned with typical LTX 2.3 Audio VAE ingest (mono/stereo PCM-friendly).
    """

    DESCRIPTION = (
        "Uses ffprobe (FFmpeg) to verify an AUDIO dict maps to a sane waveform for LTX 2.3 Audio VAE: "
        "audio stream present, duration > 0, 1–2 channels, plausible sample rate. "
        "Optional strict sample-rate and linear-PCM-only modes."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": (
                    "AUDIO",
                    {"tooltip": "ComfyUI AUDIO dict (waveform + sample_rate). Path keys → probe file; else temp WAV from tensor."},
                ),
            },
            "optional": {
                "require_linear_pcm": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If true, codec must be pcm_* or flac (reject mp3/aac/opus etc.).",
                    },
                ),
                "strict_sample_rate": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "If true, stream sample_rate must equal target_sample_rate.",
                    },
                ),
                "target_sample_rate": (
                    "INT",
                    {
                        "default": 44100,
                        "min": 8000,
                        "max": 192000,
                        "step": 1,
                        "tooltip": "Expected Hz when strict_sample_rate is on (LTX stacks commonly use 44100).",
                    },
                ),
                "min_sample_rate": (
                    "INT",
                    {
                        "default": 16000,
                        "min": 8000,
                        "max": 192000,
                        "step": 1,
                        "tooltip": "Reject streams reported below this rate.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("ok",)
    OUTPUT_TOOLTIPS = ("True if ffprobe reports a valid mono/stereo audio stream passing configured checks.",)
    FUNCTION = "probe"
    CATEGORY = "Video Tiler"

    def probe(
        self,
        audio,
        require_linear_pcm=False,
        strict_sample_rate=False,
        target_sample_rate=44100,
        min_sample_rate=16000,
    ):
        req_pcm = _scalar_bool(require_linear_pcm, False)
        strict_sr = _scalar_bool(strict_sample_rate, False)
        tgt_sr = _scalar_int(target_sample_rate, 44100)
        min_sr = _scalar_int(min_sample_rate, 16000)

        def _fail(msg: str):
            print(f"[Video Tiler] Audio FFprobe (LTX): FAIL — {msg}")
            return (False,)

        audio = _unwrap_audio(audio)
        if not isinstance(audio, dict):
            return _fail("input is not an AUDIO dict")
        wf = audio.get("waveform")
        sr = audio.get("sample_rate")
        if wf is None or not isinstance(wf, torch.Tensor):
            return _fail("missing waveform tensor")
        if sr is None:
            return _fail("missing sample_rate")
        try:
            sr_int = int(sr)
        except (TypeError, ValueError):
            return _fail("sample_rate is not an integer")

        if wf.dim() != 3:
            return _fail(f"waveform rank must be 3 [B,C,T], got {tuple(wf.shape)}")
        b, c, t = wf.shape
        if b < 1:
            return _fail("empty batch")
        if int(c) not in (1, 2):
            return _fail(f"channel count must be 1 or 2 for this check, got C={c}")
        if int(t) < 1:
            return _fail("zero-length waveform")
        if not torch.isfinite(wf).all():
            return _fail("waveform contains NaN/Inf")

        path = _path_from_audio_dict(audio)
        tmp_path: str | None = None
        if path is None:
            try:
                tmp_path = _write_temp_wav_from_tensor(wf, sr_int)
                path = tmp_path
            except Exception as e:
                return _fail(f"could not materialize WAV for ffprobe: {e}")

        try:
            data = _ffprobe_json(path)
            if data is None:
                return _fail("ffprobe unavailable or failed")
            streams = data.get("streams") or []
            if not streams:
                return _fail("no audio stream (a:0)")
            st0 = streams[0]
            if st0.get("codec_type") != "audio":
                return _fail("first stream is not audio")

            codec = (st0.get("codec_name") or "").lower()
            if req_pcm:
                if not (codec.startswith("pcm_") or codec == "flac"):
                    return _fail(f"codec {codec!r} is not linear PCM/flac")

            ch = _parse_int(st0.get("channels"))
            if ch is None or ch not in (1, 2):
                return _fail(f"channels must be 1 or 2, ffprobe reports {st0.get('channels')}")

            r_hz = _parse_int(st0.get("sample_rate"))
            if r_hz is None or r_hz < min_sr:
                return _fail(f"sample_rate too low or missing ({st0.get('sample_rate')})")

            if strict_sr and r_hz != tgt_sr:
                return _fail(f"sample_rate {r_hz} != target {tgt_sr}")

            dur_stream = _parse_positive_float(st0.get("duration"))
            fmt = data.get("format") or {}
            dur_fmt = _parse_positive_float(fmt.get("duration"))
            if dur_stream is None and dur_fmt is None:
                return _fail("zero / unknown duration")

            print(
                f"[Video Tiler] Audio FFprobe (LTX): OK — codec={codec}, {r_hz} Hz, ch={ch}, "
                f"dur≈{dur_fmt or dur_stream:.3f}s"
            )
            return (True,)
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
