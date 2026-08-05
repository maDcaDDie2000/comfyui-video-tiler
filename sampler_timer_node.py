"""ComfyUI nodes for measuring sampler execution time.

The setup/result split is intentional.  ComfyUI evaluates a graph from inputs to
outputs, so a node which patches the model necessarily runs before the sampler.
The result node takes the sampler's LATENT output to guarantee that its FLOAT
outputs are read only after sampling has completed.
"""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
import uuid
from typing import Any


def _synchronize_cuda() -> None:
    """Wait for visible CUDA work so wall-clock measurements are accurate."""
    try:
        import torch

        if not torch.cuda.is_available():
            return
        for device_index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device_index)
    except (ImportError, RuntimeError):
        # CPU, MPS, DirectML, or an unavailable CUDA runtime needs no CUDA sync.
        return


def _step_count(wrapper_args: tuple[Any, ...]) -> int:
    """Get the number of sigma intervals from a SAMPLER_SAMPLE wrapper call."""
    # Current ComfyUI calls the wrapper as (guider, sigmas, extra_args, ...).
    candidates = wrapper_args[1:2] if len(wrapper_args) > 1 else ()
    for value in candidates:
        try:
            return max(0, len(value) - 1)
        except TypeError:
            pass
    return 0


@dataclass(frozen=True)
class SamplerTimingSnapshot:
    total_seconds: float
    sampler_calls: int
    total_steps: int

    @property
    def average_seconds(self) -> float:
        if self.sampler_calls == 0:
            return 0.0
        return self.total_seconds / self.sampler_calls

    @property
    def average_seconds_per_step(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_seconds / self.total_steps


class SamplerTimerState:
    """Prompt-local, thread-safe timing data shared by the two nodes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_seconds = 0.0
        self._sampler_calls = 0
        self._total_steps = 0

    def sampler_wrapper(self, executor, *args, **kwargs):
        """ComfyUI SAMPLER_SAMPLE wrapper that records successful calls."""
        _synchronize_cuda()
        started = time.perf_counter()
        succeeded = False
        try:
            result = executor(*args, **kwargs)
            succeeded = True
            return result
        finally:
            _synchronize_cuda()
            if succeeded:
                elapsed = time.perf_counter() - started
                steps = _step_count(args)
                with self._lock:
                    self._total_seconds += elapsed
                    self._sampler_calls += 1
                    self._total_steps += steps

    def snapshot(self) -> SamplerTimingSnapshot:
        with self._lock:
            return SamplerTimingSnapshot(
                total_seconds=self._total_seconds,
                sampler_calls=self._sampler_calls,
                total_steps=self._total_steps,
            )


class VideoTileSamplerTimerStart:
    """Clone a MODEL and attach prompt-local sampler timing instrumentation."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": (
                    "MODEL",
                    {
                        "tooltip": (
                            "Connect the output MODEL to KSampler or SamplerCustom. "
                            "The input model is cloned and is not modified."
                        )
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "SAMPLER_TIMER")
    RETURN_NAMES = ("model", "sampler_timer")
    OUTPUT_TOOLTIPS = (
        "Instrumented model to connect to a sampler.",
        "Timing handle to connect to Sampler Timing Result.",
    )
    FUNCTION = "start"
    CATEGORY = "Video Tiler/Timing"
    DESCRIPTION = "Instruments a cloned model to measure its sampler calls."

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # Timing must start from an empty state on every queued prompt, even when
        # the source MODEL is cached.
        return float("nan")

    def start(self, model):
        try:
            from comfy.patcher_extension import WrappersMP
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                "Sampler Timing requires a current ComfyUI version with sampler wrappers."
            ) from exc

        if not hasattr(model, "clone") or not hasattr(model, "add_wrapper_with_key"):
            raise RuntimeError(
                "Sampler Timing expected a ComfyUI MODEL supporting sampler wrappers. "
                "Please update ComfyUI."
            )

        timer = SamplerTimerState()
        timed_model = model.clone()
        wrapper_key = f"video_tiler_sampler_timer_{uuid.uuid4().hex}"
        timed_model.add_wrapper_with_key(
            WrappersMP.SAMPLER_SAMPLE,
            wrapper_key,
            timer.sampler_wrapper,
        )
        return timed_model, timer


class VideoTileSamplerTimerResult:
    """Read timing data after the connected sampler LATENT is available."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": (
                    "LATENT",
                    {"tooltip": "Connect the LATENT output of the timed sampler."},
                ),
                "sampler_timer": (
                    "SAMPLER_TIMER",
                    {"tooltip": "Connect Sampler Timer Start's sampler_timer output."},
                ),
            }
        }

    RETURN_TYPES = ("LATENT", "FLOAT", "FLOAT", "FLOAT", "INT", "INT")
    RETURN_NAMES = (
        "samples",
        "average_seconds",
        "total_seconds",
        "seconds_per_step",
        "sampler_calls",
        "total_steps",
    )
    OUTPUT_TOOLTIPS = (
        "Unchanged sampler LATENT passthrough.",
        "Average wall-clock seconds per completed sampler call.",
        "Total wall-clock seconds across completed sampler calls.",
        "Average wall-clock seconds per sigma interval.",
        "Number of completed sampler calls included in the average.",
        "Total sigma intervals included in seconds_per_step.",
    )
    FUNCTION = "read"
    CATEGORY = "Video Tiler/Timing"
    DESCRIPTION = (
        "Reads sampler timing after sampling. FLOAT outputs can feed any "
        "compatible workflow node."
    )

    def read(self, samples, sampler_timer):
        if not isinstance(sampler_timer, SamplerTimerState):
            raise TypeError("sampler_timer must come from Sampler Timer Start.")

        timing = sampler_timer.snapshot()
        if timing.sampler_calls == 0:
            raise RuntimeError(
                "No sampler call was measured. Connect Sampler Timer Start's MODEL "
                "to the sampler, and connect that sampler's LATENT to this node."
            )

        return (
            samples,
            float(timing.average_seconds),
            float(timing.total_seconds),
            float(timing.average_seconds_per_step),
            int(timing.sampler_calls),
            int(timing.total_steps),
        )
