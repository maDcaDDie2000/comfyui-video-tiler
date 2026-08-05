from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from sampler_timer_node import (
    SamplerTimerState,
    VideoTileSamplerTimerResult,
    VideoTileSamplerTimerStart,
)


class _Executor:
    def __call__(self, *args, **kwargs):
        return args, kwargs


class _FakeModel:
    def __init__(self):
        self.registered = None

    def clone(self):
        return _FakeModel()

    def add_wrapper_with_key(self, wrapper_type, key, wrapper):
        self.registered = wrapper_type, key, wrapper


class SamplerTimerTests(unittest.TestCase):
    def test_timing_nodes_use_the_main_video_tiler_category(self):
        self.assertEqual(VideoTileSamplerTimerStart.CATEGORY, "Video Tiler")
        self.assertEqual(VideoTileSamplerTimerResult.CATEGORY, "Video Tiler")

    def test_wrapper_averages_calls_and_sigma_intervals(self):
        timer = SamplerTimerState()
        executor = _Executor()

        timer.sampler_wrapper(executor, object(), [4, 3, 2, 1, 0])
        timer.sampler_wrapper(executor, object(), [2, 1, 0])

        timing = timer.snapshot()
        self.assertEqual(timing.sampler_calls, 2)
        self.assertEqual(timing.total_steps, 6)
        self.assertGreaterEqual(timing.total_seconds, 0.0)
        self.assertEqual(timing.average_seconds, timing.total_seconds / 2)
        self.assertEqual(
            timing.average_seconds_per_step,
            timing.total_seconds / 6,
        )

    def test_result_returns_connectable_float_values(self):
        timer = SamplerTimerState()
        timer.sampler_wrapper(_Executor(), object(), [1, 0])
        samples = {"samples": object()}

        result = VideoTileSamplerTimerResult().read(samples, timer)

        self.assertIs(result[0], samples)
        self.assertIsInstance(result[1], float)
        self.assertIsInstance(result[2], float)
        self.assertIsInstance(result[3], float)
        self.assertEqual(result[4:], (1, 1))

    def test_result_rejects_an_unmeasured_sampler(self):
        with self.assertRaisesRegex(RuntimeError, "No sampler call was measured"):
            VideoTileSamplerTimerResult().read({}, SamplerTimerState())

    def test_start_clones_model_and_registers_sampler_wrapper(self):
        comfy_module = types.ModuleType("comfy")
        patcher_module = types.ModuleType("comfy.patcher_extension")
        patcher_module.WrappersMP = types.SimpleNamespace(
            SAMPLER_SAMPLE="sampler_sample"
        )
        comfy_module.patcher_extension = patcher_module
        source_model = _FakeModel()

        with patch.dict(
            sys.modules,
            {
                "comfy": comfy_module,
                "comfy.patcher_extension": patcher_module,
            },
        ):
            timed_model, timer = VideoTileSamplerTimerStart().start(source_model)

        self.assertIsNone(source_model.registered)
        self.assertIsInstance(timer, SamplerTimerState)
        self.assertEqual(timed_model.registered[0], "sampler_sample")
        self.assertTrue(
            timed_model.registered[1].startswith("video_tiler_sampler_timer_")
        )
        self.assertTrue(callable(timed_model.registered[2]))


if __name__ == "__main__":
    unittest.main()
