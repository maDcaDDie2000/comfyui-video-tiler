from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest

import torch


_PACKAGE = "video_tiler_disk_tests"
if _PACKAGE not in sys.modules:
    package = types.ModuleType(_PACKAGE)
    package.__path__ = [str(Path(__file__).resolve().parent)]
    sys.modules[_PACKAGE] = package

disk_nodes = importlib.import_module(f"{_PACKAGE}.disk_nodes")


def _config():
    return (
        4,
        4,
        2,
        2,
        1,
        1,
        0,
        0,
        (
            ("normal", 0, 0, 2, 2, 0, 0, 0),
            ("normal", 2, 0, 2, 2, 1, 0, 1),
        ),
    )


def _write_job(root: Path, saved_indices=(0, 1)) -> Path:
    job_dir = root / "review_job"
    job_dir.mkdir(parents=True)
    manifest = {
        "manifest_version": 1,
        "job_name": "review_job",
        # Deliberately stale: readers should use the manifest's current folder.
        "job_dir": "Z:/old/location/review_job",
        "tile_config": _config(),
        "tile_count": 2,
        "saved_tiles": {},
    }
    for index in saved_indices:
        value = 0.25 if index == 0 else 0.75
        tile = torch.full((3, 2, 2, 3), value, dtype=torch.float32)
        path = job_dir / f"tile_{index:05d}.pt"
        torch.save({"tile": tile, "tile_index": index}, path)
        manifest["saved_tiles"][str(index)] = {"path": path.name}
    manifest_path = job_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


class DiskFolderFlowTests(unittest.TestCase):
    def test_open_job_accepts_parent_folder_and_reports_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = _write_job(Path(temp))

            result = disk_nodes.VideoTileDiskOpenJob().open_job(temp)

            self.assertEqual(result[0], str(manifest_path))
            self.assertEqual(result[2:4], (2, 2))
            self.assertIn("complete", result[4])

    def test_standalone_folder_merge_builds_image_batch_without_tile_job_input(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = _write_job(Path(temp))

            image, status, tile_count = disk_nodes.VideoTileDiskFolderMerge().merge_folder(
                str(manifest_path.parent),
                0.0,
                merge_device="cpu",
            )

            self.assertEqual(tuple(image.shape), (3, 2, 4, 3))
            self.assertTrue(torch.allclose(image[:, :, :2, :], torch.full((3, 2, 2, 3), 0.25)))
            self.assertTrue(torch.allclose(image[:, :, 2:, :], torch.full((3, 2, 2, 3), 0.75)))
            self.assertEqual(tile_count, 2)
            self.assertIn("2/2", status)

    def test_standalone_folder_merge_waits_for_every_expected_tile(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = _write_job(Path(temp), saved_indices=(1,))

            with self.assertRaisesRegex(RuntimeError, "saved 1/2; missing 0"):
                disk_nodes.VideoTileDiskFolderMerge().merge_folder(str(manifest_path), 0.125)

    def test_preview_uses_nearest_saved_tile_and_selects_one_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = _write_job(Path(temp), saved_indices=(1,))

            result = disk_nodes.VideoTileDiskPreview().preview(
                str(manifest_path.parent),
                tile_index=0,
                frame_index=2,
                missing_tile="nearest_available",
            )

            preview, actual_tile, actual_frame, saved_count, tile_count, tile_path, status = result
            self.assertEqual(tuple(preview.shape), (1, 2, 2, 3))
            self.assertTrue(torch.allclose(preview, torch.full((1, 2, 2, 3), 0.75)))
            self.assertEqual((actual_tile, actual_frame, saved_count, tile_count), (1, 2, 1, 2))
            self.assertTrue(tile_path.endswith("tile_00001.pt"))
            self.assertIn("nearest saved tile used", status)

    def test_optional_partial_merge_skips_missing_files(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest_path = _write_job(Path(temp), saved_indices=(1,))

            image = disk_nodes.VideoTileDiskMerge().merge(
                str(manifest_path),
                0.0,
                merge_device="cpu",
                require_all_tiles=False,
            )[0]

            self.assertTrue(torch.count_nonzero(image[:, :, :2, :]) == 0)
            self.assertTrue(torch.allclose(image[:, :, 2:, :], torch.full((3, 2, 2, 3), 0.75)))


if __name__ == "__main__":
    unittest.main()
