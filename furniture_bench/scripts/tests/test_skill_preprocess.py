"""Runtime smoke tests for maintained dataset-oriented CLIs."""

from __future__ import annotations

import pickle
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from furniture_bench.scripts import (
    download_dataset,
    preprocess_data,
    show_trajectory,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOP_LEVEL_SCRIPT_ROOT = REPO_ROOT / "scripts"


class MaintainedScriptSmokeTests(unittest.TestCase):
    def assert_script_help(self, script_name: str) -> None:
        completed = subprocess.run(
            [sys.executable, str(TOP_LEVEL_SCRIPT_ROOT / script_name), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, msg=completed.stderr)
        self.assertIn("usage:", completed.stdout.lower())

    def _robot_state(self, offset: float = 0.0) -> dict[str, np.ndarray | float]:
        return {
            "ee_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32) + offset,
            "ee_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            "ee_pos_vel": np.array([0.01, 0.02, 0.03], dtype=np.float32),
            "ee_ori_vel": np.array([0.04, 0.05, 0.06], dtype=np.float32),
            "gripper_width": 0.07 + offset,
        }

    def _raw_observation(self, index: int) -> dict[str, object]:
        return {
            "color_image1": np.full((4, 4, 3), 10 + index, dtype=np.uint8),
            "color_image2": np.full((4, 4, 3), 20 + index, dtype=np.uint8),
            "robot_state": self._robot_state(offset=float(index)),
            "parts_poses": np.full((2, 4, 4), index, dtype=np.float32),
        }

    def test_preprocess_data_main_smoke(self) -> None:
        no_action = np.array([0, 0, 0, 0, 0, 0, 1, -1], dtype=np.float32)
        move_action = np.array(
            [0.01, -0.02, 0.03, 0, 0, 0, 1, -1], dtype=np.float32
        )
        rotate_action = np.array(
            [0.04, 0.05, -0.01, 0, 0, 0, -1, -1], dtype=np.float32
        )

        trajectory = {
            "furniture": "one_leg",
            "success": True,
            "observations": [self._raw_observation(i) for i in range(4)],
            "actions": [no_action, move_action, rotate_action],
            "rewards": [0.0, 0.0, 1.0],
            "skills": [0, 0, 1],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            in_dir = tmp_path / "raw"
            out_dir = tmp_path / "processed"
            in_dir.mkdir()
            pickle_path = in_dir / "00000.pkl"
            with open(pickle_path, "wb") as file_obj:
                pickle.dump(trajectory, file_obj)

            preprocess_data.main(
                [
                    "--in-data-path",
                    str(in_dir),
                    "--out-data-path",
                    str(out_dir),
                    "--save-last-step",
                    "--norm-pos-acts",
                ]
            )

            output_path = out_dir / "00000.pkl"
            self.assertTrue(output_path.exists())

            with open(output_path, "rb") as file_obj:
                processed = pickle.load(file_obj)

            self.assertEqual(processed["furniture"], "one_leg")
            self.assertEqual(len(processed["actions"]), 2)
            self.assertEqual(len(processed["observations"]), 3)
            self.assertEqual(
                processed["observations"][0]["color_image1"].shape,
                (3, 4, 4),
            )
            self.assertEqual(
                processed["observations"][0]["color_image2"].shape,
                (3, 4, 4),
            )
            self.assertIsInstance(
                processed["observations"][0]["robot_state"], np.ndarray
            )
            self.assertEqual(processed["observations"][0]["robot_state"].shape[-1], 14)
            self.assertLess(np.abs(processed["actions"][0][:3]).max(), 1.0)
            self.assertGreaterEqual(processed["actions"][1][6], 0.0)

    def test_show_trajectory_main_smoke(self) -> None:
        trajectory = {
            "furniture": "one_leg",
            "observations": [
                {
                    "color_image1": np.zeros((3, 4, 4), dtype=np.uint8),
                    "color_image2": np.ones((3, 4, 4), dtype=np.uint8),
                    "robot_state": np.arange(14, dtype=np.float32),
                }
            ],
            "actions": [np.zeros(8, dtype=np.float32)],
            "rewards": [0.0],
            "skills": [0],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            pickle_path = Path(tmpdir) / "trajectory.pkl"
            with open(pickle_path, "wb") as file_obj:
                pickle.dump(trajectory, file_obj)

            with (
                mock.patch.object(show_trajectory.cv2, "imshow") as imshow,
                mock.patch.object(
                    show_trajectory.cv2, "waitKey", return_value=27
                ) as wait_key,
                mock.patch.object(
                    show_trajectory.cv2, "destroyAllWindows"
                ) as destroy,
                mock.patch.object(show_trajectory.time, "sleep"),
            ):
                show_trajectory.main(
                    [
                        "--data-path",
                        str(pickle_path),
                        "--channel-first",
                    ]
                )

            imshow.assert_called_once()
            wait_key.assert_called_once()
            destroy.assert_called_once()

    def test_download_dataset_main_dispatches_all_rclone_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                mock.patch.object(
                    download_dataset, "download_folder_rclone"
                ) as rclone,
                mock.patch.object(
                    download_dataset, "download_file_gdown"
                ) as gdown,
                mock.patch.object(download_dataset, "maybe_untar") as untar,
            ):
                download_dataset.main(
                    [
                        "--randomness",
                        "low",
                        "--furniture",
                        "all",
                        "--out_dir",
                        tmpdir,
                        "--use-rclone",
                        "--untar",
                    ]
                )

            self.assertEqual(rclone.call_count, len(download_dataset.ALL_FURNITURE))
            self.assertEqual(untar.call_count, len(download_dataset.ALL_FURNITURE))
            gdown.assert_not_called()

    def test_download_file_gdown_creates_expected_target(self) -> None:
        recorded_kwargs: dict[str, object] = {}

        def fake_download(**kwargs: object) -> str:
            recorded_kwargs.update(kwargs)
            return str(kwargs["output"])

        fake_gdown = types.SimpleNamespace(download=fake_download)

        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            with mock.patch.dict(sys.modules, {"gdown": fake_gdown}):
                download_dataset.download_file_gdown("low", "one_leg", out_dir)

            self.assertEqual(
                recorded_kwargs["id"],
                download_dataset.DOWNLOAD_LINKS["low"]["one_leg"],
            )
            self.assertEqual(
                Path(str(recorded_kwargs["output"])),
                out_dir / "low" / "one_leg.tar.gz",
            )
            self.assertTrue((out_dir / "low").exists())

    def test_download_folder_rclone_skips_existing_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir)
            target_path = out_dir / "low" / "one_leg.tar.gz"
            target_path.parent.mkdir(parents=True)
            target_path.write_bytes(b"already downloaded")

            with mock.patch.object(download_dataset.subprocess, "run") as run:
                download_dataset.download_folder_rclone("low", "one_leg", out_dir)

            run.assert_not_called()

    def test_top_level_diagnostic_script_help(self) -> None:
        self.assert_script_help("import_furniture_bench.py")
        self.assert_script_help("import_franka_sapien.py")
        self.assert_script_help("launch_sim_gymasium.py")
        self.assert_script_help("parallel_rendering_test.py")


if __name__ == "__main__":
    unittest.main()
