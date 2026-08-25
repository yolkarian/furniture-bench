import copy
import unittest

import numpy as np

from furniture_bench.furniture.parts.part import Part


class _DummyPart(Part):
    def __init__(self, part_config: dict, part_idx: int = 0):
        super().__init__(part_config, part_idx)
        self.reset_x_len = 0.1
        self.reset_y_len = 0.1


class TestPartConfigIsolation(unittest.TestCase):
    def test_randomization_does_not_mutate_source_or_baseline_config(self) -> None:
        config = {
            "name": "dummy",
            "asset_file": "dummy.urdf",
            "ids": [0],
            "reset_pos": [np.array([0.0, 0.24, -0.015], dtype=np.float32)],
            "reset_ori": [np.eye(4, dtype=np.float32)],
        }
        source_before = copy.deepcopy(config)
        part = _DummyPart(config)
        baseline_before = copy.deepcopy(part.part_config)

        np.random.seed(7)
        part.randomize_init_pose(pos_range=[-0.015, 0.015], rot_range=15)

        np.testing.assert_array_equal(
            config["reset_pos"][0], source_before["reset_pos"][0]
        )
        np.testing.assert_array_equal(
            config["reset_ori"][0], source_before["reset_ori"][0]
        )
        np.testing.assert_array_equal(
            part.part_config["reset_pos"][0], baseline_before["reset_pos"][0]
        )
        np.testing.assert_array_equal(
            part.part_config["reset_ori"][0], baseline_before["reset_ori"][0]
        )
        self.assertFalse(
            np.shares_memory(part.reset_pos[0], config["reset_pos"][0])
        )
        self.assertFalse(
            np.shares_memory(part.reset_pos[0], part.part_config["reset_pos"][0])
        )

    def test_new_part_starts_from_clean_source_after_other_part_randomizes(self) -> None:
        config = {
            "name": "dummy",
            "asset_file": "dummy.urdf",
            "ids": [0],
            "reset_pos": [np.array([0.0, 0.24, -0.015], dtype=np.float32)],
            "reset_ori": [np.eye(4, dtype=np.float32)],
        }
        first = _DummyPart(config)
        np.random.seed(11)
        first.randomize_init_pose(pos_range=[-0.015, 0.015], rot_range=15)

        second = _DummyPart(config)
        np.testing.assert_array_equal(second.reset_pos[0], config["reset_pos"][0])
        np.testing.assert_array_equal(second.reset_ori[0], config["reset_ori"][0])


if __name__ == "__main__":
    unittest.main()
