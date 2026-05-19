import numpy as np

from furniture_bench.config import config
from furniture_bench.furniture.parts.part import Part


class ScannedPart(Part):
    def __init__(self, part_config: dict, part_idx: int):
        super().__init__(part_config, part_idx)

        reset_pos_base = np.asarray(part_config["reset_pos"], dtype=np.float64)
        reset_ori_base = [
            np.asarray(ori, dtype=np.float64) for ori in part_config["reset_ori"]
        ]

        # Scan reset poses are authored in the robot base frame. Convert them once to
        # AprilTag coordinates and also update self.part_config so inherited
        # randomize_init_pose() samples around the converted reset pose instead of
        # the out-of-bounds robot-frame pose.
        self.reset_pos = []
        self.reset_ori = []
        tag_from_robot = np.linalg.inv(config["robot"]["tag_base_from_robot_base"])
        for pos_base, ori_base in zip(reset_pos_base, reset_ori_base):
            reset_pose_base = ori_base.copy()
            reset_pose_base[:-1, -1] = pos_base

            reset_pose_tag = tag_from_robot @ reset_pose_base
            ori = np.eye(4)
            ori[:-1, :-1] = reset_pose_tag[:-1, :-1]

            self.reset_pos.append(reset_pose_tag[:-1, -1].tolist())
            self.reset_ori.append(ori)

        self.part_config["reset_pos"] = [pos.copy() for pos in self.reset_pos]
        self.part_config["reset_ori"] = [ori.copy() for ori in self.reset_ori]
        self.reset_x_len = float(part_config.get("reset_x_len", 0.0))
        self.reset_y_len = float(part_config.get("reset_y_len", 0.0))
