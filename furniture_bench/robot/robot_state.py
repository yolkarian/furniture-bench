"""Define the types of robot state."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import numpy.typing as npt

from furniture_bench.data.trajectory_types import ArrayLike, RobotStateDict

# List of robot state we are going to use during training and testing.
ROBOT_STATES: list[str] = [
    "ee_pos",
    "ee_quat",
    "ee_pos_vel",
    "ee_ori_vel",
    "gripper_width",
]

ROBOT_STATE_DIMS: dict[str, int] = {
    "ee_pos": 3,
    "ee_quat": 4,
    "ee_pos_vel": 3,
    "ee_ori_vel": 3,
    "joint_positions": 7,
    "joint_velocities": 7,
    "joint_torques": 7,
    "gripper_width": 1,
    "gripper_finger_1_pos": 1,
    "gripper_finger_2_pos": 1,
}


def filter_and_concat_robot_state(robot_state: RobotStateDict) -> ArrayLike:
    current_robot_state: list[ArrayLike] = []
    for rs in ROBOT_STATES:
        if rs not in robot_state:
            continue

        value = robot_state[rs]
        if rs == "gripper_width":
            value = np.asarray([value]).reshape(1)
        else:
            value = np.asarray(value)
        current_robot_state.append(value)
    return np.concatenate(current_robot_state, axis=-1)


@dataclass
class PandaState:
    """Define state of Panda arm and end-effector."""

    ee_pos: npt.NDArray[np.float64]
    ee_quat: npt.NDArray[np.float64]
    ee_pos_vel: npt.NDArray[np.float64]
    ee_ori_vel: npt.NDArray[np.float64]
    joint_positions: npt.NDArray[np.float64]
    joint_velocities: npt.NDArray[np.float64]
    joint_torques: npt.NDArray[np.float64]
    gripper_width: npt.NDArray[np.float64]


class PandaError(Enum):
    OLD_GRIPPER_ERROR = 1
    OK = "Successful"
    Gripper = "Panda gripper server stopped."
    Arm = 2
