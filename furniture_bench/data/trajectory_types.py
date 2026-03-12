"""Typed schemas for serialized FurnitureBench trajectories."""

from __future__ import annotations

from typing import TypeAlias, TypedDict

import numpy as np
import numpy.typing as npt

ArrayLike: TypeAlias = npt.NDArray[np.generic]


class RobotStateDict(TypedDict, total=False):
    """Robot-state fields stored in trajectory observations."""

    ee_pos: ArrayLike
    ee_quat: ArrayLike
    ee_pos_vel: ArrayLike
    ee_ori_vel: ArrayLike
    joint_positions: ArrayLike
    joint_velocities: ArrayLike
    joint_torques: ArrayLike
    gripper_width: float | ArrayLike
    gripper_finger_1_pos: ArrayLike
    gripper_finger_2_pos: ArrayLike


class ObservationDict(TypedDict, total=False):
    """Observation payload saved in raw or preprocessed trajectories."""

    color_image1: ArrayLike
    color_image2: ArrayLike
    color_image3: ArrayLike
    depth_image1: ArrayLike
    depth_image2: ArrayLike
    depth_image3: ArrayLike
    image1: ArrayLike
    image2: ArrayLike
    image_size: tuple[int, int]
    parts_poses: ArrayLike
    robot_state: RobotStateDict | ArrayLike


class TrajectoryDict(TypedDict, total=False):
    """Serialized trajectory structure used by maintained scripts."""

    furniture: str
    observations: list[ObservationDict]
    actions: list[ArrayLike]
    rewards: list[float]
    skills: list[int]
    success: bool
    error: bool
    error_description: str
    cam1_intr: ArrayLike
    cam2_intr: ArrayLike
    cam3_intr: ArrayLike
    cam2_to_base: ArrayLike
    cam3_to_base: ArrayLike
