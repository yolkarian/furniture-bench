"""Differential inverse-kinematics controller used by FurnitureBench."""

from __future__ import annotations

from typing import Any

import torch

import furniture_bench.utils.control as C

DEFAULT_EE_POS = torch.zeros(3, dtype=torch.float32)
DEFAULT_EE_QUAT_XYZW = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32)


def diffik_factory(
    real_robot: bool = True, *args: Any, **kwargs: Any
) -> torch.nn.Module:
    """Build a DiffIK controller for either the real robot or simulation.

    The real-robot path uses ``torchcontrol.PolicyModule`` so that Polymetis can
    stream goal updates through ``update_desired_ee_pose``. The simulation path
    uses a regular ``torch.nn.Module`` and receives goals through ``set_goal``.
    """

    if real_robot:
        import torchcontrol as toco

        base = toco.PolicyModule
    else:
        base = torch.nn.Module

    class DiffIKController(base):
        """Solve for joint targets that realize a desired EE pose delta."""

        def __init__(
            self,
            ee_pos_current: torch.Tensor | None = None,
            ee_quat_current: torch.Tensor | None = None,
            position_limits: torch.Tensor | None = None,
            pos_scalar: float = 1.0,
            rot_scalar: float = 1.0,
            dt: float = 0.1,
            **_: Any,
        ) -> None:
            super().__init__()

            ee_pos_init = (
                ee_pos_current.clone().detach()
                if ee_pos_current is not None
                else DEFAULT_EE_POS.clone()
            )
            ee_quat_init = (
                ee_quat_current.clone().detach()
                if ee_quat_current is not None
                else DEFAULT_EE_QUAT_XYZW.clone()
            )

            self.position_limits = position_limits
            self.pos_scalar = float(pos_scalar)
            self.rot_scalar = float(rot_scalar)
            self.dt = float(dt)
            self.use_parameter_goals = real_robot

            # These parameters are intentionally kept for Polymetis compatibility.
            # ``RobotInterface.update_desired_ee_pose`` updates them in-place.
            self.ee_pos_desired = torch.nn.Parameter(ee_pos_init)
            self.ee_quat_desired = torch.nn.Parameter(ee_quat_init)

            # Simulation updates the controller through ``set_goal`` instead.
            self.goal_pos = ee_pos_init.clone()
            self.goal_ori = ee_quat_init.clone()

        def _resolve_goal(
            self, reference_device: torch.device
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Pick the latest goal source while keeping both APIs supported."""
            if self.use_parameter_goals:
                goal_pos = self.ee_pos_desired
                goal_ori = self.ee_quat_desired
            else:
                goal_pos = self.goal_pos
                goal_ori = self.goal_ori

            goal_pos = goal_pos.to(reference_device)
            goal_ori = goal_ori.to(reference_device)

            # Real-robot control uses a single 3D target, so clipping here keeps the
            # requested pose within the configured workspace limits.
            if self.position_limits is not None and goal_pos.ndim == 1:
                goal_pos = C.set_goal_position(
                    self.position_limits.to(reference_device), goal_pos.clone()
                )

            return goal_pos, goal_ori

        def forward(
            self, state_dict: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            """Map the end-effector pose error to desired joint positions."""
            joint_pos_current = state_dict["joint_positions"]
            jacobian = state_dict["jacobian_diffik"]
            ee_pos = state_dict["ee_pos"]
            ee_quat_xyzw = state_dict["ee_quat"]

            goal_pos, goal_ori_xyzw = self._resolve_goal(ee_pos.device)

            # Broadcast single-pose goals to vectorized simulation batches.
            if goal_pos.ndim < ee_pos.ndim:
                goal_pos = goal_pos.unsqueeze(0).expand_as(ee_pos)
            if goal_ori_xyzw.ndim < ee_quat_xyzw.ndim:
                goal_ori_xyzw = goal_ori_xyzw.unsqueeze(0).expand_as(ee_quat_xyzw)

            position_error = goal_pos - ee_pos

            ee_mat = C.quaternion_to_matrix(ee_quat_xyzw)
            goal_mat = C.quaternion_to_matrix(goal_ori_xyzw)
            mat_error = torch.matmul(goal_mat, ee_mat.transpose(-1, -2))
            ee_delta_axis_angle = C.matrix_to_axis_angle(mat_error)

            ee_pos_vel = position_error * self.pos_scalar / self.dt
            ee_rot_vel = ee_delta_axis_angle * self.rot_scalar / self.dt
            ee_velocity_desired = torch.cat((ee_pos_vel, ee_rot_vel), dim=-1)

            joint_vel_desired = torch.linalg.lstsq(
                jacobian, ee_velocity_desired
            ).solution
            joint_pos_desired = joint_pos_current + joint_vel_desired * self.dt
            return {"joint_positions": joint_pos_desired}

        def set_goal(self, goal_pos: torch.Tensor, goal_ori: torch.Tensor) -> None:
            """Update the desired goal explicitly.

            Simulation calls this method directly. When the goal shape matches the
            Polymetis parameter tensors we mirror the values there as well so both
            interfaces stay synchronized.
            """
            if (
                self.goal_pos.shape == goal_pos.shape
                and self.goal_pos.device == goal_pos.device
                and self.goal_pos.dtype == goal_pos.dtype
            ):
                self.goal_pos.copy_(goal_pos)
            else:
                self.goal_pos = goal_pos.clone()
            if (
                self.goal_ori.shape == goal_ori.shape
                and self.goal_ori.device == goal_ori.device
                and self.goal_ori.dtype == goal_ori.dtype
            ):
                self.goal_ori.copy_(goal_ori)
            else:
                self.goal_ori = goal_ori.clone()

            with torch.no_grad():
                if self.ee_pos_desired.shape == goal_pos.shape:
                    self.ee_pos_desired.copy_(goal_pos)
                if self.ee_quat_desired.shape == goal_ori.shape:
                    self.ee_quat_desired.copy_(goal_ori)

        def reset(self) -> None:
            """Reset controller state.

            DiffIK is stateless between steps, so there is nothing to clear.
            """
            return None

    return DiffIKController(*args, **kwargs)
