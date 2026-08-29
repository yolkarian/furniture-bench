import unittest
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import numpy as np
import torch

from furniture_bench.envs import furniture_sim_env
from furniture_bench.envs.furniture_sim_env import (
    FurnitureSimRLEnv,
    _same_physical_gpu,
)
from furniture_bench.utils.sapien.camera import MINIMAL_SHADER_CONFIG


class _FakeCudaArray:
    def __init__(self, tensor: torch.Tensor, calls: list[str]):
        self.tensor = tensor
        self.calls = calls

    def torch(self) -> torch.Tensor:
        self.calls.append("torch")
        return self.tensor


class _FakeCameraGroup:
    def __init__(self, render_group: "_FakeRenderSystemGroup"):
        self.render_group = render_group

    def get_picture_cuda(self, _picture_name: str) -> _FakeCudaArray:
        self.render_group.calls.append("get_picture_cuda")
        self.render_group.assert_initialized()
        return _FakeCudaArray(torch.ones((2, 4, 4, 4)), self.render_group.calls)

    def set_pose_mode(self, _camera: object, mode: str) -> None:
        self.render_group.calls.append(f"set_pose_mode:{mode}")


class _FakeRenderSystemGroup:
    instances: ClassVar[list["_FakeRenderSystemGroup"]] = []

    def __init__(self, _render_systems: list[object]):
        self.calls: list[str] = []
        self.initialized = False
        self.instances.append(self)

    def set_cuda_poses(self, _poses: object) -> None:
        self.calls.append("set_cuda_poses")

    def create_camera_group(
        self, _sensors: list[object], picture_names: list[str]
    ) -> _FakeCameraGroup:
        self.calls.append("create_camera_group")
        self.picture_names = picture_names
        return _FakeCameraGroup(self)

    def gpu_init(self) -> None:
        self.calls.append("gpu_init")
        self.initialized = True

    def assert_initialized(self) -> None:
        if not self.initialized:
            raise RuntimeError("gpu_init must run first")

    def update_render(self) -> None:
        self.calls.append("update_render")
        self.assert_initialized()


class TestRenderLifecycle(unittest.TestCase):
    def test_same_physical_gpu_prefers_uuid_and_falls_back_to_cuda_id(self) -> None:
        self.assertTrue(
            _same_physical_gpu(
                SimpleNamespace(uuid="gpu-a", cuda_id=0),
                SimpleNamespace(uuid="gpu-a", cuda_id=3),
            )
        )
        self.assertFalse(
            _same_physical_gpu(
                SimpleNamespace(uuid="gpu-a", cuda_id=0),
                SimpleNamespace(uuid="gpu-b", cuda_id=0),
            )
        )
        self.assertTrue(
            _same_physical_gpu(
                SimpleNamespace(uuid=None, cuda_id=2),
                SimpleNamespace(uuid=None, cuda_id=2),
            )
        )

    def test_group_gpu_init_precedes_cuda_image_export_and_update(self) -> None:
        env = object.__new__(FurnitureSimRLEnv)
        env.num_envs = 2
        env.scenes = [SimpleNamespace(render_system=object()) for _ in range(2)]
        env.physx_system = SimpleNamespace(cuda_rigid_body_data=object())
        env.sensors = {"wrist": [object(), object()]}
        env.sensor_keys = {"wrist": {"Color"}}
        env._require_direct_sensor_interop = lambda: None
        _FakeRenderSystemGroup.instances.clear()

        with patch.object(
            furniture_sim_env.sapien.render,
            "RenderSystemGroup",
            _FakeRenderSystemGroup,
        ):
            env._init_render()

        group = _FakeRenderSystemGroup.instances[-1]
        self.assertEqual(
            group.calls,
            [
                "set_cuda_poses",
                "create_camera_group",
                "gpu_init",
                "get_picture_cuda",
                "torch",
                "update_render",
            ],
        )
        self.assertEqual(group.picture_names, ["Color"])

    def test_one_scene_uses_direct_gpu_group_with_static_free_camera(self) -> None:
        env = object.__new__(FurnitureSimRLEnv)
        env.num_envs = 1
        env.scenes = [SimpleNamespace(render_system=object())]
        env.physx_system = SimpleNamespace(cuda_rigid_body_data=object())
        env.sensors = {"front": [object()]}
        env.sensor_keys = {"front": {"Color"}}
        env._require_direct_sensor_interop = lambda: None
        _FakeRenderSystemGroup.instances.clear()

        with patch.object(
            furniture_sim_env.sapien.render,
            "RenderSystemGroup",
            _FakeRenderSystemGroup,
        ):
            env._init_render()

        group = _FakeRenderSystemGroup.instances[-1]
        self.assertEqual(
            group.calls,
            [
                "set_cuda_poses",
                "create_camera_group",
                "set_pose_mode:static",
                "gpu_init",
                "get_picture_cuda",
                "torch",
                "update_render",
            ],
        )

    def test_render_only_step_updates_kinematics_before_render(self) -> None:
        env = object.__new__(FurnitureSimRLEnv)
        calls: list[str] = []
        env.num_envs = 1
        env.set_franka = lambda _qpos: calls.append("set_franka")
        env.set_parts_env = lambda _index, _poses: calls.append("set_parts_env")
        env._apply_all = lambda: calls.append("apply_all")
        env.physx_system = SimpleNamespace(
            gpu_update_articulation_kinematics=lambda: calls.append(
                "update_kinematics"
            ),
            gpu_fetch_articulation_link_pose=lambda: calls.append(
                "fetch_link_pose"
            ),
        )
        env.update_render = lambda: calls.append("update_render")
        env.get_observation = lambda: {"color_image1": torch.zeros(1)}
        env.record = False

        env.render_only_step(torch.zeros((1, 9)), np.zeros((1, 21)))

        self.assertEqual(
            calls,
            [
                "set_franka",
                "set_parts_env",
                "apply_all",
                "update_kinematics",
                "fetch_link_pose",
                "update_render",
            ],
        )

    def test_physx_step_applies_viewer_interactions_first(self) -> None:
        calls: list[str] = []
        env = object.__new__(FurnitureSimRLEnv)
        env.viewer = SimpleNamespace(
            apply_interactions=lambda: calls.append("apply_interactions")
        )
        env.physx_system = SimpleNamespace(step=lambda: calls.append("physx_step"))

        env._step_physx()

        self.assertEqual(calls, ["apply_interactions", "physx_step"])

    def test_gpu_viewer_submission_does_not_sync_cpu_poses(self) -> None:
        calls: list[str] = []
        env = object.__new__(FurnitureSimRLEnv)
        env._rendering_enabled = True
        env.viewer = SimpleNamespace(
            update_render=lambda: calls.append("viewer_update"),
            render=lambda: calls.append("viewer_render"),
        )
        env.step_sensor = lambda: calls.append("sensor_update")

        env.update_render()

        self.assertEqual(
            calls,
            ["viewer_update", "viewer_render", "sensor_update"],
        )

    def test_physics_only_update_skips_render_paths(self) -> None:
        calls: list[str] = []
        env = object.__new__(FurnitureSimRLEnv)
        env._rendering_enabled = False
        env.step_viewer = lambda: calls.append("viewer_update")
        env.step_sensor = lambda: calls.append("sensor_update")

        env.update_render()

        self.assertEqual(calls, [])

    def test_fetches_only_control_and_observation_state(self) -> None:
        calls: list[str] = []
        env = object.__new__(FurnitureSimRLEnv)
        env.physx_system = SimpleNamespace(
            gpu_fetch_rigid_dynamic_data=lambda: calls.append("rigid_dynamic"),
            gpu_fetch_articulation_link_pose=lambda: calls.append("link_pose"),
            gpu_fetch_articulation_link_velocity=lambda: calls.append(
                "link_velocity"
            ),
            gpu_fetch_articulation_qpos=lambda: calls.append("qpos"),
            gpu_fetch_articulation_qvel=lambda: calls.append("qvel"),
        )

        env._fetch_all()

        self.assertEqual(
            calls,
            ["rigid_dynamic", "link_pose", "link_velocity", "qpos", "qvel"],
        )

    def test_viewer_auto_selects_same_gpu_direct_transport(self) -> None:
        calls: list[str] = []

        class FakeViewer:
            def __init__(self) -> None:
                self.control_window = SimpleNamespace()
                self.pose_transport = "direct"
                self.pose_transfer_bytes = 0

            def set_scene(self, _scene: object) -> None:
                calls.append("set_scene")

            def configure_physx_gpu_rendering(
                self,
                _system: object,
                *,
                transport: str,
            ) -> None:
                calls.append(f"configure:{transport}")

            def set_camera_pose(self, _pose: object) -> None:
                calls.append("set_camera_pose")

            def update_render(self) -> None:
                calls.append("viewer_update")

            def render(self) -> None:
                calls.append("viewer_render")

        env = object.__new__(FurnitureSimRLEnv)
        env.viewer_shader = SimpleNamespace(shader_dir="minimal")
        env.scenes = [object()]
        env.physx_system = object()
        env._direct_cuda_vulkan_interop = True
        env._same_gpu_render_device = True

        with (
            patch.object(furniture_sim_env, "Viewer", FakeViewer),
            patch.object(
                furniture_sim_env.sapien.render,
                "set_viewer_shader_dir",
            ),
            patch.object(furniture_sim_env, "set_shader"),
        ):
            env._init_viewer()

        self.assertEqual(
            calls,
            [
                "set_scene",
                "configure:auto",
                "set_camera_pose",
                "viewer_update",
                "viewer_render",
            ],
        )

    def test_sensor_observation_does_not_alias_tracked_cuda_image(self) -> None:
        env = object.__new__(FurnitureSimRLEnv)
        raw = torch.ones((2, 4, 4, 4), dtype=torch.float32)
        env.render_system_group = object()
        env._sensor_cuda_tensor_views = {"wrist": {"Color": raw}}
        env.camera_names_dict = {"1": "wrist"}
        env.camera_shader = MINIMAL_SHADER_CONFIG
        env.obs_keys = ["color_image1"]

        obs = env.get_sensor_obs()["color_image1"]
        self.assertNotEqual(obs.data_ptr(), raw.data_ptr())
        raw.zero_()
        self.assertTrue(torch.all(obs == 1))


if __name__ == "__main__":
    unittest.main()
