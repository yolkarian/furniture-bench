from __future__ import annotations

from typing import Any

import pytest
import sapien

from furniture_bench.envs.furniture_sim_env import FurnitureSimRLEnv


@pytest.mark.parametrize(
    ("rendering_enabled", "expected_render_systems"),
    [(False, 0), (True, 1)],
)
def test_scene_systems_only_constructs_renderer_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    rendering_enabled: bool,
    expected_render_systems: int,
) -> None:
    env = object.__new__(FurnitureSimRLEnv)
    env.physx_system = object()
    env.sapien_device = object()
    env._rendering_enabled = rendering_enabled
    render_system = object()
    calls: list[Any] = []

    def create_render_system(device: Any) -> object:
        calls.append(device)
        return render_system

    monkeypatch.setattr(sapien.render, "RenderSystem", create_render_system)

    systems = env._scene_systems()

    assert systems[0] is env.physx_system
    assert len(calls) == expected_render_systems
    assert systems[1:] == ([render_system] if rendering_enabled else [])
