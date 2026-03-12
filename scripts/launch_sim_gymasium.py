"""Instantiate the Gymnasium `FurnitureSim-v0` environment and run a short probe.

The historical file name intentionally keeps the original ``gymasium`` spelling.
"""

from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the Gymnasium simulator demo CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="FurnitureSim-v0")
    parser.add_argument("--furniture", default="one_leg")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--parallel-in-single-scene", action="store_true")
    parser.add_argument("--gripper-torque", type=float, default=0.002)
    parser.add_argument(
        "--camera-shader",
        choices=["default", "minimal", "rt"],
        default="default",
    )
    parser.add_argument(
        "--viewer-shader",
        choices=["default", "minimal", "rt"],
        default="rt",
    )
    parser.add_argument("--move-start-step", type=int, default=200)
    parser.add_argument("--reset-step", type=int, default=1000)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1200,
        help="Maximum simulator steps to run. Use a negative value to run until closed.",
    )
    parser.set_defaults(init_assembled=True)
    parser.add_argument("--init-assembled", action="store_true", dest="init_assembled")
    parser.add_argument(
        "--no-init-assembled",
        action="store_false",
        dest="init_assembled",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def should_continue(step_count: int, max_steps: int) -> bool:
    """Return whether the bounded demo loop should keep running."""
    return max_steps < 0 or step_count < max_steps


def viewer_is_open(env: object) -> bool:
    """Check whether the interactive viewer is still open."""
    viewer = getattr(env, "viewer", None)
    if viewer is None:
        return True
    return not bool(getattr(viewer, "closed", False))


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Gymnasium-registered FurnitureSim demo."""
    args = parse_args(argv)

    # Delay heavy imports until after parsing so ``--help`` stays cheap.
    import gymnasium as gym
    import torch

    import furniture_bench  # noqa: F401  Registers Gymnasium environments.
    from furniture_bench.envs.observation import FULL_OBS
    from furniture_bench.sim_config import sim_config

    env: object | None = None
    original_gripper_torque = sim_config["robot"]["gripper_torque"]
    sim_config["robot"]["gripper_torque"] = args.gripper_torque

    try:
        env = gym.make(
            id=args.env_id,
            furniture=args.furniture,
            num_envs=args.num_envs,
            parallel_in_single_scene=args.parallel_in_single_scene,
            headless=args.headless,
            obs_keys=FULL_OBS,
            init_assembled=args.init_assembled,
            enable_sensor=True,
            camera_shader=args.camera_shader,
            viewer_shader=args.viewer_shader,
            action_type="delta",
        )
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        env.reset()

        action = torch.zeros_like(base_env.act_low, device=base_env.device)
        did_extra_reset = False
        step_count = 0

        while should_continue(step_count, args.max_steps) and viewer_is_open(base_env):
            env_step = int(base_env.env_steps[0])
            if env_step >= args.move_start_step:
                action[:, -1] -= 0.001
                action[:, 0] -= 0.0005 * base_env.dt

            if env_step >= args.reset_step and not did_extra_reset:
                # Reset once mid-run so the Gymnasium path also exercises re-init.
                env.reset()
                action.zero_()
                did_extra_reset = True

            env.step(action)
            step_count += 1

        print(f"Completed {step_count} Gymnasium simulator steps.")
    finally:
        sim_config["robot"]["gripper_torque"] = original_gripper_torque
        if env is not None:
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()


if __name__ == "__main__":
    main()
