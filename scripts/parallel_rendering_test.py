"""Probe ManiSkill's parallel-in-single-scene rendering path."""

from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the parallel-rendering demo CLI."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default="PickCube-v1")
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--control-mode", default="pd_joint_delta_pos")
    parser.add_argument("--sim-backend", default="gpu")
    parser.add_argument("--render-backend", default="gpu")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--shader-pack", default="rt-fast")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum environment steps to run. Use a negative value to run forever.",
    )
    parser.set_defaults(parallel_in_single_scene=True, render=True)
    parser.add_argument(
        "--parallel-in-single-scene",
        action="store_true",
        dest="parallel_in_single_scene",
    )
    parser.add_argument(
        "--no-parallel-in-single-scene",
        action="store_false",
        dest="parallel_in_single_scene",
    )
    parser.add_argument("--render", action="store_true", dest="render")
    parser.add_argument("--no-render", action="store_false", dest="render")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def should_continue(step_count: int, max_steps: int) -> bool:
    """Return whether the bounded demo loop should keep running."""
    return max_steps < 0 or step_count < max_steps


def main(argv: Sequence[str] | None = None) -> None:
    """Run the ManiSkill rendering probe."""
    args = parse_args(argv)

    # Delay optional runtime imports until after parsing so ``--help`` stays cheap.
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  Registers ManiSkill environments.

    env = gym.make(
        args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        num_envs=args.num_envs,
        parallel_in_single_scene=args.parallel_in_single_scene,
        viewer_camera_configs={"shader_pack": args.shader_pack},
    )

    try:
        env.reset()
        base_env = env.unwrapped if hasattr(env, "unwrapped") else env
        print(base_env.scene.px.cuda_articulation_qpos.torch().shape)
        print(
            len(base_env.scene.sub_scenes[-1].get_entities()),
            len(base_env.scene.sub_scenes[0].get_entities()),
        )

        step_count = 0
        while should_continue(step_count, args.max_steps):
            env.step(env.action_space.sample())
            if args.render:
                env.render_human()
            step_count += 1

        print(f"Completed {step_count} ManiSkill steps.")
    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            close_fn()


if __name__ == "__main__":
    main()
