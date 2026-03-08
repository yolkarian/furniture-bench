"""Replay a recorded simulator trajectory and optionally regenerate RGB outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import safetensors.numpy
import torch


def build_parser() -> argparse.ArgumentParser:
    """Build the replay CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", "-t", type=str, default="one_leg")
    parser.add_argument("--record-path", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", "-o", type=str, default=None)
    parser.add_argument("--save-output", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def resolve_record_path(record_path: str | None) -> str:
    """Resolve the input safetensors path.

    When no explicit path is provided we keep the previous behavior of scanning
    the local ``scripts/`` directory for the first available replay file.
    """
    if record_path is not None:
        return record_path

    script_dir = Path(__file__).resolve().parent
    replay_candidates = sorted(
        path for path in script_dir.iterdir() if path.suffix == ".safetensors"
    )
    if not replay_candidates:
        raise ValueError("Cannot find any path to replay!")
    return str(replay_candidates[0])


def main(argv: Sequence[str] | None = None) -> None:
    """Replay a recorded trajectory with the SAPIEN simulator."""
    args = parse_args(argv)
    record_path = resolve_record_path(args.record_path)
    output_path = (
        args.output
        or f"{record_path.rsplit('.', maxsplit=1)[0]}_color_images.safetensors"
    )

    # Delay simulator imports until after parsing so ``--help`` is lightweight.
    from furniture_bench.envs.furniture_sim_env import FurnitureSimRLEnv
    from furniture_bench.envs.observation import DEFAULT_REPLAY_KEYS, FULL_OBS

    tensor_dict = safetensors.numpy.load_file(record_path)

    num_epoch = tensor_dict["nobs"].shape[0]
    max_steps = tensor_dict["nobs"].shape[1]
    num_envs = 1  # The replay format currently stores a single environment.

    data_keys = list(tensor_dict.keys())
    if not all(key in data_keys for key in DEFAULT_REPLAY_KEYS):
        raise ValueError("Recorded data does not contain joint info.")
    if "parts_poses" not in data_keys:
        raise ValueError("Recorded data does not contain parts info.")

    qpos_list = [tensor_dict[key] for key in DEFAULT_REPLAY_KEYS]
    qpos = np.concatenate(qpos_list, axis=-1).astype(np.float32)
    parts_poses = tensor_dict["parts_poses"].astype(np.float32)

    env = FurnitureSimRLEnv(
        furniture=args.task,
        num_envs=num_envs,
        parallel_in_single_scene=False,
        headless=args.headless,
        obs_keys=FULL_OBS,
        init_assembled=False,
        enable_sensor=True,
        camera_shader="rt",
        viewer_shader="rt",
        action_type="delta",
        april_tags=True,
        record=True,
    )
    observation = env.reset()

    if args.save_output:
        color_image1 = np.zeros(
            (num_epoch, max_steps, num_envs, *observation["color_image1"].shape[-3:]),
            dtype=np.uint8,
        )
        color_image2 = np.zeros(
            (num_epoch, max_steps, num_envs, *observation["color_image2"].shape[-3:]),
            dtype=np.uint8,
        )

    epoch_idx = 0
    for step_idx in range(max_steps):
        # Preserve the original replay randomization so regenerated renders match
        # the previously supported workflow.
        env.rand_parts_rendering(0.6)
        env.rand_light(0.6)
        env.rand_franka_rendering(0.2)
        env.rand_obstacle_rendering(0.3)

        observation = env.render_only_step(
            torch.from_numpy(qpos[epoch_idx, step_idx]).to("cuda"),
            parts_poses[epoch_idx, step_idx],
        )
        if args.save_output:
            color_image1[epoch_idx, step_idx] = observation["color_image1"].cpu()
            color_image2[epoch_idx, step_idx] = observation["color_image2"].cpu()

    if args.save_output:
        safetensors.numpy.save_file(
            {
                "color_image1": color_image1,
                "color_image2": color_image2,
            },
            output_path,
        )

    del env


if __name__ == "__main__":
    main()
