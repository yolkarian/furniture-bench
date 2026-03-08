"""Instantiate a simulator environment and step it with a selected policy."""

from __future__ import annotations

import argparse
import pickle
from typing import Sequence

import numpy as np
import torch


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the simulator smoke-test script."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--furniture", default="square_table")
    parser.add_argument(
        "--file-path", help="Demo path to replay (data directory or pickle)"
    )
    parser.add_argument(
        "--scripted", action="store_true", help="Execute hard-coded assembly script."
    )
    parser.add_argument("--no-action", action="store_true")
    parser.add_argument("--random-action", action="store_true")
    parser.add_argument(
        "--input-device",
        help="Device to control the robot.",
        choices=["keyboard", "oculus", "keyboard-oculus"],
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--init-assembled",
        action="store_true",
        help="Initialize the environment with the assembled furniture.",
    )
    parser.add_argument(
        "--save-camera-input",
        action="store_true",
        help="Save camera input of the simulator at the beginning of the episode.",
    )
    parser.add_argument(
        "--record", action="store_true", help="Record the video of the simulator."
    )
    parser.add_argument(
        "--high-res",
        action="store_true",
        help="Use high resolution images for the camera input.",
    )
    parser.add_argument(
        "--randomness",
        default="low",
        help="Randomness level of the environment.",
    )
    parser.add_argument(
        "--high-random-idx",
        default=0,
        type=int,
        help="The index of high_randomness.",
    )
    parser.add_argument(
        "--env-id",
        default="FurnitureSim-v0",
        help="Environment id of FurnitureSim.",
    )
    parser.add_argument(
        "--replay-path", type=str, help="Path to the saved data to replay action."
    )
    parser.add_argument(
        "--act-rot-repr",
        type=str,
        help="Rotation representation for action space.",
        choices=["quat", "axis", "rot_6d"],
        default="quat",
    )
    parser.add_argument(
        "--compute-device-id",
        type=int,
        default=0,
        help="GPU device ID used for simulation.",
    )
    parser.add_argument(
        "--graphics-device-id",
        type=int,
        default=0,
        help="GPU device ID used for rendering.",
    )
    parser.add_argument("--num-envs", type=int, default=1)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def action_tensor(
    action: list[float] | np.ndarray | torch.Tensor, device: torch.device, num_envs: int
) -> torch.Tensor:
    """Normalize action inputs to a batched tensor on the simulator device."""
    if isinstance(action, (list, np.ndarray)):
        batched_action = torch.tensor(action, dtype=torch.float32, device=device)
        if batched_action.ndim == 1:
            batched_action = batched_action[None, :]
        return batched_action.tile(num_envs, 1)

    batched_action = action.clone()
    if batched_action.ndim == 1:
        batched_action = batched_action[None, :]
    return batched_action.tile(num_envs, 1).float().to(device)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the simulator with teleop, replay, scripted, or no-op actions."""
    args = parse_args(argv)

    # Delay simulator imports until after argument parsing so ``--help`` stays cheap.
    from furniture_bench.device import make_device
    from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv

    # ``env-id`` is kept for CLI compatibility even though this script directly
    # instantiates ``FurnitureSimEnv`` like the previous implementation did.
    _ = args.env_id

    env = FurnitureSimEnv(
        furniture=args.furniture,
        num_envs=args.num_envs,
        resize_img=not args.high_res,
        init_assembled=args.init_assembled,
        record=args.record,
        headless=args.headless,
        enable_reward=True,
        enable_sensor=True,
        save_camera_input=args.save_camera_input,
        randomness=args.randomness,
        high_random_idx=args.high_random_idx,
        act_rot_repr=args.act_rot_repr,
        compute_device_id=args.compute_device_id,
        graphics_device_id=args.graphics_device_id,
        april_tags=True,
    )

    env.reset()
    done = False

    if args.input_device is not None:
        # Teleoperation keeps stepping until the environment reports done.
        device_interface = make_device(args.input_device)
        while not done:
            action, _ = device_interface.get_action()
            _, _, done, _ = env.step(action_tensor(action, env.device, args.num_envs))
    elif args.no_action or args.init_assembled:
        # Use the action-space neutral element to render an idle episode.
        while True:
            if args.act_rot_repr == "quat":
                action = [0, 0, 0, 0, 0, 0, 1, -1]
            else:
                action = [0, 0, 0, 0, 0, 0, -1]
            env.step(action_tensor(action, env.device, args.num_envs))
    elif args.random_action:
        import tqdm

        pbar = tqdm.tqdm()
        while True:
            sampled = env.action_space.sample()
            env.step(action_tensor(sampled, env.device, args.num_envs))
            pbar.update(args.num_envs)
    elif args.file_path is not None:
        # Replay actions stored in a demonstration pickle.
        with open(args.file_path, "rb") as file_obj:
            data = pickle.load(file_obj)
        for action in data["actions"]:
            env.step(action_tensor(action, env.device, args.num_envs))
    elif args.scripted:
        # Execute the hard-coded assembly policy exposed by the environment.
        while not done:
            action, _ = env.get_assembly_action()
            _, _, done, _ = env.step(action_tensor(action, env.device, args.num_envs))
    elif args.replay_path is not None:
        # Restore the initial simulator state before replaying logged actions.
        with open(args.replay_path, "rb") as file_obj:
            data = pickle.load(file_obj)
        env.reset_to([data["observations"][0]])
        for action in data["actions"]:
            _, _, done, _ = env.step(action_tensor(action, env.device, args.num_envs))
    else:
        raise ValueError("No action source specified.")

    print("done")


if __name__ == "__main__":
    main()
