"""Convert raw demonstration pickles into model-training trajectories."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset-preprocessing CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-data-path", help="Path to directory to load the data", required=True
    )
    parser.add_argument(
        "--out-data-path", help="Path to directory to save the data", required=True
    )
    parser.add_argument(
        "--save-last-step",
        action="store_true",
        help="Whether to save last step of the trajectory. (For example, for Off-policy learning)",
    )
    parser.add_argument(
        "--no-robot-state",
        action="store_true",
        help="Do not use robot state.",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Only use successful trajectories",
    )
    parser.add_argument(
        "--done-when-assembled",
        action="store_true",
        help="Terminate converting when all the parts are assembled.",
    )
    parser.add_argument("--sum-rew", type=int)
    parser.add_argument(
        "--from-skill", type=int, help="Where evaluation starts in skill benchmark"
    )
    parser.add_argument(
        "--to-skill", type=int, help="Where evaluation ends in skill benchmark"
    )
    parser.add_argument(
        "--skill-margin", type=int, help="Margin of skill benchmark", default=10
    )
    parser.add_argument(
        "--use-all-cam",
        action="store_true",
        help="Use all of images from three cameras.",
    )
    parser.add_argument(
        "--stack-cam", action="store_true", help="Stack images from three cameras."
    )
    parser.add_argument(
        "--norm-pos-acts",
        action="store_true",
        help="Do not normalize positional actions. [-1 to 1]",
    )
    parser.add_argument(
        "--norm-pos-x",
        type=float,
        help="Normalization factor of x position.",
        default=0.1001,
    )
    parser.add_argument(
        "--norm-pos-y",
        type=float,
        help="Normalization factor of y position.",
        default=0.1001,
    )
    parser.add_argument(
        "--norm-pos-z",
        type=float,
        help="Normalization factor of z position.",
        default=0.1001,
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def trim_leading_noops(trajectory: dict[str, Any]) -> int:
    """Drop the leading neutral actions that often precede teleoperation."""
    no_action = np.array([0, 0, 0, 0, 0, 0, 1, -1], dtype=np.float32)
    num_skipped = 0
    for idx, action in enumerate(trajectory["actions"]):
        if np.isclose(action, no_action).all():
            num_skipped = idx + 1
        else:
            break

    if num_skipped > 0:
        trajectory["observations"] = trajectory["observations"][num_skipped:]
        trajectory["actions"] = trajectory["actions"][num_skipped:]
        trajectory["rewards"] = trajectory["rewards"][num_skipped:]
        trajectory["skills"] = trajectory["skills"][num_skipped:]

    return num_skipped


def truncate_when_assembled(trajectory: dict[str, Any], total_reward: float) -> int:
    """Stop the trajectory after the final assembly reward is observed."""
    reward_sum = 0.0
    len_traj = len(trajectory["actions"])
    terminal_reward_idx = len(trajectory["rewards"]) - 1
    for idx, reward in enumerate(trajectory["rewards"]):
        reward_sum += reward
        if reward_sum == total_reward:
            terminal_reward_idx = idx
            break

    # Keep the historical behavior: retain the action that produced the final
    # reward and one extra observation after it.
    done_idx = (
        terminal_reward_idx + 1 if terminal_reward_idx + 2 < len_traj else len_traj - 1
    )
    trajectory["observations"] = trajectory["observations"][: done_idx + 1]
    trajectory["actions"] = trajectory["actions"][:done_idx]
    trajectory["rewards"] = trajectory["rewards"][:done_idx]
    trajectory["skills"] = trajectory["skills"][:done_idx]
    return done_idx


def truncate_skill_window(
    trajectory: dict[str, Any],
    from_skill: int,
    to_skill: int | None,
    skill_margin: int,
) -> None:
    """Crop the trajectory to a skill-conditioned evaluation window."""
    len_traj = len(trajectory["actions"])
    if len_traj == 0:
        return

    target_to_skill = len_traj - 1 if to_skill is None else to_skill
    skill = 0
    from_skill_idx = 0
    skill_done_idx = len_traj - 1

    for idx, skill_complete in enumerate(trajectory["skills"]):
        if skill_complete == 1:
            skill += 1
            if skill == from_skill:
                from_skill_idx = idx
            if skill == target_to_skill:
                skill_done_idx = idx
                break

    start_idx = max(from_skill_idx - skill_margin, 0)
    done_idx = min(skill_done_idx + skill_margin + 1, len_traj - 1)

    trajectory["observations"] = trajectory["observations"][start_idx : done_idx + 1]
    trajectory["actions"] = trajectory["actions"][start_idx:done_idx]
    trajectory["rewards"] = trajectory["rewards"][start_idx:done_idx]
    trajectory["skills"] = trajectory["skills"][start_idx:done_idx]


def move_images_channel_first(trajectory: dict[str, Any]) -> None:
    """Convert RGB observations to channel-first tensors expected by training code."""
    for observation in trajectory["observations"]:
        for image_key in ["color_image1", "color_image2"]:
            observation[image_key] = np.moveaxis(observation[image_key], -1, 0)


def simplify_observations(
    trajectory: dict[str, Any], use_all_cam: bool, no_robot_state: bool
) -> None:
    """Keep only the observation fields needed by downstream pipelines."""
    from furniture_bench.robot.robot_state import filter_and_concat_robot_state

    if use_all_cam:
        trajectory["observations"] = [
            {
                "color_image1": obs["color_image1"],
                "color_image2": obs["color_image2"],
                "color_image3": obs["color_image3"],
                "robot_state": filter_and_concat_robot_state(obs["robot_state"]),
            }
            for obs in trajectory["observations"]
        ]
    else:
        trajectory["observations"] = [
            {
                "color_image1": obs["color_image1"],
                "color_image2": obs["color_image2"],
                "robot_state": filter_and_concat_robot_state(obs["robot_state"]),
            }
            for obs in trajectory["observations"]
        ]

    if no_robot_state:
        for observation in trajectory["observations"]:
            observation.pop("robot_state")


def normalize_actions(trajectory: dict[str, Any], args: argparse.Namespace) -> None:
    """Normalize position deltas into the [-1, 1] interval when requested."""
    norm_eps = 1e-5
    for index, action in enumerate(trajectory["actions"]):
        normalized = np.array(action, copy=True)
        if normalized[6] < 0:
            normalized[3:7] = -normalized[3:7]
        if args.norm_pos_acts:
            normalized[0] /= args.norm_pos_x
            normalized[1] /= args.norm_pos_y
            normalized[2] /= args.norm_pos_z
            normalized = np.clip(normalized, -1 + norm_eps, 1 - norm_eps)
        trajectory["actions"][index] = normalized


def main(argv: Sequence[str] | None = None) -> None:
    """Preprocess every pickle file under the input directory."""
    args = parse_args(argv)

    # Delay config import until after parsing so ``--help`` stays lightweight.
    from furniture_bench.config import config

    # ``sum_rew`` and ``stack_cam`` are preserved for CLI compatibility.
    _ = args.sum_rew, args.stack_cam

    files = list(Path(args.in_data_path).rglob("*.pkl"))
    if not files:
        raise ValueError("Data path is empty")

    out_dir = Path(args.out_data_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    for file_index, file_path in enumerate(sorted(files), start=1):
        print(f"[{file_index} / {len(files)}] converting {file_path}")

        with open(file_path, "rb") as file_obj:
            try:
                data = pickle.load(file_obj)
            except Exception:
                print(f"Fail to load: {file_path}")
                continue

        if args.success_only and not data["success"]:
            print(f"Skip failed trajectory: {file_path}")
            continue
        if len(data["observations"]) == 0:
            print(f"Skip empty trajectory: {file_path}")
            continue

        new_traj: dict[str, Any] = {
            "furniture": data["furniture"],
            "observations": data["observations"].copy(),
            "actions": data["actions"].copy(),
            "rewards": data["rewards"].copy(),
            "skills": data["skills"].copy(),
        }

        num_skipped = trim_leading_noops(new_traj)
        print(f"Number of skipped actions: {num_skipped}")

        done_idx = -1
        if args.done_when_assembled:
            done_idx = truncate_when_assembled(
                new_traj,
                config["furniture"][data["furniture"]]["total_reward"],
            )

        if args.from_skill is not None:
            truncate_skill_window(
                new_traj,
                from_skill=args.from_skill,
                to_skill=args.to_skill,
                skill_margin=args.skill_margin,
            )

        # For standard imitation-learning trajectories we keep obs = actions + 1.
        if not args.save_last_step and new_traj["observations"]:
            new_traj["observations"].pop()

        if args.done_when_assembled and done_idx >= 0:
            print(
                "Number of truncated last steps: "
                f"{max(len(data['actions']) - (done_idx + 1), 0)}"
            )
        else:
            print("Number of truncated last steps: 0")
        print(f"Length of new trajectory {len(new_traj['actions'])}")

        move_images_channel_first(new_traj)
        simplify_observations(
            new_traj,
            use_all_cam=args.use_all_cam,
            no_robot_state=args.no_robot_state,
        )
        normalize_actions(new_traj, args)

        if args.norm_pos_acts:
            print(
                "Normalization factor: "
                f"({args.norm_pos_x}, {args.norm_pos_y}, {args.norm_pos_z})"
            )

        new_file = out_dir / file_path.name
        print(f">> save to {new_file}")
        with open(new_file, "wb") as file_obj:
            pickle.dump(new_traj, file_obj)


if __name__ == "__main__":
    main()
