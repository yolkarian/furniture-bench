"""Collect demonstrations with the SpaceMouse-based workflow."""

from __future__ import annotations

import argparse
import os
import os.path as osp
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without importing simulator dependencies."""
    parser = argparse.ArgumentParser(description="Collect IL data")
    parser.add_argument(
        "--out-data-path", help="Path to directory to save the data", required=True
    )
    parser.add_argument(
        "--furniture",
        help="Name of the furniture",
        required=True,
    )
    parser.add_argument(
        "--is-sim",
        action="store_true",
        help="Use simulator, else use real world environment.",
    )
    parser.add_argument(
        "--scripted",
        action="store_true",
        help="Use scripted function for getting action.",
    )
    parser.add_argument(
        "--pkl-only",
        action="store_true",
        help="Only save the pickle file, not .mp4 and .pngs",
    )
    parser.add_argument(
        "--save-failure",
        action="store_true",
        help="Save failure trajectories.",
    )
    parser.add_argument(
        "--headless", help="With front camera view", action="store_true"
    )
    parser.add_argument(
        "--draw-marker", action="store_true", help="Draw AprilTag marker"
    )
    parser.add_argument(
        "--manual-label",
        action="store_true",
        help="Manually label the reward",
    )
    parser.add_argument("--randomness", default="low", choices=["low", "med", "high"])
    parser.add_argument("--gpu-id", default=0, type=int)
    parser.add_argument("--num-demos", default=100, type=int)
    parser.add_argument(
        "--obs-type",
        type=str,
        choices=["state", "full", "image"],
        default="image",
        help="Observation mode to store in the dataset.",
    )
    parser.add_argument(
        "--ctrl-mode",
        type=str,
        help=(
            "Type of low level controller to use. 'osc' is kept as a compatibility "
            "alias and maps to the remaining DiffIK controller in simulation."
        ),
        choices=["osc", "diffik"],
        default="osc",
    )
    parser.add_argument(
        "--no-ee-laser",
        action="store_false",
        help="If set, will not show the laser coming from the end effector",
        dest="ee_laser",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run SpaceMouse data collection."""
    args = parse_args(argv)

    # Delay heavy imports so ``--help`` can run without simulator / hardware libs.
    from furniture_bench.config import config
    from furniture_bench.data.data_collector_sm import DataCollectorSpaceMouse
    from furniture_bench.device import make_device

    if args.furniture not in config["furniture"]:
        raise ValueError(f"Unknown furniture: {args.furniture}")

    # The SpaceMouse pipeline still uses the keyboard interface for labels and
    # episode-control hotkeys, so initialize it once here.
    keyboard_device_interface = make_device("keyboard")

    data_path = osp.join(args.out_data_path, args.furniture)
    if not osp.isdir(data_path):
        os.makedirs(data_path)

    data_collector = DataCollectorSpaceMouse(
        is_sim=args.is_sim,
        data_path=data_path,
        device_interface=keyboard_device_interface,
        furniture=args.furniture,
        headless=args.headless,
        draw_marker=args.draw_marker,
        manual_label=args.manual_label,
        obs_type=args.obs_type,
        resize_img_after_sim=False,
        # Raw simulator images are already downsampled in this path.
        small_sim_img_size=True,
        scripted=args.scripted,
        randomness=args.randomness,
        gpu_id=args.gpu_id,
        pkl_only=args.pkl_only,
        save_failure=args.save_failure,
        num_demos=args.num_demos,
        ctrl_mode=args.ctrl_mode,
        ee_laser=args.ee_laser,
    )
    data_collector.collect()


if __name__ == "__main__":
    main()
