"""Collect demonstrations with the classic keyboard / Oculus interfaces."""

from __future__ import annotations

import argparse
import os
import os.path as osp
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser without importing heavy runtime dependencies."""
    parser = argparse.ArgumentParser(description="Collect IL data")
    parser.add_argument(
        "--out-data-path", help="Path to directory to save the data", required=True
    )
    parser.add_argument(
        "--input-device",
        help="Device to control the robot.",
        choices=["keyboard", "oculus", "keyboard-oculus"],
        default="keyboard-oculus",
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
    parser.add_argument("--num-demos", default=100, type=int)
    parser.add_argument("--resize-sim-img", action="store_true")
    parser.add_argument(
        "--ctrl-mode",
        default="osc",
        type=str,
        choices=["osc", "diffik"],
        help=(
            "Low-level controller selection. 'osc' is kept as a compatibility alias "
            "and maps to the remaining DiffIK controller in simulation."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the data-collection entry point."""
    args = parse_args(argv)

    # Import runtime dependencies after parsing so ``--help`` stays lightweight.
    from furniture_bench.config import config
    from furniture_bench.data.data_collector import DataCollector
    from furniture_bench.device import make_device

    if args.furniture not in config["furniture"]:
        raise ValueError(f"Unknown furniture: {args.furniture}")

    # Scripted collection does not need a teleoperation device.
    if args.scripted:
        if not args.is_sim:
            raise ValueError("--scripted is only supported together with --is-sim.")
        device_interface = None
    else:
        device_interface = make_device(args.input_device)

    # Keep the on-disk layout unchanged: one subdirectory per furniture type.
    data_path = osp.join(args.out_data_path, args.furniture)
    if not osp.isdir(data_path):
        os.makedirs(data_path)

    data_collector = DataCollector(
        is_sim=args.is_sim,
        data_path=data_path,
        device_interface=device_interface,
        furniture=args.furniture,
        headless=args.headless,
        draw_marker=args.draw_marker,
        manual_label=args.manual_label,
        scripted=args.scripted,
        randomness=args.randomness,
        compute_device_id=args.compute_device_id,
        graphics_device_id=args.graphics_device_id,
        pkl_only=args.pkl_only,
        save_failure=args.save_failure,
        num_demos=args.num_demos,
        resize_sim_img=args.resize_sim_img,
        ctrl_mode=args.ctrl_mode,
    )
    data_collector.collect()


if __name__ == "__main__":
    main()
