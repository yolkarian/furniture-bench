"""Move the real robot to the reset pose and then lift the end effector."""

from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build a minimal parser so ``--help`` works without hardware imports."""
    return argparse.ArgumentParser(description="Reset the robot and move the arm up.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Reset the robot and move it to a safer Z height."""
    parse_args(argv)

    # Delay hardware imports until after argument parsing.
    from furniture_bench.config import config
    from furniture_bench.robot.panda import Panda

    robot = Panda(config["robot"])
    print("Reset the robot")

    # First return to the standard reset pose.
    robot.reset()
    # Then lift the end effector to create extra clearance above the table.
    robot.z_move(0.2)

    print("Reset done.")


if __name__ == "__main__":
    main()
