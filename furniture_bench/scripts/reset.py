"""Move the real robot back to its reset pose."""

from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build a minimal parser so ``--help`` works without hardware imports."""
    return argparse.ArgumentParser(description="Reset the robot to the home pose.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Reset the robot using the configured hardware server."""
    parse_args(argv)

    # Delay hardware imports until after argument parsing.
    from furniture_bench.config import config
    from furniture_bench.robot.panda import Panda

    robot = Panda(config["robot"])
    print("Reset the robot")
    robot.reset()
    print("Reset done.")


if __name__ == "__main__":
    main()
