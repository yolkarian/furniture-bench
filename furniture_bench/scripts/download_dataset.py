"""Download demonstration datasets from Google Drive."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Sequence

DOWNLOAD_LINKS: dict[str, dict[str, str]] = {
    "high": {
        "cabinet": "1RHbo27phzXVJDjMXPI91QKPHWByCSQU4",
        "chair": "1D8j1s4v9NL02V03PwEU6moOjn9v5qzFn",
        "desk": "1xOhzI96-BORgjyqF7rBdYRPc_-RxYr_Z",
        "drawer": "1QNqA48y9fFE4251xMmCMaUOdmNcnaZ8T",
        "lamp": "1Ia0SHIACIoqwzVjhc_dgzEMsiKffIO6T",
        "one_leg": "19iLUPDAvrRzevVggfD09nK2ayZ3GrVfC",
        "round_table": "1qS2lIiPdqq8pTJsPN2txU1lzQJ4seWCr",
        "square_table": "1Wq1MVCUSXxi6wJk7CQW-3LMGdUGJdiAR",
        "stool": "1QeEb4Ajz-qN820Y_DVuxU7UdvrZBv7hJ",
    },
    "med": {
        "cabinet": "1LRexjymeP0szZucTEt40ZL-2VoLYXKKo",
        "chair": "1wXGloFr4aVJ3ChYz4qKD_zRheuivM9rc",
        "desk": "1edLqFAxKRAPcnNgDkRBmw9AilN8zZSqs",
        "drawer": "1nFdVpUERi90zNNthfOR2sdYCjrW7Rg_c",
        "lamp": "1awqLazZlNOqDhnOuElttOwDOol9oWY0C",
        "one_leg": "1zRqpz3WLztpOo7ULYC6Ik3rWyYtoo9ch",
        "round_table": "1gJ_HmhpgE4nJNBMmEKHHx7mKjYUQadRA",
        "square_table": "1T4QLiCaJQjzsLANUR8jPssGsVZFChMog",
        "stool": "1IstEhReeRri2s2y7vJrcv1oQ3wUm4kqT",
    },
    "low": {
        "cabinet": "1zjMLlRlXVZDGri1QUINV540DTG7-jAa-",
        "chair": "1swulRnjB7rU1u-TuG6-WOrci9o8WZaEI",
        "desk": "1aJEqENTUCvnHhoAlwd9rks38YzkgeecL",
        "drawer": "121seXYws04z3UowpUdT-7uxg-y0l4Qb2",
        "lamp": "1kD9Fxj49Df4mgZPVkBa_b_L3dqzEQROF",
        "one_leg": "1E121w1Q9-SzFN3Bf6wC_NF-7kDA57RZf",
        "round_table": "1SjSg2tzQZ4fsN6z_xLT1vEOuTTzrQCun",
        "square_table": "1ogI5VkFcGeJsFje9_0AS_fFSwhJX6zQR",
        "stool": "1Z1ewa62pkWehC4biodoDdPfWDJdCNPtb",
    },
}

ALL_FURNITURE = [
    "lamp",
    "square_table",
    "desk",
    "round_table",
    "stool",
    "chair",
    "drawer",
    "cabinet",
    "one_leg",
]


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset-download CLI parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--randomness", help="Randomness of initial state", required=True
    )
    parser.add_argument(
        "--furniture",
        help="Name of the furniture. --all to download all furniture datasets.",
        required=True,
    )
    parser.add_argument("--out_dir", help="Path to output directory", required=True)
    parser.add_argument(
        "--use-rclone", action="store_true", help="Use rclone to download data"
    )
    parser.add_argument(
        "--untar", action="store_true", help="Untar the downloaded file"
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    return build_parser().parse_args(argv)


def download_folder_rclone(randomness: str, furniture: str, out_dir: Path) -> None:
    """Use rclone to copy a tarball from the configured Google Drive remote."""
    remote_path = f"dataset/{randomness}_compressed/{furniture}.tar.gz"
    target_path = out_dir / randomness / f"{furniture}.tar.gz"

    if target_path.exists():
        print(f"Skipped {furniture}: file already exists at {target_path}")
        return

    print(f"Start downloading Google Drive folder {remote_path}")
    (out_dir / randomness).mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["rclone", "copy", "-P", f"furniture:{remote_path}", str(out_dir / randomness)],
        check=True,
    )
    print(
        f"Finished downloading Google Drive folder {remote_path} to {out_dir / randomness}"
    )


def download_file_gdown(randomness: str, furniture: str, out_dir: Path) -> None:
    """Use gdown to fetch a single tarball from Google Drive."""
    import gdown

    if randomness not in DOWNLOAD_LINKS:
        raise ValueError(f"Unsupported randomness level: {randomness}")
    if furniture not in DOWNLOAD_LINKS[randomness]:
        raise ValueError(f"Unsupported furniture for {randomness}: {furniture}")

    target_dir = out_dir / randomness
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{furniture}.tar.gz"

    if target_path.exists():
        print(f"Skipped {furniture}: file already exists at {target_path}")
        return

    print(f"Start downloading file {randomness}/{furniture}")
    gdown.download(
        id=DOWNLOAD_LINKS[randomness][furniture],
        output=str(target_path),
        quiet=False,
    )


def maybe_untar(randomness: str, furniture: str, out_dir: Path) -> None:
    """Extract a downloaded tarball next to the source archive."""
    archive_path = out_dir / randomness / f"{furniture}.tar.gz"
    print(f"Untarring {furniture}")
    subprocess.run(
        [
            "tar",
            "-xvzf",
            str(archive_path),
            "-C",
            str(out_dir / randomness),
        ],
        check=True,
    )
    print(f"Finished untarring {furniture}")


def main(argv: Sequence[str] | None = None) -> None:
    """Download one or more dataset archives."""
    args = parse_args(argv)
    out_dir = Path(args.out_dir)

    download_list = ALL_FURNITURE if args.furniture == "all" else [args.furniture]

    for furniture in download_list:
        # Keep the two download backends behaviorally identical.
        if args.use_rclone:
            download_folder_rclone(args.randomness, furniture, out_dir)
        else:
            download_file_gdown(args.randomness, furniture, out_dir)

        if args.untar:
            maybe_untar(args.randomness, furniture, out_dir)


if __name__ == "__main__":
    main()
