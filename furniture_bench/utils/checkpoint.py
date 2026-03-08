"""Checkpoint path helpers for local evaluation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def download_ckpt_if_not_exists(
    ckpt_dir: str, run_name: str, seed: Optional[int] = None
) -> Path:
    """Resolve a local checkpoint directory.

    The previous implementation downloaded encoder-specific checkpoints from
    Google Drive. That behavior was removed together with the bundled encoder
    stack, so evaluation scripts now expect the checkpoint directory to exist
    locally.
    """
    run_suffix = f"{run_name}.{seed}" if seed is not None else run_name
    checkpoint_path = Path(ckpt_dir) / run_suffix
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Checkpoint directory not found: "
            f"{checkpoint_path}. Place a trained checkpoint there before evaluation."
        )
    return checkpoint_path
