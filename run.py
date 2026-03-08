"""Legacy compatibility entry point for the refactored repository."""

from __future__ import annotations

SUPPORTED_WORKFLOWS = """
The historical `run.py` entry point depended on training stacks that are no longer
bundled with this repository.

Use one of the supported workflows instead:
- `python -m furniture_bench.scripts.run_sim_env ...`
- `python -m furniture_bench.scripts.collect_data ...`
- `python -m furniture_bench.scripts.collect_data_sm ...`
- `python scripts/replay.py ...`
- `bash scripts/install_model_deps.sh`

See `docs/README.md` for the updated documentation.
""".strip()


def main() -> int:
    """Explain the current entry points after the repository cleanup."""
    print(SUPPORTED_WORKFLOWS)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
