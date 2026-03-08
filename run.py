"""Legacy compatibility entry point for the refactored repository."""

from __future__ import annotations

SUPPORTED_WORKFLOWS = """
The historical `run.py` entry point depended on the removed `rolf` package.

Use one of the supported workflows instead:
- `python -m furniture_bench.scripts.run_sim_env ...`
- `python -m furniture_bench.scripts.collect_data_sm ...`
- `python implicit_q_learning/train_offline.py ...`
- `python implicit_q_learning/test_offline.py ...`

See `docs/README.md` for the full English documentation.
""".strip()


def main() -> int:
    """Explain the new entry points after the legacy training stack removal."""
    print(SUPPORTED_WORKFLOWS)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
