# Development Rules

## First Message
If the user's first message is not concrete, read `README.md` and `docs/README.md`, then ask which area to work on.

Based on the answer, read the relevant files in parallel:
- simulator / environment changes: `docs/user-guide.md`, `furniture_bench/envs/`, `furniture_bench/sim_config.py`, `furniture_bench/config.py`
- data pipeline / trajectories: `furniture_bench/data/`, `furniture_bench/scripts/preprocess_data.py`, `furniture_bench/scripts/show_trajectory.py`, `furniture_bench/scripts/tests/test_skill_preprocess.py`, `scripts/replay.py`
- teleop / hardware / cameras: `furniture_bench/device/`, `furniture_bench/robot/`, `furniture_bench/scripts/calibration.py`, `furniture_bench/scripts/reset.py`, `furniture_bench/scripts/run_cam_april.py`, `furniture_bench/scripts/move_up.py`
- furniture/task definitions or asset wiring: `furniture_bench/furniture/`, `furniture_bench/config.py`, then the smallest relevant slice under `furniture_bench/assets/` or `furniture_bench/assets_no_tags/`
- Docker or launch flows: `docker/`, `scripts/entrypoint.sh`, `scripts/launch_client.sh`, `scripts/launch_server.sh`, `scripts/launch_daemon.sh`

If the request is already concrete, skip generic discovery and go straight to the relevant module and docs.

## Project Intent
- This repository was cleaned up to stay centered on `furniture_bench`, the maintained script layers, and project-owned docs.
- Do not reintroduce removed bundled workflows such as the old offline RL / IQL stack, vendored wheel artifacts, or the old top-level `config/` directory unless the user explicitly asks.
- Keep substantive shell helpers under `scripts/`; do not recreate root-level duplicates as a convenience shortcut.

## Repo Boundaries
- Primary editable surface: `furniture_bench/`, `scripts/`, `docs/`, and `docker/` when the task requires it.
- Core package map:
  `furniture_bench/envs` for simulator and real-world environments,
  `furniture_bench/data` for data collection and trajectory handling,
  `furniture_bench/device` for keyboard / Oculus / SpaceMouse interfaces,
  `furniture_bench/furniture` for task definitions and part logic,
  `furniture_bench/robot` for robot helpers,
  `furniture_bench/scripts` for maintained Python CLIs,
  `furniture_bench/utils` for shared math and support code.
- Prefer maintained CLIs under `furniture_bench/scripts/` and `scripts/`.
- If local `ManiSkill/` or `SAPIEN/` trees are present, treat them as reference-only unless the user explicitly asks for dependency-management work. Maintained code should keep targeting the installed `sapien` package rather than a vendored source tree.
- `pyproject.toml` is the source of truth for Python compatibility: `>=3.9,<3.11`.
- Repomix output is dominated by huge mesh assets. Avoid sweeping searches over `furniture_bench/assets/`, `furniture_bench/assets_no_tags/`, or `repomix-output.xml` unless the task explicitly targets them.
- `assets/` and `assets_no_tags/` overlap but are not identical. Never bulk-sync, dedupe, or rename across them without an explicit migration plan.
- DiffIK is the only maintained controller path. `osc` is a compatibility alias and should keep mapping to DiffIK unless the user explicitly asks for a breaking CLI change.
- New controller work should target DiffIK directly in both simulator and real-robot paths. Prefer `furniture_bench/utils/control.py` for shared control math instead of reviving removed controller helpers.
- The maintained smoke tests live under `furniture_bench/scripts/tests/`. ROS-style tests under `furniture_bench/assets*/franka_description_ros/...` are asset-specific and usually not the right validation target for normal repo work.

## Code Quality
- Preserve lightweight `--help` behavior in CLI scripts: parse arguments before importing simulator, camera, or robot dependencies.
- Keep public CLI flags and on-disk data layout stable where practical.
- Prefer explicit type annotations on public helpers and entry points.
- Use consistent formatting and document non-obvious behavior inline near the logic it explains.
- Keep inline comments only for behavior-critical logic or compatibility details.
- Preserve existing CLI flags, data layout, and replay / preprocessing conventions unless the user approves a breaking change.
- When touching dataset code, keep observation shapes, channel ordering, robot-state filtering, reward/skill slicing, and action normalization semantics consistent with existing scripts and tests.
- When touching config or task definitions, remember many behaviors are keyed from `furniture_bench/config.py`; update the code and referenced asset paths together.
- When changing simulation scaling or memory capacity logic, keep `GPUMemoryConfig` dynamic: define base capacities in `furniture_bench/sim_config.py`, scale from `num_envs`, round up to the next power of two, and instantiate config per environment instead of patching a shared global in place.

## Commands
- Setup: `pip install -e .`
- Compatibility setup: `bash scripts/install_model_deps.sh`
- Cheap syntax check after Python changes: `python -m compileall furniture_bench scripts`
- Shell syntax check after bash script changes: `bash -n scripts/entrypoint.sh scripts/install_model_deps.sh scripts/launch_client.sh scripts/launch_daemon.sh scripts/launch_server.sh`
- Maintained smoke tests: `python -m unittest furniture_bench.scripts.tests.test_skill_preprocess`
- CLI checks for touched maintained entry points: `python -m furniture_bench.scripts.run_sim_env --help`, `python -m furniture_bench.scripts.collect_data --help`, `python -m furniture_bench.scripts.collect_data_sm --help`, `python -m furniture_bench.scripts.download_dataset --help`, `python -m furniture_bench.scripts.preprocess_data --help`, `python -m furniture_bench.scripts.show_trajectory --help`, `python scripts/replay.py --help`
- Only run a headless simulator smoke test when the change actually needs runtime coverage and the environment has the required simulator dependencies: `python -m furniture_bench.scripts.run_sim_env --furniture one_leg --scripted --headless`
- Prefer the maintained smoke test over real dataset downloads; `download_dataset.py` shells out to `gdown`, `rclone`, and `tar`.
- This verification baseline does not replace hardware or end-to-end integration testing.

## Do Not Run By Default
- Do not run interactive hardware or camera flows unless the user explicitly asks and the required devices are available: `furniture_bench.scripts.calibration`, `furniture_bench.scripts.run_cam_april`, `furniture_bench.scripts.reset`, `furniture_bench.scripts.move_up`, real-world collection modes
- Do not start Docker launch scripts or tmux hardware sessions unless requested: `scripts/launch_client.sh`, `scripts/launch_server.sh`, `scripts/launch_daemon.sh`
- Do not edit giant mesh, image, or binary assets (`.obj`, `.dae`, `.stl`, `.usd`, large `.png` / `.jpg`) unless the task is explicitly about assets
