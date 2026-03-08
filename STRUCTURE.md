# Repository Structure

This document summarizes the **current project-owned repository layout** after cleanup.

## Status labels

- **Core**: part of the maintained workflow
- **Supporting**: useful helper or operational tooling
- **Reference-only**: vendored upstream code kept mainly for local reference

---

## Top-level overview

| Path | Approx. size | Status | What it contains |
|---|---:|---|---|
| `furniture_bench/` | 211M | Core | Main Python package: environments, task definitions, devices, scripts, utilities, and assets |
| `docs/` | 20K | Core | Maintained repository documentation |
| `scripts/` | 32K | Core | Top-level helper scripts: replay plus moved shell entry points |
| `docker/` | 36K | Supporting | Dockerfiles and NVIDIA container config |
| `demo/` | 20K | Supporting | Small import / rendering / sanity-check scripts |
| `ManiSkill/` | 1.3G | Reference-only | Vendored ManiSkill tree kept as a local reference |
| `SAPIEN/` | 863M | Reference-only | Vendored SAPIEN tree kept as a local reference |

Removed during cleanup:
- `implicit_q_learning/`
- `wheels/`
- `config/`
- generated `furniture_bench.egg-info/`
- controller files other than DiffIK

---

## Root files

| File | Role |
|---|---|
| `README.md` | Main project overview and quick-start guide |
| `pyproject.toml` | Python packaging metadata and dependencies |
| `environment.yml` | Conda environment specification |
| `MANIFEST.in` | Package-data inclusion rules |
| `run.py` | Legacy compatibility entry point; now prints current guidance |
| `STRUCTURE.md` | This repository structure summary |
| `TODO.md` | Small backlog / maintenance notes |
| `mypy.ini` | Type-checker config |
| `LICENSE` | Project license |
| `furniture_bench_banner.jpg` | Repository image asset |

---

## 1. `furniture_bench/` — main package

This is the primary maintained codebase.

### Main subdirectories

| Path | Purpose |
|---|---|
| `furniture_bench/envs/` | Gymnasium/SAPIEN environments |
| `furniture_bench/furniture/` | Furniture task classes, part logic, and scanned-task definitions |
| `furniture_bench/controllers/` | Controller package reduced to DiffIK only |
| `furniture_bench/robot/` | Panda robot helpers and robot-state definitions |
| `furniture_bench/perception/` | AprilTag and RealSense utilities |
| `furniture_bench/device/` | Input-device interfaces |
| `furniture_bench/data/` | Data collection helpers |
| `furniture_bench/scripts/` | Maintained Python CLI scripts |
| `furniture_bench/utils/` | Shared utilities, including control math |
| `furniture_bench/assets/` | Tagged furniture assets, calibration data, Franka description |
| `furniture_bench/assets_no_tags/` | Alternate asset tree used by the simulator |

### Important functional areas inside `furniture_bench/`

#### `envs/`
Main environment implementations and observation definitions.

Key files:
- `furniture_bench/envs/furniture_sim_env.py`
- `furniture_bench/envs/furniture_rl_sim_env.py`
- `furniture_bench/envs/furniture_bench_env.py`
- `furniture_bench/envs/observation.py`

#### `furniture/`
Task definitions for each assembly or scanned task, plus reusable part classes.

Main groups:
- classic furniture tasks: `chair.py`, `desk.py`, `drawer.py`, `lamp.py`, `one_leg.py`, `round_table.py`, `square_table.py`, `stool.py`, `cabinet.py`
- reusable part classes: `furniture_bench/furniture/parts/`
- scanned / factory tasks: `furniture_bench/furniture/scans/`

#### `controllers/`
Only the DiffIK controller remains:
- `furniture_bench/controllers/diffik.py`

Shared control math lives in:
- `furniture_bench/utils/control.py`

#### `scripts/`
Maintained package-level runnable scripts.

Main entry points:
- `run_sim_env.py`
- `collect_data.py`
- `collect_data_sm.py`
- `preprocess_data.py`
- `download_dataset.py`
- `show_trajectory.py`
- `run_cam_april.py`
- `calibration.py`
- `reset.py`
- `move_up.py`

---

## 2. `scripts/` — top-level helpers

This directory now contains the substantive top-level helper scripts.

Files:
- `scripts/replay.py`
- `scripts/install_model_deps.sh`
- `scripts/entrypoint.sh`
- `scripts/launch_client.sh`
- `scripts/launch_server.sh`
- `scripts/launch_daemon.sh`

Use these helper scripts directly from `scripts/`.

---

## 3. `docs/` — maintained documentation

Files:
- `docs/README.md`
- `docs/user-guide.md`
- `docs/developer-guide.md`
- `docs/migration.md`

These files describe the maintained repository surface, not the vendored third-party trees.

---

## 4. `docker/` — container tooling

Contains:
- `client.Dockerfile`
- `client_gpu.Dockerfile`
- `server.Dockerfile`
- `sapien.Dockerfile`
- NVIDIA JSON config files

Notable cleanup:
- Dockerfiles no longer depend on the removed `wheels/` directory
- the Docker entrypoint now comes from `scripts/entrypoint.sh`

---

## 5. `demo/` — small experiments

Contains local developer/demo scripts rather than main documented entry points:
- `import_furniture_bench.py`
- `import_franka_sapien.py`
- `launch_sim_gymasium.py`
- `parallel_rendering_test.py`

These are supporting developer aids, not part of the core maintained workflow.

---

## 6. Vendored reference trees

### `ManiSkill/`
A full, large local copy of ManiSkill kept as a reference codebase.

### `SAPIEN/`
A full, large local copy of SAPIEN kept as a reference codebase.

These two directories still dominate repository size, but they are not the primary editable target of this project.

---

## 7. Main structural observations

1. **Repository size is still dominated by vendored references.**
   - `ManiSkill/`: ~1.3G
   - `SAPIEN/`: ~863M

2. **The active maintained code surface is much smaller.**
   - core workflow: `furniture_bench/`, `scripts/`, `docs/`
   - controller surface: DiffIK only

3. **Shell entry points were consolidated into `scripts/`.**
   - real files now live under `scripts/`
   - the old duplicated root-level shell files were removed

4. **Generated and clearly unnecessary tracked artifacts were removed.**
   - no `furniture_bench.egg-info/`
   - no checked-in `__pycache__/` under `furniture_bench/`

---

## Suggested next step

If cleanup continues, the biggest remaining decision points are:
- whether `ManiSkill/` should remain vendored
- whether `SAPIEN/` should remain vendored
- whether `demo/` and parts of `docker/` are still worth carrying
- whether `assets/` and `assets_no_tags/` can be consolidated
