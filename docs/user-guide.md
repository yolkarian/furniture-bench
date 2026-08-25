# User Guide

## 1. Requirements

Python requirement:
- Python `3.11` only

The project uses `gymnasium==0.29.1` and SAPIEN for physics simulation.

## 2. Installation

The recommended setup uses uv:

```bash
uv sync --locked
```

This creates a project-local `.venv` from `pyproject.toml` and `uv.lock`.
The locked simulator stack uses PyTorch `2.7.1+cu128` and CUDA `12.8` runtime
wheels through the `pytorch-cu128` uv index.

Run commands through uv:

```bash
uv run python -m furniture_bench.scripts.run_sim_env --help
```

Or activate the environment explicitly:

```bash
source .venv/bin/activate
```

## 3. Supported workflows

The maintained repository supports:
- simulator execution
- real-world and simulator data collection
- recorded-trajectory replay
- dataset download
- dataset preprocessing

The repository no longer bundles offline RL / IQL training code.

## 4. Supported observation families

The maintained observation families are:
- `state`
- `full`
- `image`

## 5. Running the simulator

Run a scripted one-leg episode:

```bash
python -m furniture_bench.scripts.run_sim_env --furniture one_leg --scripted
```

Run the same smoke test headlessly:

```bash
python -m furniture_bench.scripts.run_sim_env --furniture one_leg --scripted --headless
```

Run with a different task and multiple environments:

```bash
python -m furniture_bench.scripts.run_sim_env --furniture square_table --num-envs 4
```

Useful flags:
- `--scripted`: use the built-in scripted assembly policy
- `--headless`: disable the interactive viewer
- `--num-envs`: choose vectorized simulation size
- `--record`: record simulator video
- `--high-res`: use higher-resolution camera images
- `--random-action`: step using random sampled actions
- `--no-action`: keep stepping with neutral actions

Important notes:
- `GPUMemoryConfig` scales from `num_envs`.
- The maintained controller path is DiffIK.
- Some older CLIs still accept `osc`; it now acts only as a compatibility alias to DiffIK.
- Long-lived launchers that run several jobs in one Python process must call
  `env.close()` after each job and then `sapien.shutdown()`. Furniture part reset
  poses are copied away from immutable task config, so reset randomization cannot
  contaminate spawn centers used by later jobs.
- If you are changing the SAPIEN simulator internals, read
  [Avoiding steady CPU memory growth in SAPIEN environments](sapien-memory.md)
  before modifying reset or controller hot paths.

## 6. Collecting demonstrations

### 6.1 SpaceMouse workflow

Collect simulator demonstrations with image observations:

```bash
python -m furniture_bench.scripts.collect_data_sm \
  --out-data-path data/demos \
  --furniture one_leg \
  --is-sim \
  --obs-type image
```

Collect state-only simulator demonstrations:

```bash
python -m furniture_bench.scripts.collect_data_sm \
  --out-data-path data/demos \
  --furniture one_leg \
  --is-sim \
  --obs-type state
```

Important flags:
- `--obs-type {state,full,image}`
- `--gpu-id`
- `--num-demos`
- `--pkl-only`
- `--save-failure`
- `--headless`

### 6.2 Classic keyboard / Oculus workflow

```bash
python -m furniture_bench.scripts.collect_data \
  --out-data-path data/demos \
  --furniture one_leg \
  --is-sim
```

Important flags:
- `--input-device {keyboard,oculus,keyboard-oculus}`
- `--scripted`
- `--compute-device-id`
- `--graphics-device-id`
- `--resize-sim-img`
- `--num-demos`

## 7. Replaying recorded trajectories

Replay a previously saved simulator record:

```bash
python scripts/replay.py --task one_leg --record-path /path/to/record.safetensors
```

Save regenerated rendered images:

```bash
python scripts/replay.py \
  --task one_leg \
  --record-path /path/to/record.safetensors \
  --save-output
```

This script is useful for:
- visual inspection
- replay debugging
- regenerating rendered camera observations from recorded state

## 8. Dataset utilities

### 8.1 Download datasets

Download a single dataset tarball:

```bash
python -m furniture_bench.scripts.download_dataset \
  --randomness low \
  --furniture one_leg \
  --out_dir data
```

Download and untar it immediately:

```bash
python -m furniture_bench.scripts.download_dataset \
  --randomness low \
  --furniture one_leg \
  --out_dir data \
  --untar
```

### 8.2 Preprocess collected pickles

```bash
python -m furniture_bench.scripts.preprocess_data \
  --in-data-path data/raw \
  --out-data-path data/processed
```

Common options:
- `--success-only`
- `--save-last-step`
- `--no-robot-state`
- `--use-all-cam`
- `--done-when-assembled`
- `--norm-pos-acts`

## 9. Shell helpers

The maintained top-level helper entry points live under `scripts/`:
- `scripts/replay.py`
- `scripts/import_furniture_bench.py`
- `scripts/import_franka_sapien.py`
- `scripts/launch_sim_gymasium.py`

Container helpers live under `docker/`:
- `docker/build.sh`
- `docker/run.sh`

## 10. Removed workflows

The following bundled components were intentionally removed:
- offline RL / IQL training and evaluation code
- vendored wheel artifacts under `wheels/`
- the unused top-level `config/` directory
- legacy controller implementations other than DiffIK

If you need those workflows again, add them back as separate optional integrations rather than as bundled repository dependencies.
