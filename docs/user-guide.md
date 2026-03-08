# User Guide

## 1. Installation

Install the core package:

```bash
pip install -e .
```

Install the remaining offline-learning extras:

```bash
bash install_model_deps.sh
```

## 2. What is supported

The refactored project supports three observation families:
- `state`
- `full`
- `image`

The previous encoder-backed `feature` observation path is no longer part of the supported workflow.

## 3. Running the simulator

Run the built-in simulator script:

```bash
python -m furniture_bench.scripts.run_sim_env --furniture one_leg --num-envs 4
```

Important notes:
- `GPUMemoryConfig` now scales from `num_envs`.
- Multi-environment runs do not rely on a fixed GPU buffer allocation anymore.
- `SAPIEN` and `ManiSkill` remain in the repository, but they are reference codebases rather than editable targets of this refactor.

## 4. Collecting demonstrations

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

The collector now accepts only the supported observation modes:
- `state`
- `full`
- `image`

## 5. Replaying recorded trajectories

Replay a previously saved simulator record:

```bash
python scripts/replay.py --task one_leg --record-path /path/to/record.safetensors
```

This script is useful for visual inspection and regenerating rendered camera observations.

## 6. Offline IQL training

Train on a local dataset:

```bash
python implicit_q_learning/train_offline.py \
  --env_name FurnitureSimState-v0/one_leg \
  --data_path data/Image/one_leg.pkl \
  --run_name one_leg_state_iql
```

Evaluate a local checkpoint:

```bash
python implicit_q_learning/test_offline.py \
  --env_name FurnitureSimState-v0/one_leg \
  --save_dir checkpoints \
  --run_name one_leg_state_iql
```

Checkpoint behavior changed during the refactor:
- the repository no longer downloads pretrained checkpoints automatically
- evaluation expects the checkpoint directory to already exist locally

## 7. Removed workflows

The following workflows were intentionally removed:
- bundled `r3m` feature extraction
- bundled `vip` feature extraction
- bundled `rolf` training and runtime entrypoints
- simulator-side image-feature environments

If you need those workflows again, reintroduce them as separate optional integrations rather than as bundled repository dependencies.
