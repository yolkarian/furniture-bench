# FurnitureBench (Refactored)

This repository contains the refactored SAPIEN-based `furniture_bench` package and the scripts that are still supported by the current project layout.

## Scope of this refactor

The repository now focuses on:
- the `furniture_bench` Python package
- simulator and data-collection scripts that use `furniture_bench`
- the offline IQL workflow under `implicit_q_learning`
- Project documentation in `docs/`

The following legacy components were removed from the supported workflow:
- bundled `r3m` and `vip` feature encoders
- the bundled `rolf` training stack
- simulator-side image-feature environments and encoder-specific data paths

## Supported workflows

1. Install the package and offline-learning extras:

```bash
pip install -e .
bash install_model_deps.sh
```

2. Run a simulator environment:

```bash
python -m furniture_bench.scripts.run_sim_env --furniture one_leg --num-envs 4
```

3. Collect demonstrations with a SpaceMouse:

```bash
python -m furniture_bench.scripts.collect_data_sm \
  --out-data-path data/demos \
  --furniture one_leg \
  --is-sim \
  --obs-type image
```

4. Replay a recorded trajectory:

```bash
python scripts/replay.py --task one_leg --record-path /path/to/record.safetensors
```

5. Train or evaluate the offline IQL pipeline:

```bash
python implicit_q_learning/train_offline.py --env_name FurnitureSimState-v0/one_leg --data_path data/Image/one_leg.pkl
python implicit_q_learning/test_offline.py --env_name FurnitureSimState-v0/one_leg --run_name local_run --save_dir checkpoints
```

## Documentation

- [Project docs](docs/README.md)
- [User guide](docs/user-guide.md)
- [Developer guide](docs/developer-guide.md)
- [Migration notes](docs/migration.md)

## Notes

- `SAPIEN/` and `ManiSkill/` were kept intact and can still be used as local references while developing against this project.
- `GPUMemoryConfig` now scales from `num_envs`, so multi-environment simulator runs no longer rely on a fixed GPU buffer size.
- All new documentation added by this refactor is written in English.
