# FurnitureBench

A cleaned-up, SAPIEN-based version of `furniture_bench` focused on the maintained simulator, data-collection, replay, and dataset-preparation workflows.

## Repository scope

This repository now centers on:
- the `furniture_bench` Python package
- maintained Python entry points under `furniture_bench/scripts/`
- maintained top-level helpers under `scripts/`
- project documentation under `docs/`

The repository no longer bundles:
- the old offline RL / IQL stack
- vendored wheel artifacts
- the unused top-level `config/` directory
- legacy controller implementations other than DiffIK

## Quick start

### 1. Install

Python requirement:
- Python `>=3.9,<3.11`

Install the package directly:

```bash
pip install -e .
```

Or use the compatibility helper:

```bash
bash scripts/install_model_deps.sh
```

### 2. Run the simulator

Run a scripted one-leg assembly episode:

```bash
python -m furniture_bench.scripts.run_sim_env --furniture one_leg --scripted
```

Run a headless smoke test:

```bash
python -m furniture_bench.scripts.run_sim_env --furniture one_leg --scripted --headless
```

### 3. Collect demonstrations

SpaceMouse-based collection in simulation:

```bash
python -m furniture_bench.scripts.collect_data_sm \
  --out-data-path data/demos \
  --furniture one_leg \
  --is-sim \
  --obs-type image
```

Classic keyboard / Oculus collection:

```bash
python -m furniture_bench.scripts.collect_data \
  --out-data-path data/demos \
  --furniture one_leg \
  --is-sim
```

### 4. Replay a recorded trajectory

```bash
python scripts/replay.py --task one_leg --record-path /path/to/record.safetensors
```

### 5. Dataset utilities

Download a dataset archive:

```bash
python -m furniture_bench.scripts.download_dataset \
  --randomness low \
  --furniture one_leg \
  --out_dir data
```

Preprocess collected pickles:

```bash
python -m furniture_bench.scripts.preprocess_data \
  --in-data-path data/raw \
  --out-data-path data/processed
```

## Main behavior changes after cleanup

- DiffIK is now the only maintained controller path.
- Some CLI flags still accept `osc` as a **compatibility alias** and internally map it to DiffIK.
- Top-level shell scripts were moved into `scripts/`.
- The repository root no longer contains duplicated shell entry-point files.
- `SAPIEN/` and `ManiSkill/` are still present as local reference trees, but they are not the primary editable target of this repository.

## Documentation

- [Documentation index](docs/README.md)
- [User guide](docs/user-guide.md)
- [Development rules](AGENTS.md)
- [Migration notes](docs/migration.md)
- [Repository structure](STRUCTURE.md)

## Notes

- The maintained docs in this repository describe the project-owned code and workflows. Vendored third-party trees keep their own upstream documentation.
- Use the shell helpers directly from `scripts/`.
