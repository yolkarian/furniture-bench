# Developer Guide

## 1. Refactor goals

This refactor was designed to make the project smaller, clearer, and easier to maintain.

The core changes are:
- keep the project centered on `furniture_bench`
- remove bundled visual-encoder stacks (`r3m`, `vip`) and the legacy `rolf` training package
- remove simulator-side encoder environments
- document the supported workflows in English
- improve type annotations and comments in the touched Python entry points

## 2. Package layout

The main package areas are:
- `furniture_bench/envs`: simulator and real-world environments
- `furniture_bench/data`: data-collection helpers
- `furniture_bench/scripts`: supported command-line entry points under the package
- `furniture_bench/utils`: shared utilities

The refactor intentionally did not modify:
- `SAPIEN/`
- `ManiSkill/`

These folders are present to help developers inspect API usage and implementation patterns while working on `furniture_bench`.

## 3. Dynamic GPU memory sizing

`GPUMemoryConfig` is now dynamic.

Implementation details:
- base capacities remain defined in `furniture_bench/sim_config.py`
- the config now exposes `scale_for_envs(num_envs)`
- each capacity is scaled linearly with `num_envs`
- scaled capacities are rounded up to the next power of two
- `FurnitureSimRLEnv` creates a per-instance simulator config instead of mutating the module-level config in place

This design avoids two previous problems:
- fixed GPU allocations that were too small for larger vectorized runs
- accidental cross-run state leakage caused by mutating a shared global simulator config

## 4. Observation policy after the refactor

Supported observation families:
- `state`
- `full`
- `image`

Unsupported observation family:
- `feature`

This means:
- no bundled encoder-backed Gym registrations
- no simulator-side feature extraction environments
- no data-collection path that serializes encoder outputs as the canonical format

## 5. Script policy

The project now treats scripts in three groups:
- supported: simulator, replay, data collection, offline IQL train/eval
- legacy but documented: `run.py` now explains that the old `rolf` entry point was removed
- removed: encoder-only and fine-tuning scripts that depended on the deleted legacy stack

## 6. Typing and comments

For touched Python files, the refactor follows these rules:
- add explicit return types for entry points and utility helpers where practical
- use narrow literal types for public script options when the supported values are fixed
- place short comments above important behavior blocks rather than relying on historical context

## 7. Extending the project safely

When adding new functionality:
- keep new integrations optional and modular
- avoid mutating module-level simulator config objects in environment constructors
- prefer raw observations over bundled pretrained encoders unless there is a strong project-level reason
- keep docs in `docs/` updated at the same time as code changes
