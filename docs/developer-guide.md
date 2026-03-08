# Developer Guide

## 1. Cleanup goals

This repository cleanup was designed to make the project smaller, clearer, and easier to maintain.

The main changes were:
- keep the repository centered on `furniture_bench`
- remove the bundled offline RL / IQL stack
- remove vendored wheel artifacts and the unused top-level `config/` directory
- keep only the DiffIK controller path
- move substantive shell scripts into `scripts/`
- improve formatting, typing, and inline comments in maintained entry points

## 2. Project boundaries

The maintained project surface is:
- `furniture_bench/`
- `scripts/`
- `docs/`

The vendored trees are still present mainly as local references:
- `ManiSkill/`
- `SAPIEN/`

Those reference trees are not the primary editable target of this repository.

## 3. Package layout

The main package areas are:
- `furniture_bench/envs`: simulator and real-world environments
- `furniture_bench/data`: data-collection helpers
- `furniture_bench/device`: keyboard, Oculus, and SpaceMouse-related interfaces
- `furniture_bench/furniture`: task definitions and part logic
- `furniture_bench/robot`: real-robot helpers
- `furniture_bench/scripts`: maintained Python CLI entry points
- `furniture_bench/utils`: shared utilities, including control math

## 4. Controller policy after cleanup

Only the DiffIK controller path remains maintained.

Compatibility details:
- the simulator normalizes `osc` flags to DiffIK so old script usage does not immediately break
- `Panda` now initializes the DiffIK controller family on the real-robot side as well
- control math helpers moved from `furniture_bench/controllers/control_utils.py` to `furniture_bench/utils/control.py`

Practical implication:
- new work should target DiffIK directly
- `osc` should be treated as a transitional compatibility flag, not as an active controller implementation

## 5. Script policy

The repository now uses two maintained script layers:
- maintained Python entry points under `furniture_bench/scripts/`
- maintained top-level helpers under `scripts/`

The maintained Python scripts were updated to:
- parse arguments before importing heavy simulator or hardware dependencies
- keep public CLI flags stable where practical
- add explicit type annotations to entry points and helpers
- include inline comments around behavior-critical logic

## 6. Formatting and typing expectations

Current expectations for maintained code:
- use consistent formatting
- prefer explicit type annotations on public functions and entry points
- document non-obvious behavior inline near the logic it explains
- preserve CLI compatibility when reorganizing script internals

## 7. Dynamic GPU memory sizing

`GPUMemoryConfig` remains dynamic.

Implementation details:
- base capacities remain defined in `furniture_bench/sim_config.py`
- capacities scale from `num_envs`
- scaled capacities are rounded up to the next power of two
- simulator config is instantiated per environment rather than patched globally in-place

## 8. Verification approach

After cleanup, the maintained scripts are verified mainly by:
- Python syntax compilation
- `--help` checks for maintained Python CLIs
- shell syntax checks for maintained bash entry points
- targeted runtime smoke tests for important simulator paths

This does not replace full hardware or end-to-end integration testing, but it is the baseline validation used during repository cleanup.
