# Migration Notes

## 1. Removed directories and packages

The following bundled directories were removed:
- `implicit_q_learning/`
- `wheels/`
- `config/`

The following controller modules were removed:
- `furniture_bench/controllers/osc.py`
- `furniture_bench/controllers/diffik_qp.py`
- `furniture_bench/controllers/control_utils.py`

## 2. Structural changes

### 2.1 Environment and helper cleanup

The project environment is now managed by uv:
- dependency declarations live in `pyproject.toml`
- reproducibility lives in `uv.lock`
- the project runtime lives in `.venv/`

Maintained container helpers live under `docker/`:
- `docker/build.sh`
- `docker/run.sh`

The old duplicated shell files and conda environment manifest were removed from the repository.

### 2.2 Controller cleanup

After cleanup:
- DiffIK is the only maintained controller path
- simulator code normalizes `osc` to DiffIK as a compatibility alias
- shared control math lives in `furniture_bench/utils/control.py`

### 2.3 Script modernization

Maintained Python entry points were updated to:
- parse arguments before importing heavy runtime dependencies
- add explicit type annotations on public helpers and entry points
- include inline comments for behavior-critical logic
- preserve existing CLI surfaces where practical

## 3. Behavior changes

### 3.1 Offline-learning workflow

Before cleanup, the repository bundled an offline IQL stack.

After cleanup:
- the repository no longer ships offline RL training or evaluation code
- dataset download and preprocessing utilities remain
- external training code should now live outside this repository

### 3.2 Controller flags

Before cleanup, several scripts exposed `osc` and DiffIK as peer controller choices.

After cleanup:
- DiffIK is the active implementation
- `osc` may still appear in some CLIs, but only as a compatibility alias

### 3.3 Root shell entry points

Before cleanup, substantive shell scripts lived directly at the repository root.

After cleanup:
- the maintained script files live under `scripts/`
- the old root-level copies were removed

## 4. Recommended replacements

Use these replacements for removed workflows:
- environment setup: `uv sync --locked`
- running project commands: `uv run <command>`
- containers: `bash docker/build.sh` and `bash docker/run.sh`
- offline RL code: keep it in a separate repository or optional integration
- vendored wheel installs: declare package dependencies in `pyproject.toml` and let uv resolve them
