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

### 2.1 Shell scripts moved into `scripts/`

The substantive shell scripts now live under `scripts/`:
- `scripts/install_model_deps.sh`
- `scripts/entrypoint.sh`
- `scripts/launch_client.sh`
- `scripts/launch_server.sh`
- `scripts/launch_daemon.sh`

The old duplicated shell files were removed from the repository root.

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
- offline RL code: keep it in a separate repository or optional integration
- vendored wheel installs: install `dt-apriltags` from package management during environment setup
- historical shell helpers: call the maintained files under `scripts/`
