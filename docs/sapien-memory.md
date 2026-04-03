# Avoiding Steady CPU Memory Growth in SAPIEN Environments

This note summarizes a real issue that was found and fixed in FurnitureBench's
SAPIEN simulator environment. The goal is practical: if you build or modify a
GPU-backed SAPIEN environment, avoid patterns that make CPU RSS (resident set
size, meaning the process memory currently resident in RAM) climb a little bit
after every episode or reset.

The examples below come from the maintained simulator code in
[`furniture_bench/envs/furniture_sim_env.py`](../furniture_bench/envs/furniture_sim_env.py)
and the DiffIK controller in
[`furniture_bench/controllers/diffik.py`](../furniture_bench/controllers/diffik.py).

## Symptom

The failure mode looked like this:

- an episode finished
- `reset()` was called
- GPU memory stayed flat
- Python heap stayed mostly flat
- CPU RSS still increased by a small amount
- after many episodes, the process looked like it had a memory leak

This is easy to misread because many users check system memory right after a
reset. In practice, the growth can come from either:

- work performed during the previous episode, or
- allocations inside the reset hot path itself

You need multi-episode measurements to tell the difference.

## What Happened In FurnitureBench

In the original SAPIEN path, the steady CPU growth came from several small
allocation sources that compounded over time.

### 1. Repeated `.torch()` wrapper creation on SAPIEN CUDA buffers

Calling `cuda_*_buffer.torch()` repeatedly in `step()` and `reset()` creates
new Torch wrapper/view objects. Even though the underlying simulation state is
GPU-resident, wrapper churn on the host side can still make CPU RSS drift.

The fix was to cache the wrappers once after `gpu_init()` and reuse them:

- cache rigid-body state, force, and torque views
- cache articulation `qpos`, `qvel`, `qf`, and target views
- release those cached views in `close()`

### 2. Tiny temporary tensors inside per-env reset loops

In `_reset_franka()`, repeated patterns such as `torch.zeros_like(slice)` were
executed for each environment during every reset. The individual tensors were
small, but at high `num_envs` they created a stable CPU-side RSS staircase.

The fix was to switch to direct in-place zeroing on cached views:

- write `= 0` into the target slices
- avoid constructing temporary tensors just to clear data

### 3. Per-part object churn in `_reset_parts()`

The original reset path rebuilt many short-lived objects for every part in every
environment:

- `sapien.Pose`
- numpy homogeneous matrices
- small Torch tensors for positions and quaternions

That pattern is especially expensive in vectorized environments.

The fix was to batch the reset path:

- convert AprilTag coordinates to simulator coordinates with cached transforms
- assemble part positions and quaternions into small numpy arrays per env
- perform one indexed write for all parts in the env
- do the same for obstacle positions

### 4. Controller hot-path allocation churn

The reset issue was the user-visible symptom, but step-side warm-up also came
from repeated allocation in the controller path:

- cloned goal tensors every step
- fresh Jacobian buffers every step
- an unnecessary matrix inverse in DiffIK

The fix was to:

- reuse controller-side buffers for goals, Jacobians, and target joint tensors
- update goals in-place when shapes already match
- replace `inv(R)` with `R.transpose(-1, -2)` for rotation matrices

## Patterns To Avoid

When writing a SAPIEN environment, avoid these patterns in `step()` and
`reset()`:

- calling `.torch()` on the same SAPIEN CUDA buffer in inner loops
- creating many tiny tensors inside per-env or per-part loops
- using `torch.zeros_like(slice)` when direct in-place zeroing is enough
- rebuilding `sapien.Pose` objects in the hot path when raw batched writes are possible
- repeatedly cloning controller goal tensors instead of updating in-place
- treating every CPU RSS increase as a Python leak without checking native-side churn

## Recommended Pattern

Use this design instead:

1. After `gpu_init()`, cache every frequently used SAPIEN CUDA buffer view.
2. Reuse those cached views everywhere in step/reset code.
3. Preallocate hot-path work buffers once in `__init__`.
4. Batch state writes by environment whenever possible.
5. Prefer in-place updates over constructing replacement tensors.
6. Clear cached views in `close()` if the simulator is being torn down.

In FurnitureBench, the relevant examples are:

- cached CUDA views in
  [`furniture_bench/envs/furniture_sim_env.py`](../furniture_bench/envs/furniture_sim_env.py)
- batched part reset writes in
  [`furniture_bench/envs/furniture_sim_env.py`](../furniture_bench/envs/furniture_sim_env.py)
- in-place DiffIK goal updates in
  [`furniture_bench/controllers/diffik.py`](../furniture_bench/controllers/diffik.py)

## Profiling Advice

To diagnose this class of issue correctly:

- measure CPU RSS across many episodes, not just one reset
- separate `after_episode` from `after_reset`
- treat the first rollout as warm-up; it often includes one-time allocations
- measure GPU memory separately with `nvidia-smi`
- do not rely only on `torch.cuda.memory_reserved()`, because SAPIEN GPU usage
  may not appear there
- use `tracemalloc` only as a Python-heap signal; a flat Python heap does not
  rule out native-side RSS growth

## Validation Checklist

After a change, verify all of the following:

- `reset-only` stabilizes after the initial warm-up reset
- `episode -> reset -> next episode` no longer shows a steady RSS staircase
- GPU memory is stable after warm-up
- scripted and RL environment paths behave the same way
- both `num_envs=1` and a high parallel count are stable

## Measured Outcome In This Repository

The issue was reproduced most clearly at `num_envs=256`.

Before the fix:

- `reset-only` showed about `+2.14 MB/reset` in steady state
- full episodes also showed a visible CPU step at each reset boundary

After the fix:

- `reset-only` had one warm-up jump, then stabilized at approximately `0 MB/reset`
- the RL path at `num_envs=256` showed `0 MB` reset growth after warm-up
- the RL path at `num_envs=1` also stabilized after the first episode
- the scripted path stabilized after the first episode as well

That is the behavior you want: one-time warm-up is acceptable, but steady
per-episode RSS growth is not.
