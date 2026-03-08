# Migration Notes

## Removed modules and packages

The following bundled packages were removed from the repository:
- `r3m/`
- `vip/`
- `rolf/`

The following package-level environments were removed:
- `FurnitureBenchImageFeature-v0`
- `FurnitureSimImageFeature-v0`
- `FurnitureImageFeatureDummy-v0`

The following legacy scripts were removed:
- `implicit_q_learning/extract_feature.py`
- `implicit_q_learning/train_finetune.py`

## Behavioral changes

### Dynamic GPU memory sizing

Before the refactor, simulator GPU buffer sizes were fixed and then patched in-place for a few special cases.

After the refactor:
- buffer sizes are derived from `num_envs`
- scaling happens per environment instance
- factory tasks still apply their task-specific overrides before scaling

### Checkpoint handling

Before the refactor, evaluation could trigger automatic downloads of historical checkpoints.

After the refactor:
- checkpoint resolution is local only
- evaluation scripts require the checkpoint directory to already exist

### Data collection

Before the refactor, the data collector exposed a `feature` mode tied to bundled encoders.

After the refactor, supported modes are:
- `state`
- `full`
- `image`

## Recommended replacements

Use these replacements for removed workflows:
- feature extraction: store raw images instead of bundled encoder outputs
- `rolf` training: use the maintained offline IQL scripts if they match your workflow
- legacy training entrypoint: follow the documented scripts in `docs/user-guide.md`
