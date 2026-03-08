# Refactored FurnitureBench Documentation

This documentation describes the supported workflows of the refactored repository.

## Document map

- [User guide](user-guide.md): installation, simulator usage, data collection, replay, and offline training.
- [Developer guide](developer-guide.md): package structure, dynamic GPU-memory sizing, and refactor boundaries.
- [Migration notes](migration.md): removed legacy modules and the new supported replacements.

## Repository boundaries

The refactor intentionally focuses on:
- `furniture_bench`
- scripts that directly use `furniture_bench`
- the offline `implicit_q_learning` workflow that still operates on raw observations

The following folders were intentionally left untouched and can be used as local implementation references only:
- `SAPIEN/`
- `ManiSkill/`
