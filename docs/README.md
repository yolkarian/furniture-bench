# Documentation Index

This documentation covers the **project-owned** parts of the cleaned-up repository.

It focuses on:
- the `furniture_bench` package
- maintained Python entry points under `furniture_bench/scripts/`
- maintained top-level helpers under `scripts/`
- repository cleanup and compatibility behavior

It does **not** attempt to rewrite the vendored upstream documentation under:
- `ManiSkill/`
- `SAPIEN/`

## Document map

- [User guide](user-guide.md)
  - installation
  - simulator usage
  - data collection
  - replay
  - dataset download and preprocessing

- [Developer guide](developer-guide.md)
  - package boundaries
  - controller cleanup
  - script conventions
  - typing / formatting expectations
  - verification approach

- [Migration notes](migration.md)
  - removed directories and modules
  - shell-script relocation
  - compatibility aliases
  - recommended replacements for removed workflows

- [Repository structure](../STRUCTURE.md)
  - current top-level layout
  - core vs supporting vs reference-only areas
