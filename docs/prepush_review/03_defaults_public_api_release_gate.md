# Defaults, Public API, And Release Gate

## Install Docs And `pyproject.toml`

README now advertises uv-native extras such as `--extra torch`, `--extra llm`,
and `--extra all`. `pyproject.toml` defines matching
`[project.optional-dependencies]` entries and keeps the dev dependency group
for contributor installs.

Fix:

- Keep README and `pyproject.toml` aligned when changing extras.
- Do not ship pip commands; v0.1.0 is uv-first.

## Core Dependency Claim

README and `pyproject.toml` now agree that the core install is only `numpy`.
`langextract` and `tiktoken` live in non-minimal extras such as `llm`,
`runtime`, `research`, `train`, and `all`.

Fix:

- Update README or move dependencies behind real extras.

## Server-Script Scope Claim

README says local server management is excluded, but server scripts exist in
the repo.

Fix:

- Either remove the exclusion claim or move scripts out of the minimal public
  surface.

## Stale Docs

`docs/training_defaults.md` still has pre-port paths such as `src/ctreepo/...`
and links that should now point under `src/treepo/_research/...`,
`configs/research/methods/...`, or `examples/research/methods/...`.

Fix:

- Rewrite stale paths.
- Run a link checker or equivalent file-existence scan.

## methods Version And Workspace Language

`src/treepo/methods/__init__.py` still describes methods as a parallel workspace and has
its own stale version.

Fix:

- Remove "eventual merge" language.
- Prefer package metadata version or omit a methods-specific version.

## Lazy Exports

`import treepo` is checked for heavy imports, but the release gate does not
resolve every lazy export.

Fix:

- In the release gate, iterate `_LAZY_EXPORTS` and call `getattr(treepo, name)`
  for each symbol in a subprocess.

## Core-Light Allowlist

Public modules such as `objective.py`, `manifest.py`, `certificate.py`,
`honesty.py`, `sampling.py`, and `paths.py` should be guarded against heavy
imports if they are public-light modules.

Fix:

- Extend `_is_core_light_path`.

## methods Must Enter The Release Gate

`src/treepo/methods/` is currently in `MIGRATION_TIER_PREFIXES`, which excludes the
primary public dispatcher from several hygiene checks.

Fix:

1. Remove `src/treepo/methods/` from the migration-tier exclusions.
2. Run the release gate.
3. Fix whatever stale paths/imports surface.

Keep `_research`, scripts, tests, and fixture configs quarantined as needed.

## Top-Level `NormalizedOutput`

`ExperimentContext.record()` returns `NormalizedOutput`; `treepo.core` exports
it, but top-level `treepo` does not.

Fix:

- Re-export `NormalizedOutput` at top level if `ExperimentContext` is part of
  the top-level public API.

## Changelog

`CHANGELOG.md` should list:

- shipped public surface;
- known Lean/certificate limitations;
- HLL scalar-vs-state caveat;
- `_research` quarantine;
- status of large LDA bench modules.
