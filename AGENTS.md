# AGENTS.md - treepo Agent Quick Start

This checkout is the standalone `treepo` package. For detailed guidance, read `docs/llm_guide.md` after this file.

## Working Rules

- Check `git status --short` before editing; preserve unrelated user changes.
- Use `rg` for search and inspect nearby tests before changing behavior.
- Keep the package boundary small. Large application workflows, model-serving orchestration, publication grids, and private datasets belong downstream.
- Preserve the single public learning surface: `treepo.fit(...)`.
- Use existing public records before creating new shapes: `TaskState`, `TreeRecord`, `TreeUnitRef`, `PreferenceDataset`, `LocalLawAuditRow`, `ObjectiveSpec`, and `FitResult`.
- Keep top-level `import treepo` light. Heavy optional stacks must remain behind extras and lazy imports.
- Keep examples small, provider-neutral, and runnable without model servers unless explicitly documented as downstream-owned.

## Common Commands

```bash
uv sync
uv run pytest -q
uv run pytest -q tests/test_package_layers.py tests/test_release_gates.py tests/test_unified_contracts.py
uv run treepo-bench check release --json
uv run python -m treepo.release
```

## Package Map

- `src/treepo/learning.py`: package-level `treepo.fit(...)` wrapper.
- `src/treepo/methods/`: fit contracts, family registry, alternating runtime, built-in families.
- `src/treepo/local_law.py`: canonical C1/C2/C3 row arithmetic and audit summaries.
- `src/treepo/state.py` and `src/treepo/tree.py`: JSONable state and tree artifact boundaries.
- `src/treepo/finetune.py`: trainer-neutral export views from `PreferenceDataset`.
- `src/treepo/bench/`: `treepo-bench` runs and release checks.
- `examples/`: small package-native examples.
- `inventory.yaml`: release-checked package boundary inventory.

## Before Handoff

Run the focused tests for touched areas. If you cannot run a relevant check, say so with the exact reason. Do not hide existing dirty-worktree or broader-suite failures.
