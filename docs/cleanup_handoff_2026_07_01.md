# treepo public-surface cleanup — handoff (2026-07-01)

Read this before continuing readability/cleanup work on the `treepo` package. It
records the state after a public-surface cleanup pass, what changed, what is
intentionally left alone, and how to verify.

## What `treepo` is

- The **official package** (`~/treepo`, `import treepo`). One centralized API:
  `treepo.fit()` → `treepo.methods.learning.fit` → `run_alternating_family`
  (`src/treepo/methods/runtime.py`). The family contract is
  `FamilyRuntime` (runtime-checkable Protocol in `src/treepo/methods/contracts.py`);
  all families (`classical_sketch, dspy, fno, learnable_constant, llm,
  neural_operator, oracle`) implement it.
- Examples live in `examples/methods/*.py`, each: frozen dataclass config ← paired
  `.toml` → centralized `make_*` fixture → `treepo.fit()` → result JSON + one-line
  status. Bench examples in `examples/bench/*.yaml`, run via `treepo-bench`.
- The manifesto comparison the project cares about (compare examples across leaf
  granularity × supervision scope, all through one API) **already exists cleanly**:
  `examples/methods/run_manifesto_replications.py` sweeps
  `leaf_unit_counts` × `supervision_grid` (from `manifesto_replications.toml`)
  with one `fit()` call per cell. Backend lives in
  `src/treepo/tasks/manifesto/replication.py`.

## Important context: the tree is a large cleanup diff

The working tree was heavily dirty when the cleanup began and now contains the
full uncommitted public-surface/methods streamline. The old methods dispatch,
estimator, g-estimator, and application stub modules were removed from the
package surface. The delete phase is reference-clean: there are no dangling
imports to removed methods modules anywhere in src/tests/examples.

## What this cleanup pass changed (all uncommitted, in the worktree)

Duplication lifted into the package:
- New `src/treepo/tasks/sampling_artifacts.py` — `write_sampling_rows_jsonl`,
  `sampling_summary`, `write_sampling_artifacts`. Replaces the triplicated
  `_write_jsonl` / `_sampling_summary` / `_write_sampling_artifacts` in the 3
  manifesto example scripts and standardizes the `sampling/` output subdir
  (on-disk file locations unchanged).
- `src/treepo/methods/preference.py` — added `summarize_preference_views`
  (replaces `_optimizer_preview` ×3) and `export_adapter_views` (replaces
  `_adapter_exports` ×2).
- Top-level `_jsonable` / `stable_digest` consolidated to `common.jsonable` +
  `manifest.stable_digest`. **`evidence._jsonable` was intentionally kept
  separate** — its body genuinely differs (omits Enum conversion, expands
  dataclasses before `to_dict`); merging would change serialization output.

Dead code removed:
- `fno.py` root-model path (`_build_root_model`, `_RootNeuralOp`, `_RootConv1D`,
  `_root_head`, ~85 lines, superseded by `_TreeFGModel`'s readout).
- Misleading `_1_7` metric aliases in `runtime.py` + `learning.py` (they
  duplicated native values under a name implying a 1–7 rescaling that never
  happened; scale is carried by `metrics_scale`).
- `wrote_any` no-op tail in `learning._write_prediction_records`.
- Empty skeleton dirs `bench/learned/`, `bench/reports/`, `bench/suites/`.

Readability:
- Numeric-feature hashing extracted to `src/treepo/methods/_numeric_features.py`
  (`add_numeric_sequence_features`, with docstring); shrinks the 1484-line
  `fno.py`.
- `llm._call_predict_fn` uses `inspect.signature` instead of nested
  `try/except TypeError` arity probing (genuine callee `TypeError`s now
  propagate).
- `finetune._register_builtin_adapters` data-driven via `_BUILTIN_ADAPTER_SPECS`.
- `run_manifesto_replications._run_cell` no longer takes ~10 injected callables.
- Module docstrings added to 9 top-level modules; public-function docstrings
  added in `objective`, `honesty`, `certificate`, `release`.

Doc drift fixed:
- `CHANGELOG.md`: `run()`/`suite/report` CLI → `fit()`/`run/check`.
- `inventory.yaml`: removed deleted `dispatch` from the `treepo.methods` note.

## 2026-07-02 stabilization update

The follow-on methods-wide streamline plan is fully implemented:

- Method fixtures, Manifesto/RILE helpers, fit-result assembly, alternating
  runtime, preference records/views/IO, and neural-operator/FNO internals are
  split behind stable facades.
- `family="fno"` is a concrete FNO route. `family="neural_operator"` is the
  generic route and accepts explicit operator kinds, including
  `operator_kind="fno"` and the `operator_kind="fourier"` alias.
- Application runtimes can register family factories through
  `treepo.methods.families.register_family(...)` without adding branches to
  `treepo`.
- Examples are vignette-style: setup/data loading is centralized under
  `examples/methods/example_setup`, and runnable example files mostly read
  config, call `treepo.fit(...)`, and write outputs.
- Release-boundary docs now distinguish built-in provider-neutral wrappers
  (`llm`, `dspy`) from downstream application runtimes.

## Open items

No known cleanup-plan blockers remain. Further work should shift from splitting
to review/release hygiene:

1. Review the large uncommitted diff and organize it into sensible commits.
2. Keep future changes small unless a focused audit exposes a real duplication
   or boundary problem.
3. Remaining borderline long functions in top-level modules
   (`tree.local_law_rows_from_tree_records`,
   `objective.resolve_root_local_objective_weights`) were intentionally left
   alone; they read fine with keyword-only signatures.

## Verify

Use the project venv (`uv run`), not a bare `python`.

```
cd ~/treepo
uv run python -c "import treepo; import treepo.methods; print('ok')"
uv lock --check
uv run pytest -q -p no:randomly
uv run treepo-bench check release --json
uv run python -m treepo.release
uv build --wheel --sdist --out-dir /tmp/treepo_release_artifacts
```

As of the 2026-07-02 stabilization pass:

- `uv lock --check` passes.
- `uv run pytest -q -p no:randomly` passes: 162 passed, 1 skipped.
- `uv run treepo-bench check release --json` passes.
- `uv run python -m treepo.release` passes.
- `uv build --wheel --sdist --out-dir /tmp/treepo_release_artifacts` passes.

## Conventions

- Legacy/superseded files are renamed with an `OLD_` prefix + a header note —
  never deleted, never edited/imported. (No `OLD_*` currently in `treepo`.)
- Keep `import treepo.methods`, `import treepo.llm`, and top-level
  `import treepo` import-light; torch/neuralop/DSPy/OpenAI/TRL stay lazy.
- Do not add per-node `.cpu()`/`.item()` calls in any FNO forward path; keep the
  FNO channel invariant (in_channels 1 for f / 2 for g, out 1).
- Do not touch the digest/`jsonable` serialization semantics — hashes and
  manifest output must stay byte-stable.
