# Package Minimization Status

Date: 2026-05-23

Scope: this repository checkout, v0.1.0 source release preparation.

This note records the current package boundary after the `cld`/`cdx` cleanup,
uv migration, example relocation, and optional-dependency split. It is meant to
be the short operational status page for what is now minimal and what still
needs a policy decision before the v0.1.0 push.

## Current Boundary

The public package is intentionally small:

- Base install depends only on `numpy`.
- `import treepo` is expected to remain import-light.
- Public-facing surfaces are `treepo`, core certificate/manifest/objective
  modules, `treepo.methods`, `treepo.local_law`, `treepo.bench`, and the
  small task/HLL helpers documented in `README.md`.
- Research code is quarantined under `treepo._research`.
- Source-tree examples and configs live under `examples/research/` and
  `configs/research/`.
- The published wheel should not rely on source-tree examples/configs being
  installed.

The package is uv-first. Release docs should use `uv sync`, `uv run`,
`uv lock`, and `uv build`; bare pip commands are intentionally not part of the
v0.1.0 workflow.

## Dependency Model

Minimal runtime:

- `numpy`

Optional extras:

- `bench`: YAML config IO for benchmark/example configs.
- `reports`: matplotlib/pandas reporting path.
- `sketches`: datasketches-backed sketch adapters.
- `sklearn`: sklearn/proxy baselines.
- `torch`: learned sketch and state-model modules.
- `llm`: OpenAI/DSPy clients plus `langextract` and `tiktoken`.
- `runtime`: LongBench/RULER runtime helpers plus `langextract`.
- `research`: non-LLM research utilities plus `langextract`, pandas, sklearn,
  and token utilities.
- `train`: DSPy/TRL/state-model training hooks, including LLM utilities.
- `all`: every optional runtime/research stack for simple uv installs.

`langextract` is deliberately absent from the base install and present in the
non-minimal stacks where it is likely to be useful: `llm`, `runtime`,
`research`, `train`, `all`, and the default dev dependency group.

## Source Layout After Cleanup

Moved out of the public-looking root:

- `examples/cardinality_recovery.yaml`
- `examples/hll_merge_learning.yaml`
- `examples/classical_sketches.yaml`
- `examples/runtime_*.yaml`
- `examples/longbench_v2_tiny.yaml`
- `examples/methods/*`
- `configs/methods/*`

Current research fixture homes:

- `examples/research/bench/`
- `examples/research/runtime/`
- `examples/research/methods/`
- `configs/research/methods/`

The older `cld` naming has been removed from the package-facing layout. The
current public methods surface is under `treepo.methods`; research and legacy
implementation detail remains under `treepo._research`.

## Import-Light Status

The public import path was checked so that `treepo.list_methods()` does not
pull in the heavy/non-base stack. The checked modules stayed unloaded before
and after `list_methods()`:

- `yaml`
- `langextract`
- `tiktoken`
- `dspy`
- `openai`
- `torch`
- `transformers`
- `datasets`
- `pandas`

This should remain a release invariant: public discovery APIs must not make
optional extras mandatory.

## Verification Snapshot

Most recent focused checks after the minimization pass:

```bash
uv lock --check
uv run pytest tests/test_examples.py tests/test_release_gates.py \
  tests/test_package_layers.py tests/methods/test_canonical_defaults_drift.py -q
uv run pytest -q
uv run python -m treepo.release
uv build --wheel --out-dir /tmp/treepo_wheel_qc
```

Observed results:

- Focused checks: `41 passed`.
- Full tests: `268 passed, 10 skipped`.
- Release gate: `ok: true`.
- Wheel build: succeeded.
- Wheel spot check: source-tree `examples/`, `configs/`, `treepo/runtime`,
  `treepo/sketches`, and public `treepo/bench/lda` paths were absent.

## What Remains To Minimize

### 1. Decide Whether `_research` Ships In The Wheel

Current state:

- `_research` is not public, but it is still packaged.
- Some public or semi-public research examples import config classes and helpers
  from `treepo._research`.
- The release gate treats `_research` as a quarantined migration tier.

Options:

- Keep `_research` in the v0.1.0 source/wheel as internal, unsupported code.
- Exclude `_research` from the wheel and make research examples source-tree
  only.
- Promote the small subset needed by public APIs out of `_research`, then
  exclude the rest.

Recommendation for v0.1.0: keep `_research` packaged but clearly unsupported.
Move toward exclusion in v0.2 after public APIs no longer depend on internal
research dataclasses or helpers.

### 2. Keep `treepo.bench` Narrow

Current state:

- Benchmark CLI and paper suites are still package-visible.
- LDA, runtime, sketch adapter, and report-heavy paths have been moved away
  from the public root.

Remaining work:

- Re-check that every `treepo.bench` import is light unless its command has
  explicitly entered an extra-backed path.
- Keep LDA and large simulation modules under `_research` or source-tree
  research examples.
- Avoid adding new benchmark families directly under public `treepo.bench`
  unless they are small release-facing smoke tests.

### 3. Make Wheel Policy Explicit

Current state:

- Source release is the immediate target.
- README says examples/docs are part of the GitHub checkout and not guaranteed
  from a wheel install.

Remaining work:

- Add a short packaging policy before PyPI: which docs, examples, and configs
  are intentionally excluded from wheels.
- Decide whether `treepo-bench` should support wheel-only installs or require
  source checkout fixtures for research suites.

### 4. Tighten Release-Gate Coverage

Current state:

- Release gate passes.
- Heavy import checks cover the important public import path.

Remaining work:

- Resolve every lazy top-level export in a subprocess during the gate.
- Keep expanding the core-light allowlist as public modules stabilize.
- Add a wheel-content assertion if the exclusion policy becomes strict.

### 5. Finish Documentation Consistency

Current state:

- README, changelog, training defaults, and pre-push notes were updated for uv,
  research examples, optional extras, and the new methods path.

Remaining work:

- Run a source-tree link/path scan over Markdown files.
- Remove any stale references to `cld`, `cdx`, root-level examples, or
  old `configs/methods` paths.
- Keep install docs uv-only.

### 6. Separate Contributor Defaults From User Defaults

Current state:

- `uv sync` installs the dev group, which includes research and LLM utilities.
- `uv sync --no-dev` is the minimal user install.
- `uv sync --extra all --no-dev` is the broad non-dev install.

Remaining work:

- Keep this distinction explicit in README and release notes.
- Avoid implying that default contributor sync is the same thing as the minimal
  package dependency surface.

## Current Release Posture

The v0.1.0 package is in a good minimal-source-release posture:

- base dependency surface is small;
- optional extras are explicit;
- examples/configs are quarantined under research;
- LLM/langextract utilities are available in non-minimal stacks;
- public import discovery is light;
- tests and release gate pass.

The main unresolved minimization question is not a correctness blocker: whether
internal `_research` should ship in v0.1.0 or be excluded from wheels after a
larger public/private split.

## Update 2026-06-11: `_research` Pruned To Its Reachable Closure

The main unresolved question above ("does `_research` ship?") was resolved by
pruning instead: `_research` now contains only the transitive import closure
reachable from the public package surface, `tests/`, `examples/`, and
`scripts/` — computed by AST scan over static imports, lazy
`import_module(...)` string literals, and f-string module-prefix patterns
(conservatively keeping whole subtrees for dynamic prefixes).

- Files: 585 → 369 `.py` (216 pruned; ~22M → ~8.7M).
- Entire subtrees removed: `datasets/`, `embeddings/`, `pipelines/`,
  `harness.py`; large cuts in `ctreepo/` (sim suites), `training/`, `tasks/`,
  `runtime/`, `experiments/`.
- A complete pre-prune snapshot lives at `~/OLD_treepo` (frozen archive,
  `.venv` excluded; recreate with `uv sync`). Restore from there if a pruned
  module turns out to be needed.
- Verification after the prune: full suite `283 passed, 10 skipped`
  (includes release gates, examples smoke, package layers, methods
  reproduction cells); `uv build --wheel` succeeds; the parent
  ThinkingTrees workspace's treepo-coupled suites pass against the
  minimized package (its venv installs this repo editable).

Known dynamic-import seams the closure handles explicitly: the lazy task
registry (`_research/tasks/__init__.py` `_DEFAULT_TASK_MODULES`), the
`treepo.methods` family factories, and the bench runner / learning
`import_module` calls. Any NEW lazy import added by string must either be a
literal (the closure scan catches it) or be accompanied by a test, or it
risks being pruned in a future pass.
