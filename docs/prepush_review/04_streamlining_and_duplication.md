# Streamlining And Duplication

## Canonical Local-Law Module

Use one canonical module, `treepo.local_law`, with:

- `LawKind`;
- Lean/paper alias map;
- scalar `corrected_local_law_loss`;
- mode normalization;
- scalar depth weight validation;
- all-row influence overlap.

Current package shape:

- `treepo.training.local_law` keeps torch wrappers only;
- `treepo.methods` imports and re-exports canonical law symbols directly;
- the old audit/local-law shim modules have been deleted.

## Finite-Float Helper

`_finite_float` / `_finite` is duplicated across several modules.

Fix:

- Add `finite_float(value, *, name)` to `treepo.common`.
- Import it everywhere.

## Law Alias Map

Law aliases are duplicated and inconsistent.

Fix:

- Define one mapping:
  - `c1`, `l1`, `leaf` -> `leaf_preservation`
  - `c2`, `l3`, `idempotence` -> `on_range_idempotence`
  - `c3`, `l2`, `merge` -> `merge_preservation`

## `methods/dispatch.py` Is Too Dense

It mixes:

- registry;
- fit/oracle/audit handlers;
- oracle fixtures;
- config allowlists.

Minimum split:

- keep sketch factories out of public `treepo.methods`.

Better split:

- create `methods/handlers/{fit,oracle,audit}.py`.

## `learning.py` Is A Public-API Decision

`src/treepo/learning.py` has no internal imports, but top-level `treepo`
lazily exports `FitConfig`, `FitResult`, and `fit` from it.

Fix:

- If methods is canonical, repoint top-level fit exports to methods before deleting
  `learning.py`.
- If `treepo.learning` remains public, add direct tests and README coverage.

## `_optional.py`

`src/treepo/_optional.py` is unused.

Fix:

- Delete it, or use it consistently for optional dependency errors after real
  extras are defined.

## Bench Data Generation

Zipf/token generation is duplicated between HLL merge learning and cardinality
recovery.

Fix:

- Extract shared utilities to `src/treepo/bench/common_data_gen.py`.

## Training Objective Vs Certificate Radius

Training code returns normalized losses for optimization. Lean certificates
consume influence-weighted error envelopes.

Fix:

- Separate training objectives from certificate-radius construction.
- Add docstrings stating which module returns a backprop loss and which returns
  a theorem-facing bound.

## Import Path Mutation

Release-facing examples and `methods/canonical_defaults.py` no longer mutate
`sys.path`. Some migrated tests still insert the repo root to call source-tree
scripts directly.

Fix:

- Keep package-relative imports through `treepo._research`.
- Prefer `uv run ...` / installed-package execution for examples.
- Leave source-tree path setup only in tests that invoke standalone scripts.
