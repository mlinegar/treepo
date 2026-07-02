# `treepo.methods` Module Layout

This is the reference for how the internal `treepo.methods` package is organized
and the convention every cleanup pass follows. Read it before splitting or
merging modules under `src/treepo/methods/`.

The guiding idea: **isolate the messy data work, keep the entry point lean.**
`treepo.fit(...)` and the family runtimes should read top-to-bottom as an
orchestration vignette; the awkward per-row parsing, encoding, and projection
logic lives in focused, clearly-named siblings that the lean surface calls into.

## The convention

Every large module in `treepo.methods` is decomposed the same way:

1. **Keep the public import path stable.** Consumers import from a small set of
   stable surfaces (`treepo.fit`, `treepo.methods.fno`,
   `treepo.methods.preference`, `treepo.methods.runtime`, etc.). Those paths
   never move.
2. **Split implementation by responsibility**, along the *data-prep vs.
   model-logic seam*:
   - **Data-prep** (the messy half): row parsing, scalar/JSON coercion, leaf
     extraction, embedding, target extraction, projection to export shapes.
   - **Model logic / machinery** (the lean half): the family runtime, torch
     models, config coercion, dependency discovery, the data model.
3. **Facades import from responsibility modules directly.** A public facade
   (`preference.py`, `fno.py`, `runtime.py`) imports the real symbols from its
   responsibility modules and re-exports them via `__all__`. Add an
   intermediate hub module only when an existing internal import surface
   genuinely needs preserving; currently every facade imports directly.
4. **Keep the dependency graph acyclic.** Responsibility modules import
   downward (leaf → model → ops); the facade imports from all of them and is
   imported by none of them internally. Break the rare back-edge with a
   deferred (function-local) import.
5. **Preserve laziness.** Keep torch/neuralop imports inside functions and
   methods so `import treepo` and `import treepo.methods` stay light.

### Anti-pattern: re-export stubs over a surviving monolith

A file that is only `from _core import (...)` plus `__all__`, while the
implementation stays in a 40KB+ `_core.py`, adds indirection and makes the tree
*look* decomposed while the work still lives in one monolith. When a `_core`
module holds most of the defs, finish the decomposition: move the real code
into responsibility files and keep `_core` as the lean runtime that calls into
them.

## Stable public surfaces

These import paths are contracts. Changing them is a breaking change.

- `treepo.fit(...)` / `treepo.methods.learning.fit`
- `treepo.methods.runtime.{IterationRecord, SplitMetrics, evaluate_splits, run_alternating_family}`
- `treepo.methods.preference.*`
- `treepo.methods.fno.*`
- `treepo.methods.fixtures.*`
- `treepo.tasks.manifesto.*`
- Family names (including `"neural_operator"`, which resolves through
  `families.py`), config keys, artifact keys, metric keys, manifest paths, and
  preference export row shapes.

## The central fit path

`fit()` is deliberately tiny — load inputs, resolve the family, run the ladder,
assemble the result:

- `learning.py` — `fit(spec)`: the single public learning surface (~60 lines).
- `_fit_inputs.py` — family/runtime resolution, sequence coercion, objective
  resolution.
- `_preference_traces.py` — `PreferenceDataset` → f/g training trace conversion.
- `_fit_result.py` — `FitResult` assembly, final-metric flattening, split-metric
  payloads, and prediction JSONL writing/collection.
- `_run_manifest.py` — methods manifest writing and JSON default handling.
- `_coerce.py` — scalar/vector coercion helpers shared across methods modules.
- `families.py` — the small family-name registry; downstream packages register
  additional runtimes here, keeping `treepo` branch-free.

## Alternating runtime

- `runtime.py` — the public facade for the runtime surface.
- `_runtime_types.py` — runtime dataclasses (`IterationRecord`, `SplitMetrics`).
- `_runtime_evaluation.py` — split evaluation, prediction rows, scalar/vector
  metric math.
- `_runtime_loop.py` — the alternating f/g orchestration loop, the
  train-f/train-g stage schedule, and statistic / local-law payload extraction.

## Neural-operator / FNO stack

`_neural_operator_core.py` is the **lean family runtime**: it holds
`NeuralOperatorFamily`, `FNOFamily`, and the two `build_*_family` functions,
and delegates the messy work to the responsibility modules below. `fno.py` is
the public facade over it; the `"neural_operator"` family name resolves to the
same runtime through `families.py`.

Dependency order (top imports downward):

```
_fno_config        constants, Config dataclasses, coercion, clamp/safe_float   (bottom)
   ▲
_fno_neuralop      torch/neuralop discovery, constructor kwargs, validation    (machinery)
   ▲
_fno_models        _TreeFGModel + leaf operators                               (model logic)
_fno_encoding      leaf text/token extraction, embedding, numeric features     (DATA-PREP)
_numeric_features  bit-hashing encoder for numeric-token leaf sequences        (DATA-PREP, used by _fno_encoding)
_fno_targets       scalar/vector target extraction                             (DATA-PREP)
_fno_transition    numeric transition-state targets/loss/rows                  (DATA-PREP)
_fno_statistic     _NeuralOperatorStatistic adapter                            (adapter)
   ▲
_neural_operator_core   lean family runtime + build_* factories                (lean top)
   ▲
fno.py             public facade
```

`_fno_statistic` takes the family as an argument rather than importing the
family class, so it stays below the runtime with no cycle.

## Preference stack

The public facade `preference.py` imports directly from the responsibility
modules. Split by responsibility:

```
_preference_normalize   row parsing, scalar/JSON coercion, field constants     (DATA-PREP, leaf)
   ▲
_preference_views       supervised/DPO/reward/GRPO projections                 (DATA-PREP)
   ▲
_preference_dataset     Candidate / PreferenceRecord / PreferenceDataset        (data model)
   ▲
_preference_io          dataset IO, export, adapter-view fan-out, previews      (ops)
_preference_tree        TreeRecord-derived units                                (ops)
   ▲
preference.py           public facade
```

The rare back-edge between layers is resolved with a function-local import,
keeping the model/ops layers acyclic.

## Other modules

The remaining files are single-responsibility already: `contracts.py` (fit and
family contracts), `canonical_defaults.py` (the generic `load_dataclass` TOML
loader), `grid.py` (reproducible method-grid enumeration and JSON/CSV output),
`oracles.py` (built-in oracle scorers), `learnable.py` (the
`learnable_constant` family), `sketch.py` (the `classical_sketch` family),
`lda.py` (the LDA fixture baseline helpers), and `llm.py` / `dspy.py` (the
provider-neutral wrapper families).

## Already-decomposed task/fixture modules

These follow the same pattern and are good templates for the seam:

- `fixtures/` — `hll.py`, `lda.py`, `markov.py` per DGP, `common.py` for shared
  helpers, `__init__.py` as the facade.
- `tasks/manifesto/` — `documents.py`, `sampling.py`, `trees.py`,
  `preferences.py`, `exports.py`, `prompts.py`, `state.py`, `rile.py`;
  `__init__.py` is the facade.

## Adding to the package

- **New helper in an existing stack** → put it in the responsibility module that
  matches its role (data-prep vs. machinery) and add it to that module's
  `__all__`; add it to the facade's re-export list only when the public surface
  needs it.
- **New family** → implement the `FamilyRuntime` protocol
  (`contracts.py`), add a `build_*` factory, and register it in `families.py`.
  Application-heavy families register from downstream packages.
- **Legacy code** → archive by renaming with an `OLD_` prefix and a header
  note, and keep `OLD_*` modules import-free; delete verified-dead code
  outright (git history preserves it).

## Checks

Run the focused suite for the surface you touched plus the release gates:

```bash
uv run pytest tests/methods -q
uv run pytest -q            # full suite + release gates
```

Verify laziness is intact:

```bash
uv run python -c "import sys, treepo; assert 'torch' not in sys.modules and 'neuralop' not in sys.modules"
```
