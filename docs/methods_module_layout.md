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
3. **Use a re-export hub, not stubs.** Where an existing import surface must be
   preserved (e.g. `_neural_operator_core`, `_preference_core`), that module
   becomes a thin **hub** that imports the real symbols from the responsibility
   modules and re-exports them via `__all__`. The hub contains no orphan logic.
4. **Keep the dependency graph acyclic.** Responsibility modules import
   downward (leaf → model → ops); the hub imports from all of them and is
   imported by none of them internally. Break the rare back-edge with a
   deferred (function-local) import, not a restructure.
5. **Preserve laziness.** `import treepo` and `import treepo.methods` must not
   pull in torch or neuralop. Heavy imports stay inside functions/methods.

### Anti-pattern: re-export stubs over a surviving monolith

Do **not** create a file that is only `from _core import (...)` + `__all__`
while the implementation stays in a 40KB+ `_core.py`. That adds indirection
without improving readability, and it makes the tree *look* decomposed when it
is not. If a `_core` module still holds most of the defs, it has not been
decomposed — invert it so `_core` is the hub and the real code lives in the
responsibility files. (This is exactly what the 2026-07-02 pass fixed for the
FNO and preference stacks; see
[`internal_streamline_plan_2026_07_01.md`](internal_streamline_plan_2026_07_01.md).)

## Stable public surfaces

These import paths are contracts. Changing them is a breaking change.

- `treepo.fit(...)` / `treepo.methods.learning.fit`
- `treepo.methods.runtime.{IterationRecord, SplitMetrics, evaluate_splits, run_alternating_family}`
- `treepo.methods.preference.*`
- `treepo.methods.fno.*`
- `treepo.methods.neural_operator.*`
- `treepo.methods.fixtures.*`
- `treepo.tasks.manifesto.*`
- Family names, config keys, artifact keys, metric keys, manifest paths, and
  preference export row shapes.

## The central fit path

`fit()` is deliberately tiny — load inputs, resolve the family, run the ladder,
assemble the result:

- `learning.py` — `fit(spec)`: the single public learning surface (~60 lines).
- `_fit_inputs.py` — family/runtime resolution, sequence coercion, objective
  resolution.
- `_preference_traces.py` — `PreferenceDataset` → f/g training trace conversion.
- `_fit_result.py` — `FitResult` assembly, final-metric flattening, split-metric
  payloads.
- `_prediction_records.py` — prediction JSONL writing/collection.
- `_run_manifest.py` — methods manifest writing and JSON default handling.
- `families.py` — the small family-name registry; downstream packages can
  register additional runtimes without adding branches to `treepo`.

## Alternating runtime

- `runtime.py` — compatibility facade for the runtime surface.
- `_runtime_types.py` — runtime dataclasses (`IterationRecord`, `SplitMetrics`).
- `_runtime_schedule.py` — stage names, labels, train-f/train-g schedule.
- `_runtime_evaluation.py` — split evaluation, prediction rows, scalar/vector
  metric math.
- `_runtime_statistics.py` — statistic / local-law payload extraction.
- `_runtime_loop.py` — the alternating f/g orchestration loop.

## Neural-operator / FNO stack

`_neural_operator_core.py` is the **lean family runtime + re-export hub**: it
holds `NeuralOperatorFamily`, `FNOFamily`, and the two `build_*_family`
functions, and re-exports the helpers below so existing
`from treepo.methods._neural_operator_core import ...` call sites stay stable.
`fno.py` and `neural_operator.py` are the public facades over it.

Dependency order (top imports downward):

```
_fno_config        constants, Config dataclasses, coercion, clamp/safe_float   (bottom)
   ▲
_fno_neuralop      torch/neuralop discovery, constructor kwargs, validation    (machinery)
   ▲
_fno_models        _TreeFGModel + leaf operators                               (model logic)
_fno_encoding      leaf text/token extraction, embedding, numeric features     (DATA-PREP)
_fno_targets       scalar/vector target extraction                             (DATA-PREP)
_fno_transition    numeric transition-state targets/loss/rows                  (DATA-PREP)
_fno_statistic     _NeuralOperatorStatistic adapter                            (adapter)
   ▲
_neural_operator_core   family runtime + build_* + re-export hub               (lean top)
   ▲
fno.py / neural_operator.py   public facades
```

`_fno_statistic` takes the family as an argument rather than importing the
family class, so it stays below the runtime with no cycle.

## Preference stack

`_preference_core.py` is a **pure re-export hub** — no logic of its own. The
public facade `preference.py` and the lazy `treepo.methods` exports resolve
through it. Split by responsibility:

```
_preference_normalize   row parsing, scalar/JSON coercion, field constants     (DATA-PREP, leaf)
   ▲
_preference_views       supervised/DPO/reward/GRPO projections                 (DATA-PREP)
_preference_adapters    export_adapter_views fan-out (injected fn, no model)    (leaf)
   ▲
_preference_dataset     Candidate / PreferenceRecord / PreferenceDataset        (data model)
   ▲
_preference_io          dataset IO, export, view previews                       (ops)
_preference_tree        TreeRecord-derived units, tree filtering                (ops)
   ▲
_preference_core        re-export hub
   ▲
preference.py           public facade
```

`PreferenceDataset.filter_tree` calls `filter_units_for_tree`, which lives in
`_preference_tree` and imports the data model — a back-edge. It is resolved with
a **function-local import** inside `filter_tree`, keeping the model/ops layers
acyclic. This is the sanctioned way to handle such edges; do not "fix" it by
merging the modules.

## Already-decomposed task/fixture modules

These follow the same pattern and are good templates for the seam:

- `fixtures/` — `hll.py`, `lda.py`, `markov.py` per DGP, `common.py` for shared
  helpers, `__init__.py` as the facade.
- `tasks/manifesto/` — `documents.py`, `sampling.py`, `trees.py`,
  `preferences.py`, `exports.py`, `prompts.py`, `state.py`, `rile.py`;
  `replication.py` and `__init__.py` are facades.

## Adding to the package

- **New helper in an existing stack** → put it in the responsibility module that
  matches its role (data-prep vs. machinery), add it to that module's `__all__`,
  and — only if an existing import surface needs it — add it to the hub's
  re-export list.
- **New family** → implement the `FamilyRuntime` protocol
  (`contracts.py`), add a `build_*` factory, and register it in `families.py`.
  Application-heavy families register from downstream packages instead of being
  bundled.
- **Legacy code** → archive by renaming with an `OLD_` prefix and a header note;
  never delete outright, never import `OLD_*`.

## Checks

Run the focused suite for the surface you touched plus the release gates:

```bash
.venv/bin/python -m pytest tests/methods -q
.venv/bin/python -m pytest -q            # full suite + release gates
```

Verify laziness is intact:

```bash
.venv/bin/python -c "import sys, treepo; assert 'torch' not in sys.modules and 'neuralop' not in sys.modules"
```
