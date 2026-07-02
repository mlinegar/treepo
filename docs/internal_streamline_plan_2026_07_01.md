# Internal Streamline Plan

Date: 2026-07-01

This follows the example cleanup pattern: keep the public-facing entry point
small, move setup and one-off mechanics into focused modules, and preserve
existing import surfaces through facade modules.

## Pattern

For each cleanup target:

1. Keep the current public import path stable.
2. Move grouped implementation details into small files by responsibility.
3. Keep orchestration functions readable: load inputs, call core API, collect
   outputs.
4. Avoid new public record shapes when existing package records already fit.
5. Run focused tests for the moved surface plus release gates.

## Planned Passes

### 1. Method Fixtures

Status: completed in this pass.

Previous file: `src/treepo/methods/fixtures.py`

Target shape:

- `src/treepo/methods/fixtures/__init__.py`: compatibility facade.
- `src/treepo/methods/fixtures/hll.py`: HLL item fixture.
- `src/treepo/methods/fixtures/lda.py`: LDA topic fixture.
- `src/treepo/methods/fixtures/markov.py`: Markov changepoint fixture.
- `src/treepo/methods/fixtures/common.py`: shared exact-score metadata, leaf
  slicing, and CUDA validation helpers.

Success criteria:

- Existing imports like `from treepo.methods.fixtures import make_lda_topic_trees`
  still work.
- No behavior change to generated metadata or fixture determinism.
- Focused fixture, FNO, grid, example, and release checks pass.

### 2. Manifesto Task Helpers

Status: completed in this pass.

Current file: `src/treepo/tasks/manifesto/replication.py`

Target shape:

- `documents.py`: dataclasses and packaged tiny corpus.
- `trees.py`: tree and `TreeRecord` conversion.
- `sampling.py`: document and qsentence sampling rows.
- `preferences.py`: root/unit preference records.
- `exports.py`: scoped reward-view export helpers.
- `prompts.py`: provider-neutral prompt and oracle predict helper.
- `common.py`: shared private propensity, root-label, sampling, and slug helpers.
- `replication.py` or `__init__.py`: compatibility facade.

Success criteria:

- Existing `treepo.tasks.manifesto` imports stay stable.
- Sampling propensities and preference output JSON remain byte-shape stable.
- Manifesto example smoke tests pass.

### 3. Methods-Wide Streamline Roadmap

Status: completed in this pass.

Scope rule: all methods cleanup is facade-only and behavior-preserving. Public
imports, family names, config keys, artifact keys, metric keys, manifest paths,
and preference export shapes stay stable.

Pass order:

1. Central fit/runtime path.
2. Preference boundary.
3. Neural operator/FNO internals.
4. Final methods-wide release checks.

Stable public surfaces:

- `treepo.fit(...)`
- `treepo.methods.learning.fit`
- `treepo.methods.runtime.IterationRecord`
- `treepo.methods.runtime.SplitMetrics`
- `treepo.methods.runtime.evaluate_splits`
- `treepo.methods.runtime.run_alternating_family`
- `treepo.methods.preference.*`
- `treepo.methods.fno.*`
- `treepo.methods.neural_operator.*`

### 4. Learning Result Assembly

Status: completed in this pass.

Current file: `src/treepo/methods/learning.py`

Target shape:

- `_fit_inputs.py`: family/runtime resolution, sequence coercion, objective
  resolution.
- `_preference_traces.py`: `PreferenceDataset` to f/g training trace
  conversion.
- `_fit_result.py`: `FitResult` assembly, final metric flattening, split-metric
  payloads.
- `_prediction_records.py`: prediction JSONL writing/collection.
- `_run_manifest.py`: methods manifest writing and JSON default handling.

Success criteria:

- `treepo.fit(...)` remains the single public learning surface.
- Result dicts, manifest paths, and artifact keys remain stable.
- Unified-contract and release-gate tests pass.

### 5. Alternating Runtime

Status: completed in this pass.

Current file: `src/treepo/methods/runtime.py`

Target shape:

- `_runtime_types.py`: runtime dataclasses.
- `_runtime_schedule.py`: stage names, labels, train-f/train-g schedule.
- `_runtime_evaluation.py`: split evaluation, prediction rows, scalar/vector
  metric math.
- `_runtime_statistics.py`: statistic/local-law payload extraction.
- `_runtime_loop.py`: alternating f/g orchestration.

Success criteria:

- `run_alternating_family(...)` reads as the internal training vignette.
- No changes to metric names or history payloads.
- Method-family tests pass.

### 6. Preference Boundary

Status: completed in this pass.

Current file: `src/treepo/methods/preference.py`

Target shape:

- `preference.py`: compatibility facade for public records, dataset, exports,
  and tree helpers.
- `_preference_core.py`: behavior-preserving private implementation core.
- `_preference_normalize.py`: row parsing, JSON coercion, optional scalar
  helpers.
- `_preference_views.py`: supervised/DPO/reward/GRPO projection helpers.
- `_preference_io.py`: dataset save/load and export file writing helpers.
- `_preference_tree.py`: `TreeRecord` to preference units and tree filtering.
- `_preference_adapters.py`: trainer adapter export wrapper.

Success criteria:

- `PreferenceDataset` behavior stays stable.
- Export file names and row shapes stay stable.
- Unified-contract, fine-tune view, and example tests pass.

### 7. Neural Operator Family

Status: completed in this pass.

Current file: `src/treepo/methods/fno.py`

Target shape:

- `fno.py`: compatibility facade for public classes and builders.
- `_neural_operator_core.py`: behavior-preserving private implementation core.
- `_fno_config.py`: config coercion and operator-kind normalization.
- `_fno_neuralop.py`: neuralop discovery, constructor kwargs, compatibility
  checks.
- `_fno_models.py`: torch model/leaf operator definitions.
- `_fno_encoding.py`: leaf text/token extraction and embedding/numeric
  encoding.
- `_fno_targets.py`: scalar/vector target extraction.
- `_fno_transition.py`: numeric transition state targets/loss/local-law rows.
- `_fno_statistic.py`: composable statistic wrapper.

Success criteria:

- Top-level `import treepo` stays light.
- Torch remains lazy.
- `family="fno"` stays the concrete FNO route over the shared neural-operator
  runtime.
- `family="neural_operator"` is the generic route for explicit operator
  selection, including `operator_kind="fno"` and the `operator_kind="fourier"`
  alias.
- FNO family tests and example smoke tests pass.

### 8. Excluded Application Hooks

Status: completed in this pass.

Review small stubs that exist only to redirect large workflows outside the
package. Prefer docs plus clear family-registration errors over extra public-ish
functions when possible.

Success criteria:

- Public docs explain the extension boundary.
- Release gates still pass.

## Addendum (2026-07-02): real decomposition of passes 6 and 7

Passes 6 (preference) and 7 (neural operator) were originally landed as
facade-only: the `_preference_*` and `_fno_*` files were pure re-export stubs
(`from _core import ...` + `__all__`, zero real defs), and all implementation
still lived in the 45KB `_preference_core.py` and 51KB `_neural_operator_core.py`
monoliths. That added indirection without improving readability.

These two passes are now genuinely decomposed along the data-prep vs.
model-logic seam (matching the fixtures/manifesto pattern), while keeping every
public and `_core` import surface stable via re-export hubs.

Neural operator (`_neural_operator_core.py` is now a lean family runtime +
re-export hub):

- `_fno_config.py`: config dataclasses, coercion, operator-kind normalization.
- `_fno_neuralop.py`: torch/neuralop discovery and validation.
- `_fno_models.py`: the f/g torch model and leaf operators.
- `_fno_encoding.py`: leaf extraction and embedding (data-prep).
- `_fno_targets.py`: supervision target extraction (data-prep).
- `_fno_transition.py`: numeric transition-state supervision (data-prep).
- `_fno_statistic.py`: composable-statistic adapter.

Preference (`_preference_core.py` is now a pure re-export hub):

- `_preference_normalize.py`: row parsing, scalar/JSON coercion (data-prep).
- `_preference_views.py`: supervised/DPO/reward/GRPO projections (data-prep).
- `_preference_dataset.py` (new): the `Candidate`/`PreferenceRecord`/
  `PreferenceDataset` data model.
- `_preference_io.py`: dataset IO, export, view previews.
- `_preference_tree.py`: `TreeRecord`-derived units and tree filtering.
- `_preference_adapters.py`: trainer-adapter export fan-out.

`PreferenceDataset.filter_tree` uses a deferred import of `filter_units_for_tree`
to keep the data-model/tree modules acyclic. `import treepo` stays torch- and
neuralop-free. Full suite: 162 passed, 1 skipped.
