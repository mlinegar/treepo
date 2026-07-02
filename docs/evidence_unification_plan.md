# Evidence Unification Plan

`treepo.fit(...)` returns one `FitResult`. The evidence story uses the same
shape: each run records one compact, JSONable evidence artifact that gathers
root metrics, preference data, statistic metadata, local-law diagnostics, and
prediction records without adding a second framework.

## Goal

Use one evidence shape for all built-in families and examples:

```python
result.artifacts["evidence"]
```

This artifact is a read-only summary of data already produced by the run. It is
not a trainer, dataset type, fit selector, or benchmark framework.

## Current Inputs

The package produces the pieces the evidence artifact needs.

- Root metrics live in `FitResult.metrics` and split metrics live in
  `FitResult.summary["split_metrics"]`.
- Preference supervision and preference exports live under
  `FitResult.artifacts["preference_data"]` when `preference_data` is supplied.
- Executable or diagnostic state metadata lives under
  `FitResult.artifacts["statistic"]` when a family exposes a
  `ComposableStatistic`.
- Prediction rows live under `FitResult.artifacts["prediction_records"]` when
  a fit run writes predictions.
- Local-law summaries can already be computed by `treepo.local_law.audit_local_laws(...)`.

`treepo.evidence.build_evidence(...)` names these pieces consistently and `treepo.fit(...)` stores the assembled artifact at `result.artifacts["evidence"]`.

## Evidence Shape

```json
{
  "version": "0.1",
  "run": {
    "family": "neural_operator",
    "schedule": "fg",
    "status": "success",
    "n_iterations": 2,
    "output_dir": "..."
  },
  "root": {
    "metrics": {},
    "split_metrics": {}
  },
  "preferences": {
    "present": true,
    "summary": {},
    "counts": {},
    "files": {}
  },
  "statistic": {
    "present": true,
    "info": {},
    "local_law_summary": {}
  },
  "local_laws": {
    "present": true,
    "summary": {},
    "by_law_kind": {}
  },
  "predictions": {
    "present": true,
    "files": []
  }
}
```

Sections are always present. Each section has `present: false` when the run did
not produce that evidence kind. This keeps downstream code simple and makes it
clear when a family has no concrete statistic or no local-law rows.

## Semantics

### Root

`root` contains scalar root-level evidence. It mirrors `FitResult.metrics` and
`FitResult.summary["split_metrics"]`. This is the common result lane for all
families, including neural operators, sketches, LLM/DSPy wrappers, and oracles.

### Preferences

`preferences` describes the `PreferenceDataset` supplied to the run and the
projection files exported by `export_preference_records(...)`. It does not add a
new supervision concept. Root labels, node labels, scored candidates, pairwise
preferences, and ranked groups remain projections of `PreferenceDataset`.

### Statistic

`statistic` describes executable state when a family exposes a
`ComposableStatistic`. Exact sketches and trained neural operators can fill this
section. LLM/DSPy wrappers can leave it absent unless a downstream task exposes
an executable task-specific statistic.

### Local Laws

`local_laws` contains summaries produced by `audit_local_laws(...)`. It should
use the same `LocalLawAuditRow` vocabulary everywhere: `law_kind`, `observed`,
`propensity`, `effective_propensity`, `node_weight`, `depth`, and metadata.

This section is the right place for C1/C2/C3 evidence. It should not be hidden
inside a benchmark-only report.

### Predictions

`predictions` points to prediction record files and can include compact counts.
Large row payloads stay on disk; the evidence artifact records where they are.

## Current Code Surface

`treepo.evidence.build_evidence(...)` assembles JSONable dictionaries from existing objects. `treepo.methods.learning._build_result(...)` calls it after metrics, preference artifacts, statistic artifacts, and prediction records are known. The returned `FitResult` includes `artifacts["evidence"]`.

Examples should read the same evidence fields rather than inventing their own summary vocabulary.

## Next Implementation Steps

1. Add a focused local-law example:

```text
examples/methods/run_local_law_certificate.py
```

This example should build a small set of C1/C2/C3 `LocalLawAuditRow` records,
call `audit_local_laws(...)`, write the same evidence shape without needing a
model service, and include the component-radius certificate ledger.

2. Extend tests for fit-level evidence coverage:

- `treepo.fit(...)` returns `artifacts["evidence"]` for LDA and Manifesto smoke examples.
- The local-law example writes the same evidence shape.
- Release checks reject examples that create separate evidence vocabularies.

## Example Mapping

| Example | Evidence that should be present |
| --- | --- |
| HLL sketch | root metrics, statistic, exact local-law summary |
| Markov neural operator | root metrics, preferences, learned statistic metadata |
| LDA neural operator | root metrics, preferences if supplied, statistic metadata |
| Manifesto | root metrics, preferences for qsentence or document-unit supervision |
| Local-law certificate example | local-law summary, per-law-kind diagnostics, and certificate components |

## Non-Goals

- No new fit selector.
- No new dataset type.
- No new benchmark command.
- No automatic oracle sampling runtime.
- No attempt to make LLM/DSPy wrappers expose an executable statistic by default.
- No large row payloads embedded directly in `FitResult`.

## Open Questions

1. Should `local_law_rows` be passed through `backend_config`, returned by a
   family, or accepted as a top-level `fit` config key? The conservative first
   step is to keep direct calls to `audit_local_laws(...)` and only attach rows
   when they are already available from a statistic.
2. Should evidence also be written to `evidence.json` in the run output
directory? This is useful for examples and release artifacts, and it keeps
large downstream runners from needing to import Python objects.
3. Should `treepo-bench` rows include a compact evidence summary column? This
   is useful, but only after `FitResult.artifacts["evidence"]` is stable.

## Suggested Next Code Pass

Add `examples/methods/run_local_law_certificate.py`, write its output through the evidence shape, and add a smoke test that verifies C1/C2/C3 summaries appear under `local_laws.by_law_kind`. After that, update LDA and Manifesto smoke tests to assert the evidence sections they already produce.
