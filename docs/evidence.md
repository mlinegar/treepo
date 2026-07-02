# Evidence Artifact

`treepo.evidence.build_evidence(...)` assembles one compact, JSONable evidence
artifact per run from objects the run already produced: root metrics,
preference exports, statistic metadata, local-law summaries, and prediction
records. `treepo.fit(...)` calls it while assembling the result and stores the
artifact at `result.artifacts["evidence"]`.

## Shape

```json
{
  "version": "0.1",
  "run": {"family": "neural_operator", "schedule": "fg", "status": "success", "n_iterations": 2, "output_dir": "..."},
  "root": {"metrics": {}, "split_metrics": {}},
  "preferences": {"present": true, "summary": {}, "counts": {}, "files": {}},
  "statistic": {"present": true, "info": {}, "local_law_summary": {}},
  "local_laws": {"present": true, "summary": {}, "by_law_kind": {}},
  "predictions": {"present": true, "files": []}
}
```

Every section is always present; a section carries `present: false` when the
run produced no evidence of that kind. Downstream readers branch on the flag.

## Semantics

- **root** — scalar root-level evidence, mirroring `FitResult.metrics` and
  `FitResult.summary["split_metrics"]`. This is the common result lane for
  every family: neural operators, sketches, LLM/DSPy wrappers, and oracles.
- **preferences** — describes the `PreferenceDataset` supplied to the run and
  the projection files exported by `export_preference_records(...)`. Root
  labels, node labels, scored candidates, pairwise preferences, and ranked
  groups are all projections of the same dataset.
- **statistic** — executable-state metadata when a family exposes a
  `ComposableStatistic`. Exact sketches and trained neural operators fill this
  section; LLM/DSPy wrappers fill it when a downstream task supplies an
  executable task-specific statistic.
- **local_laws** — summaries produced by
  `treepo.local_law.audit_local_laws(...)`, using the `LocalLawAuditRow`
  vocabulary throughout: `law_kind`, `observed`, `propensity`,
  `effective_propensity`, `node_weight`, `depth`, and metadata. C1/C2/C3
  evidence lives here.
- **predictions** — pointers to prediction record files plus compact counts.
  Large row payloads stay on disk; the artifact records where they live.

Examples read these same evidence fields, so every run in the package shares
one summary vocabulary.
`examples/methods/run_local_law_certificate.py` is the smallest end-to-end
walkthrough: it builds sampled C1/C2/C3 rows, audits them, and writes this
evidence shape plus a component-radius certificate ledger.
