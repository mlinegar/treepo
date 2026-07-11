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

## Two-channel partial-observation certificates

`treepo.certificate.build_two_channel_error_certificate(...)` packages the
partially observed tree vocabulary into the existing component-radius ledger:
leaf-up residuals map to `local_law_radius`, root-down aggregate residuals map
to `calibration_radius`, and overidentification plus conditional-average
envelopes map to `estimation_radius` with semantic metadata.

For local-law based runs, prefer
`treepo.local_law.triangle_local_law_residual_from_audit(...)` or
`treepo.local_law.build_triangle_local_law_error_certificate(...)`. These
helpers connect the audited C1/C2/C3 objective to the leaf-up/merge-triangle
transport channel and then route document-level root controls through the same
two-channel ledger.

There are two non-additive envelope adapters:

- `CommonMechanismEnvelopeEvidence` uses observed root-error bounds. Under the
  explicit assumptions that the same `f` and `g` are used at roots and internal
  nodes, and that local laws transport those calls into a common mechanism, it
  records `amplification * observed_root_radius + slack` as a hidden-degradation
  radius. In the emitted metadata, `transport_source:
  merge_triangle_local_laws` means the caller is invoking the same C1/C2/C3
  local-law transport that underwrites the merge-triangle proposition, while
  `root_control_source: audit_bound` means the root-control premise is the
  audited local-distortion bound.
- `ConditionalAverageEnvelopeEvidence` records a one-sided radius supplied by an
  external Bayesian/MRP or small-area workflow. Diagnostics
  (`posterior_predictive_fit`, `psis_loo_stable`, and `rank_calibrated`) are
  explicit assumptions; by default the builder rejects an envelope unless all
  three are true.

The package does not fit a Bayesian multilevel or MRP model and does not derive
a Gelman-style bound from data. It can, however, use document-level observations
as a worst-case common-mechanism envelope when the caller asserts the shared
`f,g` and local-law transport conditions.
