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
  "run": {"family": "neural_operator", "schedule": "fg", "g_mode": "learned", "g_contract": {"operator": "learned_shared", "fit_status": "fitted_this_run", "learned_this_run": true, "g_update_count": 1}, "status": "success", "n_iterations": 3, "output_dir": "..."},
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

- **root** — root-level scalar or named-vector evidence, mirroring
  `FitResult.metrics` and `FitResult.summary["split_metrics"]`. This is the
  common result lane for every family: neural operators, sketches, LLM/DSPy wrappers, and oracles.
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
  `effective_propensity`, `node_weight`, `depth`, and metadata. C1/C2/C3-
  targeted evidence lives here. Rows should identify whether they are semantic
  witnesses, separating probes, or readout proxies; the law tag alone does not
  establish the relational law.
- **predictions** — pointers to prediction record files plus compact counts.
  Large row payloads stay on disk; the artifact records where they live.

## Representation and operator status

Evidence must not infer a learned `g` merely because the family protocol has a
`train_g(...)` method or because a `g` slot exists in an artifact.
`run.g_mode` is the configured policy (`identity`, `fixed`, or `learned`);
`run.g_contract` records the realized outcome:

Topology is recorded independently. For a binary C-Tree, `L >= 1` and
`M = L - 1`; the derived path is `full_doc_direct`,
`ctree_base_summary`, or `ctree_recursive`. `full_doc` is full-span singleton
geometry and does not imply identity, while `ctree` includes the singleton
base case. A grid's multi-leaf `ctree` arm is a narrower experimental choice,
not the package definition.

`reduce_g` is not an independently learned object. It is the recursive fold
induced by the single recorded `g` artifact:
`reduce_g(Leaf(b)) = g(b)` and
`reduce_g(Node(T_L,T_R)) = g(reduce_g(T_L) concat reduce_g(T_R))`.
Any leaf-call and internal-call counts describe support on which that one
operator was exercised; they do not identify separate parameters, artifacts,
or learners.

- `operator=fixed_identity` for an identity/elided state mechanism, independent
  of whether the declared topology has zero or positive merge applications;
- `operator=fixed_nonidentity_or_family_owned` for an explicit concrete
  deterministic/frozen operator supplied under `g_mode=fixed`;
- `operator=learned_shared` only when `learned_this_run=true` and
  `g_update_count>0`; and
- `operator=trainable_not_updated_this_run` when `g_mode=learned` but this
  fit executed no `g` update. A warm-start artifact may have prior training
  provenance, so `initial_g_artifact_present` remains explicit.

`train_g_call_count` counts attempts; `g_update_count` counts realized
updates. When artifact equality cannot reveal an in-place update, a family
returns `treepo.methods.GTrainOutcome` as the authoritative update signal.
A no-op call never licenses `learned_shared`. Fixed artifacts are concrete,
nontrainable, family-validated operators; identity is package-owned.

Missing mode provenance is `undeclared`, never silently `learned`.

A singleton C-Tree has one leaf and zero merge applications, regardless of
`g_mode`. `full_doc_direct` uses `f(X)` and may elide the identity leaf
invocation. `ctree_base_summary` uses `f(g(X))`: its sole nonidentity `g` call
has a realized C1 population, while its realized C3 population is empty. That
empty stratum is structural absence and must not be read as a C3 pass.

A realized singleton `g` update is evidence that the shared operator was
updated from leaf-call support; it is not learned-composition or internal-call
evidence. Only a `ctree_recursive` run that exercises the updated `g` at
internal nodes and carries appropriate C3 evidence can support that narrower
claim. Fixed analytic rows may diagnose a control, but they do not become
evidence about a learned operator. The operator status, derived execution
path, leaf count, merge-application count, and call-stratum evidence
populations belong in run provenance whenever a downstream report compares
these cells.

The generic provider-neutral `llm` family does not optimize `g`; its fixed
text-state/concatenation behavior and any artifact-only `train_g(...)` call are
not update evidence. The optimizer-backed `dspy` family learns one shared `g`
program for leaf and merge roles. Even there, configured trainability is not
evidence: the run must return an explicit realized update outcome.

`K=1`, `K=3`, and `K=57` retain the same ordered named-vector prediction rows
and sum-L1 metric. Mirroring the sole K=1 coordinate into scalar compatibility
fields does not change its evidence population or estimand.

Examples read these same evidence fields, so every run in the package shares
one summary vocabulary.
`examples/methods/run_local_law_certificate.py` is the smallest end-to-end
walkthrough: it builds sampled C1/C2/C3-targeted rows, audits them, and writes
this evidence shape plus a component-radius ledger.

## Two-channel partial-observation certificates

`treepo.certificate.build_two_channel_error_certificate(...)` packages the
partially observed tree vocabulary into the existing component-radius ledger:
leaf-up residuals map to `local_law_radius`, root-down aggregate residuals map
to `calibration_radius`, and overidentification plus conditional-average
envelopes map to `estimation_radius` with semantic metadata.

For local-law based runs, prefer
`treepo.local_law.triangle_local_law_residual_from_audit(...)` or
`treepo.local_law.build_triangle_local_law_error_certificate(...)`. These
helpers package the audited objective and a caller-supplied
leaf-up/merge-triangle bound into the two-channel ledger. They do not derive
semantic task sufficiency, a concentration radius, or local-to-root transport
from the raw point objective.

There are two non-additive envelope adapters:

- `CommonMechanismEnvelopeEvidence` uses observed root-error bounds. Under the
  explicit assumptions that the same `f` and `g` are used at roots and internal
  nodes, and that a separately justified local-law transport controls those
  calls through a common mechanism, it
  records `amplification * observed_root_radius + slack` as a hidden-degradation
  radius. In the emitted metadata, `transport_source:
  merge_triangle_local_laws` is a caller assertion invoking a supplied
  local-law transport; it is not derived from metadata or point rows. Meanwhile,
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
