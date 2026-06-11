# Execution Roadmap

## Suggested Pre-Push Sequence

1. Fix docs/package claims:
   - README extras vs `pyproject.toml`;
   - core dependency claim;
   - stale `docs/training_defaults.md` paths;
   - `_optional.py` decision;
   - `learning.py` public/repoint/delete decision.
2. Fix HLL defaults:
   - precision `12 -> 14`;
   - fixture `leaf_token_count 12 -> 24`;
   - keep HLL sketch defaults in research configs.
3. Replace observed-only public overlap with all-row Lean-compatible overlap.
   Done in `treepo.local_law`.
4. Fix row propensities in manifest and HLL generated rows.
   Done on the public/local-law paths.
5. Collapse duplicate audit/local-law schemas.
   Done by deleting the old audit/local-law shim modules.
6. Rename scalar HLL `lean_*` fields for v0.1.
7. Add invariant checks:
   - `0 <= gamma_depth <= 1`;
   - nonzero law weights when local laws are enabled;
   - convex objective mixture or explicit non-convex opt-in;
   - DSPy two-leaf context budget;
   - FNO channel shape.
8. Bring `src/treepo/methods/` into the release gate.
9. Add lazy-export validation, core-light coverage, top-level
   `NormalizedOutput`, README public-surface note, and real CHANGELOG.
10. Document the Lipschitz and measurement certificate gap.

## Verification Checklist

- `treepo.run("sketch", ...)` is no longer a public package path.
- Classical-sketch oracle auto-fixture resolves `leaf_token_count=24`.
- One canonical `LocalLawAuditRow` exists on the public path.
- One canonical `compute_influence_weighted_overlap` exists and computes
  all-row `D_lambda` / `W_lambda`.
- Observed-only sample ESS, if retained, has an explicitly diagnostic name.
- No manifest or audit row has `propensity == 0.0`.
- No public field is named `lean_*` unless it is backed by the relevant
  state-level law check.
- `certificate.py` and `objective.py` document or instantiate Lipschitz and
  measurement terms.
- `ObjectiveSpec(local_law_estimator=..., local_law_component_weights={})`
  raises when local laws are enabled.
- `ObjectiveSpec(root_share=0.8, local_law_weight=2.0)` raises unless an
  explicit non-convex opt-in exists.
- `_depth_weight(..., gamma_depth=1.5)` raises.
- DSPy construction raises when two-leaf concat budget exceeds context.
- FNO construction raises when f/g channel counts violate the paper invariant.
- `python -m treepo.release` passes with methods included.
- Every lazy top-level export resolves in the release gate.
- `from treepo import NormalizedOutput` works if top-level `ExperimentContext`
  is public.
- README install commands work as written.

## Large Bench Scope

The LDA bench modules are not minimal-release public API. They are now
quarantined as research code:

- `src/treepo/_research/bench/lda/segment_lda_ops_weight_recovery.py`
- `src/treepo/_research/bench/lda/segmented_lda_ctreepo.py`

v0.1 status:

- Public `treepo-bench` no longer advertises or dispatches LDA experiments.
- LDA configs, examples, reports, and runner scripts are under research-only
  paths.

v0.2 work:

- Split into config, data generation, estimators, evaluation, and runner.
- Add per-estimator E2E tests.

## v0.2 Roadmap

1. Instantiate Lipschitz and measurement-error certificate terms.
2. Add HLL state-level register-max C3 certification.
3. Split `methods/dispatch.py` into registry and handler modules.
4. Extract common bench data generation.
5. Finish removing test-only `sys.path` setup where standalone script fixtures
   no longer need source-tree execution.
