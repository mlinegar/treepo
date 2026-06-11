# Release Blockers

These should be fixed before pushing v0.1.0 because they affect theorem
alignment, reproducibility, or public API honesty.

## 1. Audit Overlap Must Be All-Row

Lean defines influence design effect over the full finite audit row space:

- `FormalProofs/OPT/InfluenceWeightedLocalLaws.lean:89`:
  `sum_a lambda(a)^2 / pi(a)`.
- `FormalProofs/OPT/InfluenceWeightedLocalLaws.lean:104`: every row has
  `0 < pi(a)`.
- `FormalProofs/OPT/InfluenceWeightedLocalLaws.lean:211`: root error is
  controlled by influence-weighted residual mass.

Status:

- fixed for v0.1. Public exports now come from `treepo.local_law`.
- `compute_influence_weighted_overlap` includes unobserved rows in `D_lambda`
  and `W_lambda`.
- observed-only ESS remains only as `observed_effective_sample_size`.

## 2. Row Propensity Semantics Are Wrong

Lean separates design propensity from the observation indicator.

Status:

- fixed for v0.1 on the public manifest/local-law paths.
- HLL generated local-law rows use positive design propensities.

Fix:

1. `propensity` is the design inclusion probability and must be positive for
   every row.
2. `observed` is the binary sampled/not-sampled indicator.
3. `effective_propensity` is the clipped numerical value, with clipping metadata
   if needed.

## 3. Duplicate Audit Implementations Conflict

Status:

- fixed for v0.1.
- `src/treepo/local_law.py` is the single scalar implementation.
- `src/treepo/training/local_law.py` keeps tensor wrappers only.
- the old audit/local-law shim modules were deleted.

## 4. Canonical Defaults Are Not Wired Through Factories

Paper-canonical defaults:

- HLL sketch defaults are now research-only.

Factory fallbacks currently use older values:

- HLL sketch precision fallback is no longer public API.
- Classical-sketch oracle fixture `leaf_token_count` fallback is `12`.

Fix:

1. Keep HLL sketch defaults under `configs/research/`.
2. Avoid public-path tests for `treepo.run("sketch", ...)`; that method no longer exists.
   and oracle auto-fixtures.

## 5. Certificate Omits Lean Bound Terms

`src/treepo/certificate.py` mirrors the additive certificate shell, but it does
not instantiate:

- `L * epsilon_fiber + 2 * epsilon_readout` from
  `LipschitzReadoutFactorization.lean`.
- `(L1 * K * budget) + L2 * dist(featureHat, feature)` from
  `TheoremBackingApproxMeasurementError.lean`.

Fix for v0.1:

- Document clearly in `certificate.py`, `objective.py`, README, and CHANGELOG
  that the emitted certificate is a component-radius ledger unless the user has
  already included those theorem constants in the supplied radius.

Fix for v0.2:

- Add first-class Lipschitz and measurement-error certificate components.

## 6. HLL `lean_*` Fields Overclaim

The learned HLL path trains scalar readout loss via differentiable HLL
estimates. Lean explicitly warns that scalar readout equality does not imply
register-state merge exactness.

Fix for v0.1:

- Rename `lean_adjusted_loss`, `lean_merge_adapter`, and
  `lean_projection_target` to `scalar_adjusted_loss`,
  `scalar_merge_adapter`, and `scalar_projection_target`.
- Add a docstring citing `ClassicalSketchLocalLaws.lean`.

Fix for v0.2:

- Add state-level register-max C3 residuals and reserve `lean_*` naming for
  state-level checks.

## 7. Proof-Side Invariants Are Not Enforced

Add Python checks for:

- `0 <= gamma_depth <= 1`.
- local-law estimator enabled implies at least one positive law component.
- objective weights form the promised convex combination, or require an
  explicit non-convex opt-in.
- DSPy two-leaf concat budget fits the context window.
- FNO `f`/`g` channel counts match the paper invariant.
