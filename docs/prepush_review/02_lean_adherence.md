# Lean Adherence

## Verdict

The structural skeleton is faithful:

- C1/C2/C3 exist.
- Corrected local-law / AIPW arithmetic exists.
- Depth discounting exists.
- Unified certificate radii exist.

The theorem-completing pieces still outside v0.1 are:

- Lipschitz and measurement-error constants are absent from the certificate;
- HLL learned diagnostics are scalar readout checks, not state-level register
  law certificates.

## Local-Law Crosswalk

| Paper | Lean | Meaning | Python status |
|---|---|---|---|
| C1 | L1 | leaf preservation | present |
| C2 | L3 | idempotence / on-range inertness | present |
| C3 | L2 | merge preservation | present |

Status:

- fixed for v0.1. `treepo.local_law` owns the alias map and row schema.

Keep:

- Define one alias map and import it everywhere.

## Corrected Local-Law Arithmetic

Formula is present in one scalar module plus tensor wrappers:

- `src/treepo/local_law.py`
- `src/treepo/training/local_law.py`

Status:

- fixed for v0.1. Scalar helpers live in `treepo.local_law`; training imports
  them and adds torch-specific wrappers.

## Discounted Tree Objective

Lean bracket theorems use `0 <= gamma <= 1`.

Python only rejects `gamma < 0`.

Fix:

- Reject `gamma_depth > 1` wherever users can supply it.

## Influence-Weighted Root Control

Lean consumes an influence-weighted residual mass over all audit rows.

Status:

- fixed for v0.1. `treepo.local_law` is canonical and public.
- observed-only metrics are diagnostics.

## Unified Certificate

Python matches the additive certificate shell:

```text
abs(reported_estimate)
+ local_law_radius
+ calibration_radius
+ estimation_radius
+ clipping_radius
```

But it does not build theorem-complete component radii. In particular, it lacks
first-class Lipschitz readout and measurement-error terms.

Fix:

- v0.1: document the gap.
- v0.2: instantiate the missing terms.
