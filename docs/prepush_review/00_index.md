# Systematic Pre-Push Review Packet

Date: 2026-05-22

Scope: `/home/mlinegar/treepo`, reviewed as the minimal public TreePO/C-TreePO
package before first push.

This packet consolidates:

- the direct review against Lean contracts in
  `/home/mlinegar/ThinkingTrees/lean3/FormalProofs`;
- the prior combined review packet;
- useful additions from an independent v0.1.0 pre-push review packet.

## Files

- [01_release_blockers.md](01_release_blockers.md): pre-push blockers and
  theorem-facing correctness issues.
- [02_lean_adherence.md](02_lean_adherence.md): Lean-to-Python mapping and
  certificate gaps.
- [03_defaults_public_api_release_gate.md](03_defaults_public_api_release_gate.md):
  paper defaults, docs, packaging, public API, and release-gate coverage.
- [04_streamlining_and_duplication.md](04_streamlining_and_duplication.md):
  duplicate logic and simplification targets.
- [05_execution_roadmap.md](05_execution_roadmap.md): surgical sequence,
  verification checklist, and v0.2 follow-ups.
- [../package_minimization_status.md](../package_minimization_status.md):
  current package boundary, optional extras, verification snapshot, and
  remaining minimization decisions.

## Executive Summary

Fix before push:

1. Make audit overlap and row propensities match Lean: all rows have positive
   design propensities; `observed` is a separate sampling indicator.
2. Resolve the duplicate audit/local-law implementations around the all-row
   Lean-compatible semantics. This is now done through `treepo.local_law`;
   the old audit shim was deleted after its semantics were preserved.
3. Wire canonical defaults through public factories: HLL precision `14`, HLL
   fixture `leaf_token_count=24`.
4. Either instantiate or clearly document the missing Lipschitz and measurement
   terms in the Python certificate.
5. Rename scalar HLL `lean_*` fields for v0.1 or add state-level register-law
   certification.
6. Enforce proof-side invariants in Python: `0 <= gamma <= 1`, nonzero enabled
   law weights, objective mixture constraints, DSPy two-leaf context budget, and
   FNO channel shape.
7. Correct README/pyproject/docs contradictions, bring methods into the release
   gate, and validate every lazy public export.

Prior focused tests passed before these docs were written:

```bash
.venv/bin/python -m pytest -q \
  tests/test_unified_contracts.py \
  tests/methods/test_local_law_audit.py \
  tests/training/test_local_law.py \
  tests/methods/test_error_estimation.py \
  tests/methods/test_canonical_defaults_drift.py \
  tests/test_release_gates.py
```

Result: `74 passed, 1 skipped` at packet creation; later release checks cover
the corrected all-row overlap behavior.
