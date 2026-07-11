#!/usr/bin/env python3
"""Build sampled local-law evidence and a compact certificate ledger.

This source-tree example is deliberately small: it does not train a model.
It shows the public evidence ingredients that a real run should persist:

* root-level metrics,
* unit/candidate preference exports,
* statistic metadata,
* sampled C1/C2/C3 local-law audit rows with logged propensities,
* a two-channel component-radius certificate ledger.

For partially observed trees, ``build_triangle_local_law_error_certificate``
maps the audited C1/C2/C3 transport residual, document-level root controls,
overidentification, and optional hidden-degradation envelopes into the ledger.
Root-share identification weights belong in ``node_weight`` metadata/objective
weights, not in logged propensities.
"""

from __future__ import annotations

from example_setup import (
    parse_output_dir,
    toy_certificate_preferences,
    toy_error_certificate,
    toy_local_law_rows,
    toy_root_metrics,
    toy_statistic_artifact,
    write_json,
    write_local_law_rows,
)


def main() -> int:
    from treepo.evidence import build_evidence
    from treepo.local_law import audit_local_laws
    from treepo.methods.preference import export_preference_records

    output_dir = parse_output_dir()
    rows = toy_local_law_rows()
    rows_path = output_dir / "sampled_local_law_rows.jsonl"
    write_local_law_rows(rows_path, rows)

    audit_dir = output_dir / "local_law_audit"
    audit = audit_local_laws(rows, output_dir=audit_dir)
    preferences = toy_certificate_preferences()
    preference_artifacts = export_preference_records(preferences, output_dir / "preference")
    statistic_artifact = toy_statistic_artifact(
        audit=audit,
        rows_path=rows_path,
        audit_dir=audit_dir,
        row_count=len(rows),
    )
    root_metrics = toy_root_metrics()
    summary = {
        "family": "example",
        "schedule": "evidence_only",
        "n_iterations": 0,
        "output_dir": str(output_dir),
        "split_metrics": {"all": root_metrics},
    }
    artifacts = {
        "preference_data": preference_artifacts,
        "statistic": statistic_artifact,
        "local_laws": {
            "summary": audit["local_law_objective"],
            "by_law_kind": audit["by_law_kind"],
            "source": "sampled_rows",
            "files": statistic_artifact["files"],
        },
    }
    evidence = build_evidence(
        status="success",
        metrics=root_metrics,
        summary=summary,
        artifacts=artifacts,
        local_law_rows=rows,
    )
    certificate = toy_error_certificate(root_metrics=root_metrics, audit=audit)

    evidence_path = output_dir / "evidence.json"
    certificate_path = output_dir / "certificate.json"
    result_path = output_dir / "local_law_certificate_result.json"
    write_json(evidence_path, evidence)
    write_json(certificate_path, certificate.to_dict())
    write_json(
        result_path,
        {
            "audit": audit,
            "certificate": certificate.to_dict(),
            "evidence": evidence,
            "files": {
                "rows": str(rows_path),
                "audit_summary": str(audit_dir / "audit_summary.json"),
                "evidence": str(evidence_path),
                "certificate": str(certificate_path),
            },
            "preferences": preference_artifacts,
            "statistic": statistic_artifact,
        },
    )

    objective = float(audit["local_law_objective"]["objective"])
    observed = int(audit["local_law_objective"]["observed_count"])
    total = int(audit["local_law_objective"]["row_count"])
    print(
        "status=success family=example "
        f"rows={total} observed={observed} "
        f"local_law_objective={objective:.6f} "
        f"certificate_radius={certificate.radius_sum:.6f} output={result_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
