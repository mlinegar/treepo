from __future__ import annotations

import json
from pathlib import Path

from treepo.bench.reports import lda_leafnoise, learned_g_overnight, publication_progress


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_lda_leafnoise_scan_smoke(tmp_path: Path) -> None:
    out_root = tmp_path
    p = out_root / "segmented_lda_ctreepo" / "equivalence" / "x" / "seed_0.json"
    _write_json(
        p,
        {
            "config": {
                "topic_process": "bag_of_words",
                "topic_phi_estimator": "sklearn_lda",
                "leaf_theta_estimator": "sklearn_lda",
                "eval_leaf_query_rate": 0.0,
                "eval_internal_query_rate": 0.0,
                "n_books_train": 16,
                "fixed_leaf_tokens": 32,
                "calibration_leaf_query_rate": 0.1,
                "seed": 0,
                "n_topics": 4,
                "vocab_size": 64,
                "min_segments": 1,
                "max_segments": 1,
                "min_seg_tokens": 32,
                "max_seg_tokens": 32,
                "alpha_topic": 0.2,
                "beta_word": 0.1,
                "segment_concentration": 80.0,
                "segment_background": 2.0,
            },
            "topic_meta": {"topic_phi_l2_error_mean": 0.5, "leaf_theta_l1_mean": 0.1, "corpus_signature_test": "abc"},
            "metrics": {"estimated_calibrated_budgeted": {"root_l1_mean": 0.25}},
        },
    )
    rows = lda_leafnoise._scan(out_root)  # type: ignore[attr-defined]
    assert len(rows) == 1


def test_publication_progress_scan_smoke(tmp_path: Path) -> None:
    out_root = tmp_path
    p = (
        out_root
        / "segmented_lda_ctreepo"
        / "equivalence"
        / "lda"
        / "k8_v512"
        / "lane_lda_direct"
        / "tp_bag_of_words"
        / "phi_sklearn_lda"
        / "train_128"
        / "lt_32"
        / "cal_0p1"
        / "leaf_0"
        / "int_0"
        / "seed_0.json"
    )
    _write_json(
        p,
        {
            "config": {
                "n_books_train": 128,
                "fixed_leaf_tokens": 32,
                "calibration_leaf_query_rate": 0.1,
                "eval_leaf_query_rate": 0.0,
                "eval_internal_query_rate": 0.0,
                "seed": 0,
            },
            "topic_meta": {"topic_phi_l2_error_mean": 0.5},
            "metrics": {"estimated_calibrated_budgeted": {"root_l1_mean": 0.25}},
        },
    )
    rows = publication_progress._scan_rows(out_root)  # type: ignore[attr-defined]
    assert len(rows) == 1
    assert rows[0].regime == "lda"
    assert rows[0].lane.startswith("lane_")


def test_learned_g_overnight_scan_smoke(tmp_path: Path) -> None:
    out_root = tmp_path
    p1 = out_root / "round_000" / "learned-segment-lda-ops-g" / "c00" / "seed_0" / "summary.json"
    p2 = out_root / "round_000" / "learned-segmented-lda-theta-g" / "c00" / "seed_0" / "summary.json"

    _write_json(
        p1,
        {
            "metrics": {
                "root_mae": 0.9,
                "merge_mae": 0.3,
                "schedule_spread_mean": 1.2,
                "leaf_mae": 0.5,
                "schedule_spread_p95": 2.0,
                "leaf_violation_rate": 0.1,
                "merge_violation_rate": 0.2,
            }
        },
    )
    _write_json(
        p2,
        {
            "metrics": {
                "root_mae": 0.2,
                "merge_mae": 0.1,
                "schedule_spread_mean": 0.3,
                "leaf_mae": 0.25,
                "schedule_spread_p95": 0.5,
                "leaf_violation_rate": 0.05,
                "merge_violation_rate": 0.05,
            }
        },
    )

    rows = learned_g_overnight._scan_runs(out_root)  # type: ignore[attr-defined]
    assert len(rows) == 2
    assert {r.experiment for r in rows} == {"learned-segment-lda-ops-g", "learned-segmented-lda-theta-g"}

    rc = learned_g_overnight.main(
        [
            "--output-root",
            str(out_root),
            "--out-dir",
            str(out_root / "figures" / "learned_g_overnight"),
            "--no-emit-pdf",
        ]
    )
    assert rc == 0
    assert (out_root / "figures" / "learned_g_overnight" / "learned_g_overnight_latest.md").exists()
    assert (out_root / "figures" / "learned_g_overnight" / "learned_g_overnight_latest_diagnostics.json").exists()
