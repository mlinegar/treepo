"""Optional full Manifesto Project sampling integration test.

This test uses the real local Manifesto Project CSVs from the companion
ThinkingTrees checkout. It is skipped by default because the corpus has more
than two million qsentence rows and is not part of the installed package.
"""

from __future__ import annotations

import csv
import os
import random
import sys
from collections import Counter
from pathlib import Path

import pytest

from treepo.sampling import ObservationUnitKind, SamplingMetadata

DEFAULT_MANIFESTO_PROJECT_ROOT = Path("data/raw/manifesto_project_full")
RUN_FULL_MANIFESTO = os.environ.get("TREEPO_RUN_MANIFESTO_PROJECT_FULL") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_FULL_MANIFESTO,
    reason=(
        "set TREEPO_RUN_MANIFESTO_PROJECT_FULL=1 to run the full local "
        "Manifesto Project CSV integration test"
    ),
)


def test_full_manifesto_project_corpus_supports_known_propensity_sampling() -> None:
    root = Path(os.environ.get("TREEPO_MANIFESTO_PROJECT_ROOT", DEFAULT_MANIFESTO_PROJECT_ROOT))
    corpus_path = root / "manifesto_corpus_df.csv"
    main_path = root / "manifesto_maindataset.csv"
    if not corpus_path.exists() or not main_path.exists():
        pytest.skip(f"Manifesto Project CSVs not found under {root}")

    main_ids, n_main_rows, n_main_with_rile = _read_manifesto_main_ids(main_path)
    counts_by_doc, n_qsentences, n_coded, n_uncoded = _read_corpus_counts(corpus_path)
    doc_ids = sorted(counts_by_doc)

    assert n_main_rows == 5285
    assert n_main_with_rile == 5186
    assert len(main_ids) == 5285
    assert len(doc_ids) == 3327
    assert n_qsentences == 2_271_892
    assert n_coded == 2_170_571
    assert n_uncoded == 101_321
    assert set(doc_ids) <= main_ids
    assert sum(counts_by_doc.values()) == n_qsentences
    assert min(counts_by_doc.values()) == 1
    assert max(counts_by_doc.values()) == 11_587

    doc_sample_size = 128
    qsentence_sample_size = 16
    sampled_doc_ids = set(random.Random(17).sample(doc_ids, doc_sample_size))
    document_propensity = doc_sample_size / len(doc_ids)
    document_rows = [
        {
            "doc_id": doc_id,
            "observed": doc_id in sampled_doc_ids,
            "inclusion_probability": document_propensity,
        }
        for doc_id in doc_ids
    ]
    assert len(document_rows) == len(doc_ids)
    assert sum(1 for row in document_rows if row["observed"]) == doc_sample_size

    sampled_qsentence_population = sum(counts_by_doc[doc_id] for doc_id in sampled_doc_ids)
    observed_qsentences = sum(
        min(qsentence_sample_size, counts_by_doc[doc_id])
        for doc_id in sampled_doc_ids
    )
    assert sampled_qsentence_population > observed_qsentences > doc_sample_size

    # Check the two-stage DSL propensities over real document sizes without
    # materializing a multi-million-row qsentence sidecar.
    for doc_id in sorted(sampled_doc_ids)[:32]:
        n_units = counts_by_doc[doc_id]
        unit_propensity = min(qsentence_sample_size, n_units) / n_units
        sampling = SamplingMetadata(
            document_propensity=document_propensity,
            unit_propensity=unit_propensity,
            label_propensity=1.0,
            joint_propensity=document_propensity * unit_propensity,
            sampling_scheme="uniform_without_replacement",
            policy_name="manifesto_project_full_qsentence_uniform",
            unit_kind=ObservationUnitKind.LEAF,
            metadata={"manifesto_id": doc_id, "qsentence_count": n_units},
        )
        assert sampling.effective_joint_propensity() == pytest.approx(
            document_propensity * unit_propensity
        )
        assert sampling.ipw_weight() == pytest.approx(
            1.0 / (document_propensity * unit_propensity)
        )

    assert {
        "document_population_count": len(doc_ids),
        "document_observed_count": doc_sample_size,
        "qsentence_population_count": n_qsentences,
        "sampled_doc_qsentence_population_count": sampled_qsentence_population,
        "qsentence_observed_count": observed_qsentences,
    } == {
        "document_population_count": 3327,
        "document_observed_count": 128,
        "qsentence_population_count": 2_271_892,
        "sampled_doc_qsentence_population_count": 65_718,
        "qsentence_observed_count": 1_283,
    }


def _read_manifesto_main_ids(path: Path) -> tuple[set[str], int, int]:
    ids: set[str] = set()
    n_rows = 0
    n_with_rile = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n_rows += 1
            ids.add(f"{row['party']}_{row['date']}")
            if row.get("rile") not in {"", "NA", None}:
                n_with_rile += 1
    return ids, n_rows, n_with_rile


def _read_corpus_counts(path: Path) -> tuple[Counter[str], int, int, int]:
    csv.field_size_limit(sys.maxsize)
    counts: Counter[str] = Counter()
    n_rows = 0
    n_coded = 0
    n_uncoded = 0
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            n_rows += 1
            manifesto_id = str(row.get("manifesto_id") or "")
            counts[manifesto_id] += 1
            if row.get("cmp_code") in {"", "NA", None}:
                n_uncoded += 1
            else:
                n_coded += 1
    return counts, n_rows, n_coded, n_uncoded
