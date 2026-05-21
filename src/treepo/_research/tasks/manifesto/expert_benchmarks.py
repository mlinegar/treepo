"""
Benoit replication-archive loader.

Reads the AJPS Dataverse files shipped with Benoit et al. 2026
(doi:10.7910/DVN/XY1FFE) from `data/examples/benoit_dataverse/`:

- `data_experts.rda`              — expert ensemble means, keyed on
                                    (manifesto, issue).
- `data_mp.rda`                   — MP metadata per manifesto; provides
                                    (party, year) <-> Benoit `manifesto`
                                    string crosswalk.
- `data_llms_all_reported.rds`    — their main proprietary-LLM scores.
- `data_llms_all_openweight.rds`  — their open-weight-LLM scores.
- `data_llms_all_replication.rds` — their 3-month re-run of proprietary.

Using their own files instead of raw CHES trend files gives us the exact
benchmark they used (same rescaling, same party-year join, same NA
treatment), so our Pearson r values are directly stackable next to their
Figure 1 and Table 6.

Verified: re-computing their Figure 1 correlations from
`data_llms_all_reported.rds` × `data_experts.rda` reproduces the published
numbers to within 0.005 (Economic .872 vs .87; Decentralization .495 vs
.49; etc.). See `scripts/reproduce_benoit_figure1.py`.

Important scale convention: Benoit's headline correlations use the released
``expert_mean`` values directly. Those values are not all bounded to 1-7
(the codebook describes expert-survey ranges as varying by survey/issue).
For calibration or supervised targets that require the LLM's 1-7 range, use
the derived ``expert_mean_1_7`` column added by ``load_benoit_expert_means``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .dimensions import BENOIT_DIMENSIONS, PolicyDimension
from .expert_scale import normalize_benoit_expert_mean

logger = logging.getLogger(__name__)


_DEFAULT_DATAVERSE_DIR = Path(__file__).resolve().parents[3] / "data" / "examples" / "benoit_dataverse"


def _dataverse_path(dir_: Optional[Path] = None) -> Path:
    return dir_ if dir_ is not None else _DEFAULT_DATAVERSE_DIR


def _require_pyreadr():
    try:
        import pyreadr  # type: ignore
    except ImportError as e:
        raise ImportError(
            "pyreadr is required to read Benoit's .rda/.rds files. "
            "Install with `pip install pyreadr`."
        ) from e
    return pyreadr


@dataclass(frozen=True)
class ManifestoCrosswalkRow:
    """Join row: Benoit manifesto string <-> MP (party, year) key."""

    manifesto: str         # Benoit's verbose key, e.g. "Ireland - IRL 1989 txt - IRL 1989 Fianna Fail"
    country: int
    year: int
    party: int             # MP party code (matches `ManifestoSample.party_id`)
    partyname: str
    partyabbrev: str
    rile: float


def load_benoit_experts(
    *,
    dataverse_dir: Optional[Path] = None,
    issues: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Load `data_experts.rda`. Returns one row per (manifesto, issue, expert_survey),
    preserving Benoit's released `expert_mean` column.

    Parameters
    ----------
    issues : iterable of Benoit issue codes (e.g. `{"taxspend", "eu"}`) to keep.
             When `None`, all six issues are returned.
    """
    pyreadr = _require_pyreadr()
    path = _dataverse_path(dataverse_dir) / "data_experts.rda"
    if not path.exists():
        raise FileNotFoundError(
            f"Expected Benoit replication file at {path}. "
            f"Unzip data/examples/dataverse_files.zip into data/examples/benoit_dataverse/."
        )
    df = pyreadr.read_r(str(path))["data_experts"]
    if issues is not None:
        df = df[df["issue"].isin(list(issues))]
    return df.reset_index(drop=True)


def load_benoit_expert_means(
    dimension: PolicyDimension,
    *,
    dataverse_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Per-manifesto expert mean for one dimension.

    Returns columns: manifesto, issue, expert_mean, expert_mean_raw, and
    expert_mean_1_7. `expert_mean` is kept as Benoit's released raw benchmark
    value for exact replication. `expert_mean_1_7` is the explicit derived
    target on the same 1-7 range as LLM scores.
    """
    issue_code = BENOIT_DIMENSIONS[dimension].benoit_issue_code
    df = load_benoit_experts(dataverse_dir=dataverse_dir, issues={issue_code})
    collapsed = (
        df.dropna(subset=["expert_mean"])
        .groupby(["manifesto", "issue"], as_index=False)["expert_mean"]
        .mean()
    )
    collapsed["expert_mean_raw"] = collapsed["expert_mean"]
    collapsed["expert_mean_1_7"] = collapsed["expert_mean"].map(
        lambda value: normalize_benoit_expert_mean(value, dimension)
    )
    return collapsed


def load_benoit_mp_crosswalk(
    *,
    dataverse_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Return Benoit's `manifesto` string <-> MP (party, year) crosswalk.

    Source is `data_mp.rda` which contains both the Benoit `manifesto` column
    and standard MP fields. Use this to resolve our `ManifestoSample.party_id`
    + `year` to the string key Benoit uses in `data_experts` and
    `data_llms_all_*`.
    """
    pyreadr = _require_pyreadr()
    path = _dataverse_path(dataverse_dir) / "data_mp.rda"
    if not path.exists():
        raise FileNotFoundError(f"Expected Benoit replication file at {path}.")
    df = pyreadr.read_r(str(path))["data_mp"]
    cols = ["manifesto", "country", "year", "party", "partyname", "partyabbrev", "rile"]
    return df[cols].dropna(subset=["party", "year"]).reset_index(drop=True)


def load_benoit_llm_scores(
    kind: str = "reported",
    *,
    dataverse_dir: Optional[Path] = None,
    dimension: Optional[PolicyDimension] = None,
    include_coalitions: bool = False,
) -> pd.DataFrame:
    """
    Load Benoit's LLM scores.

    Parameters
    ----------
    kind : one of "reported" (proprietary baseline), "openweight", "replication".
    dimension : restrict to a single dimension's `benoit_issue_code`.
    include_coalitions : include `run` values starting with "coalitions_".
    """
    if kind not in {"reported", "openweight", "replication"}:
        raise ValueError(f"kind must be reported | openweight | replication; got {kind!r}")
    pyreadr = _require_pyreadr()
    path = _dataverse_path(dataverse_dir) / f"data_llms_all_{kind}.rds"
    if not path.exists():
        raise FileNotFoundError(f"Expected Benoit replication file at {path}.")
    df = pyreadr.read_r(str(path))[None]
    if not include_coalitions:
        df = df[~df["run"].astype(str).str.startswith("coalition")]
    if dimension is not None:
        code = BENOIT_DIMENSIONS[dimension].benoit_issue_code
        df = df[df["issue"] == code]
    return df.reset_index(drop=True)


_MASKED_ISSUE_TO_DIM = {
    "taxation": PolicyDimension.ECONOMIC,
    "lifestyle": PolicyDimension.SOCIAL,
    "immigration": PolicyDimension.IMMIGRATION,
    "european_union": PolicyDimension.EU,
    "environment": PolicyDimension.ENVIRONMENT,
    "decentralization": PolicyDimension.DECENTRALIZATION,
}


def load_benoit_masked_summaries(
    dimension: Optional[PolicyDimension] = None,
    *,
    dataverse_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Parse data_masked.csv and return one row per (manifesto, issue) with the
    anonymized summary extracted from the serialized LangChain prompt.

    Benoit's archive ships 482 manifestos × 6 issues × GPT-4o zero-shot. This
    lets us score Benoit-authored summaries with our own scorer without needing
    any MP text.
    """
    import ast

    path = _dataverse_path(dataverse_dir) / "data_masked.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected Benoit replication file at {path}.")
    df = pd.read_csv(path)

    rows = []
    for r in df.itertuples():
        # r.prompt is a nested list of LangChain-serialized messages; the first
        # inner list holds the SystemMessage + HumanMessage for one call.
        try:
            outer = ast.literal_eval(r.prompt)
        except (ValueError, SyntaxError):
            continue
        messages = outer[0] if outer and isinstance(outer[0], list) else outer
        system_text = ""
        human_text = ""
        for m in messages:
            if not isinstance(m, dict):
                continue
            kind = m.get("id", [""])[-1] if isinstance(m.get("id"), list) else ""
            content = m.get("kwargs", {}).get("content", "")
            if kind == "SystemMessage":
                system_text = content
            elif kind == "HumanMessage":
                human_text = content

        # Human message has form: "Analyze the following political text:\n\n<SUMMARY>"
        summary = human_text.split("\n\n", 1)[1].strip() if "\n\n" in human_text else human_text.strip()
        issue_code = str(r.issue)
        dim = _MASKED_ISSUE_TO_DIM.get(issue_code)
        benoit_manifesto_key = str(r.original_file).removesuffix(".txt")
        rows.append(
            {
                "original_file": r.original_file,
                "manifesto_stem": benoit_manifesto_key,
                "masked_issue": issue_code,
                "dimension": dim.value if dim is not None else None,
                "summary": summary,
                "benoit_score": r.score,
                "system_rubric": system_text,
            }
        )

    out = pd.DataFrame(rows)
    if dimension is not None:
        out = out[out["dimension"] == dimension.value]
    return out.reset_index(drop=True)


def _train_lookup_for_pool(dim: PolicyDimension, pool: str) -> dict:
    """Return {manifesto_stem: label} for a given (dim, pool)."""
    if pool == "openweight":
        scores = load_benoit_llm_scores(kind="openweight", dimension=dim)
        ensemble = benoit_ensemble_mean(scores)
        ensemble["manifesto_stem"] = ensemble["manifesto"].astype(str).str.removesuffix(".txt")
        return dict(zip(ensemble["manifesto_stem"], ensemble["score_llm_mean"]))
    if pool == "expert":
        experts = load_benoit_expert_means(dim)
        return {
            str(r.manifesto).removesuffix(".txt"): float(r.expert_mean_1_7)
            for r in experts.itertuples()
        }
    raise ValueError(f"unknown pool {pool!r}")


def load_joint_train_pairs(
    pool: str = "openweight",
    *,
    test_keys_per_dim: Optional[dict[PolicyDimension, set[str]]] = None,
    global_holdout_keys: Optional[set[str]] = None,
    dataverse_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Pool training examples across all 6 dimensions for joint optimization.

    Returns columns: manifesto_stem, dimension (str), summary, label.

    `test_keys_per_dim` is an optional mapping of `PolicyDimension ->
    {manifesto_stem, ...}` that should be excluded from the training pool
    (the held-out test set per dim).

    `global_holdout_keys` can be used to exclude the union of held-out
    manifestos across all dimensions so a pooled shared scorer never trains
    on a manifesto that appears in any dimension's evaluation set.

    When both are omitted, the full pool is returned and the caller is
    responsible for deduplication.
    """
    rows = []
    global_holdout = set(global_holdout_keys or ())
    for dim in PolicyDimension:
        summaries = load_benoit_masked_summaries(dimension=dim, dataverse_dir=dataverse_dir)
        lookup = _train_lookup_for_pool(dim, pool)
        summaries["label"] = summaries["manifesto_stem"].map(lookup)
        summaries = summaries.dropna(subset=["label"]).copy()
        holdout = set()
        if test_keys_per_dim is not None:
            holdout.update(test_keys_per_dim.get(dim, set()))
        holdout.update(global_holdout)
        if holdout:
            summaries = summaries[~summaries["manifesto_stem"].isin(holdout)]
        summaries["dimension"] = dim.value
        rows.append(summaries[["manifesto_stem", "dimension", "summary", "label"]])
    return pd.concat(rows, ignore_index=True)


def benoit_ensemble_mean(llm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse a raw LLM-score DataFrame (one row per scorer * summarizer * run)
    to the (manifesto, issue) ensemble mean matching Benoit Figure 1.
    """
    return (
        llm_df.groupby(["manifesto", "issue"], as_index=False)
        .agg(
            score_llm_mean=("score_llm", lambda x: x.dropna().mean()),
            n_scores=("score_llm", "size"),
            n_non_na=("score_llm", lambda x: x.notna().sum()),
        )
    )
