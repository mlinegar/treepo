from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from treepo._research.bench.lda.segmented_lda_ctreepo import SegmentedLDACtreePOConfig
from treepo.bench.runner import RunSpec


EXPERIMENT_SEGMENTED = "segmented-lda-ctreepo"


def _parse_items(text: str) -> List[str]:
    out: List[str] = []
    for raw in str(text).replace(",", " ").split():
        x = raw.strip()
        if x:
            out.append(x)
    return out


def _parse_ints(text: str) -> List[int]:
    return [int(x) for x in _parse_items(text)]


def _base_segmented_config() -> Dict[str, object]:
    return dict(asdict(SegmentedLDACtreePOConfig()))


def _fmt_float(x: float) -> str:
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")


def _identifiable_zero_paths_segmented(
    *,
    output_root: Path,
    configs: Sequence[Dict[str, object]],
) -> List[Tuple[Path, Path, Path]]:
    """Return the historical identifiable-zero LDA output layout."""

    proc_values = {str(c.get("topic_process", "segments")).strip().lower() for c in configs}
    theta_values = {str(c.get("leaf_theta_estimator", "lstsq")).strip().lower() for c in configs}
    docs_values = {int(c.get("topic_phi_docs", 0) or 0) for c in configs}
    default_seed_frac = float(SegmentedLDACtreePOConfig().neural_topic_seed_fraction)
    seed_fracs_global = {
        float(c.get("neural_topic_seed_fraction", default_seed_frac))
        for c in configs
        if str(c.get("topic_phi_estimator", "")).strip().lower().startswith("neural_")
    }
    multiple_seed_fracs = len(seed_fracs_global) > 1

    out: List[Tuple[Path, Path, Path]] = []
    for c in configs:
        proc = str(c.get("topic_process", "segments")).strip().lower()
        theta = str(c.get("leaf_theta_estimator", "lstsq")).strip().lower()
        est = str(c.get("topic_phi_estimator", "")).strip()
        est_norm = est.strip().lower()
        is_neural = bool(est_norm.startswith("neural_"))

        phi_docs = int(c.get("topic_phi_docs", 0) or 0)
        td = int(c.get("n_books_train"))
        fixed_leaf_tokens = int(c.get("fixed_leaf_tokens"))
        cal = float(c.get("calibration_leaf_query_rate"))
        el = float(c.get("eval_leaf_query_rate"))
        ei = float(c.get("eval_internal_query_rate"))
        seed = int(c.get("seed"))

        proc_prefix = ""
        if len(proc_values) > 1 or proc != "segments":
            proc_prefix = f"tp_{proc}/"

        theta_prefix = ""
        if len(theta_values) > 1 or theta != "lstsq":
            theta_prefix = f"theta_{theta}/"

        docs_component = ""
        if len(docs_values) > 1 or phi_docs != 0:
            docs_component = f"/docs_{phi_docs}"

        seed_component = ""
        if is_neural:
            seed_frac = float(c.get("neural_topic_seed_fraction", default_seed_frac))
            if multiple_seed_fracs or seed_frac != default_seed_frac:
                seed_component = f"/seedfrac_{_fmt_float(seed_frac)}"

        sub = (
            f"{proc_prefix}{theta_prefix}phi_{est}{docs_component}{seed_component}"
            f"/train_{td}/lt_{fixed_leaf_tokens}"
            f"/cal_{_fmt_float(cal)}/leaf_{_fmt_float(el)}/int_{_fmt_float(ei)}"
        )
        base = Path(output_root) / sub / f"seed_{seed}"
        out.append((base.with_suffix(".json"), base.with_suffix(".csv"), base.with_suffix(".config.yaml")))
    return out


def _build_segmented_specs(
    *,
    output_root: Path,
    configs: Sequence[Dict[str, object]],
    skip_existing: bool,
) -> List[RunSpec]:
    paths = _identifiable_zero_paths_segmented(output_root=Path(output_root), configs=configs)
    specs: List[RunSpec] = []
    for cfg, (json_out, csv_out, cfg_out) in zip(configs, paths):
        if skip_existing and Path(json_out).exists() and Path(csv_out).exists():
            continue
        specs.append(
            RunSpec(
                experiment=EXPERIMENT_SEGMENTED,
                config=cfg,
                json_out=Path(json_out),
                csv_out=Path(csv_out),
                config_out=Path(cfg_out),
            )
        )
    return specs


def build_identifiable_zero_dtm_lda(
    *,
    out_root: Path,
    skip_existing: bool,
    seeds: Optional[str] = None,
    topic_phi_estimators: Optional[str] = None,
) -> List[RunSpec]:
    """
    Replicates `scripts/run_identifiable_zero_dtm_lda_overnight.sh`.
    """
    seeds_list = _parse_ints(seeds) if seeds is not None else [0, 1, 2, 3, 4, 5]
    est_list = _parse_items(topic_phi_estimators) if topic_phi_estimators is not None else ["tensor_lda", "sklearn_lda"]

    output_root = Path(out_root) / "segmented_lda_ctreepo" / "equivalence" / "lda"

    train_docs = [256, 512, 1024, 2048, 4096]
    calib_rates = [0.02, 0.05, 0.1, 0.2, 0.4]
    q_rates = [0.0, 0.5]

    base = _base_segmented_config()
    base.update(
        {
            "topic_process": "bag_of_words",
            "leaf_theta_estimator": "lstsq",
            "topic_phi_docs": 0,
            "n_topics": 4,
            "vocab_size": 256,
            "min_segments": 6,
            "max_segments": 6,
            "min_seg_tokens": 24,
            "max_seg_tokens": 48,
            "fixed_leaf_tokens": 32,
            "alpha_topic": 0.20,
            "beta_word": 0.10,
            "segment_concentration": 80.0,
            "segment_background": 2.0,
            "topic_phi_permute": True,
            "calibration_policy": "uniform",
            "eval_internal_query_design": "risk",
            "n_books_test": 5000,
            "selection_audit_trials": 0,
        }
    )

    configs: List[Dict[str, object]] = []
    for q in q_rates:
        for td in train_docs:
            for cal in calib_rates:
                for est in est_list:
                    for seed in seeds_list:
                        c = dict(base)
                        c.update(
                            {
                                "topic_phi_estimator": str(est),
                                "n_books_train": int(td),
                                "calibration_leaf_query_rate": float(cal),
                                "eval_leaf_query_rate": float(q),
                                "eval_internal_query_rate": float(q),
                                "seed": int(seed),
                            }
                        )
                        configs.append(c)

    return _build_segmented_specs(output_root=output_root, configs=configs, skip_existing=skip_existing)


def build_identifiable_zero_lda_leafnoise(
    *,
    out_root: Path,
    skip_existing: bool,
    seeds: Optional[str] = None,
) -> List[RunSpec]:
    """
    Replicates `scripts/run_identifiable_zero_lda_leafnoise_overnight.sh`.
    """
    seeds_list = _parse_ints(seeds) if seeds is not None else [0, 1, 2, 3, 4, 5]
    output_root = Path(out_root) / "segmented_lda_ctreepo" / "equivalence" / "lda_leafnoise"

    train_docs = [16, 32, 64, 128, 256, 512, 1024, 2048]
    leaf_tokens = [2048, 512, 128, 32, 8]
    calib_rates = [0.0, 0.1]

    base = _base_segmented_config()
    base.update(
        {
            "topic_process": "bag_of_words",
            "leaf_theta_estimator": "sklearn_lda",
            "topic_phi_estimator": "sklearn_lda",
            "topic_phi_docs": 0,
            "n_topics": 4,
            "vocab_size": 256,
            "min_segments": 1,
            "max_segments": 1,
            "min_seg_tokens": 2048,
            "max_seg_tokens": 2048,
            "alpha_topic": 0.20,
            "beta_word": 0.10,
            "segment_concentration": 80.0,
            "segment_background": 2.0,
            "topic_phi_permute": True,
            "calibration_policy": "uniform",
            "eval_internal_query_design": "risk",
            "n_books_test": 2000,
            "eval_leaf_query_rate": 0.0,
            "eval_internal_query_rate": 0.0,
            "selection_audit_trials": 0,
        }
    )

    configs: List[Dict[str, object]] = []
    for lt in leaf_tokens:
        for td in train_docs:
            for cal in calib_rates:
                for seed in seeds_list:
                    c = dict(base)
                    c.update(
                        {
                            "fixed_leaf_tokens": int(lt),
                            "n_books_train": int(td),
                            "calibration_leaf_query_rate": float(cal),
                            "seed": int(seed),
                        }
                    )
                    configs.append(c)

    return _build_segmented_specs(output_root=output_root, configs=configs, skip_existing=skip_existing)


def build_identifiable_zero_publication_ctreepo(
    *,
    out_root: Path,
    skip_existing: bool,
    seeds: Optional[str] = None,
) -> List[RunSpec]:
    """
    Replicates `scripts/run_identifiable_zero_publication_ctreepo_cpu_pass.sh`.
    """
    seeds_list = _parse_ints(seeds) if seeds is not None else list(range(8))

    Q_RATES = [0.0, 0.25, 0.5]
    Q_RATES_UPPER = [0.0, 0.25]

    output_root = Path(out_root) / "segmented_lda_ctreepo" / "equivalence"

    specs: List[RunSpec] = []

    # ------------------------------------------------------------------
    # Regime A: LDA (bag_of_words) | k=8, v=512
    # ------------------------------------------------------------------
    train_docs_lda = [128, 256, 512, 1024, 2048, 4096]
    leaf_tokens_lda = [32, 16, 8]
    cal_rates_lda = [0.0, 0.05, 0.1]
    n_books_test_lda = 4000

    base_lda = _base_segmented_config()
    base_lda.update(
        {
            "topic_process": "bag_of_words",
            "topic_phi_docs": 0,
            "n_topics": 8,
            "vocab_size": 512,
            "min_segments": 1,
            "max_segments": 1,
            "min_seg_tokens": 2048,
            "max_seg_tokens": 2048,
            "alpha_topic": 0.20,
            "beta_word": 0.10,
            "segment_concentration": 80.0,
            "segment_background": 2.0,
            "topic_phi_permute": True,
            "calibration_policy": "uniform",
            "eval_internal_query_design": "risk",
            "n_books_test": int(n_books_test_lda),
            "selection_audit_trials": 0,
        }
    )

    def _lane(
        lane_root: Path,
        *,
        leaf_theta_estimator: str,
        topic_phi_estimator: str,
        neural_overrides: Optional[Dict[str, object]] = None,
        seed_fracs: Optional[Sequence[float]] = None,
        q_rates: Sequence[float] = Q_RATES,
        cal_rates: Sequence[float] = cal_rates_lda,
        train_docs: Sequence[int] = train_docs_lda,
        leaf_tokens: Sequence[int] = leaf_tokens_lda,
    ) -> None:
        configs: List[Dict[str, object]] = []
        for lt in leaf_tokens:
            for q in q_rates:
                for td in train_docs:
                    for cal in cal_rates:
                        fracs = list(seed_fracs) if seed_fracs is not None else [float(base_lda.get("neural_topic_seed_fraction", 0.35))]
                        if not str(topic_phi_estimator).strip().lower().startswith("neural_"):
                            fracs = [float(base_lda.get("neural_topic_seed_fraction", 0.35))]
                        for frac in fracs:
                            for seed in seeds_list:
                                c = dict(base_lda)
                                c.update(
                                    {
                                        "fixed_leaf_tokens": int(lt),
                                        "eval_leaf_query_rate": float(q),
                                        "eval_internal_query_rate": float(q),
                                        "n_books_train": int(td),
                                        "calibration_leaf_query_rate": float(cal),
                                        "leaf_theta_estimator": str(leaf_theta_estimator),
                                        "topic_phi_estimator": str(topic_phi_estimator),
                                        "neural_topic_seed_fraction": float(frac),
                                        "seed": int(seed),
                                    }
                                )
                                if neural_overrides:
                                    c.update(dict(neural_overrides))
                                configs.append(c)
        specs.extend(_build_segmented_specs(output_root=lane_root, configs=configs, skip_existing=skip_existing))

    lda_root = output_root / "lda" / "k8_v512"
    _lane(
        lda_root / "lane_lda_direct",
        leaf_theta_estimator="sklearn_lda",
        topic_phi_estimator="sklearn_lda",
    )
    _lane(
        lda_root / "lane_phi_base",
        leaf_theta_estimator="lstsq",
        topic_phi_estimator="tensor_lda",
    )
    _lane(
        lda_root / "lane_neural_weak",
        leaf_theta_estimator="lstsq",
        topic_phi_estimator="neural_ctreepo",
        seed_fracs=[0.125],
        neural_overrides={
            "neural_topic_base_estimator": "tensor_lda",
            "neural_topic_operator_boost": 0.6,
            "neural_topic_seed_llm_min_weight": 0.02,
            "neural_topic_seed_llm_max_weight": 0.15,
            "neural_topic_mix_samples": 64,
        },
    )
    _lane(
        lda_root / "lane_neural_default",
        leaf_theta_estimator="lstsq",
        topic_phi_estimator="neural_ctreepo",
        seed_fracs=[0.25, 0.5],
        neural_overrides={
            "neural_topic_base_estimator": "tensor_lda",
            "neural_topic_operator_boost": 1.0,
            "neural_topic_seed_llm_min_weight": 0.10,
            "neural_topic_seed_llm_max_weight": 0.35,
            "neural_topic_mix_samples": 128,
        },
    )

    # ------------------------------------------------------------------
    # Regime B: hard (segments) | k=12, v=1024
    # ------------------------------------------------------------------
    train_docs_hard = [128, 256, 512, 1024, 2048]
    train_docs_upper = [1024, 2048, 4096]
    leaf_tokens_hard = [16, 8]
    cal_rates_hard = [0.05, 0.1, 0.2]
    n_books_test_hard = 5000

    base_hard = dict(base_lda)
    base_hard.update(
        {
            "topic_process": "segments",
            "n_topics": 12,
            "vocab_size": 1024,
            "min_segments": 10,
            "max_segments": 12,
            "min_seg_tokens": 16,
            "max_seg_tokens": 32,
            "alpha_topic": 0.35,
            "beta_word": 0.40,
            "segment_concentration": 18.0,
            "segment_background": 6.0,
            "n_books_test": int(n_books_test_hard),
        }
    )

    def _lane_hard(
        lane_root: Path,
        *,
        topic_phi_estimator: str,
        seed_fracs: Optional[Sequence[float]] = None,
        q_rates: Sequence[float] = Q_RATES,
        cal_rates: Sequence[float] = cal_rates_hard,
        train_docs: Sequence[int] = train_docs_hard,
        neural_overrides: Optional[Dict[str, object]] = None,
    ) -> None:
        configs: List[Dict[str, object]] = []
        for lt in leaf_tokens_hard:
            for q in q_rates:
                for td in train_docs:
                    for cal in cal_rates:
                        fracs = list(seed_fracs) if seed_fracs is not None else [float(base_hard.get("neural_topic_seed_fraction", 0.35))]
                        if not str(topic_phi_estimator).strip().lower().startswith("neural_"):
                            fracs = [float(base_hard.get("neural_topic_seed_fraction", 0.35))]
                        for frac in fracs:
                            for seed in seeds_list:
                                c = dict(base_hard)
                                c.update(
                                    {
                                        "fixed_leaf_tokens": int(lt),
                                        "eval_leaf_query_rate": float(q),
                                        "eval_internal_query_rate": float(q),
                                        "n_books_train": int(td),
                                        "calibration_leaf_query_rate": float(cal),
                                        "leaf_theta_estimator": "lstsq",
                                        "topic_phi_estimator": str(topic_phi_estimator),
                                        "neural_topic_seed_fraction": float(frac),
                                        "seed": int(seed),
                                    }
                                )
                                if neural_overrides:
                                    c.update(dict(neural_overrides))
                                configs.append(c)
        specs.extend(_build_segmented_specs(output_root=lane_root, configs=configs, skip_existing=skip_existing))

    hard_root = output_root / "hard" / "k12_v1024"
    _lane_hard(hard_root / "lane_phi_base", topic_phi_estimator="tensor_lda")
    _lane_hard(
        hard_root / "lane_neural_weak",
        topic_phi_estimator="neural_ctreepo",
        seed_fracs=[0.0833333333],
        neural_overrides={
            "neural_topic_base_estimator": "tensor_lda",
            "neural_topic_operator_boost": 0.6,
            "neural_topic_seed_llm_min_weight": 0.02,
            "neural_topic_seed_llm_max_weight": 0.15,
            "neural_topic_mix_samples": 64,
        },
    )
    _lane_hard(
        hard_root / "lane_neural_default",
        topic_phi_estimator="neural_ctreepo",
        seed_fracs=[0.2, 0.35],
        neural_overrides={
            "neural_topic_base_estimator": "tensor_lda",
            "neural_topic_operator_boost": 1.0,
            "neural_topic_seed_llm_min_weight": 0.10,
            "neural_topic_seed_llm_max_weight": 0.35,
            "neural_topic_mix_samples": 128,
        },
    )
    _lane_hard(
        hard_root / "lane_neural_upper",
        topic_phi_estimator="neural_ctreepo",
        seed_fracs=[1.0],
        q_rates=Q_RATES_UPPER,
        cal_rates=[0.1],
        train_docs=train_docs_upper,
        neural_overrides={
            "neural_topic_base_estimator": "tensor_lda",
            "neural_topic_operator_boost": 1.4,
            "neural_topic_seed_llm_min_weight": 0.35,
            "neural_topic_seed_llm_max_weight": 0.85,
            "neural_topic_mix_samples": 128,
        },
    )

    return specs
