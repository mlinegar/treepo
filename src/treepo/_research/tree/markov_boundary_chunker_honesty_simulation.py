"""
Markov boundary cost simulation bridged to the adaptive chunker + honesty split.

This module takes the locally-learnable Markov boundary cost toy problem from
`markov_boundary_honesty_simulation.py` and runs *the real adaptive chunker*
(`src.preprocessing.chunker.chunk_for_ops`) with `ChunkFeedbackSignal`s stored
in `AdaptiveChunkMemory`.

Key ablation:
- Honest: chunking consumes only boundary-role (predicted) signals.
- Leaky: chunking (incorrectly) consumes evaluation-role (oracle) signals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import torch
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for Markov chunker honesty simulations. "
        "Install with: pip install torch>=2.0.0"
    ) from e

from treepo._research.preprocessing.chunker import (
    AdaptiveChunkMemory,
    AdaptiveChunkingConfig,
    ChunkFeedbackSignal,
    HonestChunkingPolicy,
    chunk_for_ops,
)
from treepo._research.tree.markov_boundary_honesty_simulation import (
    BoundaryHarmPredictor,
    MarkovBoundaryConfig,
    MarkovDoc,
    PolicyMetrics,
    _boundary_cost_sum,
    _boundary_costs,
    _chunked_posterior,
    _kl_divergence,
    _l1_discrepancy,
    _loglik,
    _make_transition_matrices,
    _predict_boundary_costs,
    _set_global_seed,
    _train_boundary_model,
    generate_docs,
    _collect_boundary_training_data,
)


@dataclass(frozen=True)
class MarkovChunkerHonestySummary:
    config: Dict[str, object]
    boundary_model_train_loss_final: float
    metrics: Dict[str, PolicyMetrics]

    def to_json(self) -> str:
        payload = {
            "config": self.config,
            "boundary_model_train_loss_final": self.boundary_model_train_loss_final,
            "metrics": {k: asdict(v) for k, v in self.metrics.items()},
        }
        return json.dumps(payload, indent=2, sort_keys=True)


def _encode_tokens_fixed_width(tokens: Sequence[int], *, width: int) -> str:
    width = int(width)
    if width < 2:
        raise ValueError("token_char_width must be >= 2")
    fmt_width = width - 1
    parts = [f"{int(tok):0{fmt_width}d} " for tok in tokens]
    return "".join(parts)


def _signals_from_costs(
    costs: np.ndarray,
    *,
    token_char_width: int,
    source: str,
) -> List[ChunkFeedbackSignal]:
    width = int(token_char_width)
    signals: List[ChunkFeedbackSignal] = []
    for t, cost in enumerate(costs.tolist()):
        start = int((t + 1) * width)
        end = int((t + 2) * width)
        signals.append(
            ChunkFeedbackSignal(
                start_char=start,
                end_char=end,
                low_info_probability=float(max(0.0, min(1.0, cost))),
                noise_probability=0.0,
                confidence=1.0,
                source=str(source),
                metadata={"boundary_index": int(t)},
            )
        )
    return signals


def _chunk_ends_from_chunks(
    *,
    chunks: Sequence[object],
    token_char_width: int,
    n_tokens: int,
) -> List[int]:
    width = int(token_char_width)
    ends: List[int] = []
    for chunk in chunks:
        end_char = int(getattr(chunk, "end_char"))
        end_excl = int(end_char // width)
        end_excl = max(1, min(int(n_tokens), end_excl))
        ends.append(int(end_excl - 1))
    if not ends:
        return [int(n_tokens - 1)]
    if ends[-1] != int(n_tokens - 1):
        ends[-1] = int(n_tokens - 1)
    # Drop duplicates if chunker emitted any zero-width chunks.
    deduped: List[int] = []
    for end in ends:
        if not deduped or end > deduped[-1]:
            deduped.append(int(end))
    if not deduped:
        deduped = [int(n_tokens - 1)]
    if deduped[-1] != int(n_tokens - 1):
        deduped.append(int(n_tokens - 1))
    return deduped


def _evaluate_chunker_policy(
    docs: Sequence[Tuple[str, MarkovDoc, str]],
    *,
    config: MarkovBoundaryConfig,
    log_transitions: np.ndarray,
    adaptive_config: Optional[AdaptiveChunkingConfig],
    feedback_memory: AdaptiveChunkMemory,
    honest_policy: HonestChunkingPolicy,
    signal_role: str,
    token_char_width: int,
    base_max_chars: int,
    strategy: str,
) -> PolicyMetrics:
    total_cost = 0.0
    total_boundaries = 0.0
    total_l1 = 0.0
    total_kl = 0.0
    correct = 0
    n_docs = 0

    for doc_id, doc, text in docs:
        toks = np.asarray(doc.tokens, dtype=np.int64)
        if len(toks) < 2:
            continue
        ll = _loglik(toks, log_transitions)
        p_oracle = np.exp(ll - float(np.max(ll)))
        p_oracle = p_oracle / float(np.sum(p_oracle))
        costs = _boundary_costs(toks, log_transitions=log_transitions)

        if signal_role == "none":
            signals: List[ChunkFeedbackSignal] = []
        elif signal_role == "chunking":
            signals = feedback_memory.get_signals_for_chunking(doc_id, honest_policy=honest_policy)
        elif signal_role == "evaluation":
            signals = feedback_memory.get_signals_for_evaluation(doc_id, honest_policy=honest_policy)
        else:
            raise ValueError(f"unknown signal_role: {signal_role!r}")

        chunks = chunk_for_ops(
            text,
            max_chars=int(base_max_chars),
            strategy=str(strategy),
            adaptive_config=adaptive_config,
            feedback_signals=signals,
        )
        ends = _chunk_ends_from_chunks(
            chunks=chunks,
            token_char_width=int(token_char_width),
            n_tokens=len(toks),
        )

        p_chunk = _chunked_posterior(
            toks, loglik_full=ll, segment_ends=ends, log_transitions=log_transitions
        )
        total_cost += _boundary_cost_sum(costs, ends)
        total_boundaries += float(max(0, len(ends) - 1))
        total_l1 += _l1_discrepancy(p_oracle, p_chunk)
        total_kl += _kl_divergence(p_oracle, p_chunk)
        correct += int(int(np.argmax(p_chunk)) == int(doc.label))
        n_docs += 1

    denom = float(max(1, n_docs))
    return PolicyMetrics(
        mean_boundary_cost=float(total_cost / denom),
        mean_num_boundaries=float(total_boundaries / denom),
        mean_l1=float(total_l1 / denom),
        mean_kl=float(total_kl / denom),
        accuracy=float(correct) / denom,
        n_docs=int(n_docs),
    )


def run_markov_chunker_honesty_experiment(
    config: MarkovBoundaryConfig,
    *,
    token_char_width: int = 300,
    honest_policy: Optional[HonestChunkingPolicy] = None,
    adaptive_config: Optional[AdaptiveChunkingConfig] = None,
    strategy: str = "axis",
) -> MarkovChunkerHonestySummary:
    """
    Run the chunker + honesty split ablation on Markov boundary costs.

    Policies reported:
    - fixed: fixed-size chunking (adaptive disabled)
    - chunker_honest: adaptive chunking using boundary-role predicted signals
    - chunker_leaky: adaptive chunking using evaluation-role oracle signals
    """
    _set_global_seed(int(config.seed))

    if int(config.torch_threads) > 0:
        try:
            torch.set_num_threads(int(config.torch_threads))
        except RuntimeError:
            pass
        if hasattr(torch, "set_num_interop_threads"):
            try:
                torch.set_num_interop_threads(int(config.torch_threads))
            except RuntimeError:
                pass

    # Keep the model tiny; default to CPU when CUDA is unavailable.
    if config.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    rng = np.random.default_rng(int(config.seed))
    transitions = _make_transition_matrices(
        n_classes=int(config.n_classes),
        vocab_size=int(config.vocab_size),
        log_std=float(config.transition_log_std),
        sinkhorn_iters=int(config.sinkhorn_iters),
        rng=rng,
    )
    log_transitions = np.log(transitions)

    docs = generate_docs(config, transitions=transitions)
    train_docs = docs[: int(config.train_docs)]
    test_docs = docs[int(config.train_docs) :]

    x_train, y_train = _collect_boundary_training_data(
        train_docs, config=config, log_transitions=log_transitions, rng=rng
    )
    boundary_model = BoundaryHarmPredictor(
        vocab_size=int(config.vocab_size),
        window_size=int(config.window_size),
        emb_dim=int(config.boundary_emb_dim),
        hidden_dim=int(config.boundary_hidden_dim),
    )
    train_loss = _train_boundary_model(
        boundary_model, x_train=x_train, y_train=y_train, config=config, device=device
    )

    honest = honest_policy or HonestChunkingPolicy(enabled=True)
    if adaptive_config is None:
        adaptive_config = AdaptiveChunkingConfig(
            enabled=True,
            min_chars=int(config.min_leaf_tokens) * int(token_char_width),
            max_chars=int(config.max_leaf_tokens) * int(token_char_width),
            low_info_expansion_weight=1.0,
            noise_expansion_weight=0.0,
            high_info_compression_weight=0.0,
            proxy_blend=0.0,
        )

    base_max_chars = int(config.fixed_leaf_tokens) * int(token_char_width)
    if base_max_chars < 256:
        raise ValueError("token_char_width too small: base_max_chars would be < 256")

    memory = AdaptiveChunkMemory()
    encoded: List[Tuple[str, MarkovDoc, str]] = []
    for i, doc in enumerate(docs):
        doc_id = f"markov_doc_{i}"
        toks = np.asarray(doc.tokens, dtype=np.int64)
        text = _encode_tokens_fixed_width(toks, width=int(token_char_width))

        true_costs = _boundary_costs(toks, log_transitions=log_transitions)
        pred_costs = _predict_boundary_costs(
            boundary_model, toks, config=config, device=device
        )

        pred_signals = _signals_from_costs(
            pred_costs.astype(np.float64, copy=False),
            token_char_width=int(token_char_width),
            source="predicted_boundary_cost",
        )
        oracle_signals = _signals_from_costs(
            true_costs.astype(np.float64, copy=False),
            token_char_width=int(token_char_width),
            source="oracle_boundary_cost",
        )
        memory.update_signals(
            doc_id,
            pred_signals,
            honest_role=honest.boundary_role,
            replace_existing=True,
        )
        memory.update_signals(
            doc_id,
            oracle_signals,
            honest_role=honest.evaluation_role,
            replace_existing=False,
        )
        encoded.append((doc_id, doc, text))

    test_encoded = encoded[int(config.train_docs) :]

    metrics: Dict[str, PolicyMetrics] = {}
    metrics["fixed"] = _evaluate_chunker_policy(
        test_encoded,
        config=config,
        log_transitions=log_transitions,
        adaptive_config=None,
        feedback_memory=memory,
        honest_policy=honest,
        signal_role="none",
        token_char_width=int(token_char_width),
        base_max_chars=int(base_max_chars),
        strategy=str(strategy),
    )
    metrics["chunker_honest"] = _evaluate_chunker_policy(
        test_encoded,
        config=config,
        log_transitions=log_transitions,
        adaptive_config=adaptive_config,
        feedback_memory=memory,
        honest_policy=honest,
        signal_role="chunking",
        token_char_width=int(token_char_width),
        base_max_chars=int(base_max_chars),
        strategy=str(strategy),
    )
    metrics["chunker_leaky"] = _evaluate_chunker_policy(
        test_encoded,
        config=config,
        log_transitions=log_transitions,
        adaptive_config=adaptive_config,
        feedback_memory=memory,
        honest_policy=honest,
        signal_role="evaluation",
        token_char_width=int(token_char_width),
        base_max_chars=int(base_max_chars),
        strategy=str(strategy),
    )

    cfg_dict = asdict(config)
    cfg_dict["device_used"] = str(device)
    cfg_dict["token_char_width"] = int(token_char_width)
    cfg_dict["adaptive_config"] = asdict(adaptive_config)
    cfg_dict["honest_policy"] = asdict(honest)
    cfg_dict["chunk_strategy"] = str(strategy)

    return MarkovChunkerHonestySummary(
        config=cfg_dict,
        boundary_model_train_loss_final=float(train_loss),
        metrics=metrics,
    )


__all__ = [
    "MarkovChunkerHonestySummary",
    "run_markov_chunker_honesty_experiment",
]
