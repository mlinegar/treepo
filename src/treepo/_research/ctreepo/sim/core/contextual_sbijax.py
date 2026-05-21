"""JAX/sbijax contextual-sufficiency learner for unified-g experiments.

This module is intentionally separate from ``CleanUnifiedNO``.  It provides a
JAX-native lane where ``sbijax`` is the runtime dependency and the learning
problem is framed around the theorem-level query object:

    z_x = g(embed(x), null)
    R_K(x) = [query(c_i, x)]_i

The trainer learns a state map for items from fixed empirical contexts.
Markov two-sided contexts are one adapter, not the definition of the method.
``method="nasss"`` adds the sliced response-prediction objective that mirrors
the NASSS/SSS pattern; ``method="nass"`` adds a JSD-style dependence term that
mirrors the NASS infomax critic.  In both cases the supervised contextual
response readout is kept explicit so diagnostics remain comparable with the
PyTorch probe.

The heavy JAX stack is lazy-imported.  Install it with:

    pip install -e ".[contextual_sbi]"
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from treepo._research.core.ops_checks import LawKind
from treepo._research.ctreepo.sim.core.markov_local_laws import (
    MARKOV_COUNT_SKETCH_LAW_SET_ID,
    markov_approx_local_laws_bundle,
    markov_canonical_project_np,
    markov_exact_merge_np,
    markov_local_law_observation_rows,
)


HLL_REGISTER_SKETCH_LAW_SET_ID = "hll_register_sketch"


CONTEXTUAL_SBI_INSTALL_MSG = (
    "The contextual sbijax lane requires the optional JAX/SBI dependencies. "
    'Install with: pip install -e ".[contextual_sbi]"'
)


@dataclass(frozen=True)
class ContextualResponseDataset:
    """Fixed-context empirical response signatures for item states.

    ``item_tokens`` is the primary generic name.  ``span_tokens`` and the
    left/right context accessors are compatibility views for the original
    Markov two-sided caller.
    """

    item_tokens: np.ndarray
    response_signatures: np.ndarray
    context_payloads: tuple[Mapping[str, Any], ...]
    context_tensors: Mapping[str, np.ndarray]
    target_scale: float
    pad_id: int
    metadata: dict[str, Any]
    package_theta_targets: Mapping[str, np.ndarray] | None = None

    @property
    def span_tokens(self) -> np.ndarray:
        """Compatibility alias for older Markov-specific callers."""

        return self.item_tokens

    @property
    def context_left_tokens(self) -> np.ndarray:
        """Compatibility view for Markov two-sided context tensors."""

        return np.asarray(self.context_tensors.get("left_tokens", np.empty((0, 0), dtype=np.int32)))

    @property
    def context_right_tokens(self) -> np.ndarray:
        """Compatibility view for Markov two-sided context tensors."""

        return np.asarray(
            self.context_tensors.get("right_tokens", np.empty((0, 0), dtype=np.int32))
        )

    @property
    def context_left_raw(self) -> tuple[tuple[int, ...], ...]:
        """Compatibility view for Markov two-sided raw left contexts."""

        out: list[tuple[int, ...]] = []
        for payload in self.context_payloads:
            out.append(tuple(int(tok) for tok in payload.get("left_tokens", ())))
        return tuple(out)

    @property
    def context_right_raw(self) -> tuple[tuple[int, ...], ...]:
        """Compatibility view for Markov two-sided raw right contexts."""

        out: list[tuple[int, ...]] = []
        for payload in self.context_payloads:
            out.append(tuple(int(tok) for tok in payload.get("right_tokens", ())))
        return tuple(out)


class ContextualQueryProblem(Protocol):
    """Generic finite-context query adapter.

    The theorem-level object is ``query : Ctx -> X -> Y``.  This protocol
    supplies sampled items ``x``, sampled finite contexts ``c_i``, and
    empirical responses ``query(c_i, x)``.  Some adapters may also expose a
    ``predict_contextual_response`` method for enacted model-side context
    training, but that method is intentionally optional and duck-typed by
    PyTorch callers.
    """

    problem_id: str
    context_kind: str
    vocab_size: int
    target_scale: float

    def sample_item_tokens(
        self,
        source: Sequence[int],
        *,
        item_len: int,
        rng: np.random.Generator,
    ) -> Sequence[int]: ...

    def sample_contexts(
        self,
        sources: Sequence[Sequence[int]],
        *,
        n_contexts: int,
        item_len: int,
        rng: np.random.Generator,
    ) -> Sequence[Any]: ...

    def evaluate_query(self, context: Any, item_tokens: Sequence[int]) -> Any: ...

    def context_payload(self, context: Any) -> Mapping[str, Any]: ...

    def context_tensors(
        self,
        contexts: Sequence[Any],
        *,
        item_len: int,
        pad_id: int,
    ) -> Mapping[str, np.ndarray]: ...


@dataclass(frozen=True)
class MarkovTwoSidedContext:
    """A Markov context ``c = (left, right)`` for ``query(c, x)``."""

    left_tokens: tuple[int, ...]
    right_tokens: tuple[int, ...]


@dataclass(frozen=True)
class MarkovTwoSidedContextProblem:
    """Markov changepoint-count specialization of ``ContextualQueryProblem``."""

    block_by_token: Sequence[int]
    vocab_size: int
    target_scale: float
    problem_id: str = "markov_changepoint_count"
    context_kind: str = "markov_two_sided"

    def sample_item_tokens(
        self,
        source: Sequence[int],
        *,
        item_len: int,
        rng: np.random.Generator,
    ) -> Sequence[int]:
        return sample_token_fragment(source, fragment_len=int(item_len), rng=rng)

    def sample_contexts(
        self,
        sources: Sequence[Sequence[int]],
        *,
        n_contexts: int,
        item_len: int,
        rng: np.random.Generator,
    ) -> Sequence[MarkovTwoSidedContext]:
        if not sources:
            raise ValueError("need at least one source to sample contexts")
        out: list[MarkovTwoSidedContext] = []
        for _ in range(int(n_contexts)):
            left_doc = sources[int(rng.integers(0, len(sources)))]
            right_doc = sources[int(rng.integers(0, len(sources)))]
            out.append(
                MarkovTwoSidedContext(
                    left_tokens=tuple(
                        sample_token_fragment(left_doc, fragment_len=int(item_len), rng=rng)
                    ),
                    right_tokens=tuple(
                        sample_token_fragment(right_doc, fragment_len=int(item_len), rng=rng)
                    ),
                )
            )
        return tuple(out)

    def evaluate_query(
        self,
        context: MarkovTwoSidedContext,
        item_tokens: Sequence[int],
    ) -> float:
        count = exact_count_for_tokens(
            list(context.left_tokens) + list(item_tokens) + list(context.right_tokens),
            block_by_token=self.block_by_token,
        )
        return float(count) / float(self.target_scale)

    def context_payload(self, context: MarkovTwoSidedContext) -> Mapping[str, Any]:
        return {
            "kind": self.context_kind,
            "left_tokens": [int(tok) for tok in context.left_tokens],
            "right_tokens": [int(tok) for tok in context.right_tokens],
        }

    def context_tensors(
        self,
        contexts: Sequence[MarkovTwoSidedContext],
        *,
        item_len: int,
        pad_id: int,
    ) -> Mapping[str, np.ndarray]:
        return {
            "left_tokens": np.asarray(
                [
                    pad_fragment(ctx.left_tokens, fragment_len=int(item_len), pad_id=int(pad_id))
                    for ctx in contexts
                ],
                dtype=np.int32,
            ),
            "right_tokens": np.asarray(
                [
                    pad_fragment(ctx.right_tokens, fragment_len=int(item_len), pad_id=int(pad_id))
                    for ctx in contexts
                ],
                dtype=np.int32,
            ),
        }

    def predict_contextual_response(
        self,
        model: Any,
        item_tokens: np.ndarray,
        context_batch: Mapping[str, np.ndarray],
        device: Any,
    ) -> Any:
        """Execute ``f(g(g(z_left, z_x), z_right))`` through a PyTorch model."""

        import torch

        left_tokens = torch.as_tensor(
            np.asarray(context_batch["left_tokens"], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        item_tokens_t = torch.as_tensor(
            np.asarray(item_tokens, dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        right_tokens = torch.as_tensor(
            np.asarray(context_batch["right_tokens"], dtype=np.int64),
            dtype=torch.long,
            device=device,
        )
        z_left = model._encode_leaf_states_via_g(left_tokens)
        z_item = model._encode_leaf_states_via_g(item_tokens_t)
        z_right = model._encode_leaf_states_via_g(right_tokens)
        z_left_item = model._merge_state_batch_via_g(z_left, z_item)
        z_full = model._merge_state_batch_via_g(z_left_item, z_right)
        return model._score_states_via_f(z_full)


@dataclass(frozen=True)
class HLLUnionContext:
    """A cardinality context merged with an item by HLL register max."""

    tokens: tuple[int, ...]


@dataclass(frozen=True)
class HLLUnionContextProblem:
    """HyperLogLog cardinality specialization of ``ContextualQueryProblem``."""

    vocab_size: int
    target_scale: float
    precision: int = 4
    hash_bits: int = 64
    problem_id: str = "hll_cardinality"
    context_kind: str = "hll_union"

    def sample_item_tokens(
        self,
        source: Sequence[int],
        *,
        item_len: int,
        rng: np.random.Generator,
    ) -> Sequence[int]:
        return sample_token_fragment(source, fragment_len=int(item_len), rng=rng)

    def sample_contexts(
        self,
        sources: Sequence[Sequence[int]],
        *,
        n_contexts: int,
        item_len: int,
        rng: np.random.Generator,
    ) -> Sequence[HLLUnionContext]:
        if not sources:
            raise ValueError("need at least one source to sample contexts")
        out: list[HLLUnionContext] = []
        for _ in range(int(n_contexts)):
            doc = sources[int(rng.integers(0, len(sources)))]
            out.append(
                HLLUnionContext(
                    tokens=tuple(sample_token_fragment(doc, fragment_len=int(item_len), rng=rng))
                )
            )
        return tuple(out)

    def evaluate_query(
        self,
        context: HLLUnionContext,
        item_tokens: Sequence[int],
    ) -> float:
        estimate = _hll_estimate_for_tokens_np(
            list(context.tokens) + list(item_tokens),
            precision=int(self.precision),
            hash_bits=int(self.hash_bits),
        )
        return float(estimate) / float(self.target_scale)

    def context_payload(self, context: HLLUnionContext) -> Mapping[str, Any]:
        return {
            "kind": self.context_kind,
            "tokens": [int(tok) for tok in context.tokens],
            "precision": int(self.precision),
            "hash_bits": int(self.hash_bits),
        }

    def context_tensors(
        self,
        contexts: Sequence[HLLUnionContext],
        *,
        item_len: int,
        pad_id: int,
    ) -> Mapping[str, np.ndarray]:
        return {
            "context_tokens": np.asarray(
                [
                    pad_fragment(ctx.tokens, fragment_len=int(item_len), pad_id=int(pad_id))
                    for ctx in contexts
                ],
                dtype=np.int32,
            )
        }


def _contexts_from_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    expected_kind: str,
) -> tuple[MarkovTwoSidedContext, ...]:
    if str(expected_kind) != "markov_two_sided":
        raise ValueError(f"cannot rebuild contexts for context_kind={expected_kind!r}")
    return tuple(
        MarkovTwoSidedContext(
            left_tokens=tuple(int(tok) for tok in payload.get("left_tokens", ())),
            right_tokens=tuple(int(tok) for tok in payload.get("right_tokens", ())),
        )
        for payload in payloads
    )


def build_contextual_query_dataset(
    item_sources: Sequence[Sequence[int]],
    *,
    problem: ContextualQueryProblem,
    samples_per_source: int,
    item_len: int,
    n_contexts: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    contexts: Sequence[Any] | None = None,
    context_sources: Sequence[Sequence[int]] | None = None,
) -> ContextualResponseDataset:
    """Build finite response signatures for any ``query : Ctx -> X -> Y``."""

    if not item_sources:
        raise ValueError("need at least one item source")
    if int(samples_per_source) <= 0:
        raise ValueError("samples_per_source must be positive")
    if int(n_contexts) <= 0:
        raise ValueError("n_contexts must be positive")
    if rng is None:
        rng = np.random.default_rng(0 if seed is None else int(seed))

    pad_id = int(problem.vocab_size)
    if contexts is None:
        context_pool = problem.sample_contexts(
            context_sources or item_sources,
            n_contexts=int(n_contexts),
            item_len=int(item_len),
            rng=rng,
        )
    else:
        context_pool = tuple(contexts)
        if len(context_pool) != int(n_contexts):
            raise ValueError("provided contexts must have n_contexts entries")

    item_rows: list[list[int]] = []
    response_rows: list[list[np.ndarray]] = []
    source_indices: list[int] = []
    hll_fast_path = isinstance(problem, HLLUnionContextProblem)
    for source_idx, source in enumerate(item_sources):
        for _ in range(int(samples_per_source)):
            item = list(problem.sample_item_tokens(source, item_len=int(item_len), rng=rng))
            item_rows.append(pad_fragment(item, fragment_len=int(item_len), pad_id=pad_id))
            source_indices.append(int(source_idx))
            if not hll_fast_path:
                response_rows.append(
                    [
                        np.asarray(problem.evaluate_query(ctx, item), dtype=np.float32)
                        for ctx in context_pool
                    ]
                )

    context_payloads = tuple(dict(problem.context_payload(ctx)) for ctx in context_pool)
    context_tensors = dict(
        problem.context_tensors(context_pool, item_len=int(item_len), pad_id=pad_id)
    )
    item_array = np.asarray(item_rows, dtype=np.int32)
    if hll_fast_path:
        response_array = _hll_union_response_matrix_np(
            item_array,
            context_pool,
            pad_id=pad_id,
            precision=int(problem.precision),
            hash_bits=int(problem.hash_bits),
            target_scale=float(problem.target_scale),
        )
    else:
        response_array = np.asarray(response_rows, dtype=np.float32)
    metadata = {
        "problem_id": str(problem.problem_id),
        "context_kind": str(problem.context_kind),
        "n_sources": int(len(item_sources)),
        "samples_per_source": int(samples_per_source),
        "samples_per_doc": int(samples_per_source),
        "item_len": int(item_len),
        "fragment_len": int(item_len),
        "n_contexts": int(n_contexts),
        "response_signature_contexts": int(n_contexts),
        "n_items": int(len(item_rows)),
        "n_rows": int(len(item_rows)),
        "response_target_shape": list(response_array.shape[2:]),
        "response_signature_dim": int(np.prod(response_array.shape[1:])),
        "target_scale": float(problem.target_scale),
        "pad_id": int(pad_id),
        "source_indices_per_item": source_indices,
        "doc_indices_per_row": list(source_indices),
    }
    if isinstance(problem, MarkovTwoSidedContextProblem):
        metadata.update(
            {
                "block_by_token": [int(x) for x in problem.block_by_token],
                "n_regimes": int(max(problem.block_by_token)) + 1,
            }
        )
    if isinstance(problem, HLLUnionContextProblem):
        metadata.update(
            {
                "hll_precision": int(problem.precision),
                "hll_hash_bits": int(problem.hash_bits),
                "hll_register_count": int(1 << int(problem.precision)),
                "hll_max_register": int(problem.hash_bits) - int(problem.precision) + 1,
            }
        )
    return ContextualResponseDataset(
        item_tokens=item_array,
        response_signatures=response_array,
        context_payloads=context_payloads,
        context_tensors=context_tensors,
        target_scale=float(problem.target_scale),
        pad_id=int(pad_id),
        metadata=metadata,
    )


@dataclass(frozen=True)
class ContextualSBIJAXConfig:
    """Training configuration for the JAX contextual-sufficiency lane."""

    trainer: str = "repo"
    method: str = "nasss"
    package_theta: str = "response_signature"
    input_encoding: str = "normalized_token_ids"
    summary_activation: str = "relu"
    posterior_estimator: str = "npe"
    density_family: str = "mdn"
    posterior_eval_samples: int = 0
    posterior_eval_batch_size: int = 0
    posterior_sampler: str = "nuts"
    vocab_size: int = 16
    embedding_dim: int = 32
    state_dim: int = 16
    hidden_dim: int = 64
    response_signature_contexts: int = 8
    response_signature_slices: int = 4
    contextual_loss_weight: float = 1.0
    infomax_loss_weight: float = 1.0
    local_law_supervision_mode: str = "dual"
    local_law_weight: float = 1.0
    local_law_leaf_weight: float = 1.0
    local_law_merge_weight: float = 1.0
    local_law_idempotence_weight: float = 1.0
    local_law_contextual_weight: float = 1.0
    local_law_package_weight: float = 0.0
    # HLL-only auxiliary: penalize the normalized cardinality estimate implied
    # by predicted leaf/merge registers. Defaults off so Markov and prior HLL
    # local-law runs keep their original objective.
    local_law_hll_estimate_weight: float = 0.0
    local_law_leaf_rate: float = 1.0
    local_law_merge_rate: float = 1.0
    local_law_idempotence_rate: float = 1.0
    local_law_summary_family: str = "mlp"
    local_law_summary_fno_n_modes: int = 16
    local_law_summary_fno_n_layers: int = 2
    local_law_summary_fno_pooling_mode: str = "sum"
    # Count-only supervision (instead of full sketch): C1/C2 grade
    # `count_readout(rep) ≈ count_truth` rather than the full
    # ``(count, first, last)`` sketch slot-by-slot. The rep is a learned
    # sufficient statistic of width ``local_law_rep_dim`` (auto = 2*theta_dim
    # when 0). Requires ``law_architecture='fully_learned'`` because
    # arbitrary rep_dim breaks the analytic merge/decoder.
    local_law_count_only: bool = False
    local_law_rep_dim: int = 0
    # C2 merge supervision form. ``mse`` is the existing element-wise MSE
    # between predicted parent state and the merge truth. ``nass_jsd``
    # replaces it with sbijax NASS-style InfoNCE/JSD: train an internal
    # critic on (merge_state, merge_truth) pairs (positive vs permuted
    # negatives) and minimize the negative MI lower bound. This is "use
    # sbijax internally for the merge."
    local_law_merge_loss: str = "mse"
    # Merge architecture (g internally). ``mlp`` (default): asymmetric
    # concat MLP. ``fno_rep``: 1D FNO with the rep dim as spatial axis,
    # (left, right) lifted to ``merge_fno_hidden_channels`` channels;
    # ``merge_fno_n_modes`` low-freq spectral modes are kept. With
    # state_dim=256 / n_modes=32 this is a "real" FNO — qualitatively
    # different from the prior degenerate length-2 design.
    merge_family: str = "mlp"
    merge_fno_n_modes: int = 16
    merge_fno_n_layers: int = 2
    merge_fno_hidden_channels: int = 32
    # Decoder head (h on top of g). ``mlp`` (default), ``linear``.
    decoder_head: str = "mlp"
    # HLL law-package parameters. Registers are supervised as normalized
    # values in [0, 1], with exact merge = elementwise max.
    hll_precision: int = 4
    hll_hash_bits: int = 64
    # Literal paper factorization: g maps leaves/summaries into a learned
    # summary z, and an explicit f decodes z -> theorem-state theta before
    # local-law/readout losses. Defaults off for comparability with older
    # fused-theta runs where summary_net directly emitted theta-shaped states.
    local_law_explicit_state_decoder: bool = False
    local_law_summary_dim: int = 0
    local_law_state_decoder_head: str = "mlp"
    # NASSS-merge sliced contrastive: when ``local_law_merge_loss ==
    # 'nasss_jsd'`` we project merge_target onto ``merge_nasss_n_slices``
    # random unit projections and apply per-slice JSD MI bound.
    merge_nasss_n_slices: int = 16
    law_architecture: str = "analytic"
    c2_merge_target: str = "theta"
    learned_merge_hidden_dim: int = 0
    learned_decoder_hidden_dim: int = 0
    learning_rate: float = 3e-4
    lr_schedule: str = "constant"
    n_iter: int = 100
    batch_size: int = 128
    posterior_samples: int = 32
    density_components: int = 5
    seed: int = 0


@dataclass(frozen=True)
class ContextualSBIJAXResult:
    """Serializable diagnostics plus opaque JAX objects for further eval."""

    params: Any
    history: list[dict[str, float | int]]
    train_diagnostics: dict[str, float | int]
    val_diagnostics: dict[str, float | int]
    provenance: dict[str, Any]
    config: ContextualSBIJAXConfig
    slice_matrix: Any
    apply_fn: Any

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "history": self.history,
            "train_diagnostics": self.train_diagnostics,
            "val_diagnostics": self.val_diagnostics,
            "provenance": self.provenance,
            "config": asdict(self.config),
        }


@dataclass(frozen=True)
class ContextualMarkovSplits:
    """Official Markov docs flattened for contextual response learning."""

    train_docs: list[list[int]]
    val_docs: list[list[int]]
    test_docs: list[list[int]]
    train_root_counts: list[float]
    val_root_counts: list[float]
    test_root_counts: list[float]
    block_by_token: list[int]
    metadata: dict[str, Any]


def _optional_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def contextual_sbijax_available() -> bool:
    """Return whether the runtime package surface can be imported."""

    try:
        _require_contextual_sbi()
    except ImportError:
        return False
    return True


def _require_contextual_sbi() -> SimpleNamespace:
    """Import the JAX/sbijax runtime surface lazily.

    haiku is a transitive sbijax dep (``sbijax/_src/nn/nass_net.py`` defines
    ``class NASSNet(hk.Module)`` and the rest of ``sbijax/_src/nn/`` imports
    it). New learned merge/decoder modules use Flax (``flax.linen``), while
    the repo-owned JAX FNO summary is implemented directly with JAX FFTs and
    Haiku transforms so it can plug into the existing local-law optimizer.
    ``norax`` and ``pardax`` are design references only, not runtime deps.
    """

    try:
        import haiku as hk
        import flax.linen as fnn
        import jax
        from jax import numpy as jnp
        from jax import random as jr
        import optax
        import sbijax
        from sbijax.nn import (
            make_cm,
            make_cnf,
            make_maf,
            make_mdn,
            make_nass_net,
            make_nasss_net,
            make_resnet,
            make_spf,
        )
    except ImportError as exc:  # pragma: no cover - covered by non-installed envs
        raise ImportError(CONTEXTUAL_SBI_INSTALL_MSG) from exc

    missing = [
        name
        for name in ("CMPE", "FMPE", "NASS", "NASSS", "NLE", "NPE", "NRE", "SNLE")
        if not hasattr(sbijax, name)
    ]
    if missing:
        raise ImportError(f"sbijax is installed but missing expected symbols: {missing}")
    return SimpleNamespace(
        hk=hk,
        fnn=fnn,
        jax=jax,
        jnp=jnp,
        jr=jr,
        optax=optax,
        sbijax=sbijax,
        make_cm=make_cm,
        make_cnf=make_cnf,
        make_maf=make_maf,
        make_mdn=make_mdn,
        make_nass_net=make_nass_net,
        make_nasss_net=make_nasss_net,
        make_resnet=make_resnet,
        make_spf=make_spf,
    )


def contextual_sbijax_provenance(
    *,
    method: str,
    response_signature_contexts: int,
    response_signature_slices: int,
    trainer: str = "repo",
    input_encoding: str | None = None,
    summary_activation: str | None = None,
    downstream_readout: str | None = None,
) -> dict[str, Any]:
    """Package provenance recorded in every JAX contextual run."""

    installed = contextual_sbijax_available()
    out: dict[str, Any] = {
        "backend_package": "sbijax",
        "backend_version": _optional_version("sbijax"),
        "jax_version": _optional_version("jax"),
        "jaxlib_version": _optional_version("jaxlib"),
        "surjectors_version": _optional_version("surjectors"),
        "method": str(method),
        "trainer": str(trainer),
        "response_signature_contexts": int(response_signature_contexts),
        "response_signature_slices": int(response_signature_slices),
        "installed": bool(installed),
    }
    if input_encoding is not None:
        out["input_encoding"] = str(input_encoding)
    if summary_activation is not None:
        out["summary_activation"] = str(summary_activation)
    if downstream_readout is not None:
        out["downstream_readout"] = str(downstream_readout)
    if installed:
        deps = _require_contextual_sbi()
        out.update(
            {
                "sbijax_has_nass": hasattr(deps.sbijax, "NASS"),
                "sbijax_has_nasss": hasattr(deps.sbijax, "NASSS"),
                "sbijax_has_snle": hasattr(deps.sbijax, "SNLE"),
                "sbijax_class": "NASSS" if str(method) == "nasss" else "NASS",
            }
        )
    return out


def palette_block_map(*, vocab_size: int, n_regimes: int) -> list[int]:
    """Map token ids to disjoint palette regimes."""

    v = int(vocab_size)
    n = int(n_regimes)
    if v <= 0 or n <= 0:
        raise ValueError("vocab_size and n_regimes must be positive")
    block_by_token = [0 for _ in range(v)]
    start = 0
    base, extra = divmod(v, n)
    for regime_id in range(n):
        size = base + (1 if regime_id < extra else 0)
        for token_id in range(start, start + size):
            block_by_token[token_id] = regime_id
        start += size
    return block_by_token


def exact_count_for_tokens(
    tokens: Sequence[int],
    *,
    block_by_token: Sequence[int],
) -> float:
    """Count adjacent regime changes under a palette-block oracle."""

    if len(tokens) <= 1:
        return 0.0
    blocks = [int(block_by_token[int(tok)]) for tok in tokens]
    return float(sum(1 for left, right in zip(blocks, blocks[1:]) if left != right))


def markov_exact_sketch_targets_for_dataset(
    dataset: ContextualResponseDataset,
    *,
    block_by_token: Sequence[int],
    target_scale: float | None = None,
    n_regimes: int | None = None,
) -> np.ndarray:
    """Exact Markov sufficient sketch targets for package-native ``theta``.

    Each row is ``[count / target_scale, one_hot(first_regime),
    one_hot(last_regime)]``.  This uses the known Markov witness only as a
    training target for package-native NASS/NASSS comparison runs.
    """

    scale = float(dataset.target_scale if target_scale is None else target_scale)
    n_blocks = int(max(block_by_token)) + 1 if n_regimes is None else int(n_regimes)
    if n_blocks <= 0:
        raise ValueError("n_regimes must be positive")
    rows: list[np.ndarray] = []
    for row in np.asarray(dataset.item_tokens, dtype=np.int64):
        toks = [int(tok) for tok in row if int(tok) != int(dataset.pad_id)]
        if not toks:
            count_norm = 0.0
            first = 0
            last = 0
        else:
            blocks = [int(block_by_token[int(tok)]) for tok in toks]
            count = sum(1 for left, right in zip(blocks, blocks[1:]) if int(left) != int(right))
            count_norm = float(count) / scale
            first = int(blocks[0])
            last = int(blocks[-1])
        first_oh = np.zeros((n_blocks,), dtype=np.float32)
        last_oh = np.zeros((n_blocks,), dtype=np.float32)
        first_oh[first] = 1.0
        last_oh[last] = 1.0
        rows.append(
            np.concatenate(
                [np.asarray([count_norm], dtype=np.float32), first_oh, last_oh],
                axis=0,
            )
        )
    return np.asarray(rows, dtype=np.float32)


def _hll_config(*, precision: int, hash_bits: int):
    from treepo.hll import HLLConfig

    return HLLConfig(precision=int(precision), hash_bits=int(hash_bits))


def _hll_max_register(*, precision: int, hash_bits: int) -> int:
    p = int(precision)
    h = int(hash_bits)
    if not (4 <= p <= h - 2):
        raise ValueError("hll_precision must be in [4, hll_hash_bits - 2]")
    return int(h - p + 1)


def _hll_register_state_from_tokens_np(
    tokens: Sequence[int],
    *,
    precision: int,
    hash_bits: int,
) -> np.ndarray:
    from treepo.hll import HyperLogLogSketch

    config = _hll_config(precision=int(precision), hash_bits=int(hash_bits))
    sketch = HyperLogLogSketch.from_tokens(config, [int(tok) for tok in tokens])
    max_register = float(_hll_max_register(precision=int(precision), hash_bits=int(hash_bits)))
    return sketch.registers.astype(np.float32) / max_register


def _hll_estimate_from_normalized_registers_np(
    states: np.ndarray,
    *,
    precision: int,
    hash_bits: int,
    round_registers: bool = True,
) -> np.ndarray:
    from treepo.hll import _hll_alpha

    arr = np.asarray(states, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    m_int = int(1 << int(precision))
    if int(arr.shape[1]) != m_int:
        raise ValueError(f"HLL states must have {m_int} registers; got {arr.shape[1]}")
    max_register = float(_hll_max_register(precision=int(precision), hash_bits=int(hash_bits)))
    regs = np.clip(arr, 0.0, 1.0) * max_register
    if bool(round_registers):
        regs = np.rint(regs)
    m = float(m_int)
    z = np.power(2.0, -regs).sum(axis=1)
    raw = float(_hll_alpha(m_int)) * (m * m) / np.maximum(z, 1e-12)
    if bool(round_registers):
        zeros = (regs <= 0.0).sum(axis=1).astype(np.float64)
    else:
        # Differentiable proxy used by the legacy HLL formula/readout loss:
        # exact on integer registers and smooth for off-lattice predictions.
        zeros = np.clip(1.0 - regs, 0.0, None).sum(axis=1)
    small = m * np.log(m / np.maximum(zeros.astype(np.float64), 1.0))
    out = np.where((raw <= 2.5 * m) & (zeros > 0), small, raw)
    hash_space = float(2.0 ** int(hash_bits))
    large = out > hash_space / 30.0
    if np.any(large):
        clipped = np.minimum(out[large] / hash_space, 1.0 - 1e-12)
        out[large] = -hash_space * np.log1p(-clipped)
    return out.astype(np.float32)


def _hll_estimate_for_tokens_np(
    tokens: Sequence[int],
    *,
    precision: int,
    hash_bits: int,
) -> float:
    state = _hll_register_state_from_tokens_np(
        tokens,
        precision=int(precision),
        hash_bits=int(hash_bits),
    )
    return float(
        _hll_estimate_from_normalized_registers_np(
            state,
            precision=int(precision),
            hash_bits=int(hash_bits),
        )[0]
    )


def hll_register_sketch_targets_for_dataset(
    dataset: ContextualResponseDataset,
    *,
    precision: int | None = None,
    hash_bits: int | None = None,
) -> np.ndarray:
    """Exact HLL register targets for package-native local-law supervision.

    Registers are normalized by ``hash_bits - precision + 1`` so the target
    lives in a compact [0, 1] coordinate system. Exact HLL merge is therefore
    just elementwise max in these coordinates.
    """

    p = int(dataset.metadata.get("hll_precision", 4) if precision is None else int(precision))
    h = int(dataset.metadata.get("hll_hash_bits", 64) if hash_bits is None else int(hash_bits))
    _hll_max_register(precision=p, hash_bits=h)
    rows: list[np.ndarray] = []
    for row in np.asarray(dataset.item_tokens, dtype=np.int64):
        toks = [int(tok) for tok in row if int(tok) != int(dataset.pad_id)]
        rows.append(
            _hll_register_state_from_tokens_np(
                toks,
                precision=p,
                hash_bits=h,
            )
        )
    return np.asarray(rows, dtype=np.float32)


def _hll_union_response_matrix_np(
    item_array: np.ndarray,
    contexts: Sequence[HLLUnionContext],
    *,
    pad_id: int,
    precision: int,
    hash_bits: int,
    target_scale: float,
) -> np.ndarray:
    """Vectorized HLL union responses for a fixed item/context bank."""

    item_tokens = np.asarray(item_array, dtype=np.int64)
    if item_tokens.ndim != 2:
        raise ValueError(f"item_array must be 2D; got {tuple(item_tokens.shape)}")
    item_states = np.asarray(
        [
            _hll_register_state_from_tokens_np(
                [int(tok) for tok in row if int(tok) != int(pad_id)],
                precision=int(precision),
                hash_bits=int(hash_bits),
            )
            for row in item_tokens
        ],
        dtype=np.float32,
    )
    context_states = np.asarray(
        [
            _hll_register_state_from_tokens_np(
                [int(tok) for tok in ctx.tokens],
                precision=int(precision),
                hash_bits=int(hash_bits),
            )
            for ctx in contexts
        ],
        dtype=np.float32,
    )
    out = np.zeros((int(item_states.shape[0]), int(context_states.shape[0])), dtype=np.float32)
    for ctx_idx, ctx_state in enumerate(context_states):
        merged = np.maximum(item_states, ctx_state[None, :])
        out[:, ctx_idx] = _hll_estimate_from_normalized_registers_np(
            merged,
            precision=int(precision),
            hash_bits=int(hash_bits),
            round_registers=True,
        ) / float(target_scale)
    return out


def with_package_theta_target(
    dataset: ContextualResponseDataset,
    *,
    name: str,
    targets: np.ndarray,
) -> ContextualResponseDataset:
    """Return a dataset carrying an alternate package-native theta target."""

    arr = np.asarray(targets, dtype=np.float32)
    if int(arr.shape[0]) != int(dataset.item_tokens.shape[0]):
        raise ValueError(
            "package theta target row count must match dataset item rows; "
            f"got {int(arr.shape[0])} vs {int(dataset.item_tokens.shape[0])}"
        )
    targets_by_name = dict(dataset.package_theta_targets or {})
    targets_by_name[str(name)] = arr
    return replace(dataset, package_theta_targets=targets_by_name)


def markov_exact_response_predictions_for_dataset(
    dataset: ContextualResponseDataset,
    *,
    block_by_token: Sequence[int],
    target_scale: float | None = None,
    n_regimes: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct fixed-context responses from the exact Markov sketch.

    Returns ``(states, predictions)`` where states are the exact
    ``[count / scale, one_hot(first), one_hot(last)]`` rows and predictions
    match the dataset's flattened response signatures.
    """

    scale = float(dataset.target_scale if target_scale is None else target_scale)
    n_blocks = int(max(block_by_token)) + 1 if n_regimes is None else int(n_regimes)
    states = markov_exact_sketch_targets_for_dataset(
        dataset,
        block_by_token=block_by_token,
        target_scale=scale,
        n_regimes=n_blocks,
    )
    preds = np.zeros(
        (int(states.shape[0]), int(len(dataset.context_payloads))),
        dtype=np.float32,
    )
    count_norm = states[:, 0]
    first_oh = states[:, 1 : 1 + n_blocks]
    last_oh = states[:, 1 + n_blocks : 1 + 2 * n_blocks]
    for context_idx, (left, right) in enumerate(
        zip(dataset.context_left_raw, dataset.context_right_raw, strict=True)
    ):
        left_tokens = [int(tok) for tok in left]
        right_tokens = [int(tok) for tok in right]
        left_count = exact_count_for_tokens(
            left_tokens,
            block_by_token=block_by_token,
        )
        right_count = exact_count_for_tokens(
            right_tokens,
            block_by_token=block_by_token,
        )
        left_last = int(block_by_token[int(left_tokens[-1])]) if left_tokens else 0
        right_first = int(block_by_token[int(right_tokens[0])]) if right_tokens else 0
        left_boundary = (1.0 - first_oh[:, left_last]) / scale
        right_boundary = (1.0 - last_oh[:, right_first]) / scale
        preds[:, context_idx] = (
            count_norm + float(left_count + right_count) / scale + left_boundary + right_boundary
        )
    return states, preds


def markov_exact_sketch_oracle_diagnostics(
    dataset: ContextualResponseDataset,
    *,
    block_by_token: Sequence[int],
    target_scale: float | None = None,
    n_regimes: int | None = None,
) -> dict[str, Any]:
    """Diagnostics for the deterministic exact Markov sketch control."""

    states, preds = markov_exact_response_predictions_for_dataset(
        dataset,
        block_by_token=block_by_token,
        target_scale=target_scale,
        n_regimes=n_regimes,
    )
    diagnostics = _response_diagnostics(
        preds=preds,
        truths=dataset.response_signatures.reshape(dataset.response_signatures.shape[0], -1),
        states=states,
    )
    diagnostics.update(
        {
            "problem_id": str(dataset.metadata.get("problem_id", "")),
            "context_kind": str(dataset.metadata.get("context_kind", "")),
            "oracle_state": "markov_exact_sketch",
        }
    )
    return diagnostics


def sample_token_fragment(
    tokens: Sequence[int],
    *,
    fragment_len: int,
    rng: np.random.Generator,
) -> list[int]:
    """Sample one contiguous token fragment."""

    if not tokens:
        raise ValueError("cannot sample a fragment from an empty token sequence")
    length = max(1, min(int(fragment_len), len(tokens)))
    if len(tokens) == length:
        return list(map(int, tokens))
    start = int(rng.integers(0, len(tokens) - length + 1))
    return list(map(int, tokens[start : start + length]))


def pad_fragment(
    tokens: Sequence[int],
    *,
    fragment_len: int,
    pad_id: int,
) -> list[int]:
    """Pad or crop a token fragment to a fixed JAX array length."""

    out = list(map(int, tokens[: int(fragment_len)]))
    out.extend([int(pad_id)] * max(0, int(fragment_len) - len(out)))
    return out


def make_synthetic_markov_docs(
    *,
    n_docs: int,
    doc_tokens: int,
    vocab_size: int,
    n_regimes: int,
    expected_boundaries: float,
    seed: int,
) -> list[list[int]]:
    """Small sticky Markov/palette corpus for JAX smoke runs."""

    rng = np.random.default_rng(int(seed))
    block_by_token = palette_block_map(vocab_size=int(vocab_size), n_regimes=int(n_regimes))
    tokens_by_regime: list[list[int]] = [[] for _ in range(int(n_regimes))]
    for tok, regime in enumerate(block_by_token):
        tokens_by_regime[int(regime)].append(int(tok))

    switch_prob = float(max(0.0, expected_boundaries)) / max(1.0, float(int(doc_tokens) - 1))
    docs: list[list[int]] = []
    for _ in range(int(n_docs)):
        regime = int(rng.integers(0, int(n_regimes)))
        toks: list[int] = []
        for pos in range(int(doc_tokens)):
            if pos > 0 and float(rng.random()) < switch_prob:
                choices = [r for r in range(int(n_regimes)) if r != regime]
                regime = int(rng.choice(choices))
            toks.append(int(rng.choice(tokens_by_regime[regime])))
        docs.append(toks)
    return docs


def flatten_fno_count_docs(fno_docs: Sequence[Any]) -> list[list[int]]:
    """Flatten prepared FNO-count docs back to full left-to-right token lists."""

    flat_docs: list[list[int]] = []
    for doc in fno_docs:
        rows = getattr(doc, "leaf_token_ids", None)
        if rows is None:
            raise ValueError("expected prepared FNO docs with leaf_token_ids")
        tokens: list[int] = []
        for leaf in rows:
            tokens.extend(int(tok) for tok in leaf)
        expected_n = int(getattr(doc, "n_tokens", len(tokens)))
        if len(tokens) != expected_n:
            raise ValueError(
                "flattened token length does not match FNO doc n_tokens: "
                f"{len(tokens)} vs {expected_n}"
            )
        flat_docs.append(tokens)
    return flat_docs


def root_counts_from_fno_count_docs(fno_docs: Sequence[Any]) -> list[float]:
    """Extract official root count labels from prepared FNO-count docs."""

    out: list[float] = []
    for doc in fno_docs:
        if not hasattr(doc, "root_count"):
            raise ValueError("expected prepared FNO docs with root_count")
        out.append(float(getattr(doc, "root_count")))
    return out


def _flat_tokens_from_markov_docs(docs: Sequence[Any]) -> list[list[int]]:
    """Extract full token lists from saved ``MarkovOPSDataBundle`` docs."""

    flat_docs: list[list[int]] = []
    for doc in docs:
        tokens = getattr(doc, "tokens", None)
        if tokens is None:
            raise ValueError("expected Markov OPS docs with tokens")
        flat_docs.append([int(tok) for tok in tokens])
    return flat_docs


def _root_counts_from_markov_docs(docs: Sequence[Any]) -> list[float]:
    """Extract root changepoint counts from saved ``MarkovOPSDataBundle`` docs."""

    out: list[float] = []
    for doc in docs:
        boundaries = getattr(doc, "true_boundaries", None)
        if boundaries is None:
            raise ValueError("expected Markov OPS docs with true_boundaries")
        out.append(float(len(tuple(boundaries))))
    return out


def _infer_bundle_palette_shape(bundle: Any) -> tuple[int, int]:
    """Infer the global disjoint-palette shape for a saved Markov bundle."""

    metadata_payload = dict(getattr(bundle, "metadata", {}) or {})
    conditions = [dict(item or {}) for item in list(metadata_payload.get("conditions") or [])]
    condition_regimes = [
        int(item["n_regimes"])
        for item in conditions
        if "n_regimes" in item and int(item["n_regimes"]) > 0
    ]
    condition_vocab = [
        int(item.get("vocab_size", 4 * int(item["n_regimes"])))
        for item in conditions
        if "n_regimes" in item and int(item["n_regimes"]) > 0
    ]

    docs = (
        tuple(getattr(bundle, "train_docs", ()) or ())
        + tuple(getattr(bundle, "val_docs", ()) or ())
        + tuple(getattr(bundle, "test_docs", ()) or ())
    )
    observed_max_token = -1
    observed_max_regime = -1
    for doc in docs:
        tokens = getattr(doc, "tokens", ()) or ()
        regimes = getattr(doc, "token_regimes", ()) or ()
        if tokens:
            observed_max_token = max(observed_max_token, max(int(tok) for tok in tokens))
        if regimes:
            observed_max_regime = max(observed_max_regime, max(int(regime) for regime in regimes))

    n_regimes = max([4, observed_max_regime + 1, *condition_regimes])
    vocab_size = max([4 * int(n_regimes), observed_max_token + 1, *condition_vocab])
    return int(vocab_size), int(n_regimes)


def _sliced_condition_ids(
    metadata_payload: Mapping[str, Any],
    *,
    split: str,
    n_docs: int,
) -> list[str]:
    condition_ids = dict(metadata_payload.get("condition_ids") or {})
    raw = list(condition_ids.get(str(split), ()) or ())
    return [str(value) for value in raw[: int(n_docs)]]


def _condition_counts_from_ids(condition_ids: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for condition_id in condition_ids:
        key = str(condition_id)
        counts[key] = int(counts.get(key, 0)) + 1
    return dict(sorted(counts.items()))


def load_markov_contextual_splits_from_bundle(
    bundle_path: str | Path,
    *,
    train_docs: int,
    val_docs: int,
    test_docs: int,
) -> ContextualMarkovSplits:
    """Load saved paper Markov bundles for contextual-sufficiency probes.

    This is the bridge from the paper-facing hazard-panel data prep to the
    JAX/sbijax contextual lane.  It keeps the contextual learner on full
    left-to-right token sequences while preserving panel/condition metadata
    for downstream diagnostics.
    """

    from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import MarkovOPSDataBundle

    path = Path(bundle_path)
    bundle = MarkovOPSDataBundle.load(path)
    n_train = min(int(train_docs), len(bundle.train_docs))
    n_val = min(int(val_docs), len(bundle.val_docs))
    n_test = min(int(test_docs), len(bundle.test_docs))
    train_split = tuple(bundle.train_docs[:n_train])
    val_split = tuple(bundle.val_docs[:n_val])
    test_split = tuple(bundle.test_docs[:n_test])

    vocab_size, n_regimes = _infer_bundle_palette_shape(bundle)
    block_by_token = palette_block_map(
        vocab_size=int(vocab_size),
        n_regimes=int(n_regimes),
    )

    source_metadata = dict(getattr(bundle, "metadata", {}) or {})
    sliced_condition_ids = {
        "train": _sliced_condition_ids(source_metadata, split="train", n_docs=n_train),
        "val": _sliced_condition_ids(source_metadata, split="val", n_docs=n_val),
        "test": _sliced_condition_ids(source_metadata, split="test", n_docs=n_test),
    }
    condition_counts = {
        split: _condition_counts_from_ids(ids) for split, ids in sliced_condition_ids.items() if ids
    }
    doc_tokens_values = {
        len(getattr(doc, "tokens", ()) or ()) for doc in train_split + val_split + test_split
    }
    metadata_payload = {
        "data_source": "markov",
        "markov_loader": "saved_ops_count_bundle",
        "bundle_path": str(path),
        "hazard_panel_id": str(
            source_metadata.get("hazard_panel_id", source_metadata.get("panel_id", ""))
        ),
        "panel_id": str(source_metadata.get("panel_id", "")),
        "doc_tokens": (int(next(iter(doc_tokens_values))) if len(doc_tokens_values) == 1 else None),
        "doc_tokens_values": sorted(int(value) for value in doc_tokens_values),
        "leaf_tokens": None,
        "train_docs": int(n_train),
        "val_docs": int(n_val),
        "test_docs": int(n_test),
        "vocab_size": int(vocab_size),
        "n_regimes": int(n_regimes),
        "condition_ids": sliced_condition_ids,
        "condition_counts": condition_counts,
        "train_corpus_signature": str(getattr(bundle, "train_corpus_signature", "")),
        "val_corpus_signature": str(getattr(bundle, "val_corpus_signature", "")),
        "test_corpus_signature": str(getattr(bundle, "test_corpus_signature", "")),
    }
    if "conditions" in source_metadata:
        metadata_payload["conditions"] = list(source_metadata.get("conditions") or [])

    return ContextualMarkovSplits(
        train_docs=_flat_tokens_from_markov_docs(train_split),
        val_docs=_flat_tokens_from_markov_docs(val_split),
        test_docs=_flat_tokens_from_markov_docs(test_split),
        train_root_counts=_root_counts_from_markov_docs(train_split),
        val_root_counts=_root_counts_from_markov_docs(val_split),
        test_root_counts=_root_counts_from_markov_docs(test_split),
        block_by_token=block_by_token,
        metadata=metadata_payload,
    )


def _sticky_mean_segment_length(
    *,
    doc_tokens: int,
    min_segments: int,
    max_segments: int,
) -> int:
    target_segments = 0.5 * (float(min_segments) + float(max_segments))
    target_boundaries = max(1.0, target_segments - 1.0)
    return max(1, int(round(float(max(1, int(doc_tokens) - 1)) / target_boundaries)))


def _load_markov_split_docs_direct(
    *,
    doc_tokens: int,
    train_docs: int,
    val_docs: int,
    test_docs: int,
    leaf_tokens: int,
    expected_boundaries: float | None,
    seed: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], dict[str, Any]]:
    """Generate official sticky Markov docs without using a named preset."""

    from treepo._research.ctreepo.sim.core.markov_changepoint_ops_count import (
        OPSCountConfig,
        build_markov_changepoint_ops_count_data_bundle,
    )
    from treepo._research.ctreepo.sim.core.markov_neural_operator_baselines import (
        _prepare_fno_count_docs,
    )

    scale = math.sqrt(float(int(doc_tokens)) / 128.0)
    resolved_expected_boundaries = (
        float(expected_boundaries) if expected_boundaries is not None else 5.0 * scale
    )
    min_segments = max(2, int(round(2.0 * scale)))
    max_segments = max(min_segments, int(round(6.0 * scale)))
    mean_seg_len = _sticky_mean_segment_length(
        doc_tokens=int(doc_tokens),
        min_segments=int(min_segments),
        max_segments=int(max_segments),
    )
    switch_prob = float(
        max(0.0, resolved_expected_boundaries) / max(1.0, float(int(doc_tokens) - 1))
    )
    cfg = OPSCountConfig(
        generator_profile="hazard_topic",
        n_regimes=4,
        vocab_size=16,
        min_tokens=int(doc_tokens),
        max_tokens=int(doc_tokens),
        min_segments=int(min_segments),
        max_segments=int(max_segments),
        min_seg_len=int(mean_seg_len),
        max_seg_len=int(mean_seg_len),
        hazard_switch_prob=float(switch_prob),
        fixed_leaf_tokens=int(leaf_tokens),
        train_docs=int(train_docs),
        val_docs=int(val_docs),
        test_docs=int(test_docs),
        seed=int(seed),
        data_seed=int(seed),
        model_seed=int(seed),
    )
    bundle = build_markov_changepoint_ops_count_data_bundle(cfg)
    metadata = {
        "markov_loader": "direct_ops_count",
        "doc_tokens": int(doc_tokens),
        "leaf_tokens": int(leaf_tokens),
        "expected_boundaries": float(resolved_expected_boundaries),
        "min_segments": int(min_segments),
        "max_segments": int(max_segments),
        "mean_seg_len": int(mean_seg_len),
        "hazard_switch_prob": float(switch_prob),
        "vocab_size": int(cfg.vocab_size),
        "n_regimes": int(cfg.n_regimes),
    }
    return (
        tuple(_prepare_fno_count_docs(bundle.train_docs, leaf_tokens=int(leaf_tokens))),
        tuple(_prepare_fno_count_docs(bundle.val_docs, leaf_tokens=int(leaf_tokens))),
        tuple(_prepare_fno_count_docs(bundle.test_docs, leaf_tokens=int(leaf_tokens))),
        metadata,
    )


def _load_markov_split_docs_named(
    *,
    benchmark: str,
    train_docs: int,
    leaf_tokens: int,
    seed: int,
) -> tuple[tuple[Any, ...], tuple[Any, ...], tuple[Any, ...], dict[str, Any]]:
    """Load official Markov docs through the named prepared-benchmark path."""

    from treepo._research.ctreepo.sim.core.full_doc_anchor_diagnostics import (
        _load_fno_docs,
        prepare_markov_full_doc_anchor_diagnostics_data,
    )

    payload = prepare_markov_full_doc_anchor_diagnostics_data(
        benchmark_name=str(benchmark),
        seeds=(int(seed),),
        train_doc_counts=(int(train_docs),),
        use_cuda=False,
        torch_threads=1,
        config_overrides={"fixed_leaf_tokens": int(leaf_tokens)},
    )
    prepared = dict(payload["prepared"][0])
    metadata = {
        "markov_loader": "named_prepared_benchmark",
        "benchmark": str(benchmark),
        "leaf_tokens": int(leaf_tokens),
        "prepared": {
            "train_fno_docs_json": str(prepared["train_fno_docs_json"]),
            "val_fno_docs_json": str(prepared["val_fno_docs_json"]),
            "test_fno_docs_json": str(prepared["test_fno_docs_json"]),
        },
        "vocab_size": 16,
        "n_regimes": 4,
    }
    return (
        tuple(_load_fno_docs(Path(str(prepared["train_fno_docs_json"])))),
        tuple(_load_fno_docs(Path(str(prepared["val_fno_docs_json"])))),
        tuple(_load_fno_docs(Path(str(prepared["test_fno_docs_json"])))),
        metadata,
    )


def load_markov_contextual_splits(
    *,
    benchmark: str,
    doc_tokens: int,
    train_docs: int,
    val_docs: int,
    test_docs: int,
    leaf_tokens: int,
    expected_boundaries: float | None,
    seed: int,
    vocab_size: int = 16,
    n_regimes: int = 4,
) -> ContextualMarkovSplits:
    """Load official Markov train/val/test splits for the sbijax lane.

    If ``doc_tokens > 0`` this follows the direct sticky-Markov generation
    path used by the PyTorch probe.  Otherwise it uses the named prepared
    benchmark path and caps val/test after loading.
    """

    if int(doc_tokens) > 0:
        train_fno, val_fno, test_fno, metadata = _load_markov_split_docs_direct(
            doc_tokens=int(doc_tokens),
            train_docs=int(train_docs),
            val_docs=int(val_docs),
            test_docs=int(test_docs),
            leaf_tokens=int(leaf_tokens),
            expected_boundaries=expected_boundaries,
            seed=int(seed),
        )
    else:
        train_fno, val_fno, test_fno, metadata = _load_markov_split_docs_named(
            benchmark=str(benchmark),
            train_docs=int(train_docs),
            leaf_tokens=int(leaf_tokens),
            seed=int(seed),
        )
        train_fno = tuple(train_fno[: int(train_docs)])
        val_fno = tuple(val_fno[: int(val_docs)])
        test_fno = tuple(test_fno[: int(test_docs)])

    train_flat = flatten_fno_count_docs(train_fno)
    val_flat = flatten_fno_count_docs(val_fno)
    test_flat = flatten_fno_count_docs(test_fno)
    train_root_counts = root_counts_from_fno_count_docs(train_fno)
    val_root_counts = root_counts_from_fno_count_docs(val_fno)
    test_root_counts = root_counts_from_fno_count_docs(test_fno)
    resolved_vocab_size = int(metadata.get("vocab_size", int(vocab_size)))
    resolved_n_regimes = int(metadata.get("n_regimes", int(n_regimes)))
    block_by_token = palette_block_map(
        vocab_size=int(resolved_vocab_size),
        n_regimes=int(resolved_n_regimes),
    )
    metadata.update(
        {
            "data_source": "markov",
            "benchmark": str(benchmark),
            "doc_tokens": int(doc_tokens),
            "leaf_tokens": int(leaf_tokens),
            "train_docs": int(len(train_flat)),
            "val_docs": int(len(val_flat)),
            "test_docs": int(len(test_flat)),
            "vocab_size": int(resolved_vocab_size),
            "n_regimes": int(resolved_n_regimes),
            "seed": int(seed),
        }
    )
    return ContextualMarkovSplits(
        train_docs=train_flat,
        val_docs=val_flat,
        test_docs=test_flat,
        train_root_counts=train_root_counts,
        val_root_counts=val_root_counts,
        test_root_counts=test_root_counts,
        block_by_token=block_by_token,
        metadata=metadata,
    )


def build_contextual_response_dataset(
    flat_docs: Sequence[Sequence[int]],
    *,
    block_by_token: Sequence[int],
    vocab_size: int,
    target_scale: float,
    samples_per_doc: int,
    fragment_len: int,
    response_signature_contexts: int,
    seed: int,
    context_left_tokens: Sequence[Sequence[int]] | None = None,
    context_right_tokens: Sequence[Sequence[int]] | None = None,
) -> ContextualResponseDataset:
    """Build Markov two-sided response signatures.

    Compatibility wrapper around ``build_contextual_query_dataset``.  New code
    should prefer the generic builder plus an explicit ``ContextualQueryProblem``.
    """

    if not flat_docs:
        raise ValueError("need at least one document")
    if int(samples_per_doc) <= 0:
        raise ValueError("samples_per_doc must be positive")
    if int(response_signature_contexts) <= 0:
        raise ValueError("response_signature_contexts must be positive")

    if (context_left_tokens is None) != (context_right_tokens is None):
        raise ValueError("context_left_tokens and context_right_tokens must be provided together")
    problem = MarkovTwoSidedContextProblem(
        block_by_token=block_by_token,
        vocab_size=int(vocab_size),
        target_scale=float(target_scale),
    )
    if context_left_tokens is None:
        contexts = None
    else:
        left_contexts = [tuple(int(tok) for tok in x) for x in context_left_tokens]
        right_contexts = [tuple(int(tok) for tok in x) for x in context_right_tokens or ()]
        if len(left_contexts) != int(response_signature_contexts) or len(right_contexts) != int(
            response_signature_contexts
        ):
            raise ValueError("provided context banks must have response_signature_contexts entries")
        contexts = tuple(
            MarkovTwoSidedContext(left_tokens=left, right_tokens=right)
            for left, right in zip(left_contexts, right_contexts)
        )
    return build_contextual_query_dataset(
        flat_docs,
        problem=problem,
        samples_per_source=int(samples_per_doc),
        item_len=int(fragment_len),
        n_contexts=int(response_signature_contexts),
        seed=int(seed),
        contexts=contexts,
    )


def _safe_corrcoef(preds: np.ndarray, truths: np.ndarray) -> float:
    if preds.size < 2 or truths.size < 2:
        return float("nan")
    if float(np.std(preds)) <= 0.0 or float(np.std(truths)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(preds.astype(np.float64), truths.astype(np.float64))[0, 1])


def _collision_rate(
    states: np.ndarray,
    responses: np.ndarray,
    *,
    state_eps: float = 1e-5,
    response_eps: float = 1e-4,
    max_n: int = 4096,
) -> float:
    n = int(states.shape[0])
    if n < 2:
        return float("nan")
    # The full pairwise distance is O(N^2) memory; at N=10^5 the (N, N, D)
    # broadcast can want >1 TiB. Subsample to ``max_n`` so the diagnostic
    # remains tractable; the collision rate is a sanity check, not a
    # primary metric, so a uniform random subset is sufficient.
    if n > int(max_n):
        rng = np.random.default_rng(0)
        sel = rng.choice(int(n), size=int(max_n), replace=False)
        states = states[sel]
        responses = responses[sel]
        n = int(max_n)
    state_dist = np.linalg.norm(states[:, None, :] - states[None, :, :], axis=-1)
    response_dist = np.linalg.norm(responses[:, None, :] - responses[None, :, :], axis=-1)
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    bad = (state_dist <= float(state_eps)) & (response_dist > float(response_eps)) & upper
    denom = int(np.sum(upper))
    return float(np.sum(bad) / max(1, denom))


def _response_diagnostics(
    *,
    preds: np.ndarray,
    truths: np.ndarray,
    states: np.ndarray,
) -> dict[str, float | int]:
    pred = np.asarray(preds, dtype=np.float64)
    truth = np.asarray(truths, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(f"prediction/truth shape mismatch: {pred.shape} vs {truth.shape}")
    err = pred - truth
    flat_pred = pred.reshape(-1)
    flat_truth = truth.reshape(-1)
    return {
        "n": int(pred.shape[0]),
        "state_dim": int(states.shape[-1]) if np.asarray(states).ndim >= 2 else 0,
        "response_signature_contexts": int(pred.shape[1]) if pred.ndim == 2 else 0,
        "contextual_mae": float(np.mean(np.abs(err))) if err.size else float("nan"),
        "contextual_mse": float(np.mean(err * err)) if err.size else float("nan"),
        "pred_mean": float(np.mean(flat_pred)) if flat_pred.size else float("nan"),
        "pred_std": float(np.std(flat_pred)) if flat_pred.size else float("nan"),
        "truth_mean": float(np.mean(flat_truth)) if flat_truth.size else float("nan"),
        "truth_std": float(np.std(flat_truth)) if flat_truth.size else float("nan"),
        "pred_truth_corr": _safe_corrcoef(flat_pred, flat_truth),
        "collision_rate": _collision_rate(np.asarray(states, dtype=np.float64), truth),
    }


def _flatten_diagnostic_rows(
    name: str,
    values: np.ndarray,
    *,
    n_rows: int | None = None,
) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError(f"{name} must have a leading row dimension")
    if n_rows is not None and int(arr.shape[0]) != int(n_rows):
        raise ValueError(
            f"{name} row count mismatch: got {int(arr.shape[0])}, expected {int(n_rows)}"
        )
    return arr.reshape(int(arr.shape[0]), -1)


def _summary_collision_diagnostics(
    *,
    prefix: str,
    states: np.ndarray,
    responses: np.ndarray,
    state_eps: float,
    response_eps: float,
    max_n: int = 4096,
) -> dict[str, float | int]:
    n = int(states.shape[0])
    dim = int(states.shape[1]) if states.ndim == 2 else 0
    base = {
        f"{prefix}_state_dim": dim,
        f"{prefix}_state_eps": float(state_eps),
        f"{prefix}_response_eps": float(response_eps),
    }
    if n < 2:
        base.update(
            {
                f"{prefix}_state_collision_pair_count": 0,
                f"{prefix}_state_collision_rate": float("nan"),
                f"{prefix}_bad_collision_pair_count": 0,
                f"{prefix}_bad_collision_rate": float("nan"),
                f"{prefix}_bad_collision_rate_given_state_collision": float("nan"),
                f"{prefix}_mean_response_distance_at_state_collision": float("nan"),
                f"{prefix}_max_response_distance_at_state_collision": float("nan"),
            }
        )
        return base
    # Same N^2 memory bound as ``_collision_rate``. Subsample for tractability.
    if n > int(max_n):
        rng = np.random.default_rng(0)
        sel = rng.choice(int(n), size=int(max_n), replace=False)
        states = states[sel]
        responses = responses[sel]
        n = int(max_n)

    state_dist = np.linalg.norm(states[:, None, :] - states[None, :, :], axis=-1)
    response_dist = np.linalg.norm(
        responses[:, None, :] - responses[None, :, :],
        axis=-1,
    )
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    state_collisions = (state_dist <= float(state_eps)) & upper
    bad_collisions = state_collisions & (response_dist > float(response_eps))
    denom = int(np.sum(upper))
    state_collision_count = int(np.sum(state_collisions))
    bad_collision_count = int(np.sum(bad_collisions))
    response_at_collisions = response_dist[state_collisions]
    base.update(
        {
            f"{prefix}_state_collision_pair_count": state_collision_count,
            f"{prefix}_state_collision_rate": float(state_collision_count / max(1, denom)),
            f"{prefix}_bad_collision_pair_count": bad_collision_count,
            f"{prefix}_bad_collision_rate": float(bad_collision_count / max(1, denom)),
            f"{prefix}_bad_collision_rate_given_state_collision": (
                float(bad_collision_count / state_collision_count)
                if state_collision_count
                else float("nan")
            ),
            f"{prefix}_mean_response_distance_at_state_collision": (
                float(np.mean(response_at_collisions))
                if response_at_collisions.size
                else float("nan")
            ),
            f"{prefix}_max_response_distance_at_state_collision": (
                float(np.max(response_at_collisions))
                if response_at_collisions.size
                else float("nan")
            ),
        }
    )
    return base


def hybrid_summary_diagnostics(
    *,
    base_states: np.ndarray,
    neural_states: np.ndarray,
    response_signatures: np.ndarray,
    state_eps: float = 1e-5,
    response_eps: float = 1e-4,
) -> dict[str, float | int | str]:
    """Finite-response collision diagnostics for a hybrid summary.

    The helper compares base-only, neural-only, and concatenated hybrid states
    against fixed empirical response signatures.  It does not train a learner
    or estimate mutual information; it is the diagnostic analogue of the
    deterministic Lean condition that summary collisions should not change
    response signatures.
    """

    responses = _flatten_diagnostic_rows("response_signatures", response_signatures)
    n = int(responses.shape[0])
    base = _flatten_diagnostic_rows("base_states", base_states, n_rows=n)
    neural = _flatten_diagnostic_rows("neural_states", neural_states, n_rows=n)
    hybrid = np.concatenate([base, neural], axis=1)
    diagnostics: dict[str, float | int | str] = {
        "diagnostic": "hybrid_summary_finite_response_collision",
        "n": n,
        "response_signature_dim": int(responses.shape[1]),
    }
    diagnostics.update(
        _summary_collision_diagnostics(
            prefix="base",
            states=base,
            responses=responses,
            state_eps=float(state_eps),
            response_eps=float(response_eps),
        )
    )
    diagnostics.update(
        _summary_collision_diagnostics(
            prefix="neural",
            states=neural,
            responses=responses,
            state_eps=float(state_eps),
            response_eps=float(response_eps),
        )
    )
    diagnostics.update(
        _summary_collision_diagnostics(
            prefix="hybrid",
            states=hybrid,
            responses=responses,
            state_eps=float(state_eps),
            response_eps=float(response_eps),
        )
    )
    return diagnostics


def hybrid_summary_diagnostics_for_contextual_sbijax(
    *,
    params: Any,
    apply_fn: Any,
    dataset: ContextualResponseDataset,
    base_states: np.ndarray,
    state_eps: float = 1e-5,
    response_eps: float = 1e-4,
) -> dict[str, Any]:
    """Evaluate hybrid diagnostics using learned contextual sbijax states."""

    deps = _require_contextual_sbi()
    jnp = deps.jnp
    tokens = jnp.asarray(dataset.item_tokens, dtype=jnp.int32)
    responses = jnp.asarray(
        dataset.response_signatures.reshape(dataset.response_signatures.shape[0], -1),
        dtype=jnp.float32,
    )
    states, _preds, _slice_pred, _critic = apply_fn(params, tokens, responses)
    diagnostics = hybrid_summary_diagnostics(
        base_states=np.asarray(base_states),
        neural_states=np.asarray(states),
        response_signatures=dataset.response_signatures,
        state_eps=float(state_eps),
        response_eps=float(response_eps),
    )
    diagnostics.update(
        {
            "problem_id": str(dataset.metadata.get("problem_id", "")),
            "context_kind": str(dataset.metadata.get("context_kind", "")),
        }
    )
    return diagnostics


def _unit_slice_matrix(deps: SimpleNamespace, *, key: Any, n_contexts: int, n_slices: int) -> Any:
    jnp = deps.jnp
    raw = deps.jr.normal(key, (int(n_contexts), int(n_slices)))
    return raw / jnp.linalg.norm(raw, axis=0, keepdims=True).clip(min=1e-6)


def _make_forward(
    deps: SimpleNamespace,
    *,
    vocab_size: int,
    embedding_dim: int,
    state_dim: int,
    hidden_dim: int,
    response_signature_contexts: int,
    response_signature_slices: int,
):
    hk = deps.hk
    jax = deps.jax
    jnp = deps.jnp
    pad_id = int(vocab_size)
    k = int(response_signature_contexts)
    s = max(1, int(response_signature_slices))

    def forward(tokens, responses):
        embed = hk.Embed(vocab_size=int(vocab_size) + 1, embed_dim=int(embedding_dim))
        emb = embed(tokens)
        mask = (tokens != pad_id).astype(jnp.float32)[..., None]
        pooled = (emb * mask).sum(axis=1) / mask.sum(axis=1).clip(min=1.0)
        state = hk.nets.MLP(
            [int(hidden_dim), int(hidden_dim), int(state_dim)],
            activation=jax.nn.relu,
            activate_final=False,
            name="g_state",
        )(pooled)
        response_pred = hk.nets.MLP(
            [int(hidden_dim), k],
            activation=jax.nn.relu,
            activate_final=False,
            name="context_readout_f",
        )(state)
        slice_pred = hk.nets.MLP(
            [int(hidden_dim), s],
            activation=jax.nn.relu,
            activate_final=False,
            name="nasss_slice_readout",
        )(state)
        critic_in = jnp.concatenate([state, responses], axis=-1)
        critic = hk.nets.MLP(
            [int(hidden_dim), 1],
            activation=jax.nn.relu,
            activate_final=False,
            name="nass_critic",
        )(critic_in).squeeze(-1)
        return state, response_pred, slice_pred, critic

    return hk.without_apply_rng(hk.transform(forward))


def _fit_contextual_sbijax_repo(
    train: ContextualResponseDataset,
    val: ContextualResponseDataset,
    *,
    config: ContextualSBIJAXConfig,
) -> ContextualSBIJAXResult:
    """Fit the repo-owned mirrored NASS/NASSS objective."""

    method = str(config.method)
    if method not in {"nass", "nasss"}:
        raise ValueError("method must be 'nass' or 'nasss'")
    if int(train.response_signatures.shape[1]) != int(config.response_signature_contexts):
        raise ValueError("config response_signature_contexts does not match train data")
    if int(val.response_signatures.shape[1]) != int(config.response_signature_contexts):
        raise ValueError("config response_signature_contexts does not match val data")
    train_response_dim = int(np.prod(train.response_signatures.shape[1:]))
    val_response_dim = int(np.prod(val.response_signatures.shape[1:]))
    if train_response_dim != val_response_dim:
        raise ValueError("train and val response signatures must flatten to the same dimension")

    deps = _require_contextual_sbi()
    _ = deps.sbijax.NASSS if method == "nasss" else deps.sbijax.NASS
    jax = deps.jax
    jnp = deps.jnp
    jr = deps.jr
    optax = deps.optax

    n_slices = (
        int(train_response_dim)
        if int(config.response_signature_slices) <= 0
        else int(config.response_signature_slices)
    )
    forward = _make_forward(
        deps,
        vocab_size=int(config.vocab_size),
        embedding_dim=int(config.embedding_dim),
        state_dim=int(config.state_dim),
        hidden_dim=int(config.hidden_dim),
        response_signature_contexts=int(train_response_dim),
        response_signature_slices=int(n_slices),
    )
    key = jr.PRNGKey(int(config.seed))
    init_key, slice_key, train_key = jr.split(key, 3)
    slice_matrix = _unit_slice_matrix(
        deps,
        key=slice_key,
        n_contexts=int(train_response_dim),
        n_slices=int(n_slices),
    )

    train_tokens = jnp.asarray(train.item_tokens, dtype=jnp.int32)
    train_responses = jnp.asarray(
        train.response_signatures.reshape(train.response_signatures.shape[0], -1),
        dtype=jnp.float32,
    )
    val_tokens = jnp.asarray(val.item_tokens, dtype=jnp.int32)
    val_responses = jnp.asarray(
        val.response_signatures.reshape(val.response_signatures.shape[0], -1),
        dtype=jnp.float32,
    )

    init_n = min(max(1, int(config.batch_size)), int(train_tokens.shape[0]))
    params = forward.init(init_key, train_tokens[:init_n], train_responses[:init_n])
    optimizer = optax.adam(float(config.learning_rate))
    opt_state = optimizer.init(params)

    def loss_parts(params, tokens, responses, rng_key):
        state, response_pred, slice_pred, f_pos = forward.apply(params, tokens, responses)
        contextual_mse = jnp.mean((response_pred - responses) ** 2)
        if method == "nasss":
            target_slices = responses @ slice_matrix
            package_loss = jnp.mean((slice_pred - target_slices) ** 2)
        else:
            idx_neg = jr.permutation(rng_key, responses.shape[0])
            _, _, _, f_neg = forward.apply(params, tokens, responses[idx_neg])
            package_loss = jax.nn.softplus(-f_pos).mean() + jax.nn.softplus(f_neg).mean()
        total = (
            float(config.contextual_loss_weight) * contextual_mse
            + float(config.infomax_loss_weight) * package_loss
        )
        return total, (contextual_mse, package_loss, jnp.mean(state * state))

    @jax.jit
    def step(params, opt_state, tokens, responses, rng_key):
        (loss, aux), grads = jax.value_and_grad(loss_parts, has_aux=True)(
            params, tokens, responses, rng_key
        )
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, aux

    @jax.jit
    def eval_loss(params, tokens, responses, rng_key):
        loss, aux = loss_parts(params, tokens, responses, rng_key)
        return loss, aux

    history: list[dict[str, float | int]] = []
    n_train = int(train_tokens.shape[0])
    batch_size = max(1, int(config.batch_size))
    np_rng = np.random.default_rng(int(config.seed) + 11)
    for iteration in range(int(config.n_iter)):
        order = np_rng.permutation(n_train)
        train_loss_acc = 0.0
        train_contextual_acc = 0.0
        train_package_acc = 0.0
        seen = 0
        iter_key = jr.fold_in(train_key, int(iteration))
        for batch_idx, start in enumerate(range(0, n_train, batch_size)):
            idx = order[start : start + batch_size]
            b_tokens = train_tokens[jnp.asarray(idx, dtype=jnp.int32)]
            b_responses = train_responses[jnp.asarray(idx, dtype=jnp.int32)]
            params, opt_state, batch_loss, aux = step(
                params,
                opt_state,
                b_tokens,
                b_responses,
                jr.fold_in(iter_key, int(batch_idx)),
            )
            weight = int(len(idx))
            train_loss_acc += float(batch_loss) * weight
            train_contextual_acc += float(aux[0]) * weight
            train_package_acc += float(aux[1]) * weight
            seen += weight
        val_loss, val_aux = eval_loss(
            params, val_tokens, val_responses, jr.fold_in(iter_key, 99_991)
        )
        history.append(
            {
                "iteration": int(iteration + 1),
                "train_loss": float(train_loss_acc / max(1, seen)),
                "train_contextual_mse": float(train_contextual_acc / max(1, seen)),
                "train_package_loss": float(train_package_acc / max(1, seen)),
                "val_loss": float(val_loss),
                "val_contextual_mse": float(val_aux[0]),
                "val_package_loss": float(val_aux[1]),
            }
        )

    train_diag = evaluate_contextual_sbijax(
        params=params,
        apply_fn=forward.apply,
        dataset=train,
    )
    val_diag = evaluate_contextual_sbijax(
        params=params,
        apply_fn=forward.apply,
        dataset=val,
    )
    return ContextualSBIJAXResult(
        params=params,
        history=history,
        train_diagnostics=train_diag,
        val_diagnostics=val_diag,
        provenance=contextual_sbijax_provenance(
            method=method,
            response_signature_contexts=int(config.response_signature_contexts),
            response_signature_slices=int(n_slices),
            trainer="repo",
            input_encoding="learned_token_embedding_pool",
            downstream_readout="joint_haiku_mlp_mse",
        ),
        config=config,
        slice_matrix=slice_matrix,
        apply_fn=forward.apply,
    )


def _validate_fit_inputs(
    train: ContextualResponseDataset,
    val: ContextualResponseDataset,
    *,
    config: ContextualSBIJAXConfig,
) -> tuple[str, str, int, int]:
    method = str(config.method)
    if method not in {"nass", "nasss"}:
        raise ValueError("method must be 'nass' or 'nasss'")
    trainer = str(config.trainer)
    if trainer not in {
        "repo",
        "package",
        "theta_supervised",
        "identity_theta",
        "exact_zero_markov",
        "learned_local_laws",
        "posterior",
        "npe",
        "nass_nle",
    }:
        raise ValueError(
            "trainer must be 'repo', 'package', 'theta_supervised', "
            "'identity_theta', 'exact_zero_markov', 'learned_local_laws', "
            "'posterior', 'npe', or 'nass_nle'"
        )
    package_theta = str(config.package_theta)
    if package_theta not in {
        "response_signature",
        "markov_exact_sketch",
        "hll_register_sketch",
    }:
        raise ValueError(
            "package_theta must be 'response_signature', 'markov_exact_sketch', "
            "or 'hll_register_sketch'"
        )
    input_encoding = str(config.input_encoding)
    if input_encoding not in {
        "normalized_token_ids",
        "one_hot_token_ids",
        "regime_ids",
        "regime_one_hot",
        "markov_exact_sketch",
    }:
        raise ValueError(
            "input_encoding must be one of: normalized_token_ids, "
            "one_hot_token_ids, regime_ids, regime_one_hot, markov_exact_sketch"
        )
    summary_activation = str(config.summary_activation)
    if summary_activation not in {
        "relu",
        "tanh",
        "gelu",
        "swish",
        "silu",
        "elu",
        "leaky_relu",
    }:
        raise ValueError(
            "summary_activation must be one of: relu, tanh, gelu, swish, " "silu, elu, leaky_relu"
        )
    if int(train.response_signatures.shape[1]) != int(config.response_signature_contexts):
        raise ValueError("config response_signature_contexts does not match train data")
    if int(val.response_signatures.shape[1]) != int(config.response_signature_contexts):
        raise ValueError("config response_signature_contexts does not match val data")
    if int(config.posterior_samples) <= 0:
        raise ValueError("posterior_samples must be positive")
    if int(config.posterior_eval_samples) < 0:
        raise ValueError("posterior_eval_samples must be non-negative")
    if int(config.posterior_eval_batch_size) < 0:
        raise ValueError("posterior_eval_batch_size must be non-negative")
    if str(config.posterior_sampler) not in {"nuts", "slice"}:
        raise ValueError("posterior_sampler must be 'nuts' or 'slice'")
    if str(config.posterior_estimator) not in {
        "npe",
        "fmpe",
        "cmpe",
        "nle",
        "snle",
        "nre",
    }:
        raise ValueError("posterior_estimator must be one of: npe, fmpe, cmpe, nle, snle, nre")
    if str(config.density_family) not in {"mdn", "maf", "spf", "cnf", "cm", "resnet"}:
        raise ValueError("density_family must be one of: mdn, maf, spf, cnf, cm, resnet")
    if int(config.density_components) <= 0:
        raise ValueError("density_components must be positive")
    if str(config.local_law_supervision_mode) not in {
        "dense_exact",
        "sparse_ipw",
        "dual",
    }:
        raise ValueError("local_law_supervision_mode must be dense_exact, sparse_ipw, or dual")
    if str(config.local_law_summary_family) not in {
        "mlp",
        "affine_probe",
        "regime_transition_sum",
        "jax_fno",
        "norax_fno",
    }:
        raise ValueError(
            "local_law_summary_family must be 'mlp', 'affine_probe', "
            "'regime_transition_sum', 'jax_fno', or 'norax_fno'"
        )
    if int(config.local_law_summary_fno_n_modes) <= 0:
        raise ValueError("local_law_summary_fno_n_modes must be positive")
    if int(config.local_law_summary_fno_n_layers) <= 0:
        raise ValueError("local_law_summary_fno_n_layers must be positive")
    if int(config.hll_precision) < 4:
        raise ValueError("hll_precision must be at least 4")
    if int(config.hll_hash_bits) <= int(config.hll_precision) + 1:
        raise ValueError("hll_hash_bits must be at least hll_precision + 2")
    if float(config.local_law_hll_estimate_weight) < 0.0:
        raise ValueError("local_law_hll_estimate_weight must be non-negative")
    if str(config.local_law_summary_fno_pooling_mode) not in {"sum", "mean"}:
        raise ValueError("local_law_summary_fno_pooling_mode must be 'sum' or 'mean'")
    if str(config.law_architecture) not in {
        "analytic",
        "learned_merge",
        "learned_decoder",
        "fully_learned",
    }:
        raise ValueError(
            "law_architecture must be analytic, learned_merge, learned_decoder, " "or fully_learned"
        )
    if str(config.c2_merge_target) not in {"theta", "self_consistency"}:
        raise ValueError("c2_merge_target must be 'theta' or 'self_consistency'")
    if bool(config.local_law_count_only):
        if str(config.law_architecture) != "fully_learned":
            raise ValueError(
                "local_law_count_only=True requires "
                "law_architecture='fully_learned' (analytic merge/decoder "
                "expect sketch-shape state, which is incompatible with an "
                "arbitrary learned rep_dim)"
            )
    if int(config.local_law_rep_dim) < 0:
        raise ValueError("local_law_rep_dim must be non-negative")
    if int(config.local_law_summary_dim) < 0:
        raise ValueError("local_law_summary_dim must be non-negative")
    if str(config.local_law_state_decoder_head) not in {"mlp", "linear"}:
        raise ValueError("local_law_state_decoder_head must be 'mlp' or 'linear'")
    if bool(config.local_law_explicit_state_decoder):
        if bool(config.local_law_count_only):
            raise ValueError(
                "local_law_explicit_state_decoder=True is incompatible with "
                "local_law_count_only=True; count_only already uses a scalar "
                "readout from a learned rep"
            )
        if str(config.law_architecture) not in {"learned_merge", "fully_learned"}:
            raise ValueError(
                "local_law_explicit_state_decoder=True requires "
                "law_architecture='learned_merge' or 'fully_learned' because "
                "analytic merge is defined in decoded theta space, not learned "
                "summary space"
            )
    if str(config.local_law_merge_loss) not in {"mse", "nass_jsd", "nasss_jsd"}:
        raise ValueError("local_law_merge_loss must be 'mse', 'nass_jsd', or 'nasss_jsd'")
    if str(config.merge_family) not in {"mlp", "fno_rep"}:
        raise ValueError("merge_family must be 'mlp' or 'fno_rep'")
    if str(config.decoder_head) not in {"mlp", "linear"}:
        raise ValueError("decoder_head must be 'mlp' or 'linear'")
    if int(config.merge_fno_n_modes) < 1:
        raise ValueError("merge_fno_n_modes must be >= 1")
    if int(config.merge_fno_n_layers) < 1:
        raise ValueError("merge_fno_n_layers must be >= 1")
    if int(config.merge_fno_hidden_channels) < 1:
        raise ValueError("merge_fno_hidden_channels must be >= 1")
    if int(config.merge_nasss_n_slices) < 1:
        raise ValueError("merge_nasss_n_slices must be >= 1")
    if int(config.learned_merge_hidden_dim) < 0:
        raise ValueError("learned_merge_hidden_dim must be non-negative")
    if int(config.learned_decoder_hidden_dim) < 0:
        raise ValueError("learned_decoder_hidden_dim must be non-negative")
    for name in (
        "local_law_leaf_rate",
        "local_law_merge_rate",
        "local_law_idempotence_rate",
    ):
        value = float(getattr(config, name))
        if value < 0.0 or value > 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    train_response_dim = int(np.prod(train.response_signatures.shape[1:]))
    val_response_dim = int(np.prod(val.response_signatures.shape[1:]))
    if train_response_dim != val_response_dim:
        raise ValueError("train and val response signatures must flatten to the same dimension")
    n_slices = (
        int(train_response_dim)
        if int(config.response_signature_slices) <= 0
        else int(config.response_signature_slices)
    )
    return method, trainer, train_response_dim, n_slices


def _sbijax_activation(deps: SimpleNamespace, name: str):
    """Resolve a package-native summary-network activation by name."""

    jax_nn = deps.jax.nn
    activation = str(name)
    if activation == "relu":
        return jax_nn.relu
    if activation == "tanh":
        return deps.jnp.tanh
    if activation == "gelu":
        return jax_nn.gelu
    if activation in {"swish", "silu"}:
        return jax_nn.swish
    if activation == "elu":
        return jax_nn.elu
    if activation == "leaky_relu":
        return lambda x: jax_nn.leaky_relu(x, negative_slope=0.01)
    raise ValueError(f"unknown summary_activation {activation!r}")


def _flatten_response_signatures(dataset: ContextualResponseDataset) -> np.ndarray:
    return np.asarray(
        dataset.response_signatures.reshape(dataset.response_signatures.shape[0], -1),
        dtype=np.float32,
    )


def _package_theta_array(
    dataset: ContextualResponseDataset,
    *,
    package_theta: str,
) -> np.ndarray:
    if str(package_theta) == "response_signature":
        return _flatten_response_signatures(dataset)
    targets = dict(dataset.package_theta_targets or {})
    if str(package_theta) not in targets:
        raise ValueError(
            f"dataset is missing package theta target {package_theta!r}; "
            "attach it with with_package_theta_target(...)"
        )
    out = np.asarray(targets[str(package_theta)], dtype=np.float32)
    if out.ndim != 2:
        raise ValueError(f"package theta target must be 2D; got shape {tuple(out.shape)}")
    return out


def _metadata_block_by_token(dataset: ContextualResponseDataset) -> list[int] | None:
    raw = dataset.metadata.get("block_by_token")
    if raw is None:
        return None
    return [int(x) for x in raw]


def _encode_tokens_for_sbijax_package(
    deps: SimpleNamespace,
    tokens: Any,
    *,
    vocab_size: int,
    input_encoding: str,
    block_by_token: Sequence[int] | None = None,
    n_regimes: int | None = None,
    target_scale: float | None = None,
) -> Any:
    """Encode tokens for package-direct NASS/NASSS."""

    jnp = deps.jnp
    token_array = jnp.asarray(tokens, dtype=jnp.int32)
    encoding = str(input_encoding)
    if encoding == "normalized_token_ids":
        denom = float(max(1, int(vocab_size)))
        return token_array.astype(jnp.float32) / denom
    if encoding == "one_hot_token_ids":
        return jnp.reshape(
            deps.jax.nn.one_hot(token_array, int(vocab_size) + 1, dtype=jnp.float32),
            (
                int(token_array.shape[0]),
                int(token_array.shape[1]) * (int(vocab_size) + 1),
            ),
        )

    if block_by_token is None:
        raise ValueError(f"{encoding} requires block_by_token metadata")
    block_map = np.asarray([int(x) for x in block_by_token], dtype=np.int32)
    n_blocks = int(max(block_map)) + 1 if n_regimes is None else int(n_regimes)
    if n_blocks <= 0:
        raise ValueError("n_regimes must be positive for regime encodings")
    mapped = np.full((int(vocab_size) + 1,), int(n_blocks), dtype=np.int32)
    mapped[: min(int(vocab_size), int(block_map.shape[0]))] = block_map[
        : min(int(vocab_size), int(block_map.shape[0]))
    ]
    regime_ids = jnp.asarray(mapped, dtype=jnp.int32)[token_array]
    if encoding == "regime_ids":
        return regime_ids.astype(jnp.float32) / float(max(1, n_blocks))
    if encoding == "regime_one_hot":
        return jnp.reshape(
            deps.jax.nn.one_hot(regime_ids, n_blocks + 1, dtype=jnp.float32),
            (
                int(regime_ids.shape[0]),
                int(regime_ids.shape[1]) * (int(n_blocks) + 1),
            ),
        )
    if encoding == "markov_exact_sketch":
        if target_scale is None:
            raise ValueError("markov_exact_sketch encoding requires target_scale")
        valid = token_array != int(vocab_size)
        pair_valid = valid[:, 1:] & valid[:, :-1]
        transitions = (regime_ids[:, 1:] != regime_ids[:, :-1]) & pair_valid
        count_norm = jnp.sum(transitions.astype(jnp.float32), axis=1, keepdims=True) / float(
            target_scale
        )
        first_idx = jnp.argmax(valid.astype(jnp.int32), axis=1)
        lengths = jnp.sum(valid.astype(jnp.int32), axis=1)
        last_idx = jnp.maximum(lengths - 1, 0)
        row_idx = jnp.arange(token_array.shape[0])
        first_regime = regime_ids[row_idx, first_idx]
        last_regime = regime_ids[row_idx, last_idx]
        first_oh = deps.jax.nn.one_hot(first_regime, n_blocks, dtype=jnp.float32)
        last_oh = deps.jax.nn.one_hot(last_regime, n_blocks, dtype=jnp.float32)
        return jnp.concatenate([count_norm, first_oh, last_oh], axis=1)
    raise ValueError(f"unknown input_encoding {encoding!r}")


def _encode_dataset_items_for_sbijax_package(
    deps: SimpleNamespace,
    dataset: ContextualResponseDataset,
    *,
    config: ContextualSBIJAXConfig,
) -> Any:
    block_by_token = _metadata_block_by_token(dataset)
    n_regimes_raw = dataset.metadata.get("n_regimes")
    n_regimes = None if n_regimes_raw is None else int(n_regimes_raw)
    return _encode_tokens_for_sbijax_package(
        deps,
        dataset.item_tokens,
        vocab_size=int(config.vocab_size),
        input_encoding=str(config.input_encoding),
        block_by_token=block_by_token,
        n_regimes=n_regimes,
        target_scale=float(dataset.target_scale),
    )


def _dummy_response_signature_model_fns(
    deps: SimpleNamespace,
    *,
    response_signature_dim: int,
):
    """Dummy SBI prior/simulator required by the package constructor.

    The package-direct trainer below passes prepared Markov data directly to
    ``fit``.  These functions provide the constructor's expected surface but
    are not used to generate the training rows for this milestone.
    """

    try:
        from tensorflow_probability.substrates.jax import distributions as tfd
    except ImportError as exc:  # pragma: no cover - optional extra pins this
        raise ImportError(CONTEXTUAL_SBI_INSTALL_MSG) from exc

    jnp = deps.jnp

    def prior_fn():
        return tfd.JointDistributionNamed(
            {
                "theta": tfd.Independent(
                    tfd.Normal(
                        loc=jnp.zeros((int(response_signature_dim),), dtype=jnp.float32),
                        scale=jnp.ones((int(response_signature_dim),), dtype=jnp.float32),
                    ),
                    reinterpreted_batch_ndims=1,
                )
            }
        )

    def simulator_fn(seed, theta):
        del seed
        if isinstance(theta, Mapping):
            return theta["theta"]
        return theta

    return prior_fn, simulator_fn


def _make_package_readout(
    deps: SimpleNamespace,
    *,
    response_signature_dim: int,
    hidden_dim: int,
):
    hk = deps.hk
    jax = deps.jax

    def forward(states):
        return hk.nets.MLP(
            [int(hidden_dim), int(hidden_dim), int(response_signature_dim)],
            activation=jax.nn.relu,
            activate_final=False,
            name="f_readout",
        )(states)

    return hk.without_apply_rng(hk.transform(forward))


def _make_theta_summary_net(
    deps: SimpleNamespace,
    *,
    theta_dim: int,
    hidden_dim: int,
):
    hk = deps.hk
    jax = deps.jax

    def forward(features):
        return hk.nets.MLP(
            [int(hidden_dim), int(hidden_dim), int(theta_dim)],
            activation=jax.nn.relu,
            activate_final=False,
            name="g_theta_summary",
        )(features)

    return hk.without_apply_rng(hk.transform(forward))


def _make_regime_transition_sum_summary_net(
    deps: SimpleNamespace,
    *,
    n_regimes: int,
    fragment_len: int,
    hidden_dim: int,
    target_scale: float,
):
    """Weakly structured summary for flattened regime-one-hot fragments.

    The network receives regime indicators, not the exact Markov sketch. It
    learns a soft adjacent-transition detector and endpoint readouts, then
    returns the same sketch-shaped state used by the local laws.
    """

    hk = deps.hk
    jax = deps.jax
    jnp = deps.jnp
    n = int(n_regimes)
    width = int(n) + 1
    frag = int(fragment_len)

    def forward(features):
        batch = int(features.shape[0])
        x = jnp.reshape(features, (batch, frag, width))
        valid = 1.0 - x[:, :, n]
        if frag > 1:
            left = x[:, :-1, :]
            right = x[:, 1:, :]
            pair_features = jnp.concatenate([left, right], axis=-1)
            pair_valid = valid[:, :-1] * valid[:, 1:]
            edge_logits = hk.nets.MLP(
                [int(hidden_dim), int(hidden_dim), 1],
                activation=jax.nn.relu,
                activate_final=False,
                name="g_regime_transition_edges",
            )(pair_features).squeeze(-1)
            count_norm = jnp.sum(jax.nn.sigmoid(edge_logits) * pair_valid, axis=1) / float(
                target_scale
            )
        else:
            count_norm = jnp.zeros((batch,), dtype=features.dtype)

        lengths = jnp.sum(valid.astype(jnp.int32), axis=1)
        first_idx = jnp.argmax(valid.astype(jnp.int32), axis=1)
        last_idx = jnp.maximum(lengths - 1, 0)
        row_idx = jnp.arange(batch)
        first_features = x[row_idx, first_idx, :]
        last_features = x[row_idx, last_idx, :]
        first_logits = hk.nets.MLP(
            [int(hidden_dim), n],
            activation=jax.nn.relu,
            activate_final=False,
            name="g_regime_first_head",
        )(first_features)
        last_logits = hk.nets.MLP(
            [int(hidden_dim), n],
            activation=jax.nn.relu,
            activate_final=False,
            name="g_regime_last_head",
        )(last_features)
        first_probs = jax.nn.softmax(first_logits, axis=-1)
        last_probs = jax.nn.softmax(last_logits, axis=-1)
        return jnp.concatenate([count_norm[:, None], first_probs, last_probs], axis=1)

    return hk.without_apply_rng(hk.transform(forward))


def _enriched_pool(
    *,
    x,
    pool_mode: str,
    jnp,
):
    """Concatenate ``[aggregate-pool, first-position, last-position]``.

    A general (non-Markov-specific) read-off that preserves both the global
    aggregate signal (sum/mean over the spatial axis) and endpoint signals.
    Pure aggregate pools (sum or mean alone) destroy positional information
    needed to recover first/last-style endpoint summaries; concatenating
    the explicit first and last positions restores that capability without
    baking in any task-specific structure.

    ``x`` is `(B, L, C)`. Returns `(B, 3 * C)`.
    """
    if str(pool_mode) == "mean":
        agg = jnp.mean(x, axis=1)
    else:
        agg = jnp.sum(x, axis=1)
    first = x[:, 0, :]
    last = x[:, -1, :]
    return jnp.concatenate([agg, first, last], axis=-1)


def _make_jax_fno_summary_net(
    deps: SimpleNamespace,
    *,
    fragment_len: int,
    input_width: int,
    fno_width: int,
    n_modes: int,
    n_layers: int,
    pooling_mode: str,
    output_dim: int,
):
    """Internal JAX 1-D FNO summary with enriched-pool read-off.

    This is intentionally self-contained. The implementation is inspired by
    standard FNO packages such as ``norax``, but it does not import them or make
    them runtime dependencies. Inputs are flattened leaf features; the network
    reshapes them to ``(B, L, C)``, appends a normalized position channel,
    applies spectral residual blocks, then decodes
    ``[aggregate-pool, first-position, last-position]`` to the Markov sketch
    shape.
    """

    hk = deps.hk
    jax = deps.jax
    jnp = deps.jnp
    L = int(fragment_len)
    M = max(1, min(int(n_modes), L // 2 + 1))
    width_in = int(input_width)
    fwidth = int(fno_width)
    out_dim = int(output_dim)
    pool_mode = str(pooling_mode)
    n_lay = int(n_layers)

    def _spectral_block(x, layer_idx: int):
        pointwise = hk.Linear(fwidth, name=f"fno_pointwise_{layer_idx}")(x)
        x_ft = jnp.fft.rfft(x, axis=1)
        n_freq = int(x_ft.shape[1])
        m = max(1, min(M, n_freq))
        weight_scale = 1.0 / max(1.0, float(fwidth))
        real = hk.get_parameter(
            f"fno_spectral_real_{layer_idx}",
            shape=(m, fwidth, fwidth),
            init=hk.initializers.RandomNormal(stddev=weight_scale),
        )
        imag = hk.get_parameter(
            f"fno_spectral_imag_{layer_idx}",
            shape=(m, fwidth, fwidth),
            init=hk.initializers.RandomNormal(stddev=weight_scale),
        )
        weights = real + 1j * imag
        low = jnp.einsum("bmi,mio->bmo", x_ft[:, :m, :], weights)
        out_ft = jnp.zeros((x.shape[0], n_freq, fwidth), dtype=jnp.complex64)
        out_ft = out_ft.at[:, :m, :].set(low)
        spectral = jnp.fft.irfft(out_ft, n=L, axis=1)
        return jax.nn.gelu(pointwise + spectral)

    def forward(features):
        batch = int(features.shape[0])
        x = jnp.reshape(features, (batch, L, width_in))
        if L > 1:
            pos = jnp.linspace(0.0, 1.0, L, dtype=features.dtype)
        else:
            pos = jnp.zeros((1,), dtype=features.dtype)
        pos = jnp.broadcast_to(pos[None, :, None], (batch, L, 1))
        x = jnp.concatenate([x, pos], axis=-1)
        x = hk.Linear(fwidth, name="fno_lift")(x)
        for layer_idx in range(n_lay):
            x = _spectral_block(x, layer_idx)
        pooled = _enriched_pool(x=x, pool_mode=pool_mode, jnp=jnp)
        return hk.Linear(out_dim, name="fno_readout")(pooled)

    return hk.without_apply_rng(hk.transform(forward))


def _make_jax_fno_summary_package_aux_net(
    deps: SimpleNamespace,
    *,
    fragment_len: int,
    input_width: int,
    fno_width: int,
    n_modes: int,
    n_layers: int,
    pooling_mode: str,
    output_dim: int,
    hidden_dim: int,
    response_signature_dim: int,
    response_signature_slices: int,
):
    """JAX FNO summary backbone + sbijax NASS/NASSS heads.

    Same FNO + enriched-pool body as :func:`_make_jax_fno_summary_net`, but
    additionally produces ``slice_pred`` (for NASSS sliced contrastive
    supervision) and ``critic`` (for NASS InfoNCE-style contrastive
    supervision) on top of the pooled state. This lets us actually train
    against the sbijax package objective with the FNO encoder, instead of
    using sbijax NASS/NASSS purely as a post-hoc diagnostic.
    """

    hk = deps.hk
    jax = deps.jax
    jnp = deps.jnp
    L = int(fragment_len)
    M = max(1, min(int(n_modes), L // 2 + 1))
    width_in = int(input_width)
    fwidth = int(fno_width)
    out_dim = int(output_dim)
    pool_mode = str(pooling_mode)
    n_lay = int(n_layers)
    s = max(1, int(response_signature_slices))

    def _spectral_block(x, layer_idx: int):
        pointwise = hk.Linear(fwidth, name=f"fno_pointwise_{layer_idx}")(x)
        x_ft = jnp.fft.rfft(x, axis=1)
        n_freq = int(x_ft.shape[1])
        m = max(1, min(M, n_freq))
        weight_scale = 1.0 / max(1.0, float(fwidth))
        real = hk.get_parameter(
            f"fno_spectral_real_{layer_idx}",
            shape=(m, fwidth, fwidth),
            init=hk.initializers.RandomNormal(stddev=weight_scale),
        )
        imag = hk.get_parameter(
            f"fno_spectral_imag_{layer_idx}",
            shape=(m, fwidth, fwidth),
            init=hk.initializers.RandomNormal(stddev=weight_scale),
        )
        weights = real + 1j * imag
        low = jnp.einsum("bmi,mio->bmo", x_ft[:, :m, :], weights)
        out_ft = jnp.zeros((x.shape[0], n_freq, fwidth), dtype=jnp.complex64)
        out_ft = out_ft.at[:, :m, :].set(low)
        spectral = jnp.fft.irfft(out_ft, n=L, axis=1)
        return jax.nn.gelu(pointwise + spectral)

    def forward(features, responses):
        batch = int(features.shape[0])
        x = jnp.reshape(features, (batch, L, width_in))
        if L > 1:
            pos = jnp.linspace(0.0, 1.0, L, dtype=features.dtype)
        else:
            pos = jnp.zeros((1,), dtype=features.dtype)
        pos = jnp.broadcast_to(pos[None, :, None], (batch, L, 1))
        x = jnp.concatenate([x, pos], axis=-1)
        x = hk.Linear(fwidth, name="fno_lift")(x)
        for layer_idx in range(n_lay):
            x = _spectral_block(x, layer_idx)
        pooled = _enriched_pool(x=x, pool_mode=pool_mode, jnp=jnp)
        state = hk.Linear(out_dim, name="fno_readout")(pooled)
        # NASSS sliced read-off on top of the FNO state.
        slice_pred = hk.nets.MLP(
            [int(hidden_dim), s],
            activation=jax.nn.relu,
            activate_final=False,
            name="fno_nasss_slice_readout",
        )(state)
        # NASS critic: scalar score on (state, response).
        critic_in = jnp.concatenate([state, responses], axis=-1)
        critic = hk.nets.MLP(
            [int(hidden_dim), 1],
            activation=jax.nn.relu,
            activate_final=False,
            name="fno_nass_critic",
        )(critic_in).squeeze(-1)
        return state, slice_pred, critic

    return hk.without_apply_rng(hk.transform(forward))


def _fno_input_width_for_encoding(
    *,
    encoding: str,
    n_regimes: int,
    vocab_size: int,
) -> int:
    """Channels per spatial position for the FNO summary, by input encoding."""
    if encoding == "regime_one_hot":
        return int(n_regimes) + 1
    if encoding == "one_hot_token_ids":
        return int(vocab_size) + 1
    if encoding in {"regime_ids", "normalized_token_ids"}:
        return 1
    raise ValueError(
        "local_law_summary_family='jax_fno' is not supported for " f"input_encoding={encoding!r}"
    )


def _make_learned_merge_net(
    deps: SimpleNamespace,
    *,
    state_dim: int,
    hidden_dim: int,
):
    """Asymmetric learned merge ``g(s_L, s_R) -> s_M`` (Flax linen).

    The Markov sketch merge is asymmetric (``first`` comes from the left
    sub-state and ``last`` comes from the right). We concatenate
    ``[s_L, s_R]`` rather than averaging so the network is free to learn
    that asymmetry from the local-law objectives alone, without baking the
    Markov rule into the architecture.

    Implemented in Flax linen (haiku is deprecated for new modules). The
    ``init`` / ``apply`` API matches haiku's so existing call sites keep
    working: ``params = net.init(rng, left, right)`` returns a flax
    PyTree (``{'params': {...}}``) that optax updates natively, and
    ``net.apply(params, left, right)`` runs the forward pass.
    """

    fnn = deps.fnn
    jnp = deps.jnp
    state_dim_int = int(state_dim)
    hidden_dim_int = int(hidden_dim)

    class LearnedMergeNet(fnn.Module):

        @fnn.compact
        def __call__(self, left_states, right_states):
            x = jnp.concatenate([left_states, right_states], axis=-1)
            x = fnn.Dense(hidden_dim_int, name="g_learned_merge_l0")(x)
            x = fnn.relu(x)
            x = fnn.Dense(hidden_dim_int, name="g_learned_merge_l1")(x)
            x = fnn.relu(x)
            return fnn.Dense(state_dim_int, name="g_learned_merge_out")(x)

    return LearnedMergeNet()


def _make_learned_decoder_net(
    deps: SimpleNamespace,
    *,
    response_dim: int,
    hidden_dim: int,
    head: str = "mlp",
):
    """Learned ``f(state) -> response_signature`` head (Flax linen).

    Replaces the analytic Markov decoder. ``head='mlp'`` is the default
    2-layer MLP. ``head='linear'`` is a single Dense layer (no hidden
    nonlinearity) — useful for ablating whether the decoder needs to be
    expressive at all when g already produces a "good" rep.
    """

    fnn = deps.fnn
    response_dim_int = int(response_dim)
    hidden_dim_int = int(hidden_dim)
    head_kind = str(head)

    class LearnedDecoderMLP(fnn.Module):

        @fnn.compact
        def __call__(self, states):
            x = fnn.Dense(hidden_dim_int, name="f_learned_decoder_l0")(states)
            x = fnn.relu(x)
            x = fnn.Dense(hidden_dim_int, name="f_learned_decoder_l1")(x)
            x = fnn.relu(x)
            return fnn.Dense(response_dim_int, name="f_learned_decoder_out")(x)

    class LearnedDecoderLinear(fnn.Module):

        @fnn.compact
        def __call__(self, states):
            return fnn.Dense(response_dim_int, name="f_learned_decoder_linear")(states)

    if head_kind == "linear":
        return LearnedDecoderLinear()
    return LearnedDecoderMLP()


def _make_state_decoder_net(
    deps: SimpleNamespace,
    *,
    summary_dim: int,
    theta_dim: int,
    hidden_dim: int,
    head: str = "mlp",
):
    """Explicit paper-notation ``f(z) -> theta`` state decoder."""

    fnn = deps.fnn
    summary_dim_int = int(summary_dim)
    theta_dim_int = int(theta_dim)
    hidden_dim_int = int(hidden_dim)
    head_kind = str(head)

    class StateDecoderMLP(fnn.Module):

        @fnn.compact
        def __call__(self, summaries):
            x = fnn.Dense(hidden_dim_int, name="f_state_decoder_l0")(summaries)
            x = fnn.relu(x)
            x = fnn.Dense(hidden_dim_int, name="f_state_decoder_l1")(x)
            x = fnn.relu(x)
            return fnn.Dense(theta_dim_int, name="f_state_decoder_out")(x)

    class StateDecoderLinear(fnn.Module):

        @fnn.compact
        def __call__(self, summaries):
            return fnn.Dense(theta_dim_int, name="f_state_decoder_linear")(summaries)

    if summary_dim_int <= 0 or theta_dim_int <= 0:
        raise ValueError("summary_dim and theta_dim must be positive")
    if head_kind == "linear":
        return StateDecoderLinear()
    return StateDecoderMLP()


def _make_fno_merge_net(
    deps: SimpleNamespace,
    *,
    state_dim: int,
    hidden_channels: int,
    n_modes: int,
    n_layers: int,
):
    """1D-FNO merge ``g(s_L, s_R) -> s_M`` (Flax linen).

    Spatial axis = ``state_dim`` (the rep dim, length S). Channels =
    (left, right) lifted to ``hidden_channels``. With ``state_dim=256``
    and ``n_modes=32``, this is a real FNO with up to 32 spectral modes
    along the rep dim — qualitatively different from a length-2 merge
    axis, which only has mode 0 (sum) and mode 1 (diff).

    Architecture:
      1. Stack: (B, S, 2) where 2 = (left, right) input channels
      2. Lift: Dense -> (B, S, hidden_channels)
      3. FNO blocks: spectral conv over S + pointwise residual + ReLU
      4. Project: Dense -> (B, S, 1) -> squeeze -> (B, S)
    """

    fnn = deps.fnn
    jnp = deps.jnp
    state_dim_int = int(state_dim)
    hidden_int = max(1, int(hidden_channels))
    n_layers_int = max(1, int(n_layers))
    # Cap modes at the FFT-imposed limit: rfft of length S has S//2+1 real
    # frequency bins.
    max_modes = state_dim_int // 2 + 1
    n_modes_int = max(1, min(int(n_modes), max_modes))

    class FNOMergeNet(fnn.Module):

        @fnn.compact
        def __call__(self, left_states, right_states):
            # Stack as channels: (B, S, 2)
            x = jnp.stack([left_states, right_states], axis=-1)
            # Lift: (B, S, 2) -> (B, S, hidden_channels)
            x = fnn.Dense(hidden_int, name="fno_merge_lift")(x)
            for layer in range(n_layers_int):
                # Spectral conv over the spatial (rep_dim) axis.
                x_ft = jnp.fft.rfft(x, axis=1)  # (B, S//2+1, C) complex
                weight = self.param(
                    f"fno_merge_w_{layer}",
                    fnn.initializers.normal(stddev=1.0 / hidden_int),
                    (n_modes_int, hidden_int, hidden_int, 2),
                )
                w_complex = weight[..., 0] + 1j * weight[..., 1]
                truncated = x_ft[:, :n_modes_int, :]  # (B, modes, C)
                spectral = jnp.einsum("bmc,mco->bmo", truncated, w_complex)
                if n_modes_int < x_ft.shape[1]:
                    pad = jnp.zeros(
                        (x_ft.shape[0], x_ft.shape[1] - n_modes_int, hidden_int),
                        dtype=spectral.dtype,
                    )
                    spectral = jnp.concatenate([spectral, pad], axis=1)
                x_spectral = jnp.fft.irfft(spectral, n=state_dim_int, axis=1)
                # Pointwise residual.
                x_residual = fnn.Dense(hidden_int, name=f"fno_merge_pointwise_{layer}")(x)
                x = fnn.relu(x_spectral + x_residual)
            # Project channels -> 1 then squeeze: (B, S, C) -> (B, S)
            return fnn.Dense(1, name="fno_merge_out")(x).squeeze(-1)

    return FNOMergeNet()


def _make_theta_summary_package_aux_net(
    deps: SimpleNamespace,
    *,
    theta_dim: int,
    hidden_dim: int,
    response_signature_dim: int,
    response_signature_slices: int,
):
    """Theta-summary net with package-style NASS/NASSS auxiliary heads."""

    hk = deps.hk
    jax = deps.jax
    jnp = deps.jnp
    s = max(1, int(response_signature_slices))

    def forward(features, responses):
        state = hk.nets.MLP(
            [int(hidden_dim), int(hidden_dim), int(theta_dim)],
            activation=jax.nn.relu,
            activate_final=False,
            name="g_theta_summary",
        )(features)
        slice_pred = hk.nets.MLP(
            [int(hidden_dim), s],
            activation=jax.nn.relu,
            activate_final=False,
            name="nasss_slice_readout",
        )(state)
        critic_in = jnp.concatenate([state, responses], axis=-1)
        critic = hk.nets.MLP(
            [int(hidden_dim), 1],
            activation=jax.nn.relu,
            activate_final=False,
            name="nass_critic",
        )(critic_in).squeeze(-1)
        return state, slice_pred, critic

    return hk.without_apply_rng(hk.transform(forward))


def _affine_response_probe(
    theta: np.ndarray,
    responses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares affine map from theta rows to response signatures."""

    theta_arr = np.asarray(theta, dtype=np.float64)
    response_arr = np.asarray(responses, dtype=np.float64)
    design = np.concatenate(
        [theta_arr, np.ones((int(theta_arr.shape[0]), 1), dtype=np.float64)],
        axis=1,
    )
    weights, *_ = np.linalg.lstsq(design, response_arr, rcond=None)
    return (
        np.asarray(weights[:-1], dtype=np.float32),
        np.asarray(weights[-1], dtype=np.float32),
    )


def _package_loss_at(losses: np.ndarray, iteration: int, column: int) -> float:
    if losses.size == 0:
        return float("nan")
    row = min(max(0, int(iteration)), int(losses.shape[0]) - 1)
    return float(losses[row, int(column)])


def _posterior_eval_samples(config: ContextualSBIJAXConfig) -> int:
    return (
        int(config.posterior_eval_samples)
        if int(config.posterior_eval_samples) > 0
        else int(config.posterior_samples)
    )


def _posterior_eval_batch_size(config: ContextualSBIJAXConfig) -> int:
    return (
        int(config.posterior_eval_batch_size)
        if int(config.posterior_eval_batch_size) > 0
        else int(config.batch_size)
    )


def _markov_exact_decoder_setup(
    deps: SimpleNamespace,
    dataset: ContextualResponseDataset,
) -> dict[str, Any]:
    block_by_token = _metadata_block_by_token(dataset)
    if block_by_token is None:
        raise ValueError("markov_exact_sketch decoding requires block_by_token metadata")
    n_regimes_raw = dataset.metadata.get("n_regimes")
    n_regimes = int(max(block_by_token)) + 1 if n_regimes_raw is None else int(n_regimes_raw)
    scale = float(dataset.target_scale)
    left_counts = []
    right_counts = []
    left_last = []
    right_first = []
    left_has = []
    right_has = []
    for left, right in zip(
        dataset.context_left_raw,
        dataset.context_right_raw,
        strict=True,
    ):
        left_tokens = [int(tok) for tok in left]
        right_tokens = [int(tok) for tok in right]
        left_counts.append(
            exact_count_for_tokens(left_tokens, block_by_token=block_by_token) / scale
        )
        right_counts.append(
            exact_count_for_tokens(right_tokens, block_by_token=block_by_token) / scale
        )
        left_has.append(1.0 if left_tokens else 0.0)
        right_has.append(1.0 if right_tokens else 0.0)
        left_last.append(int(block_by_token[int(left_tokens[-1])]) if left_tokens else 0)
        right_first.append(int(block_by_token[int(right_tokens[0])]) if right_tokens else 0)
    jnp = deps.jnp
    return {
        "block_by_token": block_by_token,
        "n_regimes": int(n_regimes),
        "scale": float(scale),
        "left_counts": jnp.asarray(left_counts, dtype=jnp.float32),
        "right_counts": jnp.asarray(right_counts, dtype=jnp.float32),
        "left_last": jnp.asarray(left_last, dtype=jnp.int32),
        "right_first": jnp.asarray(right_first, dtype=jnp.int32),
        "left_has": jnp.asarray(left_has, dtype=jnp.float32),
        "right_has": jnp.asarray(right_has, dtype=jnp.float32),
    }


def _responses_from_markov_exact_states(
    states: Any,
    *,
    decoder: Mapping[str, Any],
) -> Any:
    n_regimes = int(decoder["n_regimes"])
    scale = float(decoder["scale"])
    count_norm = states[:, :1]
    first_oh = states[:, 1 : 1 + n_regimes]
    last_oh = states[:, 1 + n_regimes : 1 + 2 * n_regimes]
    left_boundary = decoder["left_has"] * (1.0 - first_oh[:, decoder["left_last"]]) / scale
    right_boundary = decoder["right_has"] * (1.0 - last_oh[:, decoder["right_first"]]) / scale
    return (
        count_norm
        + decoder["left_counts"]
        + decoder["right_counts"]
        + left_boundary
        + right_boundary
    )


def _hll_exact_decoder_setup(
    deps: SimpleNamespace,
    dataset: ContextualResponseDataset,
    *,
    precision: int,
    hash_bits: int,
) -> dict[str, Any]:
    p = int(dataset.metadata.get("hll_precision", int(precision)))
    h = int(dataset.metadata.get("hll_hash_bits", int(hash_bits)))
    _hll_max_register(precision=p, hash_bits=h)
    context_rows = []
    for payload in dataset.context_payloads:
        if str(payload.get("kind", "")) != "hll_union":
            raise ValueError("hll_register_sketch decoding requires hll_union contexts")
        context_rows.append(
            _hll_register_state_from_tokens_np(
                [int(tok) for tok in payload.get("tokens", ())],
                precision=p,
                hash_bits=h,
            )
        )
    jnp = deps.jnp
    return {
        "precision": int(p),
        "hash_bits": int(h),
        "target_scale": float(dataset.target_scale),
        "max_register": float(_hll_max_register(precision=p, hash_bits=h)),
        "context_registers": jnp.asarray(np.asarray(context_rows), dtype=jnp.float32),
    }


def _hll_estimate_from_normalized_registers_jnp(
    deps: SimpleNamespace,
    states: Any,
    *,
    precision: int,
    hash_bits: int,
) -> Any:
    from treepo.hll import _hll_alpha

    jax = deps.jax
    jnp = deps.jnp
    p = int(precision)
    h = int(hash_bits)
    m_int = int(1 << p)
    max_register = float(_hll_max_register(precision=p, hash_bits=h))
    regs = jnp.clip(states, 0.0, 1.0) * max_register
    m = float(m_int)
    z = jnp.sum(jnp.power(2.0, -regs), axis=-1)
    raw = float(_hll_alpha(m_int)) * (m * m) / jnp.maximum(z, 1e-12)
    # Legacy differentiable HLL readout proxy: exact for integer registers and
    # smooth for off-lattice model states.
    zeros = jnp.sum(jnp.maximum(1.0 - regs, 0.0), axis=-1)
    small = m * jnp.log(m / jnp.maximum(zeros, 1e-3))
    use_linear = jax.lax.stop_gradient((raw <= 2.5 * m) & (zeros > 0.5))
    small_or_raw = jnp.where(use_linear, small, raw)
    hash_space = float(2.0**h)
    clipped = jnp.minimum(raw / hash_space, 1.0 - 1e-12)
    large = -hash_space * jnp.log1p(-clipped)
    use_large = jax.lax.stop_gradient(raw > hash_space / 30.0)
    return jnp.where(use_large, large, small_or_raw)


def _responses_from_hll_exact_states(
    states: Any,
    *,
    decoder: Mapping[str, Any],
    deps: SimpleNamespace,
) -> Any:
    context_registers = decoder["context_registers"]
    merged = deps.jnp.maximum(states[:, None, :], context_registers[None, :, :])
    flat = deps.jnp.reshape(
        merged,
        (int(states.shape[0]) * int(context_registers.shape[0]), int(states.shape[1])),
    )
    estimates = _hll_estimate_from_normalized_registers_jnp(
        deps,
        flat,
        precision=int(decoder["precision"]),
        hash_bits=int(decoder["hash_bits"]),
    )
    return deps.jnp.reshape(
        estimates,
        (int(states.shape[0]), int(context_registers.shape[0])),
    ) / float(decoder["target_scale"])


def _markov_merge_split_arrays(
    dataset: ContextualResponseDataset,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return padded left/right item splits plus source row indices."""

    tokens = np.asarray(dataset.item_tokens, dtype=np.int32)
    if tokens.ndim != 2:
        raise ValueError(f"dataset item_tokens must be 2D; got {tuple(tokens.shape)}")
    pad_id = int(dataset.pad_id)
    width = int(tokens.shape[1])
    left_rows: list[np.ndarray] = []
    right_rows: list[np.ndarray] = []
    row_indices: list[int] = []
    for row_idx, row in enumerate(tokens):
        valid = [int(tok) for tok in row if int(tok) != pad_id]
        if len(valid) < 2:
            continue
        mid = max(1, len(valid) // 2)
        left = np.full((width,), pad_id, dtype=np.int32)
        right = np.full((width,), pad_id, dtype=np.int32)
        left[: len(valid[:mid])] = np.asarray(valid[:mid], dtype=np.int32)
        right[: len(valid[mid:])] = np.asarray(valid[mid:], dtype=np.int32)
        left_rows.append(left)
        right_rows.append(right)
        row_indices.append(int(row_idx))
    if not left_rows:
        empty = np.empty((0, width), dtype=np.int32)
        return empty, empty.copy(), np.empty((0,), dtype=np.int32)
    return (
        np.asarray(left_rows, dtype=np.int32),
        np.asarray(right_rows, dtype=np.int32),
        np.asarray(row_indices, dtype=np.int32),
    )


def _markov_exact_merge_states_jnp(
    deps: SimpleNamespace,
    left_states: Any,
    right_states: Any,
    *,
    n_regimes: int,
    target_scale: float,
) -> Any:
    """Differentiable Markov sketch merge over sketch-shaped JAX states."""

    jnp = deps.jnp
    n = int(n_regimes)
    left_count = left_states[:, 0]
    right_count = right_states[:, 0]
    left_first = left_states[:, 1 : 1 + n]
    left_last = left_states[:, 1 + n : 1 + 2 * n]
    right_first = right_states[:, 1 : 1 + n]
    right_last = right_states[:, 1 + n : 1 + 2 * n]
    same_boundary = jnp.sum(left_last * right_first, axis=1)
    join = 1.0 - same_boundary
    count = left_count + right_count + join / float(target_scale)
    return jnp.concatenate([count[:, None], left_first, right_last], axis=1)


def _markov_canonical_project_jnp(
    deps: SimpleNamespace,
    states: Any,
    *,
    n_regimes: int,
    target_scale: float,
) -> Any:
    """Project sketch-shaped JAX states to integer-count/one-hot endpoints."""

    jnp = deps.jnp
    n = int(n_regimes)
    count = jnp.round(states[:, :1] * float(target_scale)) / float(target_scale)
    first_logits = states[:, 1 : 1 + n]
    last_logits = states[:, 1 + n : 1 + 2 * n]
    first = deps.jax.nn.one_hot(jnp.argmax(first_logits, axis=1), n, dtype=jnp.float32)
    last = deps.jax.nn.one_hot(jnp.argmax(last_logits, axis=1), n, dtype=jnp.float32)
    return jnp.concatenate([count, first, last], axis=1)


def _hll_exact_merge_states_jnp(
    deps: SimpleNamespace,
    left_states: Any,
    right_states: Any,
) -> Any:
    """Differentiable exact HLL merge in normalized register coordinates."""

    return deps.jnp.maximum(left_states, right_states)


def _hll_canonical_project_jnp(
    deps: SimpleNamespace,
    states: Any,
    *,
    precision: int,
    hash_bits: int,
) -> Any:
    """Project HLL-shaped JAX states to valid normalized register values."""

    max_register = float(_hll_max_register(precision=int(precision), hash_bits=int(hash_bits)))
    clipped = deps.jnp.clip(states, 0.0, 1.0)
    return deps.jnp.round(clipped * max_register) / max_register


def _hll_canonical_project_np(
    states: np.ndarray,
    *,
    precision: int,
    hash_bits: int,
) -> np.ndarray:
    max_register = float(_hll_max_register(precision=int(precision), hash_bits=int(hash_bits)))
    clipped = np.clip(np.asarray(states, dtype=np.float32), 0.0, 1.0)
    return (np.rint(clipped * max_register) / max_register).astype(np.float32)


def _compute_jsd_sufficiency(
    *,
    rep: np.ndarray,
    theta: np.ndarray,
    deps: SimpleNamespace,
    n_iter: int = 800,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    seed: int = 0,
) -> dict[str, float]:
    """Train an sbijax-style JSD critic on ``(rep, theta)`` pairs.

    The JSD-style binary classifier from sbijax NASS gives a lower bound on
    ``I(theta; rep)`` (cf. ``sbijax/_src/nass.py:_jsd_summary_loss``). The
    converged loss is ``-MI_lower_bound``; lower (more negative) means the
    rep retains more information about ``theta`` and is closer to a
    sufficient statistic for it.

    Returns:
        dict with ``final_loss`` (= -MI_lower_bound), ``final_mi_lower_bound``,
        ``last_10_loss_mean`` (smoothed convergence value), ``init_loss``
        (loss at init), and ``n_iter`` actually run.
    """
    jax = deps.jax
    jnp = deps.jnp
    jr = deps.jr
    optax = deps.optax
    fnn = deps.fnn

    rep_arr = np.asarray(rep, dtype=np.float32)
    theta_arr = np.asarray(theta, dtype=np.float32)
    if rep_arr.ndim != 2 or theta_arr.ndim != 2:
        return {}
    if int(rep_arr.shape[0]) != int(theta_arr.shape[0]):
        return {}
    n = int(rep_arr.shape[0])
    if n < 4:
        return {}
    rep_jnp = jnp.asarray(rep_arr)
    theta_jnp = jnp.asarray(theta_arr)

    class _Critic(fnn.Module):

        @fnn.compact
        def __call__(self, rep_in, theta_in):
            x = jnp.concatenate([rep_in, theta_in], axis=-1)
            x = fnn.Dense(int(hidden_dim), name="jsd_critic_l0")(x)
            x = fnn.relu(x)
            x = fnn.Dense(int(hidden_dim), name="jsd_critic_l1")(x)
            x = fnn.relu(x)
            return fnn.Dense(1, name="jsd_critic_out")(x).squeeze(-1)

    critic = _Critic()
    rng = jr.PRNGKey(int(seed) + 4093)
    init_rng, train_rng = jr.split(rng)
    params = critic.init(init_rng, rep_jnp[:1], theta_jnp[:1])
    optimizer = optax.adam(float(learning_rate))
    opt_state = optimizer.init(params)
    bs = max(2, min(int(batch_size), n))

    def jsd_loss(params_obj, rep_b, theta_b, rng_b):
        # Mirrors sbijax `_jsd_summary_loss`: positive pairs vs randomly
        # permuted negatives, JSD-style softplus formulation.
        f_pos = critic.apply(params_obj, rep_b, theta_b)
        m = int(rep_b.shape[0])
        idx_neg = jr.permutation(rng_b, jnp.arange(m))
        f_neg = critic.apply(params_obj, rep_b, theta_b[idx_neg])
        mi = (-jax.nn.softplus(-f_pos)).mean() - jax.nn.softplus(f_neg).mean()
        return -mi

    @jax.jit
    def step(params_obj, opt_state_obj, rep_b, theta_b, rng_b):
        loss_val, grads = jax.value_and_grad(jsd_loss)(params_obj, rep_b, theta_b, rng_b)
        updates, opt_state_new = optimizer.update(grads, opt_state_obj, params_obj)
        params_new = optax.apply_updates(params_obj, updates)
        return params_new, opt_state_new, loss_val

    init_loss = float(jsd_loss(params, rep_jnp[:bs], theta_jnp[:bs], jr.fold_in(train_rng, 0)))
    losses: list[float] = []
    for it in range(int(n_iter)):
        train_rng, batch_rng, perm_rng = jr.split(train_rng, 3)
        idx = jr.permutation(batch_rng, jnp.arange(n))[:bs]
        rep_b = rep_jnp[idx]
        theta_b = theta_jnp[idx]
        params, opt_state, loss_val = step(params, opt_state, rep_b, theta_b, perm_rng)
        losses.append(float(loss_val))
    final_loss = float(np.mean(losses[-min(10, len(losses)) :]))
    return {
        "final_loss": float(losses[-1]),
        "final_mi_lower_bound": float(-losses[-1]),
        "last_10_loss_mean": final_loss,
        "init_loss": init_loss,
        "n_iter": int(n_iter),
    }


def _sufficiency_diagnostics_for_rep(
    *,
    rep: np.ndarray,
    theta_truth: np.ndarray,
    analytic_sketch: np.ndarray | None,
    deps: SimpleNamespace,
    label: str,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute sbijax-style JSD sufficiency for a learned rep, with refs.

    For a rep ``s = g(x)`` and a truth ``theta``, the sbijax NASS-style JSD
    contrastive bound on ``I(theta; s)`` is computed by training a small
    critic. Reference points:

      * **ceiling**: same diagnostic computed on the analytic Markov sketch
        as the rep — by construction sufficient for the changepoint count
        likelihood, so its bound is the highest reachable.
      * **floor**: same diagnostic on a randomly permuted version of ``rep``,
        which destroys the alignment with ``theta``. Bound ≈ 0 (no info).

    The ``sufficiency_proxy`` ∈ [0, 1] linearly maps our bound between the
    floor and the ceiling so that 1.0 means "as informative as the analytic
    sketch" and 0.0 means "no information about theta beyond random."

    Two parallel sufficiency tests are reported when ``theta_truth`` has
    width ≥ 1:

    * ``sufficiency_{label}_*`` — full sketch (count + first + last).
      Stricter than f*-only sufficiency; penalizes count-only mode where
      first/last aren't supervised.
    * ``sufficiency_{label}_count_*`` — count slot alone (the f*-aligned
      sufficiency). Honest claim for "rep is sufficient for f* = count."

    Both share the same critic family/training schedule, only the truth
    target changes.
    """
    rng = np.random.default_rng(int(seed) + 877)
    out: dict[str, Any] = {}
    rep_arr = np.asarray(rep, dtype=np.float32)
    truth_arr = np.asarray(theta_truth, dtype=np.float32)

    def _emit(
        test_label: str, rep_in: np.ndarray, truth_in: np.ndarray, ref_seed_offset: int
    ) -> dict[str, Any]:
        rep_diag_local = _compute_jsd_sufficiency(
            rep=rep_in, theta=truth_in, deps=deps, seed=seed + ref_seed_offset
        )
        if not rep_diag_local:
            return {}
        perm_local = rng.permutation(int(rep_in.shape[0]))
        floor_diag_local = _compute_jsd_sufficiency(
            rep=rep_in[perm_local],
            theta=truth_in,
            deps=deps,
            seed=seed + ref_seed_offset + 1,
        )
        ceiling_diag_local: dict[str, float] = {}
        if analytic_sketch is not None:
            sketch_arr_local = np.asarray(analytic_sketch, dtype=np.float32)
            if sketch_arr_local.shape[0] == truth_in.shape[0]:
                ceiling_diag_local = _compute_jsd_sufficiency(
                    rep=sketch_arr_local,
                    theta=truth_in,
                    deps=deps,
                    seed=seed + ref_seed_offset + 2,
                )
        sub: dict[str, Any] = {}
        sub[f"sufficiency_{label}_{test_label}jsd_loss"] = rep_diag_local.get("last_10_loss_mean")
        sub[f"sufficiency_{label}_{test_label}mi_lower_bound"] = -float(
            rep_diag_local.get("last_10_loss_mean", 0.0)
        )
        sub[f"sufficiency_{label}_{test_label}floor_jsd_loss"] = floor_diag_local.get(
            "last_10_loss_mean"
        )
        if ceiling_diag_local:
            sub[f"sufficiency_{label}_{test_label}ceiling_jsd_loss"] = ceiling_diag_local.get(
                "last_10_loss_mean"
            )
            floor_l = floor_diag_local.get("last_10_loss_mean")
            ceil_l = ceiling_diag_local.get("last_10_loss_mean")
            our_l = rep_diag_local.get("last_10_loss_mean")
            if (
                floor_l is not None
                and ceil_l is not None
                and our_l is not None
                and (floor_l - ceil_l) > 1e-9
            ):
                sub[f"sufficiency_{label}_{test_label}proxy"] = float(
                    (floor_l - our_l) / (floor_l - ceil_l)
                )
        return sub

    # 1) Full-sketch sufficiency (the strict test).
    out.update(_emit("", rep_arr, truth_arr, ref_seed_offset=0))
    # 2) f*-aligned sufficiency on the count slot only (the honest f* claim).
    if int(truth_arr.shape[1]) >= 1:
        count_truth = truth_arr[:, :1]
        out.update(_emit("count_", rep_arr, count_truth, ref_seed_offset=10))
    return out


def _probe_endpoints_from_rep(
    *,
    rep: np.ndarray,
    theta_truth: np.ndarray,
    n_regimes: int,
    deps: SimpleNamespace,
    label: str = "leaf_rep",
    n_iter: int = 1500,
    hidden_dim: int = 64,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    seed: int = 0,
) -> dict[str, float]:
    """Probe whether a frozen rep encodes first/last regime, anywhere in it.

    Direct ``theta_first_regime_accuracy`` does
    ``argmax(pred[:, 1:1+n_regimes])`` — i.e., it assumes the model placed
    first-regime probabilities in those specific slots. Laws-supervised
    cells respect that allocation; pure-sbijax cells (laws=0) have no
    reason to. If the rep is genuinely sufficient for the response
    distribution (which depends on first/last for Markov), the
    information is encoded SOMEWHERE in the rep — possibly in a different
    basis than literal slot indexing.

    A small classifier probe ``rep -> first_regime`` (and same for last)
    recovers the information by learning the right linear/nonlinear
    extraction. If the probe accuracy is high, the rep is sufficient
    (just not slot-aligned). If the probe is at chance, the rep
    genuinely does not encode endpoints anywhere — i.e., it is NOT a
    sufficient statistic.

    The probe is trained on a 60/40 split of the test rep set with a
    small Flax linen MLP and reported as held-out accuracy.
    """
    jax = deps.jax
    jnp = deps.jnp
    jr = deps.jr
    optax = deps.optax
    fnn = deps.fnn

    rep_arr = np.asarray(rep, dtype=np.float32)
    truth_arr = np.asarray(theta_truth, dtype=np.float32)
    if rep_arr.ndim != 2 or truth_arr.ndim != 2:
        return {}
    n = int(rep_arr.shape[0])
    if n < 8:
        return {}

    n_r = int(n_regimes)
    if int(truth_arr.shape[1]) < 1 + 2 * n_r:
        return {}
    first_truth = np.argmax(truth_arr[:, 1 : 1 + n_r], axis=1).astype(np.int32)
    last_truth = np.argmax(truth_arr[:, 1 + n_r : 1 + 2 * n_r], axis=1).astype(np.int32)

    rng_np = np.random.default_rng(int(seed) + 1729)
    perm = rng_np.permutation(n)
    n_train = int(0.6 * n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    rep_train_arr = rep_arr[train_idx]
    rep_val_arr = rep_arr[val_idx]
    first_train_np = first_truth[train_idx]
    first_val_np = first_truth[val_idx]
    last_train_np = last_truth[train_idx]
    last_val_np = last_truth[val_idx]

    rep_train = jnp.asarray(rep_train_arr)
    rep_val = jnp.asarray(rep_val_arr)
    first_train = jnp.asarray(first_train_np)
    first_val = jnp.asarray(first_val_np)
    last_train = jnp.asarray(last_train_np)
    last_val = jnp.asarray(last_val_np)

    class _Probe(fnn.Module):

        @fnn.compact
        def __call__(self, rep_in):
            x = fnn.Dense(int(hidden_dim), name="probe_l0")(rep_in)
            x = fnn.relu(x)
            x = fnn.Dense(int(hidden_dim), name="probe_l1")(x)
            x = fnn.relu(x)
            return fnn.Dense(int(n_r), name="probe_out")(x)

    probe = _Probe()
    bs = max(2, min(int(batch_size), int(rep_train.shape[0])))

    def _train_probe_for_target(target_train, target_val, rng_seed: int) -> dict[str, float]:
        rng = jr.PRNGKey(int(rng_seed))
        init_rng, train_rng = jr.split(rng)
        params = probe.init(init_rng, rep_train[:1])
        opt = optax.adam(float(learning_rate))
        opt_state = opt.init(params)

        def loss_fn(params_obj, rep_b, target_b):
            logits = probe.apply(params_obj, rep_b)
            log_probs = jax.nn.log_softmax(logits, axis=-1)
            target_oh = jax.nn.one_hot(target_b, int(n_r))
            return -jnp.sum(target_oh * log_probs, axis=-1).mean()

        @jax.jit
        def step(params_obj, opt_state_obj, rep_b, target_b):
            loss_val, grads = jax.value_and_grad(loss_fn)(params_obj, rep_b, target_b)
            updates, opt_state_new = opt.update(grads, opt_state_obj, params_obj)
            params_new = optax.apply_updates(params_obj, updates)
            return params_new, opt_state_new, loss_val

        n_t = int(rep_train.shape[0])
        for _ in range(int(n_iter)):
            train_rng, batch_rng = jr.split(train_rng)
            idx = jr.permutation(batch_rng, jnp.arange(n_t))[:bs]
            rep_b = rep_train[idx]
            target_b = target_train[idx]
            params, opt_state, _ = step(params, opt_state, rep_b, target_b)
        # Held-out accuracy on val.
        logits = probe.apply(params, rep_val)
        preds = jnp.argmax(logits, axis=-1)
        val_acc = float(jnp.mean(preds == target_val))
        # Train accuracy as a sanity check.
        logits_train = probe.apply(params, rep_train)
        preds_train = jnp.argmax(logits_train, axis=-1)
        train_acc = float(jnp.mean(preds_train == target_train))
        return {"val_accuracy": val_acc, "train_accuracy": train_acc}

    first_diag = _train_probe_for_target(first_train, first_val, int(seed) + 7919)
    last_diag = _train_probe_for_target(last_train, last_val, int(seed) + 7937)

    return {
        f"probed_{label}_first_regime_accuracy": first_diag["val_accuracy"],
        f"probed_{label}_last_regime_accuracy": last_diag["val_accuracy"],
        f"probed_{label}_first_regime_train_accuracy": first_diag["train_accuracy"],
        f"probed_{label}_last_regime_train_accuracy": last_diag["train_accuracy"],
        f"probed_{label}_n_regimes": int(n_r),
        f"probed_{label}_n_train": int(rep_train.shape[0]),
        f"probed_{label}_n_val": int(rep_val.shape[0]),
    }


def _markov_local_law_metrics_from_states_np(
    dataset: ContextualResponseDataset,
    *,
    states: np.ndarray,
    merge_pred: np.ndarray | None = None,
    merge_target: np.ndarray | None = None,
    supervision_mode: str = "dense_exact",
    leaf_rate: float = 1.0,
    merge_rate: float = 1.0,
    idempotence_rate: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Dense/sparse local-law diagnostics for sketch-shaped Markov states."""

    block_by_token = _metadata_block_by_token(dataset)
    targets = dict(dataset.package_theta_targets or {})
    if block_by_token is None or "markov_exact_sketch" not in targets:
        return {}
    theta = np.asarray(targets["markov_exact_sketch"], dtype=np.float32)
    states_np = np.asarray(states, dtype=np.float32)
    if states_np.ndim != 2 or states_np.shape != theta.shape:
        return {}
    n_regimes_raw = dataset.metadata.get("n_regimes")
    n_regimes = int(max(block_by_token)) + 1 if n_regimes_raw is None else int(n_regimes_raw)
    expected_dim = 1 + 2 * int(n_regimes)
    if int(states_np.shape[1]) != int(expected_dim):
        return {}
    idempotence_pred = markov_canonical_project_np(
        states_np,
        target_scale=float(dataset.target_scale),
        n_regimes=int(n_regimes),
    )
    bundle = markov_approx_local_laws_bundle(
        leaf_pred=states_np,
        leaf_target=theta,
        merge_pred=merge_pred,
        merge_target=merge_target,
        idempotence_pred=idempotence_pred,
        idempotence_target=theta,
    )
    leaf_losses = np.mean((states_np - theta) ** 2, axis=1)
    idempotence_losses = np.mean((idempotence_pred - theta) ** 2, axis=1)
    if merge_pred is None or merge_target is None:
        merge_losses = np.empty((0,), dtype=np.float32)
    else:
        merge_losses = np.mean(
            (np.asarray(merge_pred, dtype=np.float32) - np.asarray(merge_target, dtype=np.float32))
            ** 2,
            axis=1,
        )
    law_rows = markov_local_law_observation_rows(
        leaf_losses=leaf_losses.tolist(),
        merge_losses=merge_losses.tolist(),
        idempotence_losses=idempotence_losses.tolist(),
        supervision_mode=str(supervision_mode),
        leaf_rate=float(leaf_rate),
        merge_rate=float(merge_rate),
        idempotence_rate=float(idempotence_rate),
        seed=int(seed),
    )
    leaf_quantiles = (
        {
            "p25": float(np.quantile(leaf_losses, 0.25)),
            "p50": float(np.quantile(leaf_losses, 0.50)),
            "p75": float(np.quantile(leaf_losses, 0.75)),
            "p95": float(np.quantile(leaf_losses, 0.95)),
            "max": float(np.max(leaf_losses)),
        }
        if leaf_losses.size
        else {}
    )
    merge_quantiles = (
        {
            "p25": float(np.quantile(merge_losses, 0.25)),
            "p50": float(np.quantile(merge_losses, 0.50)),
            "p75": float(np.quantile(merge_losses, 0.75)),
            "p95": float(np.quantile(merge_losses, 0.95)),
            "max": float(np.max(merge_losses)),
        }
        if merge_losses.size
        else {}
    )
    idempotence_quantiles = (
        {
            "p25": float(np.quantile(idempotence_losses, 0.25)),
            "p50": float(np.quantile(idempotence_losses, 0.50)),
            "p75": float(np.quantile(idempotence_losses, 0.75)),
            "p95": float(np.quantile(idempotence_losses, 0.95)),
            "max": float(np.max(idempotence_losses)),
        }
        if idempotence_losses.size
        else {}
    )
    return {
        "law_set_id": MARKOV_COUNT_SKETCH_LAW_SET_ID,
        "eps_leaf": float(bundle.eps_leaf),
        "eps_merge": float(bundle.eps_merge),
        "eps_idemp": float(bundle.eps_idemp),
        # Per-leaf / per-merge / per-idempotence-node squared-error arrays.
        # These are already computed for the IPW row builder; surfacing them
        # here lets callers inspect the distribution per node rather than
        # only the aggregate eps_*.
        "per_leaf_law_loss": leaf_losses.astype(float).tolist(),
        "per_merge_law_loss": merge_losses.astype(float).tolist(),
        "per_idempotence_law_loss": idempotence_losses.astype(float).tolist(),
        "per_leaf_law_loss_quantiles": leaf_quantiles,
        "per_merge_law_loss_quantiles": merge_quantiles,
        "per_idempotence_law_loss_quantiles": idempotence_quantiles,
        "approx_local_laws": {
            "eps_leaf": float(bundle.eps_leaf),
            "eps_merge": float(bundle.eps_merge),
            "eps_idemp": float(bundle.eps_idemp),
            "evidence_status": bundle.evidence_status.value,
            "notes": bundle.notes,
        },
        "local_law_observation_metadata": law_rows.to_metadata(),
    }


def _apply_fn_chunked(
    apply_fn: Any,
    params: Any,
    tokens: Any,
    responses: Any,
    *,
    chunk_size: int = 1024,
) -> tuple[Any, Any, Any, Any]:
    """Run ``apply_fn(params, tokens, responses)`` in row-chunks and
    concatenate the four returned arrays along axis 0. The non-chunked
    call materializes the full FNO/MLP forward over all rows at once,
    which OOMs at N_train ~ 25k+ for jax_fno (~3 GiB intermediate at
    100k). Eval-time only; training already mini-batches."""
    deps = _require_contextual_sbi()
    jnp = deps.jnp
    n = int(tokens.shape[0])
    if n <= int(chunk_size):
        return apply_fn(params, tokens, responses)
    states_list: list[Any] = []
    preds_list: list[Any] = []
    slice_list: list[Any] = []
    critic_list: list[Any] = []
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        states_c, preds_c, slice_c, critic_c = apply_fn(
            params,
            tokens[start:end],
            responses[start:end],
        )
        states_list.append(states_c)
        preds_list.append(preds_c)
        slice_list.append(slice_c)
        critic_list.append(critic_c)
    return (
        jnp.concatenate(states_list, axis=0),
        jnp.concatenate(preds_list, axis=0),
        jnp.concatenate(slice_list, axis=0),
        jnp.concatenate(critic_list, axis=0),
    )


def _markov_local_law_eval_metrics(
    *,
    params: Any,
    apply_fn: Any,
    dataset: ContextualResponseDataset,
    deps: SimpleNamespace,
    states_np: np.ndarray,
    supervision_mode: str = "dense_exact",
    leaf_rate: float = 1.0,
    merge_rate: float = 1.0,
    idempotence_rate: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Evaluate Markov local-law residuals using the supplied state map."""

    block_by_token = _metadata_block_by_token(dataset)
    targets = dict(dataset.package_theta_targets or {})
    if block_by_token is None or "markov_exact_sketch" not in targets:
        return {}
    theta = np.asarray(targets["markov_exact_sketch"], dtype=np.float32)
    states_arr = np.asarray(states_np, dtype=np.float32)
    if states_arr.ndim != 2 or states_arr.shape != theta.shape:
        return {}
    n_regimes_raw = dataset.metadata.get("n_regimes")
    n_regimes = int(max(block_by_token)) + 1 if n_regimes_raw is None else int(n_regimes_raw)
    expected_dim = 1 + 2 * int(n_regimes)
    if int(states_arr.shape[1]) != int(expected_dim):
        return {}
    left_tokens, right_tokens, full_indices = _markov_merge_split_arrays(dataset)
    merge_pred_np = None
    merge_target_np = None
    merge_sufficiency: dict[str, Any] = {}
    if int(left_tokens.shape[0]) > 0:
        response_dim = int(np.prod(dataset.response_signatures.shape[1:]))
        dummy = deps.jnp.zeros((int(left_tokens.shape[0]), response_dim), dtype=deps.jnp.float32)
        left_states, _left_pred, left_slice, _left_critic = _apply_fn_chunked(
            apply_fn,
            params,
            deps.jnp.asarray(left_tokens, dtype=deps.jnp.int32),
            dummy,
        )
        right_states, _right_pred, right_slice, _right_critic = _apply_fn_chunked(
            apply_fn,
            params,
            deps.jnp.asarray(right_tokens, dtype=deps.jnp.int32),
            dummy,
        )
        merge_pred = _markov_exact_merge_states_jnp(
            deps,
            left_states,
            right_states,
            n_regimes=int(n_regimes),
            target_scale=float(dataset.target_scale),
        )
        merge_pred_np = np.asarray(merge_pred, dtype=np.float32)
        merge_target_np = theta[full_indices]
        # Sufficiency on the MERGED rep: same JSD MI bound but on the post-
        # composition state vs the truth sketch at the merge node. This is
        # the actual research signal for whether the merge preserves
        # sufficiency. Use slice_pred as the merged rep when in count-only
        # mode (where slice_pred carries the actual rep dim, not the
        # synthetic count-shape).
        merged_rep_np = merge_pred_np
        left_slice_np = np.asarray(left_slice)
        right_slice_np = np.asarray(right_slice)
        if (
            left_slice_np.ndim == 2
            and right_slice_np.ndim == 2
            and left_slice_np.shape == right_slice_np.shape
            and left_slice_np.shape[0] == merge_pred_np.shape[0]
            and left_slice_np.shape[1] >= 2
        ):
            # Count-only mode: synthesize a "merged rep" stand-in by averaging
            # the left/right slice_pred entries. The true merge in count-only
            # is a learned merge_net but we don't have its output here; using
            # mean(left, right) is a reasonable proxy for the post-composition
            # representation that the merge operates over.
            merged_rep_np = (left_slice_np + right_slice_np) / 2.0
        merge_sufficiency = _sufficiency_diagnostics_for_rep(
            rep=merged_rep_np,
            theta_truth=merge_target_np,
            analytic_sketch=merge_target_np,
            deps=deps,
            label="merge_rep",
        )
        # Probe whether endpoint info is encoded anywhere in the merged rep.
        if int(n_regimes) > 1:
            merge_sufficiency.update(
                _probe_endpoints_from_rep(
                    rep=merged_rep_np,
                    theta_truth=merge_target_np,
                    n_regimes=int(n_regimes),
                    deps=deps,
                    label="merge_rep",
                )
            )
    metrics = _markov_local_law_metrics_from_states_np(
        dataset,
        states=states_np,
        merge_pred=merge_pred_np,
        merge_target=merge_target_np,
        supervision_mode=supervision_mode,
        leaf_rate=leaf_rate,
        merge_rate=merge_rate,
        idempotence_rate=idempotence_rate,
        seed=seed,
    )
    metrics.update(merge_sufficiency)
    return metrics


def _hll_local_law_metrics_from_states_np(
    dataset: ContextualResponseDataset,
    *,
    states: np.ndarray,
    merge_pred: np.ndarray | None = None,
    merge_target: np.ndarray | None = None,
    supervision_mode: str = "dense_exact",
    leaf_rate: float = 1.0,
    merge_rate: float = 1.0,
    idempotence_rate: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    targets = dict(dataset.package_theta_targets or {})
    if "hll_register_sketch" not in targets:
        return {}
    theta = np.asarray(targets["hll_register_sketch"], dtype=np.float32)
    states_np = np.asarray(states, dtype=np.float32)
    if states_np.ndim != 2 or states_np.shape != theta.shape:
        return {}
    p = int(dataset.metadata.get("hll_precision", 4))
    h = int(dataset.metadata.get("hll_hash_bits", 64))
    expected_dim = int(1 << p)
    if int(states_np.shape[1]) != expected_dim:
        return {}
    idempotence_pred = _hll_canonical_project_np(
        states_np,
        precision=p,
        hash_bits=h,
    )
    leaf_losses = np.mean((states_np - theta) ** 2, axis=1)
    idempotence_losses = np.mean((states_np - idempotence_pred) ** 2, axis=1)
    if merge_pred is None or merge_target is None:
        merge_losses = np.empty((0,), dtype=np.float32)
    else:
        merge_losses = np.mean(
            (np.asarray(merge_pred, dtype=np.float32) - np.asarray(merge_target, dtype=np.float32))
            ** 2,
            axis=1,
        )

    def _quantiles(values: np.ndarray) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        if not arr.size:
            return {}
        return {
            "p25": float(np.quantile(arr, 0.25)),
            "p50": float(np.quantile(arr, 0.50)),
            "p75": float(np.quantile(arr, 0.75)),
            "p95": float(np.quantile(arr, 0.95)),
            "max": float(np.max(arr)),
        }

    law_rows = markov_local_law_observation_rows(
        leaf_losses=leaf_losses.tolist(),
        merge_losses=merge_losses.tolist(),
        idempotence_losses=idempotence_losses.tolist(),
        supervision_mode=str(supervision_mode),
        leaf_rate=float(leaf_rate),
        merge_rate=float(merge_rate),
        idempotence_rate=float(idempotence_rate),
        seed=int(seed),
        law_set_id=HLL_REGISTER_SKETCH_LAW_SET_ID,
    )
    eps_leaf = float(np.mean(np.abs(states_np - theta))) if theta.size else 0.0
    eps_merge = (
        float(np.mean(np.abs(np.asarray(merge_pred) - np.asarray(merge_target))))
        if merge_pred is not None and merge_target is not None
        else float("nan")
    )
    eps_idemp = float(np.mean(np.abs(states_np - idempotence_pred))) if states_np.size else 0.0
    return {
        "law_set_id": HLL_REGISTER_SKETCH_LAW_SET_ID,
        "eps_leaf": eps_leaf,
        "eps_merge": eps_merge,
        "eps_idemp": eps_idemp,
        "per_leaf_law_loss": leaf_losses.astype(float).tolist(),
        "per_merge_law_loss": merge_losses.astype(float).tolist(),
        "per_idempotence_law_loss": idempotence_losses.astype(float).tolist(),
        "per_leaf_law_loss_quantiles": _quantiles(leaf_losses),
        "per_merge_law_loss_quantiles": _quantiles(merge_losses),
        "per_idempotence_law_loss_quantiles": _quantiles(idempotence_losses),
        "approx_local_laws": {
            "eps_leaf": eps_leaf,
            "eps_merge": eps_merge,
            "eps_idemp": eps_idemp,
            "evidence_status": "approx_audited",
            "notes": "HLL register residuals in normalized register coordinates.",
        },
        "local_law_observation_metadata": law_rows.to_metadata(),
    }


def _hll_local_law_eval_metrics(
    *,
    params: Any,
    apply_fn: Any,
    dataset: ContextualResponseDataset,
    deps: SimpleNamespace,
    states_np: np.ndarray,
    supervision_mode: str = "dense_exact",
    leaf_rate: float = 1.0,
    merge_rate: float = 1.0,
    idempotence_rate: float = 1.0,
    seed: int = 0,
) -> dict[str, Any]:
    targets = dict(dataset.package_theta_targets or {})
    if "hll_register_sketch" not in targets:
        return {}
    theta = np.asarray(targets["hll_register_sketch"], dtype=np.float32)
    states_arr = np.asarray(states_np, dtype=np.float32)
    if states_arr.ndim != 2 or states_arr.shape != theta.shape:
        return {}
    left_tokens, right_tokens, full_indices = _markov_merge_split_arrays(dataset)
    merge_pred_np = None
    merge_target_np = None
    merge_sufficiency: dict[str, Any] = {}
    if int(left_tokens.shape[0]) > 0:
        response_dim = int(np.prod(dataset.response_signatures.shape[1:]))
        dummy = deps.jnp.zeros((int(left_tokens.shape[0]), response_dim), dtype=deps.jnp.float32)
        left_states, _left_pred, _left_slice, _left_critic = _apply_fn_chunked(
            apply_fn,
            params,
            deps.jnp.asarray(left_tokens, dtype=deps.jnp.int32),
            dummy,
        )
        right_states, _right_pred, _right_slice, _right_critic = _apply_fn_chunked(
            apply_fn,
            params,
            deps.jnp.asarray(right_tokens, dtype=deps.jnp.int32),
            dummy,
        )
        merge_pred_np = np.asarray(
            _hll_exact_merge_states_jnp(deps, left_states, right_states),
            dtype=np.float32,
        )
        merge_target_np = theta[full_indices]
        merge_sufficiency = _sufficiency_diagnostics_for_rep(
            rep=merge_pred_np,
            theta_truth=merge_target_np,
            analytic_sketch=merge_target_np,
            deps=deps,
            label="merge_rep",
        )
    p = int(dataset.metadata.get("hll_precision", 4))
    h = int(dataset.metadata.get("hll_hash_bits", 64))
    metrics = _hll_local_law_metrics_from_states_np(
        dataset,
        states=states_np,
        merge_pred=merge_pred_np,
        merge_target=merge_target_np,
        supervision_mode=supervision_mode,
        leaf_rate=leaf_rate,
        merge_rate=merge_rate,
        idempotence_rate=idempotence_rate,
        seed=seed,
    )
    pred_est = _hll_estimate_from_normalized_registers_np(
        states_arr,
        precision=p,
        hash_bits=h,
        round_registers=False,
    )
    truth_est = _hll_estimate_from_normalized_registers_np(
        theta,
        precision=p,
        hash_bits=h,
        round_registers=True,
    )
    raw_err = np.abs(pred_est - truth_est)
    metrics.update(
        {
            "hll_estimate_raw_mae": float(np.mean(raw_err)) if raw_err.size else 0.0,
            "per_leaf_hll_estimate_pred_raw": pred_est.astype(float).tolist(),
            "per_leaf_hll_estimate_truth_raw": truth_est.astype(float).tolist(),
            "per_leaf_hll_estimate_raw_abs_err": raw_err.astype(float).tolist(),
        }
    )
    metrics.update(merge_sufficiency)
    return metrics


def _theta_prediction_diagnostics(
    dataset: ContextualResponseDataset,
    *,
    theta_pred: np.ndarray,
    theta_std: np.ndarray | None = None,
    package_theta: str = "markov_exact_sketch",
) -> dict[str, float | int]:
    targets = dict(dataset.package_theta_targets or {})
    if str(package_theta) not in targets:
        return {}
    truth = np.asarray(targets[str(package_theta)], dtype=np.float64)
    pred = np.asarray(theta_pred, dtype=np.float64)
    # Count-only mode: prediction is shape (B, 1) — only the count
    # (count_norm) is predicted; (first, last) are NOT supervised in this
    # mode and we deliberately do not synthesize random argmax values for
    # them. Report count-specific diagnostics only.
    if (
        str(package_theta) == "markov_exact_sketch"
        and pred.ndim == 2
        and int(pred.shape[1]) == 1
        and truth.ndim == 2
        and int(truth.shape[1]) >= 1
        and int(pred.shape[0]) == int(truth.shape[0])
    ):
        target_scale = float(dataset.target_scale)
        count_pred_norm = pred[:, 0]
        count_truth_norm = truth[:, 0]
        count_err_norm = count_pred_norm - count_truth_norm
        count_raw_abs_err = np.abs(count_err_norm) * target_scale
        out: dict[str, float | int] = {
            "supervision_mode": "count_only",
            "theta_count_raw_mae": float(np.mean(count_raw_abs_err)),
            "theta_count_norm_mae": float(np.mean(np.abs(count_err_norm))),
            "theta_count_norm_mse": float(np.mean(count_err_norm**2)),
            "per_leaf_count_pred_norm": count_pred_norm.astype(float).tolist(),
            "per_leaf_count_truth_norm": count_truth_norm.astype(float).tolist(),
            "per_leaf_count_pred_raw": (count_pred_norm * target_scale).astype(float).tolist(),
            "per_leaf_count_truth_raw": (count_truth_norm * target_scale).astype(float).tolist(),
            "per_leaf_count_raw_abs_err": count_raw_abs_err.astype(float).tolist(),
            "target_scale": target_scale,
        }
        if count_raw_abs_err.size:
            out["per_leaf_count_raw_abs_err_quantiles"] = {
                "p25": float(np.quantile(count_raw_abs_err, 0.25)),
                "p50": float(np.quantile(count_raw_abs_err, 0.50)),
                "p75": float(np.quantile(count_raw_abs_err, 0.75)),
                "p95": float(np.quantile(count_raw_abs_err, 0.95)),
                "max": float(np.max(count_raw_abs_err)),
            }
        return out
    if pred.shape != truth.shape:
        return {}
    err = pred - truth
    out = {
        "supervision_mode": "full_sketch",
        "theta_mae": float(np.mean(np.abs(err))) if err.size else float("nan"),
        "theta_mse": float(np.mean(err * err)) if err.size else float("nan"),
    }
    if err.size:
        # Per-leaf scores: full vector and quantile summary so callers can see
        # the distribution without paying the cost of post-hoc reload (each
        # leaf in the test set is supervised independently under C1; this is
        # the right place to expose that signal).
        per_leaf_abs_err = np.mean(np.abs(err), axis=1)  # (n_leaves,)
        per_leaf_sq_err = np.mean(err * err, axis=1)
        out["per_leaf_theta_mae"] = per_leaf_abs_err.astype(float).tolist()
        out["per_leaf_theta_mse"] = per_leaf_sq_err.astype(float).tolist()
        out["per_leaf_theta_mae_quantiles"] = {
            "p25": float(np.quantile(per_leaf_abs_err, 0.25)),
            "p50": float(np.quantile(per_leaf_abs_err, 0.50)),
            "p75": float(np.quantile(per_leaf_abs_err, 0.75)),
            "p95": float(np.quantile(per_leaf_abs_err, 0.95)),
            "max": float(np.max(per_leaf_abs_err)),
        }
    if str(package_theta) == "markov_exact_sketch" and pred.shape[1] >= 3:
        n_regimes_raw = dataset.metadata.get("n_regimes")
        if n_regimes_raw is None:
            n_regimes = (int(pred.shape[1]) - 1) // 2
        else:
            n_regimes = int(n_regimes_raw)
        if pred.shape[1] >= 1 + 2 * n_regimes and n_regimes > 0:
            target_scale = float(dataset.target_scale)
            count_raw_per_leaf = np.abs(pred[:, 0] - truth[:, 0]) * target_scale
            first_pred_per_leaf = np.argmax(pred[:, 1 : 1 + n_regimes], axis=1)
            first_truth_per_leaf = np.argmax(truth[:, 1 : 1 + n_regimes], axis=1)
            last_pred_per_leaf = np.argmax(pred[:, 1 + n_regimes : 1 + 2 * n_regimes], axis=1)
            last_truth_per_leaf = np.argmax(truth[:, 1 + n_regimes : 1 + 2 * n_regimes], axis=1)
            first_correct_per_leaf = first_pred_per_leaf == first_truth_per_leaf
            last_correct_per_leaf = last_pred_per_leaf == last_truth_per_leaf
            out.update(
                {
                    "theta_count_raw_mae": float(np.mean(count_raw_per_leaf)),
                    "theta_first_regime_accuracy": float(np.mean(first_correct_per_leaf)),
                    "theta_last_regime_accuracy": float(np.mean(last_correct_per_leaf)),
                    "per_leaf_count_raw_abs_err": count_raw_per_leaf.astype(float).tolist(),
                    "per_leaf_first_regime_correct": first_correct_per_leaf.astype(bool).tolist(),
                    "per_leaf_last_regime_correct": last_correct_per_leaf.astype(bool).tolist(),
                    # Full per-leaf estimates (predicted sketch + truth) so
                    # callers can reconstruct what the model said for every
                    # test leaf without reloading the dataset.
                    "per_leaf_theta_pred": pred.astype(float).tolist(),
                    "per_leaf_theta_truth": truth.astype(float).tolist(),
                    "per_leaf_count_pred_norm": pred[:, 0].astype(float).tolist(),
                    "per_leaf_count_truth_norm": truth[:, 0].astype(float).tolist(),
                    "per_leaf_count_pred_raw": (pred[:, 0] * target_scale).astype(float).tolist(),
                    "per_leaf_count_truth_raw": (truth[:, 0] * target_scale).astype(float).tolist(),
                    "per_leaf_first_regime_pred": first_pred_per_leaf.astype(int).tolist(),
                    "per_leaf_first_regime_truth": first_truth_per_leaf.astype(int).tolist(),
                    "per_leaf_last_regime_pred": last_pred_per_leaf.astype(int).tolist(),
                    "per_leaf_last_regime_truth": last_truth_per_leaf.astype(int).tolist(),
                    "n_regimes": int(n_regimes),
                    "target_scale": target_scale,
                }
            )
            # Per-leaf structure: actual token IDs for each test leaf and the
            # implied per-token regime IDs. Lets callers correlate metric
            # quality with leaf content (length, regime composition, # of
            # boundaries) without reloading the bundle.
            try:
                item_tokens = np.asarray(dataset.item_tokens, dtype=np.int64)
            except Exception:
                item_tokens = None
            if (
                item_tokens is not None
                and item_tokens.ndim == 2
                and int(item_tokens.shape[0]) == int(pred.shape[0])
            ):
                pad_id = int(getattr(dataset, "pad_id", -1))
                out["per_leaf_tokens"] = item_tokens.tolist()
                out["pad_id"] = pad_id
                out["per_leaf_n_valid_tokens"] = (
                    (item_tokens != pad_id).sum(axis=1).astype(int).tolist()
                )
                block_by_token = _metadata_block_by_token(dataset)
                if block_by_token is not None:
                    block_arr = np.asarray(list(block_by_token), dtype=np.int64)
                    if int(block_arr.size) > 0:
                        clipped = np.clip(item_tokens, 0, int(block_arr.size) - 1)
                        regimes = block_arr[clipped]
                        valid_mask = item_tokens != pad_id
                        regimes = np.where(valid_mask, regimes, -1)
                        out["per_leaf_token_regimes"] = regimes.astype(int).tolist()
                        # Per-leaf regime composition (counts per regime).
                        comp = np.zeros((int(regimes.shape[0]), int(n_regimes)), dtype=np.int64)
                        for r in range(int(n_regimes)):
                            comp[:, r] = (regimes == r).sum(axis=1)
                        out["per_leaf_regime_counts"] = comp.astype(int).tolist()
    if str(package_theta) == "hll_register_sketch":
        p = int(dataset.metadata.get("hll_precision", 4))
        h = int(dataset.metadata.get("hll_hash_bits", 64))
        pred_est = _hll_estimate_from_normalized_registers_np(
            pred,
            precision=p,
            hash_bits=h,
            round_registers=False,
        )
        truth_est = _hll_estimate_from_normalized_registers_np(
            truth,
            precision=p,
            hash_bits=h,
            round_registers=True,
        )
        est_abs_err = np.abs(pred_est - truth_est)
        register_abs_err = np.abs(err)
        out.update(
            {
                "hll_precision": int(p),
                "hll_hash_bits": int(h),
                "hll_register_count": int(1 << p),
                "hll_register_mae": (
                    float(np.mean(register_abs_err)) if register_abs_err.size else float("nan")
                ),
                "hll_register_mse": float(np.mean(err * err)) if err.size else float("nan"),
                "hll_estimate_raw_mae": (
                    float(np.mean(est_abs_err)) if est_abs_err.size else float("nan")
                ),
                "hll_estimate_norm_mae": (
                    float(np.mean(est_abs_err) / max(float(dataset.target_scale), 1e-12))
                    if est_abs_err.size
                    else float("nan")
                ),
                "per_leaf_hll_estimate_pred_raw": pred_est.astype(float).tolist(),
                "per_leaf_hll_estimate_truth_raw": truth_est.astype(float).tolist(),
                "per_leaf_hll_estimate_raw_abs_err": est_abs_err.astype(float).tolist(),
                "per_leaf_hll_register_pred": pred.astype(float).tolist(),
                "per_leaf_hll_register_truth": truth.astype(float).tolist(),
            }
        )
    if theta_std is not None:
        std = np.asarray(theta_std, dtype=np.float64)
        if std.shape == pred.shape:
            out["posterior_std_mean"] = float(np.mean(std))
            out["posterior_std_max"] = float(np.max(std)) if std.size else float("nan")
    return out


def _sample_conditional_density_mean(
    deps: SimpleNamespace,
    density_model: Any,
    params: Any,
    x: Any,
    *,
    n_samples: int,
    batch_size: int,
    seed: int,
) -> Any:
    """Monte Carlo posterior mean from a package conditional density net."""

    jnp = deps.jnp
    jr = deps.jr
    xs = jnp.asarray(x, dtype=jnp.float32)
    n = int(xs.shape[0])
    batch = max(1, int(batch_size))
    draws = max(1, int(n_samples))
    base_key = jr.PRNGKey(int(seed))
    means = []
    for start in range(0, n, batch):
        xb = xs[start : start + batch]
        acc = None
        for sample_idx in range(draws):
            key = jr.fold_in(base_key, int(start * 1009 + sample_idx))
            draw = density_model.apply(params, key, method="sample", x=xb)
            acc = draw if acc is None else acc + draw
        means.append(acc / float(draws))
    return jnp.concatenate(means, axis=0)


def _make_package_density_estimator(
    deps: SimpleNamespace,
    *,
    n_dimension: int,
    hidden_dim: int,
    n_components: int,
    activation_name: str,
):
    """Package-native conditional density estimator used by NPE/NLE."""

    return deps.make_mdn(
        int(n_dimension),
        int(n_components),
        hidden_sizes=(int(hidden_dim), int(hidden_dim)),
        activation=_sbijax_activation(deps, str(activation_name)),
    )


_POSTERIOR_DENSITY_PAIRS = {
    ("npe", "mdn"),
    ("fmpe", "cnf"),
    ("cmpe", "cm"),
    ("nle", "mdn"),
    ("nre", "resnet"),
    ("snle", "maf"),
    ("snle", "spf"),
}


def _validate_posterior_pair(estimator: str, density_family: str) -> None:
    pair = (str(estimator), str(density_family))
    if pair not in _POSTERIOR_DENSITY_PAIRS:
        supported = ", ".join(f"{left}+{right}" for left, right in sorted(_POSTERIOR_DENSITY_PAIRS))
        raise ValueError(
            f"unsupported posterior estimator/density pair {pair[0]}+{pair[1]}; "
            f"supported pairs: {supported}"
        )


def _make_posterior_model_and_network(
    deps: SimpleNamespace,
    *,
    estimator: str,
    density_family: str,
    theta_dim: int,
    y_dim: int,
    hidden_dim: int,
    density_components: int,
    activation_name: str,
):
    estimator_name = str(estimator)
    density_name = str(density_family)
    _validate_posterior_pair(estimator_name, density_name)
    activation = _sbijax_activation(deps, str(activation_name))
    hidden = int(hidden_dim)
    theta_dimension = int(theta_dim)
    y_dimension = int(y_dim)

    if estimator_name == "npe":
        network = deps.make_mdn(
            theta_dimension,
            int(density_components),
            hidden_sizes=(hidden, hidden),
            activation=activation,
        )
        return deps.sbijax.NPE, network, "sbijax.nn.make_mdn"
    if estimator_name == "fmpe":
        network = deps.make_cnf(
            theta_dimension,
            n_layers=2,
            hidden_size=hidden,
            activation=activation,
            dropout_rate=0.0,
            do_batch_norm=False,
        )
        return deps.sbijax.FMPE, network, "sbijax.nn.make_cnf"
    if estimator_name == "cmpe":
        network = deps.make_cm(
            theta_dimension,
            n_layers=2,
            hidden_size=hidden,
            activation=activation,
            dropout_rate=0.0,
            do_batch_norm=False,
        )
        return deps.sbijax.CMPE, network, "sbijax.nn.make_cm"
    if estimator_name == "nle":
        network = deps.make_mdn(
            y_dimension,
            int(density_components),
            hidden_sizes=(hidden, hidden),
            activation=activation,
        )
        return deps.sbijax.NLE, network, "sbijax.nn.make_mdn"
    if estimator_name == "nre":
        network = deps.make_resnet(
            n_layers=2,
            hidden_size=hidden,
            activation=activation,
            dropout_rate=0.0,
            do_batch_norm=False,
        )
        return deps.sbijax.NRE, network, "sbijax.nn.make_resnet"
    if estimator_name == "snle":
        if density_name == "maf":
            network = deps.make_maf(
                y_dimension,
                n_layers=3,
                hidden_sizes=(hidden, hidden),
                activation=activation,
            )
            return deps.sbijax.SNLE, network, "sbijax.nn.make_maf"
        network = deps.make_spf(
            y_dimension,
            range_min=-0.25,
            range_max=1.25,
            n_layers=3,
            hidden_sizes=(hidden, hidden),
            activation=activation,
        )
        return deps.sbijax.SNLE, network, "sbijax.nn.make_spf"
    raise ValueError(f"unknown posterior_estimator {estimator_name!r}")


def _sample_direct_posterior_mean_and_std(
    deps: SimpleNamespace,
    *,
    model: Any,
    params: Any,
    encoded: Any,
    estimator: str,
    n_samples: int,
    batch_size: int,
    seed: int,
) -> tuple[Any, Any]:
    jnp = deps.jnp
    jr = deps.jr
    xs = jnp.asarray(encoded, dtype=jnp.float32)
    batch = max(1, int(batch_size))
    draws = max(1, int(n_samples))
    base_key = jr.PRNGKey(int(seed))
    means = []
    stds = []
    for start in range(0, int(xs.shape[0]), batch):
        xb = xs[start : start + batch]
        samples = []
        for sample_idx in range(draws):
            key = jr.fold_in(base_key, int(start * 1009 + sample_idx))
            if str(estimator) == "npe":
                draw = model.apply(params, key, method="sample", x=xb)
            else:
                draw = model.apply(
                    params,
                    key,
                    method="sample",
                    context=xb,
                    is_training=False,
                )
            samples.append(draw)
        stacked = jnp.stack(samples, axis=0)
        means.append(jnp.mean(stacked, axis=0))
        stds.append(jnp.std(stacked, axis=0))
    return jnp.concatenate(means, axis=0), jnp.concatenate(stds, axis=0)


def _sample_mcmc_posterior_mean_and_std(
    deps: SimpleNamespace,
    *,
    package_model: Any,
    params: Any,
    encoded: Any,
    n_samples: int,
    seed: int,
    sampler: str,
) -> tuple[Any, Any, float]:
    jnp = deps.jnp
    jr = deps.jr
    xs = jnp.asarray(encoded, dtype=jnp.float32)
    draws = max(2, int(n_samples))
    total_samples = max(4, 2 * draws)
    warmup = max(1, total_samples - draws)
    means = []
    stds = []
    ess_values = []
    for row_idx in range(int(xs.shape[0])):
        inference_data, diagnostics = package_model.sample_posterior(
            jr.PRNGKey(int(seed) + row_idx),
            params,
            xs[row_idx],
            n_chains=1,
            n_samples=int(total_samples),
            n_warmup=int(warmup),
            sampler=str(sampler),
        )
        samples_dict = deps.sbijax.inference_data_as_dictionary(inference_data.posterior)
        theta_samples = jnp.asarray(samples_dict["theta"], dtype=jnp.float32)
        means.append(jnp.mean(theta_samples, axis=0))
        stds.append(jnp.std(theta_samples, axis=0))
        try:
            ess_values.append(float(diagnostics["ess_bulk"].to_array().min()))
        except Exception:
            pass
    ess_min = float(np.min(ess_values)) if ess_values else float("nan")
    return jnp.stack(means, axis=0), jnp.stack(stds, axis=0), ess_min


def fit_contextual_sbijax_package_direct(
    train: ContextualResponseDataset,
    val: ContextualResponseDataset,
    *,
    config: ContextualSBIJAXConfig,
) -> ContextualSBIJAXResult:
    """Fit ``sbijax.NASS``/``NASSS`` directly, then train a JAX readout."""

    method, _trainer, train_response_dim, n_slices = _validate_fit_inputs(
        train,
        val,
        config=config,
    )
    if int(config.n_iter) <= 0:
        raise ValueError("package trainer requires n_iter > 0")
    if int(train.item_tokens.shape[0]) < 2:
        raise ValueError("package trainer requires at least two training rows")

    deps = _require_contextual_sbi()
    jax = deps.jax
    jnp = deps.jnp
    jr = deps.jr
    optax = deps.optax

    train_y = _encode_dataset_items_for_sbijax_package(
        deps,
        train,
        config=config,
    )
    val_y = _encode_dataset_items_for_sbijax_package(
        deps,
        val,
        config=config,
    )
    train_responses = jnp.asarray(_flatten_response_signatures(train), dtype=jnp.float32)
    val_responses = jnp.asarray(_flatten_response_signatures(val), dtype=jnp.float32)
    train_package_theta_np = _package_theta_array(
        train,
        package_theta=str(config.package_theta),
    )
    val_package_theta_np = _package_theta_array(
        val,
        package_theta=str(config.package_theta),
    )
    if int(train_package_theta_np.shape[1]) != int(val_package_theta_np.shape[1]):
        raise ValueError("train/val package theta targets must have the same feature dimension")
    train_package_theta = jnp.asarray(train_package_theta_np, dtype=jnp.float32)

    model_fns = _dummy_response_signature_model_fns(
        deps,
        response_signature_dim=int(train_package_theta_np.shape[1]),
    )
    summary_activation = _sbijax_activation(deps, str(config.summary_activation))
    if method == "nasss":
        summary_net = deps.make_nasss_net(
            int(config.state_dim),
            1,
            (int(config.hidden_dim), int(config.hidden_dim)),
            activation=summary_activation,
        )
        package_model = deps.sbijax.NASSS(model_fns, summary_net)
    else:
        summary_net = deps.make_nass_net(
            int(config.state_dim),
            (int(config.hidden_dim), int(config.hidden_dim)),
            activation=summary_activation,
        )
        package_model = deps.sbijax.NASS(model_fns, summary_net)

    summary_params, summary_losses = package_model.fit(
        jr.PRNGKey(int(config.seed)),
        data={"y": train_y, "theta": train_package_theta},
        optimizer=optax.adam(float(config.learning_rate)),
        n_iter=int(config.n_iter),
        batch_size=max(1, int(config.batch_size)),
        percentage_data_as_validation_set=0.1,
        n_early_stopping_patience=max(10, int(config.n_iter) + 1),
    )
    summary_losses_np = np.asarray(summary_losses, dtype=np.float64)

    train_states = package_model.summarize(
        summary_params,
        train_y,
        batch_size=max(1, int(config.batch_size)),
    )
    val_states = package_model.summarize(
        summary_params,
        val_y,
        batch_size=max(1, int(config.batch_size)),
    )

    readout = _make_package_readout(
        deps,
        response_signature_dim=int(train_response_dim),
        hidden_dim=int(config.hidden_dim),
    )
    readout_key, train_key = jr.split(jr.PRNGKey(int(config.seed) + 17), 2)
    init_n = min(max(1, int(config.batch_size)), int(train_states.shape[0]))
    readout_params = readout.init(readout_key, train_states[:init_n])
    optimizer = optax.adam(float(config.learning_rate))
    opt_state = optimizer.init(readout_params)

    def readout_loss(params, states, responses):
        pred = readout.apply(params, states)
        return jnp.mean((pred - responses) ** 2)

    @jax.jit
    def step(params, opt_state, states, responses):
        loss, grads = jax.value_and_grad(readout_loss)(params, states, responses)
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss

    @jax.jit
    def eval_readout_loss(params, states, responses):
        return readout_loss(params, states, responses)

    history: list[dict[str, float | int]] = []
    n_train = int(train_states.shape[0])
    batch_size = max(1, int(config.batch_size))
    np_rng = np.random.default_rng(int(config.seed) + 29)
    for iteration in range(int(config.n_iter)):
        order = np_rng.permutation(n_train)
        train_loss_acc = 0.0
        seen = 0
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            idx_jnp = jnp.asarray(idx, dtype=jnp.int32)
            readout_params, opt_state, batch_loss = step(
                readout_params,
                opt_state,
                train_states[idx_jnp],
                train_responses[idx_jnp],
            )
            weight = int(len(idx))
            train_loss_acc += float(batch_loss) * weight
            seen += weight
        train_readout_mse = float(train_loss_acc / max(1, seen))
        val_readout_mse = float(eval_readout_loss(readout_params, val_states, val_responses))
        history.append(
            {
                "iteration": int(iteration + 1),
                "trainer": "package",
                "train_loss": train_readout_mse,
                "train_contextual_mse": train_readout_mse,
                "train_readout_mse": train_readout_mse,
                "train_package_loss": _package_loss_at(summary_losses_np, iteration, 0),
                "val_loss": val_readout_mse,
                "val_contextual_mse": val_readout_mse,
                "val_readout_mse": val_readout_mse,
                "val_package_loss": _package_loss_at(summary_losses_np, iteration, 1),
                "summary_train_loss": _package_loss_at(summary_losses_np, iteration, 0),
                "summary_val_loss": _package_loss_at(summary_losses_np, iteration, 1),
            }
        )

    def apply_package(params, tokens, responses):
        del responses
        encoded = _encode_tokens_for_sbijax_package(
            deps,
            tokens,
            vocab_size=int(config.vocab_size),
            input_encoding=str(config.input_encoding),
            block_by_token=_metadata_block_by_token(train),
            n_regimes=train.metadata.get("n_regimes"),
            target_scale=float(train.target_scale),
        )
        states = package_model.summarize(
            params["summary"],
            encoded,
            batch_size=max(1, int(config.batch_size)),
        )
        response_pred = readout.apply(params["readout"], states)
        slice_pred = jnp.zeros((int(states.shape[0]), 0), dtype=response_pred.dtype)
        critic = jnp.zeros((int(states.shape[0]),), dtype=response_pred.dtype)
        return states, response_pred, slice_pred, critic

    params = {"summary": summary_params, "readout": readout_params}
    train_diag = evaluate_contextual_sbijax(
        params=params,
        apply_fn=apply_package,
        dataset=train,
    )
    val_diag = evaluate_contextual_sbijax(
        params=params,
        apply_fn=apply_package,
        dataset=val,
    )
    provenance = contextual_sbijax_provenance(
        method=method,
        response_signature_contexts=int(config.response_signature_contexts),
        response_signature_slices=int(n_slices),
        trainer="package",
        input_encoding=str(config.input_encoding),
        summary_activation=str(config.summary_activation),
        downstream_readout="haiku_mlp_mse",
    )
    provenance.update(
        {
            "summary_network": (
                "sbijax.nn.make_nasss_net" if method == "nasss" else "sbijax.nn.make_nass_net"
            ),
            "package_summary_loss_rows": int(summary_losses_np.shape[0]),
            "package_theta": str(config.package_theta),
            "package_theta_dim": int(train_package_theta_np.shape[1]),
            "baseline_role": "approximate_sbijax_baseline",
            "decoder_kind": "learned",
            "exact_zero_claim": False,
        }
    )
    return ContextualSBIJAXResult(
        params=params,
        history=history,
        train_diagnostics=train_diag,
        val_diagnostics=val_diag,
        provenance=provenance,
        config=config,
        slice_matrix=None,
        apply_fn=apply_package,
    )


def fit_contextual_sbijax_posterior_direct(
    train: ContextualResponseDataset,
    val: ContextualResponseDataset,
    *,
    config: ContextualSBIJAXConfig,
) -> ContextualSBIJAXResult:
    """Fit a package-native posterior/likelihood estimator on prepared Markov rows."""

    method, trainer_name, train_response_dim, n_slices = _validate_fit_inputs(
        train,
        val,
        config=config,
    )
    if int(config.n_iter) <= 0:
        raise ValueError("posterior trainer requires n_iter > 0")
    if int(train.item_tokens.shape[0]) < 2:
        raise ValueError("posterior trainer requires at least two training rows")

    deps = _require_contextual_sbi()
    jnp = deps.jnp
    jr = deps.jr
    optax = deps.optax

    estimator = "npe" if str(config.trainer) == "npe" else str(config.posterior_estimator)
    density_family = "mdn" if str(config.trainer) == "npe" else str(config.density_family)
    _validate_posterior_pair(estimator, density_family)

    train_y = _encode_dataset_items_for_sbijax_package(deps, train, config=config)
    val_y = _encode_dataset_items_for_sbijax_package(deps, val, config=config)
    train_theta_np = _package_theta_array(train, package_theta=str(config.package_theta))
    val_theta_np = _package_theta_array(val, package_theta=str(config.package_theta))
    if int(train_theta_np.shape[1]) != int(val_theta_np.shape[1]):
        raise ValueError("train/val package theta targets must have the same dimension")
    train_theta = jnp.asarray(train_theta_np, dtype=jnp.float32)

    model_fns = _dummy_response_signature_model_fns(
        deps,
        response_signature_dim=int(train_theta_np.shape[1]),
    )
    estimator_cls, network, network_name = _make_posterior_model_and_network(
        deps,
        estimator=estimator,
        density_family=density_family,
        theta_dim=int(train_theta_np.shape[1]),
        y_dim=int(train_y.shape[1]),
        hidden_dim=int(config.hidden_dim),
        density_components=int(config.density_components),
        activation_name=str(config.summary_activation),
    )
    if estimator == "npe":
        package_model = estimator_cls(
            model_fns,
            network,
            use_event_space_bijections=False,
        )
    else:
        package_model = estimator_cls(model_fns, network)

    posterior_params, posterior_losses = package_model.fit(
        jr.PRNGKey(int(config.seed) + 211),
        data={"y": train_y, "theta": train_theta},
        optimizer=optax.adam(float(config.learning_rate)),
        n_iter=int(config.n_iter),
        batch_size=max(1, int(config.batch_size)),
        percentage_data_as_validation_set=0.1,
        n_early_stopping_patience=max(10, int(config.n_iter) + 1),
    )
    posterior_losses_np = np.asarray(posterior_losses, dtype=np.float64)

    n_eval_samples = _posterior_eval_samples(config)
    eval_batch_size = max(1, _posterior_eval_batch_size(config))

    def theta_stats_for(encoded, *, seed_offset: int):
        if estimator in {"npe", "fmpe", "cmpe"}:
            theta_mean, theta_std = _sample_direct_posterior_mean_and_std(
                deps,
                model=package_model.model,
                params=posterior_params,
                encoded=encoded,
                estimator=estimator,
                n_samples=n_eval_samples,
                batch_size=eval_batch_size,
                seed=int(config.seed) + int(seed_offset),
            )
            return theta_mean, theta_std, float("nan")
        theta_mean, theta_std, ess_min = _sample_mcmc_posterior_mean_and_std(
            deps,
            package_model=package_model,
            params=posterior_params,
            encoded=encoded,
            n_samples=n_eval_samples,
            seed=int(config.seed) + int(seed_offset),
            sampler=str(config.posterior_sampler),
        )
        return theta_mean, theta_std, ess_min

    train_theta_mean, train_theta_std, train_mcmc_ess = theta_stats_for(
        train_y,
        seed_offset=307,
    )
    val_theta_mean, val_theta_std, val_mcmc_ess = theta_stats_for(
        val_y,
        seed_offset=409,
    )
    train_theta_mse = float(
        jnp.mean((train_theta_mean - jnp.asarray(train_theta_np, dtype=jnp.float32)) ** 2)
    )
    val_theta_mse = float(
        jnp.mean((val_theta_mean - jnp.asarray(val_theta_np, dtype=jnp.float32)) ** 2)
    )

    exact_decoder = None
    response_w = None
    response_b = None
    downstream_readout = "affine_theta_to_response"
    if str(config.package_theta) == "markov_exact_sketch":
        exact_decoder = _markov_exact_decoder_setup(deps, train)
        downstream_readout = "deterministic_markov_exact_sketch"
    else:
        response_w_np, response_b_np = _affine_response_probe(
            train_theta_np,
            _flatten_response_signatures(train),
        )
        response_w = jnp.asarray(response_w_np, dtype=jnp.float32)
        response_b = jnp.asarray(response_b_np, dtype=jnp.float32)

    def response_from_theta(theta_pred):
        if exact_decoder is not None:
            return _responses_from_markov_exact_states(
                theta_pred,
                decoder=exact_decoder,
            )
        return theta_pred @ response_w + response_b

    history: list[dict[str, float | int]] = []
    for iteration in range(int(posterior_losses_np.shape[0])):
        train_loss = _package_loss_at(posterior_losses_np, iteration, 0)
        val_loss = _package_loss_at(posterior_losses_np, iteration, 1)
        row: dict[str, float | int] = {
            "iteration": int(iteration + 1),
            "trainer": str(trainer_name),
            "posterior_estimator": str(estimator),
            "density_family": str(density_family),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_posterior_loss": train_loss,
            "val_posterior_loss": val_loss,
            "train_theta_mse": train_theta_mse,
            "val_theta_mse": val_theta_mse,
        }
        row[f"train_{estimator}_loss"] = train_loss
        row[f"val_{estimator}_loss"] = val_loss
        history.append(row)

    def apply_posterior(params_obj, tokens, responses):
        del responses
        encoded = _encode_tokens_for_sbijax_package(
            deps,
            tokens,
            vocab_size=int(config.vocab_size),
            input_encoding=str(config.input_encoding),
            block_by_token=_metadata_block_by_token(train),
            n_regimes=train.metadata.get("n_regimes"),
            target_scale=float(train.target_scale),
        )
        if estimator in {"npe", "fmpe", "cmpe"}:
            theta_mean, theta_std = _sample_direct_posterior_mean_and_std(
                deps,
                model=package_model.model,
                params=params_obj["posterior"],
                encoded=encoded,
                estimator=estimator,
                n_samples=n_eval_samples,
                batch_size=eval_batch_size,
                seed=int(config.seed) + 503,
            )
        else:
            theta_mean, theta_std, _ess_min = _sample_mcmc_posterior_mean_and_std(
                deps,
                package_model=package_model,
                params=params_obj["posterior"],
                encoded=encoded,
                n_samples=n_eval_samples,
                seed=int(config.seed) + 503,
                sampler=str(config.posterior_sampler),
            )
        response_pred = response_from_theta(theta_mean)
        slice_pred = jnp.zeros((int(theta_mean.shape[0]), 0), dtype=response_pred.dtype)
        return theta_mean, response_pred, slice_pred, theta_std

    params: dict[str, Any] = {"posterior": posterior_params}
    if response_w is not None and response_b is not None:
        params["response_w"] = response_w
        params["response_b"] = response_b
    train_diag = evaluate_contextual_sbijax(
        params=params,
        apply_fn=apply_posterior,
        dataset=train,
    )
    val_diag = evaluate_contextual_sbijax(
        params=params,
        apply_fn=apply_posterior,
        dataset=val,
    )
    if np.isfinite(train_mcmc_ess):
        train_diag["posterior_mcmc_ess_min"] = float(train_mcmc_ess)
    if np.isfinite(val_mcmc_ess):
        val_diag["posterior_mcmc_ess_min"] = float(val_mcmc_ess)

    provenance = contextual_sbijax_provenance(
        method=method,
        response_signature_contexts=int(config.response_signature_contexts),
        response_signature_slices=int(n_slices),
        trainer=str(trainer_name),
        input_encoding=str(config.input_encoding),
        summary_activation=str(config.summary_activation),
        downstream_readout=downstream_readout,
    )
    provenance.update(
        {
            "sbijax_class": str(estimator_cls.__name__),
            "posterior_estimator": str(estimator),
            "density_family": str(density_family),
            "density_estimator": str(network_name),
            "density_components": int(config.density_components),
            "posterior_samples": int(config.posterior_samples),
            "posterior_eval_samples": int(n_eval_samples),
            "posterior_eval_batch_size": int(eval_batch_size),
            "posterior_sampler": (
                str(config.posterior_sampler) if estimator in {"nle", "nre", "snle"} else None
            ),
            "package_theta": str(config.package_theta),
            "package_theta_dim": int(train_theta_np.shape[1]),
            "posterior_loss_rows": int(posterior_losses_np.shape[0]),
            "baseline_role": "approximate_sbijax_posterior_baseline",
            "decoder_kind": "exact" if exact_decoder is not None else "learned",
            "exact_zero_claim": False,
        }
    )
    return ContextualSBIJAXResult(
        params=params,
        history=history,
        train_diagnostics=train_diag,
        val_diagnostics=val_diag,
        provenance=provenance,
        config=config,
        slice_matrix=None,
        apply_fn=apply_posterior,
    )


def fit_contextual_sbijax_npe_direct(
    train: ContextualResponseDataset,
    val: ContextualResponseDataset,
    *,
    config: ContextualSBIJAXConfig,
) -> ContextualSBIJAXResult:
    """Backward-compatible alias for ``trainer='posterior', estimator='npe'``."""

    return fit_contextual_sbijax_posterior_direct(
        train,
        val,
        config=replace(
            config,
            trainer="npe",
            posterior_estimator="npe",
            density_family="mdn",
        ),
    )


def fit_contextual_sbijax_nass_nle(
    train: ContextualResponseDataset,
    val: ContextualResponseDataset,
    *,
    config: ContextualSBIJAXConfig,
) -> ContextualSBIJAXResult:
    """Fit package-native NASS/NASSS summaries followed by ``sbijax.NLE``."""

    result = fit_contextual_sbijax_package_direct(
        train,
        val,
        config=replace(config, trainer="package"),
    )
    deps = _require_contextual_sbi()
    jr = deps.jr
    optax = deps.optax
    jnp = deps.jnp

    train_y = _encode_dataset_items_for_sbijax_package(deps, train, config=config)
    val_y = _encode_dataset_items_for_sbijax_package(deps, val, config=config)
    train_theta_np = _package_theta_array(train, package_theta=str(config.package_theta))
    train_theta = jnp.asarray(train_theta_np, dtype=jnp.float32)
    package_model = deps.sbijax.NASSS if str(config.method) == "nasss" else deps.sbijax.NASS
    model_fns = _dummy_response_signature_model_fns(
        deps,
        response_signature_dim=int(train_theta_np.shape[1]),
    )
    summary_activation = _sbijax_activation(deps, str(config.summary_activation))
    if str(config.method) == "nasss":
        summary_net = deps.make_nasss_net(
            int(config.state_dim),
            1,
            (int(config.hidden_dim), int(config.hidden_dim)),
            activation=summary_activation,
        )
    else:
        summary_net = deps.make_nass_net(
            int(config.state_dim),
            (int(config.hidden_dim), int(config.hidden_dim)),
            activation=summary_activation,
        )
    summary_model = package_model(model_fns, summary_net)
    train_states = summary_model.summarize(
        result.params["summary"],
        train_y,
        batch_size=max(1, int(config.batch_size)),
    )
    val_states = summary_model.summarize(
        result.params["summary"],
        val_y,
        batch_size=max(1, int(config.batch_size)),
    )

    density = _make_package_density_estimator(
        deps,
        n_dimension=int(config.state_dim),
        hidden_dim=int(config.hidden_dim),
        n_components=int(config.density_components),
        activation_name=str(config.summary_activation),
    )
    nle_model = deps.sbijax.NLE(model_fns, density)
    nle_params, nle_losses = nle_model.fit(
        jr.PRNGKey(int(config.seed) + 613),
        data={"y": train_states, "theta": train_theta},
        optimizer=optax.adam(float(config.learning_rate)),
        n_iter=int(config.n_iter),
        batch_size=max(1, int(config.batch_size)),
        percentage_data_as_validation_set=0.1,
        n_early_stopping_patience=max(10, int(config.n_iter) + 1),
    )
    nle_losses_np = np.asarray(nle_losses, dtype=np.float64)
    history: list[dict[str, float | int]] = []
    for iteration, row in enumerate(result.history):
        nle_train = _package_loss_at(nle_losses_np, iteration, 0)
        nle_val = _package_loss_at(nle_losses_np, iteration, 1)
        merged = dict(row)
        merged.update(
            {
                "trainer": "nass_nle",
                "train_nle_loss": nle_train,
                "val_nle_loss": nle_val,
                "nle_train_loss": nle_train,
                "nle_val_loss": nle_val,
            }
        )
        history.append(merged)

    params = dict(result.params)
    params["nle"] = nle_params
    provenance = dict(result.provenance)
    provenance.update(
        {
            "trainer": "nass_nle",
            "likelihood_estimator": "sbijax.NLE",
            "density_estimator": "sbijax.nn.make_mdn",
            "density_components": int(config.density_components),
            "nle_loss_rows": int(nle_losses_np.shape[0]),
            "downstream_readout": "haiku_mlp_mse",
        }
    )
    return ContextualSBIJAXResult(
        params=params,
        history=history,
        train_diagnostics=result.train_diagnostics,
        val_diagnostics=result.val_diagnostics,
        provenance=provenance,
        config=config,
        slice_matrix=result.slice_matrix,
        apply_fn=result.apply_fn,
    )


def fit_contextual_sbijax_theta_supervised(
    train: ContextualResponseDataset,
    val: ContextualResponseDataset,
    *,
    config: ContextualSBIJAXConfig,
) -> ContextualSBIJAXResult:
    """Fit a supervised ``tokens -> package theta`` state map.

    This is a control lane for checking whether the chosen input encoding and
    model capacity can recover the supplied package-native statistic at all.
    The response readout is a fixed affine probe fit from exact theta rows to
    response signatures, so failure here points at state learning rather than
    downstream contextual readout optimization.
    """

    method, _trainer, train_response_dim, n_slices = _validate_fit_inputs(
        train,
        val,
        config=config,
    )
    deps = _require_contextual_sbi()
    jax = deps.jax
    jnp = deps.jnp
    jr = deps.jr
    optax = deps.optax

    train_features = _encode_dataset_items_for_sbijax_package(
        deps,
        train,
        config=config,
    )
    val_features = _encode_dataset_items_for_sbijax_package(
        deps,
        val,
        config=config,
    )
    train_theta_np = _package_theta_array(train, package_theta=str(config.package_theta))
    val_theta_np = _package_theta_array(val, package_theta=str(config.package_theta))
    if int(train_theta_np.shape[1]) != int(val_theta_np.shape[1]):
        raise ValueError("train/val theta targets must have the same dimension")
    train_responses_np = _flatten_response_signatures(train)
    val_responses_np = _flatten_response_signatures(val)
    response_w_np, response_b_np = _affine_response_probe(
        train_theta_np,
        train_responses_np,
    )
    response_w = jnp.asarray(response_w_np, dtype=jnp.float32)
    response_b = jnp.asarray(response_b_np, dtype=jnp.float32)
    train_theta = jnp.asarray(train_theta_np, dtype=jnp.float32)
    val_theta = jnp.asarray(val_theta_np, dtype=jnp.float32)
    train_responses = jnp.asarray(train_responses_np, dtype=jnp.float32)
    val_responses = jnp.asarray(val_responses_np, dtype=jnp.float32)

    summary_net = _make_theta_summary_net(
        deps,
        theta_dim=int(train_theta_np.shape[1]),
        hidden_dim=int(config.hidden_dim),
    )
    key = jr.PRNGKey(int(config.seed) + 101)
    init_n = min(max(1, int(config.batch_size)), int(train_features.shape[0]))
    params = summary_net.init(key, train_features[:init_n])
    optimizer = optax.adam(float(config.learning_rate))
    opt_state = optimizer.init(params)

    def response_from_theta(theta_pred):
        return theta_pred @ response_w + response_b

    def loss_parts(params, features, theta_target, responses):
        theta_pred = summary_net.apply(params, features)
        response_pred = response_from_theta(theta_pred)
        theta_mse = jnp.mean((theta_pred - theta_target) ** 2)
        response_mse = jnp.mean((response_pred - responses) ** 2)
        total = theta_mse + float(config.contextual_loss_weight) * response_mse
        return total, (theta_mse, response_mse)

    @jax.jit
    def step(params, opt_state, features, theta_target, responses):
        (loss, aux), grads = jax.value_and_grad(loss_parts, has_aux=True)(
            params,
            features,
            theta_target,
            responses,
        )
        updates, opt_state_new = optimizer.update(grads, opt_state, params)
        params_new = optax.apply_updates(params, updates)
        return params_new, opt_state_new, loss, aux

    @jax.jit
    def eval_loss(params, features, theta_target, responses):
        return loss_parts(params, features, theta_target, responses)

    history: list[dict[str, float | int]] = []
    n_train = int(train_features.shape[0])
    batch_size = max(1, int(config.batch_size))
    np_rng = np.random.default_rng(int(config.seed) + 113)
    for iteration in range(int(config.n_iter)):
        order = np_rng.permutation(n_train)
        train_loss_acc = 0.0
        train_theta_acc = 0.0
        train_response_acc = 0.0
        seen = 0
        for start in range(0, n_train, batch_size):
            idx = order[start : start + batch_size]
            idx_jnp = jnp.asarray(idx, dtype=jnp.int32)
            params, opt_state, batch_loss, aux = step(
                params,
                opt_state,
                train_features[idx_jnp],
                train_theta[idx_jnp],
                train_responses[idx_jnp],
            )
            weight = int(len(idx))
            train_loss_acc += float(batch_loss) * weight
            train_theta_acc += float(aux[0]) * weight
            train_response_acc += float(aux[1]) * weight
            seen += weight
        val_loss, val_aux = eval_loss(params, val_features, val_theta, val_responses)
        history.append(
            {
                "iteration": int(iteration + 1),
                "trainer": "theta_supervised",
                "train_loss": float(train_loss_acc / max(1, seen)),
                "train_theta_mse": float(train_theta_acc / max(1, seen)),
                "train_contextual_mse": float(train_response_acc / max(1, seen)),
                "train_package_loss": float(train_theta_acc / max(1, seen)),
                "val_loss": float(val_loss),
                "val_theta_mse": float(val_aux[0]),
                "val_contextual_mse": float(val_aux[1]),
                "val_package_loss": float(val_aux[0]),
            }
        )

    def apply_theta_supervised(params_obj, tokens, responses):
        del responses
        encoded = _encode_tokens_for_sbijax_package(
            deps,
            tokens,
            vocab_size=int(config.vocab_size),
            input_encoding=str(config.input_encoding),
            block_by_token=_metadata_block_by_token(train),
            n_regimes=train.metadata.get("n_regimes"),
            target_scale=float(train.target_scale),
        )
        states = summary_net.apply(params_obj["summary"], encoded)
        response_pred = states @ params_obj["response_w"] + params_obj["response_b"]
        slice_pred = jnp.zeros((int(states.shape[0]), 0), dtype=response_pred.dtype)
        critic = jnp.zeros((int(states.shape[0]),), dtype=response_pred.dtype)
        return states, response_pred, slice_pred, critic

    result_params = {
        "summary": params,
        "response_w": response_w,
        "response_b": response_b,
    }
    train_diag = evaluate_contextual_sbijax(
        params=result_params,
        apply_fn=apply_theta_supervised,
        dataset=train,
    )
    val_diag = evaluate_contextual_sbijax(
        params=result_params,
        apply_fn=apply_theta_supervised,
        dataset=val,
    )
    provenance = contextual_sbijax_provenance(
        method=method,
        response_signature_contexts=int(config.response_signature_contexts),
        response_signature_slices=int(n_slices),
        trainer="theta_supervised",
        input_encoding=str(config.input_encoding),
        downstream_readout="fixed_affine_probe_from_theta",
    )
    provenance.update(
        {
            "summary_network": "haiku_mlp_theta_mse",
            "package_theta": str(config.package_theta),
            "package_theta_dim": int(train_theta_np.shape[1]),
            "affine_probe_response_dim": int(train_response_dim),
            "baseline_role": "theta_supervised_control",
            "decoder_kind": "learned",
            "exact_zero_claim": False,
        }
    )
    return ContextualSBIJAXResult(
        params=result_params,
        history=history,
        train_diagnostics=train_diag,
        val_diagnostics=val_diag,
        provenance=provenance,
        config=config,
        slice_matrix=None,
        apply_fn=apply_theta_supervised,
    )


def fit_contextual_sbijax_identity_theta(
    train: ContextualResponseDataset,
    val: ContextualResponseDataset,
    *,
    config: ContextualSBIJAXConfig,
) -> ContextualSBIJAXResult:
    """Exact identity control for supplied package theta rows.

    This is not a learned model.  It checks the strongest possible claim in
    the Markov setting: when package input ``y`` is already the exact Markov
    sketch and package ``theta`` is the same sketch, the state map is the
    identity and the contextual responses are deterministically recoverable.
    """

    method, _trainer, train_response_dim, n_slices = _validate_fit_inputs(
        train,
        val,
        config=config,
    )
    if str(config.package_theta) != "markov_exact_sketch":
        raise ValueError("identity_theta currently requires package_theta='markov_exact_sketch'")
    if str(config.input_encoding) != "markov_exact_sketch":
        raise ValueError("identity_theta requires input_encoding='markov_exact_sketch'")

    deps = _require_contextual_sbi()
    jnp = deps.jnp

    block_by_token = _metadata_block_by_token(train)
    if block_by_token is None:
        raise ValueError("identity_theta requires Markov block_by_token metadata")
    n_regimes_raw = train.metadata.get("n_regimes")
    n_regimes = int(max(block_by_token)) + 1 if n_regimes_raw is None else int(n_regimes_raw)
    theta_dim = 1 + 2 * int(n_regimes)

    train_features = _encode_dataset_items_for_sbijax_package(
        deps,
        train,
        config=config,
    )
    val_features = _encode_dataset_items_for_sbijax_package(
        deps,
        val,
        config=config,
    )
    train_theta_np = _package_theta_array(train, package_theta=str(config.package_theta))
    val_theta_np = _package_theta_array(val, package_theta=str(config.package_theta))
    train_feature_np = np.asarray(train_features, dtype=np.float32)
    val_feature_np = np.asarray(val_features, dtype=np.float32)
    if tuple(train_feature_np.shape) != tuple(train_theta_np.shape):
        raise ValueError(
            "identity_theta requires encoded train inputs and theta targets to "
            f"have the same shape; got {train_feature_np.shape} and {train_theta_np.shape}"
        )
    if tuple(val_feature_np.shape) != tuple(val_theta_np.shape):
        raise ValueError(
            "identity_theta requires encoded val inputs and theta targets to "
            f"have the same shape; got {val_feature_np.shape} and {val_theta_np.shape}"
        )
    if int(train_feature_np.shape[1]) != int(theta_dim):
        raise ValueError(
            f"identity_theta expected exact sketch dimension {theta_dim}; "
            f"got {int(train_feature_np.shape[1])}"
        )
    train_theta_identity_mse = float(np.mean((train_feature_np - train_theta_np) ** 2))
    val_theta_identity_mse = float(np.mean((val_feature_np - val_theta_np) ** 2))
    max_identity_mse = max(train_theta_identity_mse, val_theta_identity_mse)
    if max_identity_mse > 1e-10:
        raise ValueError(
            "identity_theta found encoded inputs that do not match supplied theta "
            f"targets; max MSE={max_identity_mse:.6g}"
        )

    scale = float(train.target_scale)

    def _context_constants(dataset: ContextualResponseDataset) -> dict[str, Any]:
        left_counts = []
        right_counts = []
        left_last = []
        right_first = []
        left_has = []
        right_has = []
        for left, right in zip(
            dataset.context_left_raw,
            dataset.context_right_raw,
            strict=True,
        ):
            left_tokens = [int(tok) for tok in left]
            right_tokens = [int(tok) for tok in right]
            left_counts.append(
                exact_count_for_tokens(left_tokens, block_by_token=block_by_token) / scale
            )
            right_counts.append(
                exact_count_for_tokens(right_tokens, block_by_token=block_by_token) / scale
            )
            left_has.append(1.0 if left_tokens else 0.0)
            right_has.append(1.0 if right_tokens else 0.0)
            left_last.append(int(block_by_token[int(left_tokens[-1])]) if left_tokens else 0)
            right_first.append(int(block_by_token[int(right_tokens[0])]) if right_tokens else 0)
        return {
            "left_counts": jnp.asarray(left_counts, dtype=jnp.float32),
            "right_counts": jnp.asarray(right_counts, dtype=jnp.float32),
            "left_last": jnp.asarray(left_last, dtype=jnp.int32),
            "right_first": jnp.asarray(right_first, dtype=jnp.int32),
            "left_has": jnp.asarray(left_has, dtype=jnp.float32),
            "right_has": jnp.asarray(right_has, dtype=jnp.float32),
        }

    constants = _context_constants(train)

    def _responses_from_exact_states(states):
        count_norm = states[:, :1]
        first_oh = states[:, 1 : 1 + int(n_regimes)]
        last_oh = states[:, 1 + int(n_regimes) : 1 + 2 * int(n_regimes)]
        left_boundary = constants["left_has"] * (1.0 - first_oh[:, constants["left_last"]]) / scale
        right_boundary = (
            constants["right_has"] * (1.0 - last_oh[:, constants["right_first"]]) / scale
        )
        return (
            count_norm
            + constants["left_counts"]
            + constants["right_counts"]
            + left_boundary
            + right_boundary
        )

    def apply_identity_theta(params_obj, tokens, responses):
        del params_obj, responses
        states = _encode_tokens_for_sbijax_package(
            deps,
            tokens,
            vocab_size=int(config.vocab_size),
            input_encoding=str(config.input_encoding),
            block_by_token=block_by_token,
            n_regimes=int(n_regimes),
            target_scale=float(train.target_scale),
        )
        response_pred = _responses_from_exact_states(states)
        slice_pred = jnp.zeros((int(states.shape[0]), 0), dtype=response_pred.dtype)
        critic = jnp.zeros((int(states.shape[0]),), dtype=response_pred.dtype)
        return states, response_pred, slice_pred, critic

    result_params = {
        "identity": True,
        "context_constants": constants,
    }
    train_diag = evaluate_contextual_sbijax(
        params=result_params,
        apply_fn=apply_identity_theta,
        dataset=train,
    )
    val_diag = evaluate_contextual_sbijax(
        params=result_params,
        apply_fn=apply_identity_theta,
        dataset=val,
    )
    history: list[dict[str, float | int]] = [
        {
            "iteration": 0,
            "trainer": "identity_theta",
            "train_loss": float(train_diag["contextual_mse"]),
            "train_theta_mse": train_theta_identity_mse,
            "train_contextual_mse": float(train_diag["contextual_mse"]),
            "train_package_loss": train_theta_identity_mse,
            "val_loss": float(val_diag["contextual_mse"]),
            "val_theta_mse": val_theta_identity_mse,
            "val_contextual_mse": float(val_diag["contextual_mse"]),
            "val_package_loss": val_theta_identity_mse,
        }
    ]
    provenance = contextual_sbijax_provenance(
        method=method,
        response_signature_contexts=int(config.response_signature_contexts),
        response_signature_slices=int(n_slices),
        trainer="identity_theta",
        input_encoding=str(config.input_encoding),
        downstream_readout="deterministic_markov_exact_sketch",
    )
    provenance.update(
        {
            "summary_network": "identity",
            "package_theta": str(config.package_theta),
            "package_theta_dim": int(theta_dim),
            "affine_probe_response_dim": int(train_response_dim),
            "train_theta_identity_mse": train_theta_identity_mse,
            "val_theta_identity_mse": val_theta_identity_mse,
            "baseline_role": "identity_theta_exact_control",
            "decoder_kind": "exact",
            "exact_zero_claim": True,
        }
    )
    return ContextualSBIJAXResult(
        params=result_params,
        history=history,
        train_diagnostics=train_diag,
        val_diagnostics=val_diag,
        provenance=provenance,
        config=config,
        slice_matrix=None,
        apply_fn=apply_identity_theta,
    )


def fit_contextual_sbijax_exact_zero_markov(
    train: ContextualResponseDataset,
    val: ContextualResponseDataset,
    *,
    config: ContextualSBIJAXConfig,
) -> ContextualSBIJAXResult:
    """Exact structural Markov control from tokens to contextual responses.

    This lane owns the simple Markov sufficient statistic directly instead of
    asking package NASS/NASSS to discover it. It computes
    ``[count/scale, first one-hot, last one-hot]`` from token regimes, checks
    that this equals the attached ``markov_exact_sketch`` theta target, and
    decodes contextual responses by the deterministic two-sided Markov formula.
    """

    method, _trainer, train_response_dim, n_slices = _validate_fit_inputs(
        train,
        val,
        config=config,
    )
    if str(config.package_theta) != "markov_exact_sketch":
        raise ValueError("exact_zero_markov requires package_theta='markov_exact_sketch'")
    if tuple(train.context_left_raw) != tuple(val.context_left_raw) or tuple(
        train.context_right_raw
    ) != tuple(val.context_right_raw):
        raise ValueError(
            "exact_zero_markov requires train/val datasets to reuse the same "
            "context bank because the deterministic decoder is context-specific"
        )

    deps = _require_contextual_sbi()
    jnp = deps.jnp

    block_by_token = _metadata_block_by_token(train)
    if block_by_token is None:
        raise ValueError("exact_zero_markov requires Markov block_by_token metadata")
    n_regimes_raw = train.metadata.get("n_regimes")
    n_regimes = int(max(block_by_token)) + 1 if n_regimes_raw is None else int(n_regimes_raw)
    theta_dim = 1 + 2 * int(n_regimes)

    def _exact_features(dataset: ContextualResponseDataset):
        return _encode_tokens_for_sbijax_package(
            deps,
            dataset.item_tokens,
            vocab_size=int(config.vocab_size),
            input_encoding="markov_exact_sketch",
            block_by_token=block_by_token,
            n_regimes=int(n_regimes),
            target_scale=float(dataset.target_scale),
        )

    train_features = _exact_features(train)
    val_features = _exact_features(val)
    train_theta_np = _package_theta_array(train, package_theta=str(config.package_theta))
    val_theta_np = _package_theta_array(val, package_theta=str(config.package_theta))
    train_feature_np = np.asarray(train_features, dtype=np.float32)
    val_feature_np = np.asarray(val_features, dtype=np.float32)
    if int(train_feature_np.shape[1]) != int(theta_dim):
        raise ValueError(
            f"exact_zero_markov expected exact sketch dimension {theta_dim}; "
            f"got {int(train_feature_np.shape[1])}"
        )
    if tuple(train_feature_np.shape) != tuple(train_theta_np.shape):
        raise ValueError(
            "exact_zero_markov requires encoded train states and theta targets "
            f"to have the same shape; got {train_feature_np.shape} and "
            f"{train_theta_np.shape}"
        )
    if tuple(val_feature_np.shape) != tuple(val_theta_np.shape):
        raise ValueError(
            "exact_zero_markov requires encoded val states and theta targets "
            f"to have the same shape; got {val_feature_np.shape} and "
            f"{val_theta_np.shape}"
        )
    train_theta_identity_mse = float(np.mean((train_feature_np - train_theta_np) ** 2))
    val_theta_identity_mse = float(np.mean((val_feature_np - val_theta_np) ** 2))
    max_identity_mse = max(train_theta_identity_mse, val_theta_identity_mse)
    if max_identity_mse > 1e-10:
        raise ValueError(
            "exact_zero_markov computed states that do not match supplied theta "
            f"targets; max MSE={max_identity_mse:.6g}"
        )

    decoder = _markov_exact_decoder_setup(deps, train)

    def apply_exact_zero_markov(params_obj, tokens, responses):
        del responses
        states = _encode_tokens_for_sbijax_package(
            deps,
            tokens,
            vocab_size=int(config.vocab_size),
            input_encoding="markov_exact_sketch",
            block_by_token=params_obj["block_by_token"],
            n_regimes=int(params_obj["n_regimes"]),
            target_scale=float(params_obj["target_scale"]),
        )
        response_pred = _responses_from_markov_exact_states(
            states,
            decoder=params_obj["decoder"],
        )
        slice_pred = jnp.zeros((int(states.shape[0]), 0), dtype=response_pred.dtype)
        critic = jnp.zeros((int(states.shape[0]),), dtype=response_pred.dtype)
        return states, response_pred, slice_pred, critic

    result_params = {
        "deterministic_markov_sketch": True,
        "block_by_token": block_by_token,
        "n_regimes": int(n_regimes),
        "target_scale": float(train.target_scale),
        "decoder": decoder,
    }
    train_diag = evaluate_contextual_sbijax(
        params=result_params,
        apply_fn=apply_exact_zero_markov,
        dataset=train,
    )
    val_diag = evaluate_contextual_sbijax(
        params=result_params,
        apply_fn=apply_exact_zero_markov,
        dataset=val,
    )
    history: list[dict[str, float | int]] = [
        {
            "iteration": 0,
            "trainer": "exact_zero_markov",
            "train_loss": float(train_diag["contextual_mse"]),
            "train_theta_mse": train_theta_identity_mse,
            "train_contextual_mse": float(train_diag["contextual_mse"]),
            "train_package_loss": train_theta_identity_mse,
            "val_loss": float(val_diag["contextual_mse"]),
            "val_theta_mse": val_theta_identity_mse,
            "val_contextual_mse": float(val_diag["contextual_mse"]),
            "val_package_loss": val_theta_identity_mse,
        }
    ]
    provenance = contextual_sbijax_provenance(
        method=method,
        response_signature_contexts=int(config.response_signature_contexts),
        response_signature_slices=int(n_slices),
        trainer="exact_zero_markov",
        input_encoding=str(config.input_encoding),
        downstream_readout="deterministic_markov_exact_sketch",
    )
    provenance.update(
        {
            "summary_network": "deterministic_structural_markov_sketch",
            "effective_input_encoding": "markov_exact_sketch",
            "package_theta": str(config.package_theta),
            "package_theta_dim": int(theta_dim),
            "affine_probe_response_dim": int(train_response_dim),
            "train_theta_identity_mse": train_theta_identity_mse,
            "val_theta_identity_mse": val_theta_identity_mse,
            "baseline_role": "exact_zero_markov_control",
            "decoder_kind": "exact",
            "exact_zero_claim": True,
        }
    )
    return ContextualSBIJAXResult(
        params=result_params,
        history=history,
        train_diagnostics=train_diag,
        val_diagnostics=val_diag,
        provenance=provenance,
        config=config,
        slice_matrix=None,
        apply_fn=apply_exact_zero_markov,
    )


def fit_contextual_sbijax_learned_local_laws(
    train: ContextualResponseDataset,
    val: ContextualResponseDataset,
    *,
    config: ContextualSBIJAXConfig,
) -> ContextualSBIJAXResult:
    """Train/evaluate the repo-owned local-law sketch lane.

    Markov uses the theorem-domain ``[count/scale, first one-hot, last
    one-hot]`` sketch. HLL uses normalized registers with exact merge =
    elementwise max. In both cases the exact decoder remains deterministic
    unless ``law_architecture`` explicitly requests a learned decoder.
    """

    method, _trainer, train_response_dim, n_slices = _validate_fit_inputs(
        train,
        val,
        config=config,
    )
    law_package = str(config.package_theta)
    if law_package not in {"markov_exact_sketch", "hll_register_sketch"}:
        raise ValueError(
            "learned_local_laws requires package_theta='markov_exact_sketch' "
            "or 'hll_register_sketch'"
        )
    if tuple(train.context_payloads) != tuple(val.context_payloads):
        raise ValueError(
            "learned_local_laws requires train/val datasets to reuse the same "
            "context bank because the exact decoder is context-specific"
        )

    deps = _require_contextual_sbi()
    jax = deps.jax
    jnp = deps.jnp
    jr = deps.jr
    optax = deps.optax

    is_markov_law = law_package == "markov_exact_sketch"
    is_hll_law = law_package == "hll_register_sketch"
    hll_estimate_weight = float(config.local_law_hll_estimate_weight)
    if hll_estimate_weight != 0.0 and not is_hll_law:
        raise ValueError("local_law_hll_estimate_weight is only defined for hll_register_sketch")
    block_by_token = _metadata_block_by_token(train)
    n_regimes: int | None = None
    hll_precision = int(train.metadata.get("hll_precision", int(config.hll_precision)))
    hll_hash_bits = int(train.metadata.get("hll_hash_bits", int(config.hll_hash_bits)))
    hll_max_register = int(
        _hll_max_register(precision=int(hll_precision), hash_bits=int(hll_hash_bits))
    )
    if is_markov_law:
        if block_by_token is None:
            raise ValueError("markov_exact_sketch requires block_by_token metadata")
        n_regimes_raw = train.metadata.get("n_regimes")
        n_regimes = int(max(block_by_token)) + 1 if n_regimes_raw is None else int(n_regimes_raw)
        theta_dim = 1 + 2 * int(n_regimes)
        law_set_id = MARKOV_COUNT_SKETCH_LAW_SET_ID
    else:
        if str(config.input_encoding) in {
            "regime_ids",
            "regime_one_hot",
            "markov_exact_sketch",
        }:
            raise ValueError(
                "hll_register_sketch learned_local_laws supports "
                "normalized_token_ids or one_hot_token_ids input_encoding"
            )
        theta_dim = int(1 << int(hll_precision))
        law_set_id = HLL_REGISTER_SKETCH_LAW_SET_ID
    summary_family = str(config.local_law_summary_family)
    # Count-only mode: state_dim_effective is the learned rep width (default
    # 2 * theta_dim, "fair comparison + big d"). C1/C2 grade
    # ``count_readout(rep) ≈ count_truth_norm`` instead of full-sketch MSE,
    # so (first, last) are emergent rather than supervised.
    count_only = bool(config.local_law_count_only)
    if count_only and not is_markov_law:
        raise ValueError("local_law_count_only is only defined for markov_exact_sketch")
    if count_only:
        rep_dim_override = int(config.local_law_rep_dim)
        state_dim_effective = int(rep_dim_override) if rep_dim_override > 0 else 2 * int(theta_dim)
    else:
        state_dim_effective = int(theta_dim)
    explicit_state_decoder = bool(config.local_law_explicit_state_decoder)
    if explicit_state_decoder:
        if summary_family in {"affine_probe", "regime_transition_sum"}:
            raise ValueError(
                "local_law_explicit_state_decoder=True supports learned summary "
                "families ('mlp', 'jax_fno', 'norax_fno'); affine_probe and "
                "regime_transition_sum emit decoded theta directly"
            )
        summary_dim_override = int(config.local_law_summary_dim)
        state_dim_effective = (
            int(summary_dim_override)
            if summary_dim_override > 0
            else max(int(theta_dim), int(config.state_dim))
        )

    lazy_train_item_features = bool(
        is_hll_law
        and str(config.input_encoding) == "one_hot_token_ids"
        and summary_family == "mlp"
    )
    train_item_tokens = jnp.asarray(train.item_tokens, dtype=jnp.int32)
    n_train_items = int(train_item_tokens.shape[0])
    train_feature_dim = (
        int(train_item_tokens.shape[1]) * (int(config.vocab_size) + 1)
        if lazy_train_item_features
        else 0
    )
    if lazy_train_item_features:
        # HLL one-hot inputs can exceed 25 GiB at 102400 docs and leaf=512.
        # Keep token ids resident and materialize dense one-hot features only
        # for the current training mini-batch.
        train_features = None
    else:
        train_features = _encode_dataset_items_for_sbijax_package(
            deps,
            train,
            config=config,
        )
        train_feature_dim = int(train_features.shape[1])
    val_features = _encode_dataset_items_for_sbijax_package(
        deps,
        val,
        config=config,
    )
    train_theta_np = _package_theta_array(train, package_theta=str(config.package_theta))
    val_theta_np = _package_theta_array(val, package_theta=str(config.package_theta))
    if int(train_theta_np.shape[1]) != int(theta_dim):
        raise ValueError(
            f"learned_local_laws expected theta_dim={theta_dim}; "
            f"got {int(train_theta_np.shape[1])}"
        )
    if int(val_theta_np.shape[1]) != int(theta_dim):
        raise ValueError(
            f"learned_local_laws expected val theta_dim={theta_dim}; "
            f"got {int(val_theta_np.shape[1])}"
        )
    train_theta = jnp.asarray(train_theta_np, dtype=jnp.float32)
    val_theta = jnp.asarray(val_theta_np, dtype=jnp.float32)
    train_responses = jnp.asarray(_flatten_response_signatures(train), dtype=jnp.float32)
    val_responses = jnp.asarray(_flatten_response_signatures(val), dtype=jnp.float32)
    if is_markov_law:
        decoder = _markov_exact_decoder_setup(deps, train)
    else:
        decoder = _hll_exact_decoder_setup(
            deps,
            train,
            precision=int(hll_precision),
            hash_bits=int(hll_hash_bits),
        )
    package_aux_weight = float(config.local_law_package_weight)
    package_aux_active = bool(package_aux_weight != 0.0)
    hll_estimate_active = bool(is_hll_law and hll_estimate_weight != 0.0)
    regime_feature_width = int(n_regimes) + 1 if n_regimes is not None else 0
    regime_transition_fragment_len = 0
    if summary_family == "regime_transition_sum":
        if not is_markov_law:
            raise ValueError(
                "local_law_summary_family='regime_transition_sum' is only "
                "defined for markov_exact_sketch"
            )
        if str(config.input_encoding) != "regime_one_hot":
            raise ValueError(
                "local_law_summary_family='regime_transition_sum' requires "
                "input_encoding='regime_one_hot'"
            )
        if package_aux_active:
            raise ValueError(
                "local_law_summary_family='regime_transition_sum' does not "
                "support local_law_package_weight != 0"
            )
        feature_dim = int(train_feature_dim)
        if feature_dim % int(regime_feature_width) != 0:
            raise ValueError(
                "regime_transition_sum expected flattened regime one-hot "
                f"features divisible by n_regimes+1={regime_feature_width}; "
                f"got feature_dim={feature_dim}"
            )
        regime_transition_fragment_len = feature_dim // int(regime_feature_width)
    fno_summary_input_width = 0
    fno_summary_fragment_len = 0
    summary_family_canonical = "jax_fno" if summary_family == "norax_fno" else summary_family
    if summary_family_canonical == "jax_fno":
        # Package aux IS supported on jax_fno via
        # ``_make_jax_fno_summary_package_aux_net`` (FNO backbone + sbijax
        # NASSS slice + NASS critic heads). The previous hard-error here is
        # gone; the dispatch below picks the right factory based on
        # ``package_aux_active``.
        fno_summary_input_width = _fno_input_width_for_encoding(
            encoding=str(config.input_encoding),
            n_regimes=int(n_regimes or 0),
            vocab_size=int(config.vocab_size),
        )
        feature_dim = int(train_feature_dim)
        if feature_dim % int(fno_summary_input_width) != 0:
            raise ValueError(
                f"{summary_family} summary expected flattened features "
                f"divisible by input_width={fno_summary_input_width}; "
                f"got feature_dim={feature_dim}"
            )
        fno_summary_fragment_len = feature_dim // int(fno_summary_input_width)
        if fno_summary_fragment_len <= 0:
            raise ValueError(
                f"{summary_family} summary derived fragment_len <= 0; "
                "check feature_dim and input encoding"
            )
    package_slice_key = jr.PRNGKey(int(config.seed) + 829)
    package_slice_matrix = _unit_slice_matrix(
        deps,
        key=package_slice_key,
        n_contexts=int(train_response_dim),
        n_slices=int(n_slices),
    )

    def _encode_tokens(tokens: Any) -> Any:
        return _encode_tokens_for_sbijax_package(
            deps,
            tokens,
            vocab_size=int(config.vocab_size),
            input_encoding=str(config.input_encoding),
            block_by_token=block_by_token,
            n_regimes=None if n_regimes is None else int(n_regimes),
            target_scale=float(train.target_scale),
        )

    train_left_np, train_right_np, train_merge_idx = _markov_merge_split_arrays(train)
    val_left_np, val_right_np, val_merge_idx = _markov_merge_split_arrays(val)
    lazy_train_merge_features = bool(lazy_train_item_features)
    train_merge_source_idx = jnp.asarray(train_merge_idx, dtype=jnp.int32)
    if lazy_train_merge_features:
        # For HLL p=8 and long leaves, dense one-hot merge tensors are tens of
        # GiB each at 102400 docs. Keep compact token ids resident and encode
        # only the sampled merge mini-batch inside ``step``.
        train_left_tokens = jnp.asarray(train_left_np, dtype=jnp.int32)
        train_right_tokens = jnp.asarray(train_right_np, dtype=jnp.int32)
        train_left_features = None
        train_right_features = None
    else:
        train_left_tokens = None
        train_right_tokens = None
        train_left_features = _encode_tokens(train_left_np)
        train_right_features = _encode_tokens(train_right_np)
    val_left_features = _encode_tokens(val_left_np)
    val_right_features = _encode_tokens(val_right_np)
    train_merge_target = jnp.asarray(train_theta_np[train_merge_idx], dtype=jnp.float32)
    val_merge_target = jnp.asarray(val_theta_np[val_merge_idx], dtype=jnp.float32)
    if lazy_train_merge_features:
        train_merge_full_features = None
    elif int(train_merge_idx.shape[0]) > 0:
        assert train_features is not None
        train_merge_full_features = train_features[jnp.asarray(train_merge_idx, dtype=jnp.int32)]
    else:
        train_merge_full_features = jnp.zeros((0, int(train_feature_dim)), dtype=jnp.float32)
    if int(val_merge_idx.shape[0]) > 0:
        val_merge_full_features = val_features[jnp.asarray(val_merge_idx, dtype=jnp.int32)]
    else:
        val_merge_full_features = jnp.zeros((0, int(val_features.shape[1])), dtype=jnp.float32)

    mode = str(config.local_law_supervision_mode)
    train_law_rows = markov_local_law_observation_rows(
        leaf_losses=[0.0] * int(train_theta_np.shape[0]),
        merge_losses=[0.0] * int(train_merge_target.shape[0]),
        idempotence_losses=[0.0] * int(train_theta_np.shape[0]),
        supervision_mode=mode,
        leaf_rate=float(config.local_law_leaf_rate),
        merge_rate=float(config.local_law_merge_rate),
        idempotence_rate=float(config.local_law_idempotence_rate),
        seed=int(config.seed) + 701,
        law_set_id=law_set_id,
    )
    val_law_rows = markov_local_law_observation_rows(
        leaf_losses=[0.0] * int(val_theta_np.shape[0]),
        merge_losses=[0.0] * int(val_merge_target.shape[0]),
        idempotence_losses=[0.0] * int(val_theta_np.shape[0]),
        supervision_mode=mode,
        leaf_rate=float(config.local_law_leaf_rate),
        merge_rate=float(config.local_law_merge_rate),
        idempotence_rate=float(config.local_law_idempotence_rate),
        seed=int(config.seed) + 709,
        law_set_id=law_set_id,
    )
    train_sparse_law_rows = None
    val_sparse_law_rows = None
    if mode == "dual":
        train_sparse_law_rows = markov_local_law_observation_rows(
            leaf_losses=[0.0] * int(train_theta_np.shape[0]),
            merge_losses=[0.0] * int(train_merge_target.shape[0]),
            idempotence_losses=[0.0] * int(train_theta_np.shape[0]),
            supervision_mode="sparse_ipw",
            leaf_rate=float(config.local_law_leaf_rate),
            merge_rate=float(config.local_law_merge_rate),
            idempotence_rate=float(config.local_law_idempotence_rate),
            seed=int(config.seed) + 719,
            law_set_id=law_set_id,
        )
        val_sparse_law_rows = markov_local_law_observation_rows(
            leaf_losses=[0.0] * int(val_theta_np.shape[0]),
            merge_losses=[0.0] * int(val_merge_target.shape[0]),
            idempotence_losses=[0.0] * int(val_theta_np.shape[0]),
            supervision_mode="sparse_ipw",
            leaf_rate=float(config.local_law_leaf_rate),
            merge_rate=float(config.local_law_merge_rate),
            idempotence_rate=float(config.local_law_idempotence_rate),
            seed=int(config.seed) + 727,
            law_set_id=law_set_id,
        )

    def _weights(rows: Sequence[Any]) -> Any:
        values = []
        for row in rows:
            if bool(row.observed):
                values.append(1.0 / max(float(row.propensity), 1e-12))
            else:
                values.append(0.0)
        return jnp.asarray(values, dtype=jnp.float32)

    train_leaf_weights = _weights(train_law_rows.rows_by_law[LawKind.L1_LEAF])
    train_merge_weights = _weights(train_law_rows.rows_by_law[LawKind.L2_MERGE])
    train_idemp_weights = _weights(train_law_rows.rows_by_law[LawKind.L3_IDEMPOTENCE])
    val_leaf_weights = _weights(val_law_rows.rows_by_law[LawKind.L1_LEAF])
    val_merge_weights = _weights(val_law_rows.rows_by_law[LawKind.L2_MERGE])
    val_idemp_weights = _weights(val_law_rows.rows_by_law[LawKind.L3_IDEMPOTENCE])

    summary_is_identity = (
        is_markov_law
        and str(config.input_encoding) == "markov_exact_sketch"
        and int(train_feature_dim) == int(theta_dim)
    )
    summary_is_affine = (not summary_is_identity) and summary_family == "affine_probe"
    summary_is_regime_transition_sum = (
        not summary_is_identity
    ) and summary_family == "regime_transition_sum"
    summary_is_jax_fno = (not summary_is_identity) and summary_family in {"jax_fno", "norax_fno"}
    law_architecture = str(config.law_architecture)
    architecture_uses_learned_merge = law_architecture in {"learned_merge", "fully_learned"}
    architecture_uses_learned_decoder = law_architecture in {"learned_decoder", "fully_learned"}
    c2_merge_target_kind = str(config.c2_merge_target)
    learned_merge_hidden_dim = int(config.learned_merge_hidden_dim) or int(config.hidden_dim)
    learned_decoder_hidden_dim = int(config.learned_decoder_hidden_dim) or int(config.hidden_dim)
    summary_net = None
    merge_net = None
    state_decoder_net = None
    decoder_net = None
    summary_params = None
    merge_params = None
    state_decoder_params = None
    decoder_params = None
    opt_state = None
    optimizer = None
    if summary_is_affine:
        # ``lazy_train_item_features`` is only enabled for MLP summaries.
        assert train_features is not None
        affine_w_np, affine_b_np = _affine_response_probe(
            np.asarray(train_features, dtype=np.float32),
            train_theta_np,
        )
        summary_params = {
            "w": jnp.asarray(affine_w_np, dtype=jnp.float32),
            "b": jnp.asarray(affine_b_np, dtype=jnp.float32),
        }
    elif not summary_is_identity:
        if summary_is_regime_transition_sum:
            summary_net = _make_regime_transition_sum_summary_net(
                deps,
                n_regimes=int(n_regimes),
                fragment_len=int(regime_transition_fragment_len),
                hidden_dim=int(config.hidden_dim),
                target_scale=float(train.target_scale),
            )
        elif summary_is_jax_fno:
            if package_aux_active:
                summary_net = _make_jax_fno_summary_package_aux_net(
                    deps,
                    fragment_len=int(fno_summary_fragment_len),
                    input_width=int(fno_summary_input_width),
                    fno_width=int(config.hidden_dim),
                    n_modes=int(config.local_law_summary_fno_n_modes),
                    n_layers=int(config.local_law_summary_fno_n_layers),
                    pooling_mode=str(config.local_law_summary_fno_pooling_mode),
                    output_dim=int(state_dim_effective),
                    hidden_dim=int(config.hidden_dim),
                    response_signature_dim=int(train_response_dim),
                    response_signature_slices=int(n_slices),
                )
            else:
                summary_net = _make_jax_fno_summary_net(
                    deps,
                    fragment_len=int(fno_summary_fragment_len),
                    input_width=int(fno_summary_input_width),
                    fno_width=int(config.hidden_dim),
                    n_modes=int(config.local_law_summary_fno_n_modes),
                    n_layers=int(config.local_law_summary_fno_n_layers),
                    pooling_mode=str(config.local_law_summary_fno_pooling_mode),
                    output_dim=int(state_dim_effective),
                )
        elif package_aux_active:
            summary_net = _make_theta_summary_package_aux_net(
                deps,
                theta_dim=int(state_dim_effective),
                hidden_dim=int(config.hidden_dim),
                response_signature_dim=int(train_response_dim),
                response_signature_slices=int(n_slices),
            )
        else:
            summary_net = _make_theta_summary_net(
                deps,
                theta_dim=int(state_dim_effective),
                hidden_dim=int(config.hidden_dim),
            )
        key = jr.PRNGKey(int(config.seed) + 811)
        init_n = min(max(1, int(config.batch_size)), int(n_train_items))
        if lazy_train_item_features:
            init_features = _encode_tokens(train_item_tokens[:init_n])
        else:
            assert train_features is not None
            init_features = train_features[:init_n]
        if package_aux_active:
            summary_params = summary_net.init(
                key, init_features, train_responses[:init_n]
            )
        else:
            summary_params = summary_net.init(key, init_features)

    if explicit_state_decoder:
        state_decoder_net = _make_state_decoder_net(
            deps,
            summary_dim=int(state_dim_effective),
            theta_dim=int(theta_dim),
            hidden_dim=int(config.hidden_dim),
            head=str(config.local_law_state_decoder_head),
        )
        state_decoder_key = jr.PRNGKey(int(config.seed) + 817)
        state_decoder_init_summary = jnp.zeros((1, int(state_dim_effective)), dtype=jnp.float32)
        state_decoder_params = state_decoder_net.init(
            state_decoder_key,
            state_decoder_init_summary,
        )

    if architecture_uses_learned_merge:
        merge_family_kind = str(config.merge_family)
        if merge_family_kind == "fno_rep":
            merge_net = _make_fno_merge_net(
                deps,
                state_dim=int(state_dim_effective),
                hidden_channels=int(config.merge_fno_hidden_channels),
                n_modes=int(config.merge_fno_n_modes),
                n_layers=int(config.merge_fno_n_layers),
            )
        else:
            merge_net = _make_learned_merge_net(
                deps,
                state_dim=int(state_dim_effective),
                hidden_dim=int(learned_merge_hidden_dim),
            )
        merge_key = jr.PRNGKey(int(config.seed) + 821)
        merge_init_n = min(
            max(1, int(config.batch_size)),
            max(1, int(train_merge_target.shape[0])),
        )
        if int(merge_init_n) == 0:
            merge_init_n = 1
        if int(train_merge_target.shape[0]) > 0:
            init_left = jnp.asarray(
                np.zeros((int(merge_init_n), int(state_dim_effective)), dtype=np.float32),
                dtype=jnp.float32,
            )
            init_right = init_left
        else:
            init_left = jnp.zeros((1, int(state_dim_effective)), dtype=jnp.float32)
            init_right = init_left
        merge_params = merge_net.init(merge_key, init_left, init_right)

    if architecture_uses_learned_decoder:
        decoder_net = _make_learned_decoder_net(
            deps,
            response_dim=int(train_response_dim),
            hidden_dim=int(learned_decoder_hidden_dim),
            head=str(config.decoder_head),
        )
        decoder_key = jr.PRNGKey(int(config.seed) + 829)
        decoder_input_dim = int(theta_dim) if explicit_state_decoder else int(state_dim_effective)
        decoder_init_state = jnp.zeros((1, int(decoder_input_dim)), dtype=jnp.float32)
        decoder_params = decoder_net.init(decoder_key, decoder_init_state)

    # Count-only readout: a small Flax linen Dense(state_dim_effective, 1)
    # mapping the learned rep -> normalized scalar count. Only instantiated
    # when ``count_only`` is True; supervises C1 leaf and C2 merge against
    # ``count_truth_norm`` rather than the full sketch.
    count_readout_net = None
    count_readout_params = None
    if count_only:

        class _CountReadoutNet(deps.fnn.Module):

            @deps.fnn.compact
            def __call__(self, rep):
                return deps.fnn.Dense(1, name="count_readout_out")(rep).squeeze(-1)

        count_readout_net = _CountReadoutNet()
        count_readout_key = jr.PRNGKey(int(config.seed) + 833)
        count_readout_init_state = jnp.zeros((1, int(state_dim_effective)), dtype=jnp.float32)
        count_readout_params = count_readout_net.init(count_readout_key, count_readout_init_state)

    # NASS-style JSD critic for the merge (when ``local_law_merge_loss ==
    # 'nass_jsd'``): supervises the merge_net to produce parent reps that
    # are sufficient summaries of the merge-node truth via a contrastive
    # bound on I(merge_state; merge_truth). This is "use sbijax internally
    # for the merge."
    merge_loss_kind = str(config.local_law_merge_loss)
    use_nass_jsd_merge = merge_loss_kind == "nass_jsd"
    use_nasss_jsd_merge = merge_loss_kind == "nasss_jsd"
    use_jsd_merge = use_nass_jsd_merge or use_nasss_jsd_merge
    merge_critic_net = None
    merge_critic_params = None
    merge_nasss_slice_matrix = None
    if use_jsd_merge:
        merge_critic_hidden = max(32, int(config.hidden_dim))
        # NASS critic takes the merge state and a target vector
        # (full-dim for nass_jsd, or a single sliced scalar broadcast to a
        # 1-vec for nasss_jsd handled as a per-slice critic via vmap below).
        # We share a single critic net that consumes ``(state, target)``
        # concatenated; for nasss_jsd ``target`` is a length-1 slice.
        critic_target_dim = 1 if use_nasss_jsd_merge else int(theta_dim)

        class _MergeJSDCriticNet(deps.fnn.Module):

            @deps.fnn.compact
            def __call__(self, merge_state, merge_truth):
                x = jnp.concatenate([merge_state, merge_truth], axis=-1)
                x = deps.fnn.Dense(merge_critic_hidden, name="merge_critic_l0")(x)
                x = deps.fnn.relu(x)
                x = deps.fnn.Dense(merge_critic_hidden, name="merge_critic_l1")(x)
                x = deps.fnn.relu(x)
                return deps.fnn.Dense(1, name="merge_critic_out")(x).squeeze(-1)

        merge_critic_net = _MergeJSDCriticNet()
        merge_critic_key = jr.PRNGKey(int(config.seed) + 911)
        init_critic_state = jnp.zeros((1, int(state_dim_effective)), dtype=jnp.float32)
        init_critic_truth = jnp.zeros((1, int(critic_target_dim)), dtype=jnp.float32)
        merge_critic_params = merge_critic_net.init(
            merge_critic_key, init_critic_state, init_critic_truth
        )
        if use_nasss_jsd_merge:
            slice_key = jr.PRNGKey(int(config.seed) + 919)
            n_merge_slices = int(config.merge_nasss_n_slices)
            raw = jr.normal(slice_key, (int(theta_dim), n_merge_slices))
            # Unit-norm columns: each slice is a unit-vector projection.
            norms = jnp.linalg.norm(raw, axis=0, keepdims=True) + 1e-8
            merge_nasss_slice_matrix = raw / norms

    needs_training = (
        (not summary_is_identity and not summary_is_affine)
        or architecture_uses_learned_merge
        or architecture_uses_learned_decoder
        or explicit_state_decoder
        or count_only
        or use_jsd_merge
    )

    params: Any = {"summary": summary_params}
    if architecture_uses_learned_merge:
        params["merge"] = merge_params
    if explicit_state_decoder:
        params["state_decoder"] = state_decoder_params
    if architecture_uses_learned_decoder:
        params["decoder"] = decoder_params
    if count_only:
        params["count_readout"] = count_readout_params
    if use_jsd_merge:
        params["merge_critic"] = merge_critic_params

    def _readout_count(params_obj, states):
        """Apply the learned count_readout to (B, state_dim_effective)."""
        return count_readout_net.apply(params_obj["count_readout"], states)

    if needs_training:
        n_train_total = int(n_train_items)
        bs_total = max(1, int(config.batch_size))
        steps_per_epoch = max(1, (n_train_total + bs_total - 1) // bs_total)
        total_steps = max(1, int(config.n_iter) * steps_per_epoch)
        schedule_kind = str(config.lr_schedule)
        if schedule_kind == "constant":
            lr_obj: Any = float(config.learning_rate)
        elif schedule_kind == "cosine":
            lr_obj = optax.cosine_decay_schedule(
                init_value=float(config.learning_rate),
                decay_steps=int(total_steps),
            )
        else:
            raise ValueError(f"lr_schedule must be 'constant' or 'cosine'; got {schedule_kind!r}")
        optimizer = optax.adam(lr_obj)
        opt_state = optimizer.init(params)

    def _state_from_features(params_obj, features):
        if summary_is_identity:
            return features
        summary_params_obj = params_obj["summary"]
        if summary_is_affine:
            raw = features @ summary_params_obj["w"] + summary_params_obj["b"]
            if is_markov_law:
                return _markov_canonical_project_jnp(
                    deps,
                    raw,
                    n_regimes=int(n_regimes or 0),
                    target_scale=float(train.target_scale),
                )
            return _hll_canonical_project_jnp(
                deps,
                raw,
                precision=int(hll_precision),
                hash_bits=int(hll_hash_bits),
            )
        assert summary_net is not None
        if package_aux_active:
            dummy_responses = jnp.zeros(
                (int(features.shape[0]), int(train_response_dim)), dtype=jnp.float32
            )
            states, _slice_pred, _critic = summary_net.apply(
                summary_params_obj, features, dummy_responses
            )
            return states
        return summary_net.apply(summary_params_obj, features)

    def _decode_state(params_obj, summaries):
        if explicit_state_decoder:
            assert state_decoder_net is not None
            return state_decoder_net.apply(params_obj["state_decoder"], summaries)
        return summaries

    def _state_and_package_aux(params_obj, features, responses):
        if package_aux_active and not summary_is_identity and not summary_is_affine:
            assert summary_net is not None
            return summary_net.apply(params_obj["summary"], features, responses)
        states = _state_from_features(params_obj, features)
        slice_pred = jnp.zeros((int(states.shape[0]), int(n_slices)), dtype=jnp.float32)
        critic = jnp.zeros((int(states.shape[0]),), dtype=jnp.float32)
        return states, slice_pred, critic

    def _merge_states(params_obj, left_states, right_states):
        if architecture_uses_learned_merge:
            assert merge_net is not None
            return merge_net.apply(params_obj["merge"], left_states, right_states)
        if is_markov_law:
            return _markov_exact_merge_states_jnp(
                deps,
                left_states,
                right_states,
                n_regimes=int(n_regimes or 0),
                target_scale=float(train.target_scale),
            )
        return _hll_exact_merge_states_jnp(deps, left_states, right_states)

    def _decode_response(params_obj, states):
        if architecture_uses_learned_decoder:
            assert decoder_net is not None
            return decoder_net.apply(params_obj["decoder"], states)
        if is_markov_law:
            return _responses_from_markov_exact_states(states, decoder=decoder)
        return _responses_from_hll_exact_states(states, decoder=decoder, deps=deps)

    def _weighted_mean(row_losses, weights):
        denom = jnp.maximum(jnp.sum(weights), 1.0)
        return jnp.sum(row_losses * weights) / denom

    def _hll_estimate_norm(states):
        return _hll_estimate_from_normalized_registers_jnp(
            deps,
            states,
            precision=int(hll_precision),
            hash_bits=int(hll_hash_bits),
        ) / float(train.target_scale)

    def loss_parts(
        params_obj,
        features,
        theta_target,
        responses,
        left_features,
        right_features,
        merge_target,
        merge_full_features,
        leaf_weights,
        merge_weights,
        idemp_weights,
    ):
        summaries, slice_pred, critic_pos = _state_and_package_aux(
            params_obj,
            features,
            responses,
        )
        theta_pred = _decode_state(params_obj, summaries)
        response_pred = _decode_response(params_obj, theta_pred)
        if count_only:
            # C1 leaf supervises only count: count_readout(rep) ≈ count_truth.
            # The rep retains arbitrary dim ``state_dim_effective`` and is
            # otherwise unsupervised (first/last become emergent, not
            # training targets). Idempotence becomes a soft "round-then-
            # distance" on the predicted normalized count.
            count_pred = _readout_count(params_obj, summaries)
            count_target = theta_target[:, 0]
            leaf_rows = (count_pred - count_target) ** 2
            idemp_rows = (
                count_pred
                - jax.lax.stop_gradient(
                    jnp.round(count_pred * float(train.target_scale)) / float(train.target_scale)
                )
            ) ** 2
        else:
            leaf_rows = jnp.mean((theta_pred - theta_target) ** 2, axis=1)
            if is_hll_law:
                projected_states = _hll_canonical_project_jnp(
                    deps,
                    theta_pred,
                    precision=int(hll_precision),
                    hash_bits=int(hll_hash_bits),
                )
                idemp_rows = jnp.mean(
                    (theta_pred - jax.lax.stop_gradient(projected_states)) ** 2,
                    axis=1,
                )
            else:
                idemp_rows = leaf_rows
        leaf_loss = _weighted_mean(leaf_rows, leaf_weights)
        idemp_loss = _weighted_mean(idemp_rows, idemp_weights)
        if hll_estimate_active:
            hll_leaf_estimate_rows = (
                _hll_estimate_norm(theta_pred)
                - jax.lax.stop_gradient(_hll_estimate_norm(theta_target))
            ) ** 2
            hll_leaf_estimate_loss = _weighted_mean(hll_leaf_estimate_rows, leaf_weights)
        else:
            hll_leaf_estimate_loss = jnp.asarray(0.0, dtype=jnp.float32)
        hll_merge_estimate_loss = jnp.asarray(0.0, dtype=jnp.float32)
        if int(merge_target.shape[0]) > 0:
            left_summaries = _state_from_features(params_obj, left_features)
            right_summaries = _state_from_features(params_obj, right_features)
            merge_summaries = _merge_states(params_obj, left_summaries, right_summaries)
            if count_only:
                # C2 merge: count_readout of the composed rep should match
                # the merge-node count truth. ``c2_merge_target='theta'``
                # uses ground-truth merge counts; ``self_consistency`` uses
                # stop-gradient(count_readout(state_from(merge_full_features)))
                # i.e., the predicted count of the full-merge-node features
                # treated as a target.
                merge_count_pred = _readout_count(params_obj, merge_summaries)
                if c2_merge_target_kind == "self_consistency":
                    full_summaries = _state_from_features(params_obj, merge_full_features)
                    effective_count_target = jax.lax.stop_gradient(
                        _readout_count(params_obj, full_summaries)
                    )
                else:
                    effective_count_target = merge_target[:, 0]
                merge_rows = (merge_count_pred - effective_count_target) ** 2
            else:
                merge_theta_pred = _decode_state(params_obj, merge_summaries)
                if c2_merge_target_kind == "self_consistency":
                    full_summaries = _state_from_features(params_obj, merge_full_features)
                    effective_merge_target = jax.lax.stop_gradient(
                        _decode_state(params_obj, full_summaries)
                    )
                else:
                    effective_merge_target = merge_target
                if hll_estimate_active:
                    hll_merge_estimate_rows = (
                        _hll_estimate_norm(merge_theta_pred)
                        - jax.lax.stop_gradient(_hll_estimate_norm(effective_merge_target))
                    ) ** 2
                    hll_merge_estimate_loss = _weighted_mean(
                        hll_merge_estimate_rows,
                        merge_weights,
                    )
                if use_nass_jsd_merge:
                    # NASS-style JSD contrastive C2: full-vector critic on
                    # (merge_state, merge_target) with rolled-batch negatives.
                    f_pos = merge_critic_net.apply(
                        params_obj["merge_critic"],
                        merge_summaries,
                        effective_merge_target,
                    )
                    neg_target = jnp.roll(effective_merge_target, shift=1, axis=0)
                    f_neg = merge_critic_net.apply(
                        params_obj["merge_critic"],
                        merge_summaries,
                        neg_target,
                    )
                    mi_lb = (-jax.nn.softplus(-f_pos)).mean() - jax.nn.softplus(f_neg).mean()
                    merge_rows = jnp.broadcast_to(-mi_lb, merge_weights.shape)
                elif use_nasss_jsd_merge:
                    # NASSS-style sliced JSD: project merge_target onto
                    # ``n_slices`` random unit vectors, run JSD per slice
                    # against rolled negatives, average. Critic input is
                    # (merge_state, single_slice_scalar_as_1vec).
                    target_slices = (
                        effective_merge_target @ merge_nasss_slice_matrix
                    )  # (B, n_slices)
                    neg_slices = jnp.roll(target_slices, shift=1, axis=0)

                    def _slice_mi(slice_pos, slice_neg):
                        f_pos_s = merge_critic_net.apply(
                            params_obj["merge_critic"],
                            merge_summaries,
                            slice_pos[:, None],
                        )
                        f_neg_s = merge_critic_net.apply(
                            params_obj["merge_critic"],
                            merge_summaries,
                            slice_neg[:, None],
                        )
                        return (-jax.nn.softplus(-f_pos_s)).mean() - jax.nn.softplus(f_neg_s).mean()

                    mi_lb_per_slice = jax.vmap(_slice_mi, in_axes=(1, 1))(target_slices, neg_slices)
                    mi_lb = jnp.mean(mi_lb_per_slice)
                    merge_rows = jnp.broadcast_to(-mi_lb, merge_weights.shape)
                else:
                    merge_rows = jnp.mean(
                        (merge_theta_pred - effective_merge_target) ** 2,
                        axis=1,
                    )
            merge_loss = _weighted_mean(merge_rows, merge_weights)
        else:
            merge_loss = jnp.asarray(0.0, dtype=jnp.float32)
        contextual_loss = jnp.mean((response_pred - responses) ** 2)
        hll_estimate_loss = hll_leaf_estimate_loss + hll_merge_estimate_loss
        local_law_loss = (
            float(config.local_law_leaf_weight) * leaf_loss
            + float(config.local_law_merge_weight) * merge_loss
            + float(config.local_law_idempotence_weight) * idemp_loss
            + hll_estimate_weight * hll_estimate_loss
        )
        if package_aux_active and not summary_is_identity and not summary_is_affine:
            if method == "nasss":
                target_slices = responses @ package_slice_matrix
                package_loss = jnp.mean((slice_pred - target_slices) ** 2)
            else:
                neg_responses = jnp.roll(responses, shift=1, axis=0)
                _neg_states, _neg_slice_pred, critic_neg = _state_and_package_aux(
                    params_obj, features, neg_responses
                )
                package_loss = (
                    jax.nn.softplus(-critic_pos).mean() + jax.nn.softplus(critic_neg).mean()
                )
        else:
            package_loss = jnp.asarray(0.0, dtype=jnp.float32)
        total = (
            float(config.local_law_weight) * local_law_loss
            + float(config.local_law_contextual_weight) * contextual_loss
            + package_aux_weight * package_loss
        )
        return total, (
            leaf_loss,
            merge_loss,
            idemp_loss,
            contextual_loss,
            local_law_loss,
            package_loss,
            hll_estimate_loss,
        )

    if needs_training:
        assert optimizer is not None

        # Step takes integer indices and gathers train tensors INSIDE the
        # JIT graph (not 10 separate kernel launches outside it). Also
        # accumulates per-step metrics on device into ``accum`` to avoid
        # one host sync per step. ``accum`` is a length-8 vector
        # (loss, leaf, merge, idemp, context, local_law, package, hll_estimate).
        @jax.jit
        def step(
            params_obj,
            opt_state_obj,
            idx,
            merge_idx,
            accum,
        ):
            if lazy_train_item_features:
                features = _encode_tokens(train_item_tokens[idx])
            else:
                assert train_features is not None
                features = train_features[idx]
            theta_target = train_theta[idx]
            responses_b = train_responses[idx]
            leaf_weights_b = train_leaf_weights[idx]
            idemp_weights_b = train_idemp_weights[idx]
            if lazy_train_merge_features:
                assert train_left_tokens is not None
                assert train_right_tokens is not None
                left_features_b = _encode_tokens(train_left_tokens[merge_idx])
                right_features_b = _encode_tokens(train_right_tokens[merge_idx])
                merge_full_b = _encode_tokens(train_item_tokens[train_merge_source_idx[merge_idx]])
            else:
                assert train_left_features is not None
                assert train_right_features is not None
                assert train_merge_full_features is not None
                left_features_b = train_left_features[merge_idx]
                right_features_b = train_right_features[merge_idx]
                merge_full_b = train_merge_full_features[merge_idx]
            merge_target_b = train_merge_target[merge_idx]
            merge_weights_b = train_merge_weights[merge_idx]
            (loss, aux), grads = jax.value_and_grad(loss_parts, has_aux=True)(
                params_obj,
                features,
                theta_target,
                responses_b,
                left_features_b,
                right_features_b,
                merge_target_b,
                merge_full_b,
                leaf_weights_b,
                merge_weights_b,
                idemp_weights_b,
            )
            updates, opt_state_new = optimizer.update(grads, opt_state_obj, params_obj)
            params_new = optax.apply_updates(params_obj, updates)
            step_metrics = jnp.stack(
                [
                    loss,
                    aux[0],
                    aux[1],
                    aux[2],
                    aux[3],
                    aux[4],
                    aux[5],
                    aux[6],
                ]
            )
            return params_new, opt_state_new, accum + step_metrics

    @jax.jit
    def eval_loss(
        params_obj,
        features,
        theta_target,
        responses,
        left_features,
        right_features,
        merge_target,
        merge_full_features,
        leaf_weights,
        merge_weights,
        idemp_weights,
    ):
        return loss_parts(
            params_obj,
            features,
            theta_target,
            responses,
            left_features,
            right_features,
            merge_target,
            merge_full_features,
            leaf_weights,
            merge_weights,
            idemp_weights,
        )

    history: list[dict[str, float | int]] = []
    n_train = int(n_train_items)
    batch_size = max(1, int(config.batch_size))
    np_rng = np.random.default_rng(int(config.seed) + 823)
    if (not needs_training) or int(config.n_iter) <= 0:
        if lazy_train_item_features:
            raise ValueError(
                "n_iter <= 0 with lazy HLL one-hot train features is not supported; "
                "run at least one training iteration or use normalized_token_ids"
            )
        assert train_features is not None
        assert train_left_features is not None
        assert train_right_features is not None
        assert train_merge_full_features is not None
        train_loss, train_aux = eval_loss(
            params,
            train_features,
            train_theta,
            train_responses,
            train_left_features,
            train_right_features,
            train_merge_target,
            train_merge_full_features,
            train_leaf_weights,
            train_merge_weights,
            train_idemp_weights,
        )
        val_loss, val_aux = eval_loss(
            params,
            val_features,
            val_theta,
            val_responses,
            val_left_features,
            val_right_features,
            val_merge_target,
            val_merge_full_features,
            val_leaf_weights,
            val_merge_weights,
            val_idemp_weights,
        )
        history.append(
            {
                "iteration": 0,
                "trainer": "learned_local_laws",
                "train_loss": float(train_loss),
                "train_l1_leaf_mse": float(train_aux[0]),
                "train_l2_merge_mse": float(train_aux[1]),
                "train_l3_idempotence_mse": float(train_aux[2]),
                "train_contextual_mse": float(train_aux[3]),
                "train_local_law_loss": float(train_aux[4]),
                "train_package_loss": float(train_aux[5]),
                "train_hll_estimate_mse": float(train_aux[6]),
                "val_loss": float(val_loss),
                "val_l1_leaf_mse": float(val_aux[0]),
                "val_l2_merge_mse": float(val_aux[1]),
                "val_l3_idempotence_mse": float(val_aux[2]),
                "val_contextual_mse": float(val_aux[3]),
                "val_local_law_loss": float(val_aux[4]),
                "val_package_loss": float(val_aux[5]),
                "val_hll_estimate_mse": float(val_aux[6]),
            }
        )
    else:
        assert opt_state is not None
        best_val_law_loss = float("inf")
        best_params = params
        best_iteration = 0
        # Per-step strategy:
        #   1. fixed-shape ``idx`` and ``merge_idx`` so JIT compiles once
        #   2. gathers happen INSIDE the JIT (10 fewer kernel launches/step)
        #   3. metrics accumulate on device — one sync per epoch, not per step
        # Drop trailing partial batches (n_train chosen so this rarely
        # matters; 1024/10240/102400 all divide evenly by batch_size=128).
        n_train_merges = int(train_merge_target.shape[0])
        merge_batch_size = min(batch_size, n_train_merges) if n_train_merges > 0 else 0
        full_batches = n_train // batch_size
        if full_batches < 1:
            full_batches = 1
        usable_train = full_batches * batch_size
        for iteration in range(int(config.n_iter)):
            order = np_rng.permutation(n_train)[:usable_train]
            # Pre-compute the epoch's index batches and (if applicable)
            # merge-index batches as fixed-shape JAX arrays once.
            idx_epoch = jnp.asarray(order.reshape(full_batches, batch_size), dtype=jnp.int32)
            if merge_batch_size > 0:
                merge_idx_np = np_rng.integers(
                    0,
                    n_train_merges,
                    size=(full_batches, merge_batch_size),
                )
                merge_idx_epoch = jnp.asarray(merge_idx_np, dtype=jnp.int32)
            else:
                merge_idx_epoch = jnp.zeros((full_batches, 0), dtype=jnp.int32)
            accum = jnp.zeros((8,), dtype=jnp.float32)
            for batch_i in range(full_batches):
                params, opt_state, accum = step(
                    params,
                    opt_state,
                    idx_epoch[batch_i],
                    merge_idx_epoch[batch_i],
                    accum,
                )
            # Single sync per epoch: pull accum to host once.
            accum_np = np.asarray(accum) / float(full_batches)
            (
                train_loss_mean,
                train_leaf_mean,
                train_merge_mean,
                train_idemp_mean,
                train_context_mean,
                train_local_law_mean,
                train_package_mean,
                train_hll_estimate_mean,
            ) = (float(v) for v in accum_np)
            val_loss, val_aux = eval_loss(
                params,
                val_features,
                val_theta,
                val_responses,
                val_left_features,
                val_right_features,
                val_merge_target,
                val_merge_full_features,
                val_leaf_weights,
                val_merge_weights,
                val_idemp_weights,
            )
            val_law_score = (
                float(val_aux[0]) + float(val_aux[1]) + float(val_aux[2]) + float(val_aux[3])
            )
            if hll_estimate_active:
                val_law_score += hll_estimate_weight * float(val_aux[6])
            if val_law_score < best_val_law_loss:
                best_val_law_loss = val_law_score
                best_params = params
                best_iteration = int(iteration + 1)
            history.append(
                {
                    "iteration": int(iteration + 1),
                    "trainer": "learned_local_laws",
                    "train_loss": train_loss_mean,
                    "train_l1_leaf_mse": train_leaf_mean,
                    "train_l2_merge_mse": train_merge_mean,
                    "train_l3_idempotence_mse": train_idemp_mean,
                    "train_contextual_mse": train_context_mean,
                    "train_local_law_loss": train_local_law_mean,
                    "train_package_loss": train_package_mean,
                    "train_hll_estimate_mse": train_hll_estimate_mean,
                    "val_loss": float(val_loss),
                    "val_l1_leaf_mse": float(val_aux[0]),
                    "val_l2_merge_mse": float(val_aux[1]),
                    "val_l3_idempotence_mse": float(val_aux[2]),
                    "val_contextual_mse": float(val_aux[3]),
                    "val_local_law_loss": float(val_aux[4]),
                    "val_package_loss": float(val_aux[5]),
                    "val_hll_estimate_mse": float(val_aux[6]),
                }
            )
        params = best_params
        history.append(
            {
                "iteration": -1,
                "trainer": "learned_local_laws",
                "best_iteration": int(best_iteration),
                "best_val_law_score": float(best_val_law_loss),
                "note": "params_returned=best_by_val_law_score",
            }
        )

    def _apply_decode(params_obj, states):
        if bool(params_obj.get("learned_decoder_active", False)):
            assert decoder_net is not None
            return decoder_net.apply(params_obj["decoder_params"], states)
        if str(params_obj.get("law_package", "markov_exact_sketch")) == "hll_register_sketch":
            return _responses_from_hll_exact_states(
                states,
                decoder=params_obj["decoder"],
                deps=deps,
            )
        return _responses_from_markov_exact_states(states, decoder=params_obj["decoder"])

    def _apply_state_decode(params_obj, summaries):
        if bool(params_obj.get("explicit_state_decoder", False)):
            assert state_decoder_net is not None
            return state_decoder_net.apply(params_obj["state_decoder_params"], summaries)
        return summaries

    def apply_learned_local_laws(params_obj, tokens, responses):
        encoded = _encode_tokens_for_sbijax_package(
            deps,
            tokens,
            vocab_size=int(config.vocab_size),
            input_encoding=str(config.input_encoding),
            block_by_token=params_obj["block_by_token"],
            n_regimes=(
                None if params_obj.get("n_regimes") is None else int(params_obj["n_regimes"])
            ),
            target_scale=float(params_obj["target_scale"]),
        )
        if bool(params_obj["identity_summary"]):
            summaries = encoded
        elif str(params_obj.get("summary_kind", "")) == "affine_probe":
            raw_states = encoded @ params_obj["summary"]["w"] + params_obj["summary"]["b"]
            if str(params_obj.get("law_package", "markov_exact_sketch")) == "hll_register_sketch":
                summaries = _hll_canonical_project_jnp(
                    deps,
                    raw_states,
                    precision=int(params_obj["hll_precision"]),
                    hash_bits=int(params_obj["hll_hash_bits"]),
                )
            else:
                summaries = _markov_canonical_project_jnp(
                    deps,
                    raw_states,
                    n_regimes=int(params_obj["n_regimes"]),
                    target_scale=float(params_obj["target_scale"]),
                )
        elif bool(params_obj.get("package_aux_active", False)):
            assert summary_net is not None
            summaries, slice_pred, critic = summary_net.apply(
                params_obj["summary"], encoded, responses
            )
            theta_states = _apply_state_decode(params_obj, summaries)
            response_pred = _apply_decode(params_obj, theta_states)
            rep_out = (
                summaries
                if bool(params_obj.get("explicit_state_decoder", False))
                else slice_pred
            )
            return theta_states, response_pred, rep_out, critic
        else:
            assert summary_net is not None
            summaries = summary_net.apply(params_obj["summary"], encoded)
        theta_states = _apply_state_decode(params_obj, summaries)
        response_pred = _apply_decode(params_obj, theta_states)
        slice_pred = jnp.zeros((int(theta_states.shape[0]), 0), dtype=response_pred.dtype)
        critic = jnp.zeros((int(theta_states.shape[0]),), dtype=response_pred.dtype)
        # Count-only mode: return ``count_pred`` (shape (B, 1)) as the
        # ``states`` output for diagnostics. Downstream first/last accuracy
        # is NOT reported (those slots aren't supervised; faking values
        # would be misleading). The actual learned rep travels via the
        # ``slice_pred`` slot for per-leaf inspection.
        if bool(params_obj.get("count_only", False)):
            assert count_readout_net is not None
            rep = summaries
            count_pred = count_readout_net.apply(params_obj["count_readout"], rep)
            return count_pred[:, None], response_pred, rep, critic
        if bool(params_obj.get("explicit_state_decoder", False)):
            return theta_states, response_pred, summaries, critic
        return theta_states, response_pred, slice_pred, critic

    result_params = {
        "summary": params["summary"],
        "identity_summary": bool(summary_is_identity),
        "summary_kind": (
            "identity_exact_input"
            if summary_is_identity
            else (
                "affine_probe"
                if summary_is_affine
                else (
                    "regime_transition_sum"
                    if summary_is_regime_transition_sum
                    else "jax_fno" if summary_is_jax_fno else "haiku_mlp"
                )
            )
        ),
        "block_by_token": block_by_token,
        "n_regimes": None if n_regimes is None else int(n_regimes),
        "law_package": law_package,
        "package_theta": law_package,
        "law_set_id": law_set_id,
        "hll_precision": int(hll_precision),
        "hll_hash_bits": int(hll_hash_bits),
        "hll_max_register": int(hll_max_register),
        "target_scale": float(train.target_scale),
        "decoder": decoder,
        "local_law_observation_metadata": train_law_rows.to_metadata(),
        "package_aux_active": bool(
            package_aux_active and not summary_is_identity and not summary_is_affine
        ),
        "law_architecture": law_architecture,
        "learned_merge_active": bool(architecture_uses_learned_merge),
        "learned_decoder_active": bool(architecture_uses_learned_decoder),
        "explicit_state_decoder": bool(explicit_state_decoder),
        "state_decoder_params": params.get("state_decoder"),
        "state_decoder_head": str(config.local_law_state_decoder_head),
        "summary_dim_effective": int(state_dim_effective),
        "theta_dim": int(theta_dim),
        "merge_params": params.get("merge"),
        "decoder_params": params.get("decoder"),
        "count_only": bool(count_only),
        "count_readout": params.get("count_readout"),
        "local_law_hll_estimate_weight": float(hll_estimate_weight),
    }
    train_diag = evaluate_contextual_sbijax(
        params=result_params,
        apply_fn=apply_learned_local_laws,
        dataset=train,
    )
    val_diag = evaluate_contextual_sbijax(
        params=result_params,
        apply_fn=apply_learned_local_laws,
        dataset=val,
    )
    train_diag["local_law_observation_metadata"] = train_law_rows.to_metadata()
    val_diag["local_law_observation_metadata"] = val_law_rows.to_metadata()
    if train_sparse_law_rows is not None and val_sparse_law_rows is not None:
        train_diag["local_law_sparse_ipw_observation_metadata"] = (
            train_sparse_law_rows.to_metadata()
        )
        val_diag["local_law_sparse_ipw_observation_metadata"] = val_sparse_law_rows.to_metadata()
    provenance = contextual_sbijax_provenance(
        method=method,
        response_signature_contexts=int(config.response_signature_contexts),
        response_signature_slices=int(n_slices),
        trainer="learned_local_laws",
        input_encoding=str(config.input_encoding),
        downstream_readout=(
            "learned_decoder_mlp"
            if architecture_uses_learned_decoder
            else (
                "deterministic_markov_exact_sketch"
                if is_markov_law
                else "deterministic_hll_register_sketch"
            )
        ),
    )
    provenance.update(
        {
            "summary_network": (
                "identity_exact_input"
                if summary_is_identity
                else (
                    "least_squares_affine_local_law"
                    if summary_is_affine
                    else (
                        "haiku_regime_transition_sum_local_law_theta"
                        if summary_is_regime_transition_sum
                        else (
                            "internal_jax_fno_local_law_theta"
                            if summary_is_jax_fno
                            else (
                                "haiku_mlp_local_law_theta_plus_package_aux"
                                if package_aux_active
                                else "haiku_mlp_local_law_theta"
                            )
                        )
                    )
                )
            ),
            "merge_network": (
                (
                    "learned_fno_merge"
                    if str(config.merge_family) == "fno_rep"
                    else "learned_asymmetric_mlp"
                )
                if architecture_uses_learned_merge
                else (
                    "differentiable_exact_markov_merge"
                    if is_markov_law
                    else "differentiable_exact_hll_max_merge"
                )
            ),
            "merge_family": str(config.merge_family),
            "merge_fno_n_modes": int(config.merge_fno_n_modes),
            "merge_fno_n_layers": int(config.merge_fno_n_layers),
            "merge_fno_hidden_channels": int(config.merge_fno_hidden_channels),
            "decoder_head": str(config.decoder_head),
            "local_law_merge_loss": str(config.local_law_merge_loss),
            "merge_nasss_n_slices": int(config.merge_nasss_n_slices),
            "package_theta": str(config.package_theta),
            "package_theta_dim": int(theta_dim),
            "paper_notation_factorization": (
                "explicit_g_summary_then_f_state_decoder"
                if explicit_state_decoder
                else "fused_theta_state"
            ),
            "g_summary_dim": int(state_dim_effective),
            "f_state_decoder_kind": (
                f"explicit_{str(config.local_law_state_decoder_head)}"
                if explicit_state_decoder
                else "identity_fused_theta"
            ),
            "affine_probe_response_dim": int(train_response_dim),
            "baseline_role": "local_law_learned",
            "decoder_kind": ("learned_mlp" if architecture_uses_learned_decoder else "exact"),
            "response_decoder_kind": (
                "learned_mlp" if architecture_uses_learned_decoder else "exact"
            ),
            "exact_zero_claim": False,
            "law_set_id": law_set_id,
            "hll_precision": int(hll_precision) if is_hll_law else None,
            "hll_hash_bits": int(hll_hash_bits) if is_hll_law else None,
            "hll_register_count": int(theta_dim) if is_hll_law else None,
            "local_law_supervision_mode": mode,
            "local_law_leaf_rate": float(config.local_law_leaf_rate),
            "local_law_merge_rate": float(config.local_law_merge_rate),
            "local_law_idempotence_rate": float(config.local_law_idempotence_rate),
            "local_law_summary_family": str(config.local_law_summary_family),
            "local_law_summary_family_canonical": str(summary_family_canonical),
            "local_law_summary_fno_n_modes": int(config.local_law_summary_fno_n_modes),
            "local_law_summary_fno_effective_n_modes": (
                int(
                    max(
                        1,
                        min(
                            int(config.local_law_summary_fno_n_modes),
                            fno_summary_fragment_len // 2 + 1,
                        ),
                    )
                )
                if summary_is_jax_fno
                else None
            ),
            "local_law_summary_fno_n_layers": int(config.local_law_summary_fno_n_layers),
            "local_law_summary_fno_pooling_mode": str(config.local_law_summary_fno_pooling_mode),
            "law_architecture": law_architecture,
            "c2_merge_target": c2_merge_target_kind,
            "learned_merge_hidden_dim": (
                int(learned_merge_hidden_dim) if architecture_uses_learned_merge else 0
            ),
            "learned_decoder_hidden_dim": (
                int(learned_decoder_hidden_dim) if architecture_uses_learned_decoder else 0
            ),
            "local_law_package_weight": float(config.local_law_package_weight),
            "local_law_package_objective": str(config.method),
            "local_law_package_aux_active": bool(
                package_aux_active and not summary_is_identity and not summary_is_affine
            ),
            "local_law_explicit_state_decoder": bool(explicit_state_decoder),
            "local_law_summary_dim": int(state_dim_effective),
            "local_law_state_decoder_head": str(config.local_law_state_decoder_head),
            "local_law_hll_estimate_weight": float(hll_estimate_weight),
            "local_law_hll_estimate_objective": (
                "normalized_hll_formula_leaf_plus_merge_mse" if is_hll_law else None
            ),
            "train_local_law_observation_metadata": train_law_rows.to_metadata(),
            "val_local_law_observation_metadata": val_law_rows.to_metadata(),
        }
    )
    if train_sparse_law_rows is not None and val_sparse_law_rows is not None:
        provenance.update(
            {
                "train_local_law_sparse_ipw_observation_metadata": (
                    train_sparse_law_rows.to_metadata()
                ),
                "val_local_law_sparse_ipw_observation_metadata": (
                    val_sparse_law_rows.to_metadata()
                ),
            }
        )
    return ContextualSBIJAXResult(
        params=result_params,
        history=history,
        train_diagnostics=train_diag,
        val_diagnostics=val_diag,
        provenance=provenance,
        config=config,
        slice_matrix=None,
        apply_fn=apply_learned_local_laws,
    )


def fit_contextual_sbijax(
    train: ContextualResponseDataset,
    val: ContextualResponseDataset,
    *,
    config: ContextualSBIJAXConfig,
) -> ContextualSBIJAXResult:
    """Fit a JAX contextual-sufficiency summary/readout model."""

    trainer = str(config.trainer)
    if trainer == "repo":
        return _fit_contextual_sbijax_repo(train, val, config=config)
    if trainer == "package":
        return fit_contextual_sbijax_package_direct(train, val, config=config)
    if trainer == "posterior":
        return fit_contextual_sbijax_posterior_direct(train, val, config=config)
    if trainer == "npe":
        return fit_contextual_sbijax_npe_direct(train, val, config=config)
    if trainer == "nass_nle":
        return fit_contextual_sbijax_nass_nle(train, val, config=config)
    if trainer == "theta_supervised":
        return fit_contextual_sbijax_theta_supervised(train, val, config=config)
    if trainer == "identity_theta":
        return fit_contextual_sbijax_identity_theta(train, val, config=config)
    if trainer == "exact_zero_markov":
        return fit_contextual_sbijax_exact_zero_markov(train, val, config=config)
    if trainer == "learned_local_laws":
        return fit_contextual_sbijax_learned_local_laws(train, val, config=config)
    raise ValueError(
        "trainer must be 'repo', 'package', 'theta_supervised', "
        "'identity_theta', 'exact_zero_markov', 'learned_local_laws', "
        "'posterior', 'npe', or 'nass_nle'"
    )


def evaluate_contextual_sbijax(
    *,
    params: Any,
    apply_fn: Any,
    dataset: ContextualResponseDataset,
) -> dict[str, Any]:
    """Evaluate contextual response prediction diagnostics."""

    deps = _require_contextual_sbi()
    jnp = deps.jnp
    tokens = jnp.asarray(dataset.item_tokens, dtype=jnp.int32)
    responses = jnp.asarray(
        dataset.response_signatures.reshape(dataset.response_signatures.shape[0], -1),
        dtype=jnp.float32,
    )
    states, preds, slice_pred, critic = _apply_fn_chunked(apply_fn, params, tokens, responses)
    states_np = np.asarray(states)
    slice_pred_np = np.asarray(slice_pred)
    critic_np = np.asarray(critic)
    theta_std_np = (
        critic_np
        if critic_np.ndim == states_np.ndim and tuple(critic_np.shape) == tuple(states_np.shape)
        else None
    )
    diagnostics = _response_diagnostics(
        preds=np.asarray(preds),
        truths=np.asarray(
            dataset.response_signatures.reshape(dataset.response_signatures.shape[0], -1)
        ),
        states=states_np,
    )
    diagnostics["contextual_raw_mae"] = float(
        diagnostics["contextual_mae"] * float(dataset.target_scale)
    )
    targets = dict(dataset.package_theta_targets or {})
    package_theta_for_diag = "markov_exact_sketch"
    if isinstance(params, Mapping) and str(params.get("package_theta", "")):
        package_theta_for_diag = str(params["package_theta"])
    elif "markov_exact_sketch" not in targets and "hll_register_sketch" in targets:
        package_theta_for_diag = "hll_register_sketch"
    diagnostics.update(
        _theta_prediction_diagnostics(
            dataset,
            theta_pred=states_np,
            theta_std=theta_std_np,
            package_theta=package_theta_for_diag,
        )
    )
    diagnostics.update(
        {
            "problem_id": str(dataset.metadata.get("problem_id", "")),
            "context_kind": str(dataset.metadata.get("context_kind", "")),
        }
    )
    # Sufficiency for the LEAF rep: sbijax-style JSD MI lower bound on
    # ``I(theta; rep)``, with floor (random-permuted rep) and ceiling
    # (analytic Markov sketch) references for context. This is the core
    # internal-to-sbijax sufficiency check for g.
    truth_sketch = targets.get("markov_exact_sketch")
    if truth_sketch is not None:
        # In count_only and explicit-f/g modes, the learned summary rep is
        # passed through the ``slice_pred`` slot; use that rather than the
        # decoded theta-shaped ``states`` when probing representation
        # sufficiency.
        leaf_rep_for_suff = (
            slice_pred_np
            if (
                slice_pred_np.ndim == 2
                and slice_pred_np.shape[0] == states_np.shape[0]
                and slice_pred_np.shape[1] >= 2
            )
            else states_np
        )
        truth_sketch_np = np.asarray(truth_sketch, dtype=np.float32)
        diagnostics.update(
            _sufficiency_diagnostics_for_rep(
                rep=leaf_rep_for_suff,
                theta_truth=truth_sketch_np,
                analytic_sketch=truth_sketch_np,
                deps=deps,
                label="leaf_rep",
            )
        )
        # Probe whether endpoint info (first, last regime) is encoded
        # ANYWHERE in the leaf rep — not just in the literal slot indices
        # the analytic decoder expects. Pure-sbijax cells have no reason
        # to put it in slots 1..2*n_regimes; the probe finds it if it's
        # there at all.
        n_regimes_meta = dataset.metadata.get("n_regimes")
        if n_regimes_meta is None and int(truth_sketch_np.shape[1]) >= 3:
            n_regimes_inferred = (int(truth_sketch_np.shape[1]) - 1) // 2
        else:
            n_regimes_inferred = int(n_regimes_meta) if n_regimes_meta is not None else 0
        if n_regimes_inferred > 1:
            diagnostics.update(
                _probe_endpoints_from_rep(
                    rep=leaf_rep_for_suff,
                    theta_truth=truth_sketch_np,
                    n_regimes=int(n_regimes_inferred),
                    deps=deps,
                    label="leaf_rep",
                )
            )
    truth_hll = targets.get("hll_register_sketch")
    if truth_hll is not None:
        truth_hll_np = np.asarray(truth_hll, dtype=np.float32)
        if states_np.ndim == 2 and int(states_np.shape[0]) == int(truth_hll_np.shape[0]):
            leaf_rep_for_hll_suff = (
                slice_pred_np
                if (
                    slice_pred_np.ndim == 2
                    and slice_pred_np.shape[0] == states_np.shape[0]
                    and slice_pred_np.shape[1] >= 2
                )
                else states_np
            )
            diagnostics.update(
                _sufficiency_diagnostics_for_rep(
                    rep=leaf_rep_for_hll_suff,
                    theta_truth=truth_hll_np,
                    analytic_sketch=truth_hll_np,
                    deps=deps,
                    label="leaf_rep",
                )
            )
    diagnostics.update(
        _markov_local_law_eval_metrics(
            params=params,
            apply_fn=apply_fn,
            dataset=dataset,
            deps=deps,
            states_np=states_np,
        )
    )
    diagnostics.update(
        _hll_local_law_eval_metrics(
            params=params,
            apply_fn=apply_fn,
            dataset=dataset,
            deps=deps,
            states_np=states_np,
        )
    )
    return diagnostics


def exact_root_witness_diagnostics(
    flat_docs: Sequence[Sequence[int]],
    *,
    block_by_token: Sequence[int],
    root_counts: Sequence[float] | None = None,
) -> dict[str, float | int]:
    """Deterministic exact-count witness for Markov docs."""

    preds = np.asarray(
        [exact_count_for_tokens(doc, block_by_token=block_by_token) for doc in flat_docs],
        dtype=np.float64,
    )
    if root_counts is None:
        truths = preds.copy()
    else:
        truths = np.asarray([float(x) for x in root_counts], dtype=np.float64)
        if truths.shape != preds.shape:
            raise ValueError(f"root_counts shape mismatch: {truths.shape} vs {preds.shape}")
    err = np.abs(preds - truths)
    return {
        "n": int(truths.size),
        "root_mae": float(np.mean(err)) if err.size else float("nan"),
        "max_abs_error": float(np.max(err)) if err.size else float("nan"),
        "truth_mean": float(np.mean(truths)) if truths.size else float("nan"),
        "truth_std": float(np.std(truths)) if truths.size else float("nan"),
    }


__all__ = [
    "CONTEXTUAL_SBI_INSTALL_MSG",
    "ContextualMarkovSplits",
    "ContextualQueryProblem",
    "ContextualResponseDataset",
    "ContextualSBIJAXConfig",
    "ContextualSBIJAXResult",
    "MarkovTwoSidedContext",
    "MarkovTwoSidedContextProblem",
    "build_contextual_query_dataset",
    "build_contextual_response_dataset",
    "contextual_sbijax_available",
    "contextual_sbijax_provenance",
    "evaluate_contextual_sbijax",
    "exact_count_for_tokens",
    "exact_root_witness_diagnostics",
    "fit_contextual_sbijax",
    "fit_contextual_sbijax_exact_zero_markov",
    "fit_contextual_sbijax_identity_theta",
    "fit_contextual_sbijax_learned_local_laws",
    "fit_contextual_sbijax_nass_nle",
    "fit_contextual_sbijax_npe_direct",
    "fit_contextual_sbijax_package_direct",
    "fit_contextual_sbijax_posterior_direct",
    "fit_contextual_sbijax_theta_supervised",
    "flatten_fno_count_docs",
    "hybrid_summary_diagnostics",
    "hybrid_summary_diagnostics_for_contextual_sbijax",
    "load_markov_contextual_splits",
    "load_markov_contextual_splits_from_bundle",
    "markov_exact_response_predictions_for_dataset",
    "markov_exact_sketch_oracle_diagnostics",
    "markov_exact_sketch_targets_for_dataset",
    "make_synthetic_markov_docs",
    "pad_fragment",
    "palette_block_map",
    "root_counts_from_fno_count_docs",
    "sample_token_fragment",
    "with_package_theta_target",
]
