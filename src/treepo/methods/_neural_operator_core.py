"""Built-in neural-operator families for :mod:`treepo.methods`.

``neural_operator`` is the generic family. It delegates named operator kinds
to ``neuralop.models`` when available and keeps ``operator_kind='conv1d'`` as
a tiny local baseline. The public ``family='fno'`` route is the concrete
Fourier neural-operator route over the same shared runtime. Use
``family='neural_operator'`` when selecting an operator kind explicitly, for
example ``operator_kind='fno'``.
Dataset-specific structure belongs in tree fixtures or registered downstream
families; this module only owns the generic neural-operator method surface.

This module is intentionally lean: it holds the family runtime and the two
build helpers, and delegates the messy work to focused siblings —

* ``_fno_config``: config dataclasses and coercion.
* ``_fno_neuralop``: torch/neuralop discovery.
* ``_fno_models``: the f/g torch model.
* ``_fno_encoding``: leaf extraction and embedding (data-prep).
* ``_fno_targets``: supervision target extraction (data-prep).
* ``_fno_transition``: numeric transition-state supervision (data-prep).
* ``_fno_statistic``: the composable-statistic adapter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from treepo.llm.embedding import EmbeddingClient, HashingEmbeddingClient
from treepo.methods._fno_config import (
    FNOFamilyConfig,
    NeuralOperatorFamilyConfig,
    _clamp,
    _coerce_config,
    _config_payload,
    _normalize_operator_kind,
    _tensor_payload,
)
from treepo.methods._fno_encoding import (
    _coerce_embedding,
    _encode_numeric_leaf_features,
    _leaf_texts,
    _leaf_token_groups,
    _tree_sequence_cache_key,
)
from treepo.methods._fno_models import _EmbeddingCoordinateFGModel, _TreeFGModel
from treepo.methods._fno_neuralop import _require_torch, _validate_operator_kind
from treepo.methods._fno_statistic import _NeuralOperatorStatistic
from treepo.methods._fno_targets import (
    _leaf_rollup_weights,
    _node_supervision_targets,
    _target_rows,
)
from treepo.methods._fno_transition import (
    _numeric_transition_law_rows,
    _numeric_transition_state_targets,
    _pairwise_merge_depths,
)
from treepo.local_law import LawKind
from treepo.objective import LOCAL_LAW_ESTIMATOR_ORACLE_STATE, ObjectiveSpec


class NeuralOperatorFamily:
    """Generic neural-operator ``FamilyRuntime`` for tree root-score prediction."""

    name = "neural_operator"
    artifact_kind = "treepo_neural_operator"
    config_cls = NeuralOperatorFamilyConfig

    def __init__(
        self,
        config: NeuralOperatorFamilyConfig | None = None,
        *,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self.config = config or self.config_cls()
        self.operator_kind = _normalize_operator_kind(self.config.operator_kind)
        _validate_operator_kind(self.operator_kind, family_name=self.name)
        if float(self.config.numeric_transition_state_weight) > 0.0 and (
            float(self.config.leaf_weight) > 0.0 or float(self.config.merge_weight) > 0.0
        ):
            raise ValueError(
                f"family={self.name!r} got both numeric_transition_state_weight > 0 "
                "and per-node supervision weights (leaf_weight/merge_weight); the "
                "two loss shapes are mutually exclusive — pick one"
            )
        root_readout = str(self.config.root_readout or "root_state").strip().lower()
        if root_readout not in {"root_state", "leaf_mean"}:
            raise ValueError(
                f"family={self.name!r} root_readout must be 'root_state' or "
                f"'leaf_mean', got {self.config.root_readout!r}"
            )
        self.config.root_readout = root_readout
        if self.config.rollup_weight_key and root_readout != "leaf_mean":
            raise ValueError(
                f"family={self.name!r} rollup_weight_key is only meaningful with "
                "root_readout='leaf_mean'"
            )
        self.embedding_client = embedding_client or HashingEmbeddingClient(
            dim=self.config.embedding_dim,
            salt=self.config.embedding_salt,
        )
        self._torch = _require_torch()
        self._torch.manual_seed(int(self.config.seed))
        self._device = self._torch.device(str(self.config.device))
        self._model = None
        self._output_dim: int | None = None
        self._target_center = None
        self._target_scale = None
        self._last_artifact: dict[str, Any] | None = None
        self._objective: ObjectiveSpec | None = None
        self._objective_law_shares: dict[str, float] = {}
        self._objective_gamma_depth: float = 1.0
        self._node_supervision_counts: dict[str, int] | None = None
        self._last_law_source: str | None = None
        # Encoding cache keyed by tree identity. Each entry pins the tree
        # objects it was built from, so the id()-based key stays valid for as
        # long as the entry lives; the cache is bounded, evicting oldest first.
        self._encoding_cache: dict[tuple[Any, ...], tuple[Any, Any, tuple[Any, ...]]] = {}
        self._encoding_cache_max_entries = 16

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        del g
        self._maybe_warmstart(f_init)
        return self._train_side(kind="f", traces=traces, iteration=iteration, output_dir=output_dir)

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        del f
        self._maybe_warmstart(g_init)
        return self._train_side(kind="g", traces=traces, iteration=iteration, output_dir=output_dir)

    def _maybe_warmstart(self, artifact: Any) -> None:
        """Load model weights from a prior artifact into a fresh family.

        Artifacts returned by ``_train_side`` carry a ``weights_path``; handing
        one back (``initial_artifacts`` in a spec, or ``f=``/``g=`` at scoring
        time) must reconstruct the trained model, not be silently ignored.
        A live in-run model always wins: within one alternating run the state
        already reflects every completed iteration.
        """

        if self._model is not None or not isinstance(artifact, Mapping):
            return
        path = artifact.get("weights_path")
        if not path:
            return
        weights_file = Path(str(path))
        if not weights_file.exists():
            raise ValueError(
                f"family={self.name!r} warmstart artifact points at missing "
                f"weights_path {str(weights_file)!r}"
            )
        artifact_operator = artifact.get("operator_kind")
        if artifact_operator is not None and str(artifact_operator) != self.operator_kind:
            raise ValueError(
                f"family={self.name!r} warmstart operator_kind mismatch: artifact "
                f"has {artifact_operator!r}, family is {self.operator_kind!r}"
            )
        self._ensure_model(output_dim=int(artifact.get("output_dim") or 1))
        assert self._model is not None
        # Artifacts persist only tensor entries (see _train_side), so the safe
        # weights-only loader applies; non-tensor module extras are rebuilt by
        # construction, hence strict=False.
        state = self._torch.load(weights_file, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state, strict=False)
        for attr, key in (("_target_center", "target_center"), ("_target_scale", "target_scale")):
            payload = artifact.get(key)
            if payload is not None:
                setattr(
                    self,
                    attr,
                    self._torch.tensor(payload, dtype=self._torch.float32, device=self._device),
                )

    def configure_objective(self, objective: ObjectiveSpec | None) -> None:
        """Adopt a resolved ``ObjectiveSpec`` as the executed training objective.

        With a law-bearing spec, every training step minimizes the convex
        combination ``root_share * root_mse + sum_c share_c * law_c``, where
        each law channel is the depth-discounted canonical objective from
        :mod:`treepo.training.local_law` over exact numeric transition-state
        rows. The supervision targets are exact, so the spec must declare
        ``local_law_estimator='oracle_state'``; the canonical corrected
        estimator then degenerates to the oracle term (observed, propensity 1).
        Mutually exclusive with the legacy additive
        ``numeric_transition_state_weight`` knob: a configured objective must
        fully describe the executed loss.
        """
        if objective is None:
            self._objective = None
            self._objective_law_shares = {}
            self._objective_gamma_depth = 1.0
            return
        if not isinstance(objective, ObjectiveSpec):
            raise TypeError(
                f"family={self.name!r} objective must be an ObjectiveSpec; "
                f"got {type(objective).__name__}"
            )
        if float(self.config.numeric_transition_state_weight) > 0.0:
            raise ValueError(
                f"family={self.name!r} got both an objective spec and "
                "numeric_transition_state_weight > 0; the objective is the "
                "single weight source — drop the legacy knob"
            )
        if float(self.config.leaf_weight) > 0.0 or float(self.config.merge_weight) > 0.0:
            raise ValueError(
                f"family={self.name!r} got both an objective spec and per-node "
                "supervision weights (leaf_weight/merge_weight); the objective "
                "is the single weight source — express node supervision through "
                "its local-law component weights (C1 = leaves, C3 = merges)"
            )
        shares: dict[str, float] = {}
        gamma = 1.0
        law_weight = float(objective.local_law_weight or 0.0)
        if law_weight > 0.0:
            if str(objective.local_law_estimator) != LOCAL_LAW_ESTIMATOR_ORACLE_STATE:
                raise ValueError(
                    f"family={self.name!r} trains laws against exact numeric "
                    "transition states; declare local_law_estimator="
                    f"'oracle_state' (got {objective.local_law_estimator!r})"
                )
            weights = dict(objective.local_law_component_weights or {})
            if float(weights.get(LawKind.C2_IDEMPOTENCE.value, 0.0)) > 0.0:
                raise ValueError(
                    f"family={self.name!r} has no C2 (on-range idempotence) law "
                    "surface; set its component weight to 0"
                )
            total = float(sum(float(v) for v in weights.values()))
            shares = {
                str(name): law_weight * float(value) / total
                for name, value in weights.items()
                if float(value) > 0.0
            }
            gamma = float(objective.gamma_depth)
        self._objective = objective
        self._objective_law_shares = shares
        self._objective_gamma_depth = gamma

    def _train_side(
        self,
        *,
        kind: str,
        traces: Sequence[Any],
        iteration: int,
        output_dir: Path | None = None,
    ) -> Mapping[str, Any]:
        train_g = kind == "g"
        trees, targets = _target_rows(traces, self.config)
        x, lengths = self._encode_trees(trees)
        y = self._torch.tensor(targets, dtype=self._torch.float32, device=self._device)
        self._ensure_model(output_dim=int(y.shape[1]))
        y_train = self._normalized_targets(y)
        node_supervision = self._prepare_node_supervision(trees, width=int(y.shape[1]))
        rollup_weights = self._rollup_weights_tensor(trees, max_leaves=int(x.shape[1]))
        last_loss = self._train_supervised(
            x,
            lengths,
            y_train,
            train_f=not train_g,
            train_g=train_g,
            trees=trees if (train_g or self._objective_law_shares) else None,
            node_supervision=node_supervision,
            rollup_weights=rollup_weights,
        )
        weights_path: Path | None = None
        if output_dir is not None and self._model is not None:
            out = Path(output_dir)
            out.mkdir(parents=True, exist_ok=True)
            weights_path = out / f"{kind}_weights_iter{int(iteration):02d}.pt"
            # Persist tensors only (parameters/buffers): keeps the file loadable
            # under torch's safe weights-only unpickler.
            tensor_state = {
                key: value
                for key, value in self._model.state_dict().items()
                if self._torch.is_tensor(value)
            }
            self._torch.save(tensor_state, weights_path)
        artifact = self._artifact_payload(
            kind=kind,
            iteration=iteration,
            n_train=int(x.shape[0]),
            loss=last_loss,
            weights_path=weights_path,
        )
        self._last_artifact = artifact
        return artifact

    def score_roots_with_f(
        self,
        *,
        f: Any,
        g: Any,
        trees: Sequence[Any],
    ) -> list[Any | None]:
        # f and g artifacts each snapshot the whole shared model at their own
        # training iteration; scoring must resume from the newest snapshot.
        candidates = [
            artifact
            for artifact in (f, g)
            if isinstance(artifact, Mapping) and artifact.get("weights_path")
        ]
        if candidates:
            self._maybe_warmstart(
                max(candidates, key=lambda artifact: int(artifact.get("iteration") or 0))
            )
        tree_list = list(trees or [])
        if not tree_list:
            return []
        if self._model is None:
            return [None] * len(tree_list)
        self._ensure_model(output_dim=self._output_dim)
        assert self._model is not None
        self._model.eval()
        x, lengths = self._encode_trees(tree_list)
        rollup_weights = self._rollup_weights_tensor(tree_list, max_leaves=int(x.shape[1]))
        eval_chunk = max(1, int(self.config.eval_batch_size or len(tree_list) or 1))
        raw_values: list[Any] = []
        with self._torch.no_grad():
            for start in range(0, len(tree_list), eval_chunk):
                stop = start + eval_chunk
                if rollup_weights is not None:
                    raw, _traces = self._model.forward_rollup(
                        x[start:stop], lengths[start:stop], rollup_weights[start:stop]
                    )
                else:
                    raw = self._model(x[start:stop], lengths[start:stop])
                raw_values.extend(self._denormalized_predictions(raw).detach().cpu().tolist())
        values = [row if isinstance(row, list) else [row] for row in raw_values]
        if (self._output_dim or 1) == 1:
            return [_clamp(float(row[0]), self.config.target_min, self.config.target_max) for row in values]
        return [
            [_clamp(float(value), self.config.target_min, self.config.target_max) for value in row]
            for row in values
        ]

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        if kind in {"f", "g"} and not isinstance(artifact, Mapping):
            raise TypeError(f"family={self.name!r} {kind} artifact must be a mapping")
        if isinstance(artifact, Mapping):
            path = artifact.get("weights_path")
            if path and not Path(str(path)).exists():
                raise ValueError(
                    f"family={self.name!r} {kind} artifact weights_path "
                    f"{str(path)!r} does not exist"
                )

    def as_statistic(self, *, f: Any = None, g: Any = None) -> Any:
        del f, g
        if self._model is None:
            return None
        return _NeuralOperatorStatistic(self)

    def _artifact_kind_for_operator(self) -> str:
        if self.operator_kind == "fno":
            return "treepo_fno"
        return self.artifact_kind

    def _artifact_payload(
        self,
        *,
        kind: str,
        iteration: int,
        n_train: int,
        loss: float | None,
        weights_path: Path | None = None,
    ) -> dict[str, Any]:
        base_kind = self._artifact_kind_for_operator()
        return {
            "kind": base_kind if kind == "f" else f"{base_kind}_g",
            "trained": kind,
            "weights_path": str(weights_path) if weights_path is not None else None,
            "operator_kind": self.operator_kind,
            "iteration": int(iteration),
            "n_train": int(n_train),
            "loss": loss,
            "normalize_targets": bool(self.config.normalize_targets),
            "target_center": _tensor_payload(self._target_center),
            "target_scale": _tensor_payload(self._target_scale),
            "numeric_transition_state_weight": float(self.config.numeric_transition_state_weight),
            "root_readout": str(self.config.root_readout),
            "rollup_weight_key": self.config.rollup_weight_key,
            "node_supervision": {
                "root_weight": float(self.config.root_weight),
                "leaf_weight": float(self.config.leaf_weight),
                "merge_weight": float(self.config.merge_weight),
                "n_trees": int((self._node_supervision_counts or {}).get("n_trees", 0)),
                "n_leaf_rows": int((self._node_supervision_counts or {}).get("n_leaf_rows", 0)),
                "n_merge_rows": int((self._node_supervision_counts or {}).get("n_merge_rows", 0)),
                "law_source": self._last_law_source,
            },
            "objective_executed": self._objective is not None,
            "objective": self._objective.to_dict() if self._objective is not None else None,
            "output_dim": int(self._output_dim or 1),
            "target_key": self.config.target_key,
            "target_vector_key": self.config.target_vector_key,
            "config": _config_payload(self.config),
        }

    def _ensure_model(self, output_dim: int | None = None) -> None:
        if output_dim is not None:
            output_dim = max(1, int(output_dim))
            if self._output_dim is not None and self._output_dim != output_dim:
                raise ValueError(
                    f"family={self.name!r} output_dim changed from {self._output_dim} to {output_dim}"
                )
            self._output_dim = output_dim
        if self._output_dim is None:
            self._output_dim = 1
        if self._model is not None:
            return
        _validate_operator_kind(self.operator_kind, family_name=self.name)
        if self.operator_kind == "fno":
            # The documented f/g FNO invariant (embedding axis as the spatial
            # dimension, 1/2 input channels, zero-last-layer identity init) —
            # the same computational object as the TT-ladder anchor runs.
            # Other operator kinds keep the generic leaf-axis model.
            self._model = _EmbeddingCoordinateFGModel(
                config=self.config,
                torch=self._torch,
                output_dim=int(self._output_dim),
            ).to(self._device)
        else:
            self._model = _TreeFGModel(
                operator_kind=self.operator_kind,
                config=self.config,
                torch=self._torch,
                output_dim=int(self._output_dim),
            ).to(self._device)

    def _normalized_targets(self, y: Any) -> Any:
        if not bool(self.config.normalize_targets):
            self._target_center = self._torch.zeros((int(y.shape[1]),), dtype=y.dtype, device=y.device)
            self._target_scale = self._torch.ones((int(y.shape[1]),), dtype=y.dtype, device=y.device)
            return y
        if bool(getattr(self._model, "bounded_output", False)):
            # A sigmoid-bounded head predicts in [0, 1]: min-max normalize onto
            # that range (the TT convention) instead of z-scoring, which would
            # put targets outside the head's codomain.
            lo = (
                float(self.config.target_min)
                if self.config.target_min is not None
                else float(y.min().detach().cpu())
            )
            hi = (
                float(self.config.target_max)
                if self.config.target_max is not None
                else float(y.max().detach().cpu())
            )
            span = max(hi - lo, 1.0e-6)
            center = self._torch.full((int(y.shape[1]),), lo, dtype=y.dtype, device=y.device)
            scale = self._torch.full((int(y.shape[1]),), span, dtype=y.dtype, device=y.device)
            self._target_center = center.detach()
            self._target_scale = scale.detach()
            return (y - center) / scale
        center = y.mean(dim=0)
        scale = y.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        self._target_center = center.detach()
        self._target_scale = scale.detach()
        return (y - center) / scale

    def _denormalized_predictions(self, y: Any) -> Any:
        if self._target_center is None or self._target_scale is None:
            return y
        return y * self._target_scale.to(device=y.device, dtype=y.dtype) + self._target_center.to(device=y.device, dtype=y.dtype)

    def _train_supervised(
        self,
        x: Any,
        lengths: Any,
        y: Any,
        *,
        train_f: bool,
        train_g: bool,
        trees: Sequence[Any] | None = None,
        node_supervision: list[tuple[Any, Any]] | None = None,
        rollup_weights: Any | None = None,
    ) -> float | None:
        assert self._model is not None
        self._set_trainable(train_f=train_f, train_g=train_g)
        params = [param for param in self._model.parameters() if bool(param.requires_grad)]
        if not params:
            return None
        self._model.train()
        opt = self._torch.optim.AdamW(
            params,
            lr=float(self.config.learning_rate),
            weight_decay=float(self.config.weight_decay),
        )
        batch_size = max(1, int(self.config.batch_size))
        epochs = max(1, int(self.config.epochs_per_iteration))
        last_loss = None
        n = int(x.shape[0])
        # The law signal enters training in one of two mutually exclusive ways:
        # a configured ObjectiveSpec drives the convex assembly on every step,
        # while the legacy additive knob applies to g-steps only.
        law_in_loss = bool(self._objective_law_shares) or (
            self._objective is None
            and train_g
            and float(self.config.numeric_transition_state_weight) > 0.0
        )
        state_targets = (
            _numeric_transition_state_targets(trees, self.config, torch=self._torch, device=self._device)
            if law_in_loss and trees is not None
            else None
        )
        # With an ObjectiveSpec, the law channels are fed by exact numeric
        # transition states when the trees carry them, otherwise by per-node
        # score targets (labeled bundles). Node terms thereby fold into the
        # convex corrected term — never a third additive slot.
        node_law = None
        if self._objective_law_shares and state_targets is None:
            node_law = node_supervision
            if node_law is None:
                raise ValueError(
                    f"family={self.name!r} objective declares a local-law weight but "
                    "the training trees carry neither numeric transition supervision "
                    "(tree metadata needs n_states and vocabulary_size) nor per-node "
                    "targets (node label / metadata 'score')"
                )
        node_weighted = (
            self._objective is None
            and node_supervision is not None
            and (float(self.config.leaf_weight) > 0.0 or float(self.config.merge_weight) > 0.0)
        )
        self._last_law_source = (
            "numeric_transition"
            if (law_in_loss and state_targets is not None)
            else ("node_targets" if node_law is not None else None)
        )
        need_trace = state_targets is not None or node_law is not None or node_weighted
        for _epoch in range(epochs):
            # Per-epoch reseed of the shuffle (the TT ladder convention): the
            # data order at epoch e is a function of (seed, e), not of how many
            # batches previous stages consumed from the global RNG stream.
            self._torch.manual_seed(int(self.config.seed) + int(_epoch))
            order = self._torch.randperm(n, device=self._device)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                micro = self.config.micro_batch_size
                per_tree = self.config.per_tree_loss_lambda is not None and node_weighted
                if per_tree or (micro is not None and 0 < int(micro) < int(idx.shape[0])):
                    if state_targets is not None or node_law is not None:
                        raise ValueError(
                            f"family={self.name!r} micro_batch_size / per-tree loss "
                            "does not support law-bearing objectives yet; drop "
                            "micro_batch_size or the objective"
                        )
                    last_loss = self._micro_batch_step(
                        x,
                        lengths,
                        y,
                        idx,
                        opt=opt,
                        params=params,
                        node_supervision=node_supervision if node_weighted else None,
                        rollup_weights=rollup_weights,
                    )
                    continue
                law_rows = None
                node_rows = None
                if not need_trace:
                    if rollup_weights is not None:
                        pred, _traces = self._model.forward_rollup(
                            x[idx], lengths[idx], rollup_weights[idx]
                        )
                    else:
                        pred = self._model(x[idx], lengths[idx])
                else:
                    if rollup_weights is not None:
                        pred, traces = self._model.forward_rollup(
                            x[idx], lengths[idx], rollup_weights[idx], collect_trace=True
                        )
                    else:
                        pred, traces = self._model.forward_with_trace(x[idx], lengths[idx])
                    idx_list = idx.detach().cpu().tolist()
                    if state_targets is not None:
                        law_rows = _numeric_transition_law_rows(
                            traces,
                            [state_targets[int(i)] for i in idx_list],
                            torch=self._torch,
                            device=self._device,
                            dtype=pred.dtype,
                        )
                    else:
                        assert node_supervision is not None
                        node_rows = self._node_supervision_rows(
                            traces,
                            [node_supervision[int(i)] for i in idx_list],
                            dtype=pred.dtype,
                        )
                        if node_law is not None:
                            law_rows = node_rows
                root_loss = self._torch.nn.functional.mse_loss(pred, y[idx])
                if node_weighted:
                    loss = self._node_weighted_loss(
                        root_loss, node_rows, batch_n=int(idx.shape[0])
                    )
                else:
                    loss = self._assemble_loss(root_loss, law_rows)
                last_loss = float(loss.detach().cpu())
                if not bool(loss.requires_grad):
                    # The batch's loss is constant w.r.t. this side's params
                    # (e.g. a g-pass over single-leaf trees: no merge exists,
                    # fg reduces to f and the root term supervises f).
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                clip = self.config.grad_clip_norm
                if clip is not None and float(clip) > 0.0:
                    self._torch.nn.utils.clip_grad_norm_(params, float(clip))
                opt.step()
        self._set_trainable(train_f=True, train_g=True)
        return last_loss

    def _micro_batch_step(
        self,
        x: Any,
        lengths: Any,
        y: Any,
        idx: Any,
        *,
        opt: Any,
        params: list[Any],
        node_supervision: list[tuple[Any, Any]] | None,
        rollup_weights: Any | None,
    ) -> float:
        """One optimizer step over ``idx`` in memory-bounded chunks — exact.

        The chunk losses are scaled so their SUM equals the full-batch loss
        (plain root MSE, or the weighted node-mean with its denominator
        computed over the whole batch up front), so accumulated gradients are
        identical to a single-pass batch and only peak activation memory
        changes.
        """

        torch = self._torch
        micro = max(1, int(self.config.micro_batch_size or 1))
        batch_n = int(idx.shape[0])
        width = int(y.shape[1])
        root_w = float(self.config.root_weight)
        leaf_w = float(self.config.leaf_weight)
        merge_w = float(self.config.merge_weight)
        idx_list = idx.detach().cpu().tolist()

        denom: float | None = None
        if node_supervision is not None and self.config.per_tree_loss_lambda is None:
            weight_sum = torch.zeros((), device=self._device)
            for i in idx_list:
                _target, mask = node_supervision[int(i)]
                n_nodes = int(mask.shape[0])
                leaf_count = (n_nodes + 1) // 2
                is_leaf = torch.arange(n_nodes, device=mask.device) < leaf_count
                row_w = torch.where(
                    is_leaf,
                    torch.full((n_nodes,), leaf_w, device=mask.device),
                    torch.full((n_nodes,), merge_w, device=mask.device),
                )
                keep = mask & (row_w > 0.0)
                weight_sum = weight_sum + row_w[keep].sum()
            denom = root_w * float(batch_n) + float(weight_sum.detach().cpu())
            if denom <= 0.0:
                raise ValueError(
                    f"family={self.name!r} weighted node-mean has zero total weight "
                    "over the batch; nothing anchors the loss"
                )

        opt.zero_grad(set_to_none=True)
        total = 0.0
        any_grad = False
        per_tree_lambda = self.config.per_tree_loss_lambda
        for cstart in range(0, batch_n, micro):
            cidx = idx[cstart : cstart + micro]
            if node_supervision is not None and per_tree_lambda is not None:
                if rollup_weights is not None:
                    pred, traces = self._model.forward_rollup(
                        x[cidx], lengths[cidx], rollup_weights[cidx], collect_trace=True
                    )
                else:
                    pred, traces = self._model.forward_with_trace(x[cidx], lengths[cidx])
                chunk_loss = self._per_tree_convex_loss(
                    pred,
                    y[cidx],
                    traces,
                    [node_supervision[int(i)] for i in cidx.detach().cpu().tolist()],
                    batch_n=batch_n,
                )
            elif node_supervision is not None:
                if rollup_weights is not None:
                    pred, traces = self._model.forward_rollup(
                        x[cidx], lengths[cidx], rollup_weights[cidx], collect_trace=True
                    )
                else:
                    pred, traces = self._model.forward_with_trace(x[cidx], lengths[cidx])
                node_rows = self._node_supervision_rows(
                    traces,
                    [node_supervision[int(i)] for i in cidx.detach().cpu().tolist()],
                    dtype=pred.dtype,
                )
                # Numerator contribution: root_w * Σ_tree (per-tree MSE over
                # width) + Σ row_w · node loss, over this chunk only.
                sse = ((pred - y[cidx]) ** 2).sum() / float(width)
                num = root_w * sse
                if node_rows is not None:
                    losses, _depths, is_leaf = node_rows
                    row_w = torch.where(
                        is_leaf,
                        torch.full_like(losses, leaf_w),
                        torch.full_like(losses, merge_w),
                    )
                    keep = row_w > 0.0
                    num = num + (row_w[keep] * losses[keep]).sum()
                chunk_loss = num / float(denom or 1.0)
            else:
                if rollup_weights is not None:
                    pred, _traces = self._model.forward_rollup(
                        x[cidx], lengths[cidx], rollup_weights[cidx]
                    )
                else:
                    pred = self._model(x[cidx], lengths[cidx])
                chunk_loss = ((pred - y[cidx]) ** 2).sum() / float(batch_n * width)
            total += float(chunk_loss.detach().cpu())
            if bool(chunk_loss.requires_grad):
                any_grad = True
                chunk_loss.backward()
        if any_grad:
            clip = self.config.grad_clip_norm
            if clip is not None and float(clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(clip))
            opt.step()
        return total

    def _per_tree_convex_loss(
        self,
        pred: Any,
        y: Any,
        traces: Sequence[Any],
        selected: Sequence[tuple[Any, Any]],
        *,
        batch_n: int,
    ) -> Any:
        """TT-ladder loss shape: mean over trees of a per-tree convex split.

        Per tree: ``(1-λ)·root_mse + λ·(leaf/merge-weighted node mean)``; a
        tree without observed node rows contributes ``(1-λ)·root_mse``. Every
        tree weighs equally regardless of node count, so chunked accumulation
        (divide by the full ``batch_n``) is exact under any chunking.
        """

        torch = self._torch
        lam = float(self.config.per_tree_loss_lambda or 0.0)
        leaf_w = float(self.config.leaf_weight)
        merge_w = float(self.config.merge_weight)
        read = getattr(self._model, "_read", None) or self._model.readout
        total = None
        for i, (trace, (target, mask)) in enumerate(zip(traces, selected)):
            root_mse = ((pred[i] - y[i]) ** 2).mean()
            n_nodes = int(trace.shape[0])
            leaf_count = (n_nodes + 1) // 2
            preds = read(trace)
            losses = ((preds - target.to(dtype=preds.dtype)) ** 2).mean(dim=-1)
            is_leaf = torch.arange(n_nodes, device=losses.device) < leaf_count
            row_w = torch.where(
                is_leaf,
                torch.full_like(losses, leaf_w),
                torch.full_like(losses, merge_w),
            )
            keep = mask & (row_w > 0.0)
            tree_loss = (1.0 - lam) * root_mse
            if bool(keep.any()):
                node_mean = (row_w[keep] * losses[keep]).sum() / row_w[keep].sum()
                tree_loss = tree_loss + lam * node_mean
            total = tree_loss if total is None else total + tree_loss
        assert total is not None
        return total / float(batch_n)

    def _rollup_weights_tensor(self, trees: Sequence[Any], *, max_leaves: int) -> Any | None:
        """Normalized ``[n_trees, max_leaves]`` rollup weights for ``leaf_mean``."""

        if str(self.config.root_readout) != "leaf_mean":
            return None
        rows = _leaf_rollup_weights(trees, self.config, max_leaves=int(max_leaves))
        return self._torch.tensor(rows, dtype=self._torch.float32, device=self._device)

    def _prepare_node_supervision(
        self, trees: Sequence[Any], *, width: int
    ) -> list[tuple[Any, Any]] | None:
        """Extract normalized per-tree node-target tensors when supervision wants them.

        Returns one ``(targets, observed)`` tensor pair per tree in canonical
        trace order, normalized with the root-target center/scale so readout
        predictions and node targets share one space. Tensors are built once
        per training call — the batch loop only masks and gathers on-device.
        """

        node_weights_active = self._objective is None and (
            float(self.config.leaf_weight) > 0.0 or float(self.config.merge_weight) > 0.0
        )
        if not node_weights_active and not self._objective_law_shares:
            self._node_supervision_counts = None
            return None
        # Extract only the channels the executed loss will consume, so the
        # recorded row counts state exactly which loss components activated.
        if node_weights_active:
            include_leaves = float(self.config.leaf_weight) > 0.0
            include_merges = float(self.config.merge_weight) > 0.0
        else:
            include_leaves = LawKind.C1_LEAF.value in self._objective_law_shares
            include_merges = LawKind.C3_MERGE.value in self._objective_law_shares
        rows = _node_supervision_targets(
            trees,
            self.config,
            width=int(width),
            include_leaves=include_leaves,
            include_merges=include_merges,
        )
        if rows is None:
            if node_weights_active:
                # Single-leaf trees have no non-root node to supervise: the
                # lone leaf IS the root, fg reduces to f, and the root term
                # carries the supervision. Only error when nodes exist (or
                # nothing anchors the loss at all).
                from treepo.tree import tree_leaves

                has_supervisable_nodes = any(
                    len(tuple(tree_leaves(tree) or ())) >= 2 for tree in trees
                )
                if has_supervisable_nodes or float(self.config.root_weight) <= 0.0:
                    raise ValueError(
                        f"family={self.name!r} has leaf_weight/merge_weight > 0 but "
                        "the training trees carry no per-node targets; nodes need a "
                        "label (or metadata 'score'/'oracle_score', or the configured "
                        "node_target_key), and single-leaf trees need root_weight > 0"
                    )
            self._node_supervision_counts = None
            return None
        torch = self._torch
        prepared: list[tuple[Any, Any]] = []
        n_leaf = 0
        n_merge = 0
        for targets, observed in rows:
            target_tensor = torch.tensor(targets, dtype=torch.float32, device=self._device)
            if self._target_center is not None and self._target_scale is not None:
                target_tensor = (target_tensor - self._target_center) / self._target_scale
            mask_tensor = torch.tensor(observed, dtype=torch.bool, device=self._device)
            prepared.append((target_tensor, mask_tensor))
            leaf_count = (len(observed) + 1) // 2
            n_leaf += sum(1 for i, seen in enumerate(observed) if seen and i < leaf_count)
            n_merge += sum(1 for i, seen in enumerate(observed) if seen and i >= leaf_count)
        self._node_supervision_counts = {
            "n_trees": len(rows),
            "n_leaf_rows": int(n_leaf),
            "n_merge_rows": int(n_merge),
        }
        return prepared

    def _node_supervision_rows(
        self,
        traces: Sequence[Any],
        selected: Sequence[tuple[Any, Any]],
        *,
        dtype: Any,
    ) -> tuple[Any, Any, Any] | None:
        """Per-node ``(loss, depths, is_leaf)`` rows for one batch, observed only.

        Predictions are the readout applied to every trace state in one call;
        targets/masks were prepared up front, so this is pure on-device
        mask-and-gather (no per-node CPU sync).
        """

        torch = self._torch
        if len(traces) != len(selected):
            raise ValueError(
                f"node supervision got {len(traces)} traces for {len(selected)} target sets"
            )
        assert self._model is not None
        target_chunks = []
        mask_chunks = []
        depth_chunks = []
        leaf_chunks = []
        for trace, (target, mask) in zip(traces, selected):
            n_nodes = int(trace.shape[0])
            if int(target.shape[0]) != n_nodes:
                raise ValueError(
                    "node supervision count mismatch: model trace has "
                    f"{n_nodes} nodes, targets have {int(target.shape[0])}"
                )
            target_chunks.append(target)
            mask_chunks.append(mask)
            leaf_count = (n_nodes + 1) // 2
            depth_chunks.append(
                torch.tensor(
                    _pairwise_merge_depths(leaf_count)[:n_nodes],
                    dtype=torch.long,
                    device=self._device,
                )
            )
            leaf_chunks.append(torch.arange(n_nodes, device=self._device) < leaf_count)
        states = torch.cat(list(traces), dim=0)
        read = getattr(self._model, "_read", None)
        preds = read(states) if callable(read) else self._model.readout(states)
        targets = torch.cat(target_chunks, dim=0).to(device=preds.device, dtype=dtype)
        mask = torch.cat(mask_chunks)
        if not bool(mask.any()):
            return None
        losses = ((preds - targets) ** 2).mean(dim=-1)
        depths = torch.cat(depth_chunks)
        is_leaf = torch.cat(leaf_chunks)
        return losses[mask], depths[mask], is_leaf[mask]

    def _node_weighted_loss(self, root_loss: Any, node_rows: Any | None, *, batch_n: int) -> Any:
        """Weighted node-mean loss: the TT ladder's root/leaf/merge weighting.

        Every supervised row — each batch item's root plus every observed
        leaf/merge node — enters one weighted mean under the config's
        ``root_weight`` / ``leaf_weight`` / ``merge_weight``. With
        leaf/merge weights at 0 this reduces exactly to the historical root
        MSE; convexity holds by construction (weights normalize to 1).
        """

        torch = self._torch
        root_w = float(self.config.root_weight)
        leaf_w = float(self.config.leaf_weight)
        merge_w = float(self.config.merge_weight)
        root_num = root_w * float(batch_n) * root_loss
        root_den = root_w * float(batch_n)
        if node_rows is None:
            if root_w <= 0.0:
                raise ValueError(
                    f"family={self.name!r} root_weight is 0 and the training batch "
                    "produced no supervised node rows; nothing anchors the loss"
                )
            return root_loss
        losses, _depths, is_leaf = node_rows
        row_weights = torch.where(
            is_leaf,
            torch.full_like(losses, leaf_w),
            torch.full_like(losses, merge_w),
        )
        keep = row_weights > 0.0
        if root_w <= 0.0 and not bool(keep.any()):
            raise ValueError(
                f"family={self.name!r} root_weight is 0 and the training batch "
                "produced no positively weighted node rows; nothing anchors the loss"
            )
        node_num = (row_weights * losses * keep.to(losses.dtype)).sum()
        node_den = (row_weights * keep.to(losses.dtype)).sum()
        return (root_num + node_num) / (root_den + node_den)

    def _assemble_loss(self, root_loss: Any, law_rows: Any | None) -> Any:
        """Combine root and law terms under the executed objective.

        ObjectiveSpec path: ``root_share * root + sum_c share_c * law_c`` with
        each channel routed through the canonical depth-discounted objective.
        Legacy path: ``root + numeric_transition_state_weight * node_mean`` —
        the law term is the unweighted node mean, the sampling contract's
        audit estimand.
        """
        # Imported lazily to keep `import treepo.methods.fno` free of torch;
        # this module only touches torch after family construction.
        from treepo.training.local_law import local_law_objective_from_losses

        torch = self._torch
        if self._objective is None:
            if law_rows is None:
                return root_loss
            proxy, _depths, _is_leaf = law_rows
            return root_loss + float(self.config.numeric_transition_state_weight) * proxy.mean()
        loss = float(self._objective.root_share) * root_loss
        if not self._objective_law_shares:
            return loss
        if law_rows is None:
            raise ValueError(
                f"family={self.name!r} objective declares a local-law weight but "
                "the training batch produced no law rows"
            )
        proxy, depths, is_leaf = law_rows
        channel_masks = {
            LawKind.C1_LEAF.value: is_leaf,
            LawKind.C3_MERGE.value: ~is_leaf,
        }
        for name, share in self._objective_law_shares.items():
            mask = channel_masks[name]
            if not bool(mask.any()):
                raise ValueError(
                    f"family={self.name!r} objective weights law channel {name!r} "
                    "but the training batch has no such nodes; set its component "
                    "weight to 0"
                )
            masked = proxy[mask]
            # Exact-state supervision: the corrected estimator degenerates to
            # the oracle term (observed rows, propensity 1, oracle == proxy).
            channel = local_law_objective_from_losses(
                proxy_loss=masked,
                oracle_loss=masked,
                observed=torch.ones(masked.shape[0], dtype=torch.bool, device=masked.device),
                propensity=torch.ones(masked.shape[0], dtype=masked.dtype, device=masked.device),
                depths=depths[mask],
                gamma_depth=float(self._objective_gamma_depth),
            )
            loss = loss + float(share) * channel
        return loss

    def _set_trainable(self, *, train_f: bool, train_g: bool) -> None:
        """f trains the leaf operator + readout; g trains the merge operator.

        This mirrors the TT-ladder semantics the anchors were recorded under
        (``freeze_for_f_training`` / ``freeze_for_g_training``): f owns how a
        single unit becomes a scored state, g owns how two states compose.
        Models may pin the split explicitly via ``f_module_names`` /
        ``g_module_names``.
        """

        assert self._model is not None
        for param in self._model.parameters():
            param.requires_grad = False
        f_names = getattr(self._model, "f_module_names", ("leaf_operator", "readout"))
        g_names = getattr(self._model, "g_module_names", ("merge",))
        names: list[str] = []
        if train_f:
            names.extend(f_names)
        if train_g:
            names.extend(g_names)
        for name in names:
            module = getattr(self._model, name)
            for param in module.parameters():
                param.requires_grad = True

    def _encode_trees(self, trees: Sequence[Any]) -> tuple[Any, Any]:
        cache_key = _tree_sequence_cache_key(trees, dim=int(self.config.embedding_dim), device=str(self._device))
        cached = self._encoding_cache.get(cache_key)
        if cached is not None:
            return cached[0], cached[1]
        numeric_groups = [_leaf_token_groups(tree) for tree in trees]
        if numeric_groups and all(group is not None for group in numeric_groups):
            encoded = _encode_numeric_leaf_features(
                numeric_groups,
                dim=int(self.config.embedding_dim),
                torch=self._torch,
                device=self._device,
            )
        else:
            leaf_groups = [_leaf_texts(tree) for tree in trees]
            lengths = [max(1, len(group)) for group in leaf_groups]
            max_leaves = max(1, max(lengths))
            matrices = []
            for group in leaf_groups:
                texts = group or [""]
                vectors = self.embedding_client.embed_texts(texts)
                matrix = [_coerce_embedding(vec, int(self.config.embedding_dim)) for vec in vectors]
                while len(matrix) < max_leaves:
                    matrix.append([0.0] * int(self.config.embedding_dim))
                matrices.append(matrix[:max_leaves])
            x = self._torch.tensor(matrices, dtype=self._torch.float32, device=self._device)
            length_tensor = self._torch.tensor(lengths, dtype=self._torch.long, device=self._device)
            encoded = (x, length_tensor)
        while len(self._encoding_cache) >= self._encoding_cache_max_entries:
            self._encoding_cache.pop(next(iter(self._encoding_cache)))
        self._encoding_cache[cache_key] = (encoded[0], encoded[1], tuple(trees))
        return encoded


class FNOFamily(NeuralOperatorFamily):
    """Concrete Fourier neural-operator ``FamilyRuntime``."""

    name = "fno"
    artifact_kind = "treepo_fno"
    config_cls = FNOFamilyConfig


def build_neural_operator_family(backend_config: Mapping[str, Any]) -> NeuralOperatorFamily:
    """Build the generic neural-operator family from method ``backend_config``."""

    payload = dict(backend_config or {})
    config = _coerce_config(
        payload.get("neural_operator_config"), payload, config_cls=NeuralOperatorFamilyConfig
    )
    embedding_client = payload.get("embedding_client")
    if embedding_client is not None and not hasattr(embedding_client, "embed_texts"):
        raise TypeError("backend_config['embedding_client'] must provide embed_texts(texts)")
    return NeuralOperatorFamily(config, embedding_client=embedding_client)


def build_fno_family(backend_config: Mapping[str, Any]) -> FNOFamily:
    """Build the concrete FNO family from method ``backend_config``."""

    payload = dict(backend_config or {})
    config = _coerce_config(payload.get("fno_config"), payload, config_cls=FNOFamilyConfig)
    requested = _normalize_operator_kind(config.operator_kind)
    if requested != "fno":
        raise ValueError(
            "family='fno' only supports operator_kind='fno'; use "
            "family='neural_operator' for other operator kinds."
        )
    config.operator_kind = "fno"
    embedding_client = payload.get("embedding_client")
    if embedding_client is not None and not hasattr(embedding_client, "embed_texts"):
        raise TypeError("backend_config['embedding_client'] must provide embed_texts(texts)")
    return FNOFamily(config, embedding_client=embedding_client)


__all__ = [
    "FNOFamily",
    "FNOFamilyConfig",
    "NeuralOperatorFamily",
    "NeuralOperatorFamilyConfig",
    "build_fno_family",
    "build_neural_operator_family",
]
