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
from treepo.local_law import LawKind
from treepo.methods._fno_config import (
    FNOFamilyConfig,
    NeuralOperatorFamilyConfig,
    _clamp,
    _coerce_config,
    _config_payload,
    _normalize_operator_kind,
    _target_schema_payload,
    _tensor_payload,
)
from treepo.methods._fno_encoding import (
    _coerce_embedding,
    _encode_numeric_leaf_features,
    _leaf_texts,
    _leaf_token_groups,
    _tree_sequence_cache_key,
)
from treepo.methods._fno_law_state import (
    LawLossRows,
    dense_law_loss_rows,
    metadata_law_state_design_rows,
    metadata_law_state_rows,
    metadata_law_state_targets,
)
from treepo.methods._fno_loss import per_row_training_loss, training_loss_definition
from treepo.methods._fno_models import (
    _SHARED_G_ARCHITECTURE_VERSION,
    _SHARED_G_INPUT_CHANNELS,
    _UnifiedGTreeModel,
)
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
        self._warmstart_compatibility: dict[str, Any] | None = None
        self._objective: ObjectiveSpec | None = None
        self._objective_law_shares: dict[str, float] = {}
        self._objective_gamma_depth: float = 1.0
        self._node_supervision_counts: dict[str, int] | None = None
        self._root_supervision_count: int = 0
        self._last_law_source: str | None = None
        self._last_objective_components: dict[str, Any] | None = None
        self._law_state_supervision: dict[str, Any] | None = None
        self._last_optimizer_step_count: int = 0
        self._last_parameter_change_count: int = 0
        self._last_g_role_activity: dict[str, bool] = {
            "shared_g_optimizer_eligible": False,
            "leaf_domain_gradient_path_present": False,
            "merge_domain_gradient_path_present": False,
        }
        self._g_execution_mode = "learned"
        # Encoding cache keyed by tree identity. Each entry pins the tree
        # objects it was built from, so the id()-based key stays valid for as
        # long as the entry lives; the cache is bounded, evicting oldest first.
        self._encoding_cache: dict[tuple[Any, ...], tuple[Any, Any, tuple[Any, ...]]] = {}
        self._encoding_cache_max_entries = 16

    def _configure_g_execution(self, artifact: Any) -> None:
        mode = "learned"
        if isinstance(artifact, Mapping):
            mode = str(artifact.get("g_mode") or mode).strip().lower()
        if mode == "fixed":
            raise ValueError(
                f"family={self.name!r} does not execute metadata-only fixed g artifacts"
            )
        if mode not in {"identity", "learned"}:
            raise ValueError(f"family={self.name!r} got unsupported g_mode={mode!r}")
        self._g_execution_mode = mode
        if self._model is not None:
            self._model.set_g_mode(mode)

    def train_f(
        self,
        *,
        f_init: Any,
        g: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Mapping[str, Any]:
        self._configure_g_execution(g)
        self._maybe_warmstart(f_init)
        self._configure_g_execution(g)
        return self._train_side(kind="f", traces=traces, iteration=iteration, output_dir=output_dir)

    def train_g(
        self,
        *,
        g_init: Any,
        f: Any,
        traces: Sequence[Any],
        output_dir: Path,
        iteration: int,
    ) -> Any:
        del f
        self._configure_g_execution(g_init)
        self._maybe_warmstart(g_init)
        self._configure_g_execution(g_init)
        artifact = self._train_side(
            kind="g", traces=traces, iteration=iteration, output_dir=output_dir
        )
        from treepo.methods.runtime import GTrainOutcome

        steps = int(self._last_optimizer_step_count)
        changed = int(self._last_parameter_change_count)
        return GTrainOutcome(
            artifact=artifact,
            update_performed=changed > 0,
            reason=(f"neural_operator_optimizer_steps={steps};changed_parameter_tensors={changed}"),
        )

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
        current_schema = _target_schema_payload(self.config)
        artifact_schema = artifact.get("target_schema")
        artifact_output_dim = max(1, int(artifact.get("output_dim") or 1))
        current_output_dim = max(
            1,
            int(
                self._output_dim
                or self.config.target_dim
                or (len(self.config.target_names) if self.config.target_names else 1)
            ),
        )
        if artifact_schema is not None and not isinstance(artifact_schema, Mapping):
            raise ValueError(f"family={self.name!r} warmstart target_schema must be a mapping")
        if current_schema is not None or artifact_schema is not None:
            if current_schema is None or artifact_schema is None:
                named_schema = current_schema or artifact_schema
                assert isinstance(named_schema, Mapping)
                target_order = named_schema.get("target_order")
                named_width = (
                    len(target_order)
                    if isinstance(target_order, Sequence)
                    and not isinstance(target_order, (str, bytes))
                    else None
                )
                singleton_bridge = (
                    named_width == 1 and artifact_output_dim == 1 and current_output_dim == 1
                )
                if not singleton_bridge:
                    raise ValueError(
                        f"family={self.name!r} warmstart named target schema is missing "
                        "on either the artifact or the current fit; only anonymous "
                        "scalar <-> named width-1 checkpoints are compatible"
                    )
                self._warmstart_compatibility = {
                    "mode": "anonymous_scalar_named_singleton",
                    "artifact_schema_kind": (
                        "named_singleton" if artifact_schema is not None else "anonymous_scalar"
                    ),
                    "current_schema_kind": (
                        "named_singleton" if current_schema is not None else "anonymous_scalar"
                    ),
                    "artifact_output_dim": int(artifact_output_dim),
                    "current_output_dim": int(current_output_dim),
                    "artifact_schema_digest": (
                        artifact_schema.get("schema_digest")
                        if isinstance(artifact_schema, Mapping)
                        else None
                    ),
                    "current_schema_digest": (
                        current_schema.get("schema_digest")
                        if isinstance(current_schema, Mapping)
                        else None
                    ),
                }
            elif artifact_schema.get("schema_digest") != current_schema.get("schema_digest"):
                raise ValueError(
                    f"family={self.name!r} warmstart target schema mismatch: "
                    f"artifact_order={artifact_schema.get('target_order')!r}, "
                    f"current_order={current_schema.get('target_order')!r}"
                )
        artifact_architecture = artifact.get("architecture_version")
        if artifact_architecture != _SHARED_G_ARCHITECTURE_VERSION:
            raise ValueError(
                f"family={self.name!r} warmstart checkpoint architecture is "
                f"{artifact_architecture!r}; expected "
                f"{_SHARED_G_ARCHITECTURE_VERSION!r}. Split leaf/merge "
                "checkpoints cannot be loaded into the single-shared-g model; "
                "retrain or use an explicit lossy migration tool."
            )
        self._ensure_model(output_dim=artifact_output_dim)
        assert self._model is not None
        # Artifacts persist only tensor entries (see _train_side), so the safe
        # weights-only loader applies. Architecture identity is checked above;
        # strict loading prevents a renamed or partial g from passing silently.
        state = self._torch.load(weights_file, map_location=self._device, weights_only=True)
        self._model.load_state_dict(state, strict=True)
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
        combination ``root_share * root_task_loss + sum_c share_c * law_c``,
        where each law channel is the depth-discounted canonical objective from
        :mod:`treepo.training.local_law` over exact task-supplied or numeric
        transition-state rows. The supervision targets are exact, so the spec must declare
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
        trees, targets, root_observed_rows = _target_rows(traces, self.config)
        x, lengths = self._encode_trees(trees)
        y = self._torch.tensor(targets, dtype=self._torch.float32, device=self._device)
        root_observed = self._torch.tensor(
            root_observed_rows, dtype=self._torch.bool, device=self._device
        )
        self._root_supervision_count = int(root_observed.sum().detach().cpu())
        self._ensure_model(output_dim=int(y.shape[1]))
        self._last_parameter_change_count = 0
        before_g = (
            {
                name: parameter.detach().clone()
                for name, parameter in self._model.g.named_parameters()
            }
            if train_g
            else {}
        )
        y_train = self._normalized_targets(y, root_observed=root_observed)
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
            root_observed=root_observed,
        )
        if train_g:
            self._last_parameter_change_count = sum(
                not self._torch.equal(before_g[name], parameter.detach())
                for name, parameter in self._model.g.named_parameters()
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
        self._configure_g_execution(g)
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
        self._configure_g_execution(g)
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
        if (self._output_dim or 1) == 1 and not self.config.target_names:
            return [
                _clamp(float(row[0]), self.config.target_min, self.config.target_max)
                for row in values
            ]
        return [
            [_clamp(float(value), self.config.target_min, self.config.target_max) for value in row]
            for row in values
        ]

    def validate_artifact(self, *, kind: str, artifact: Any) -> None:
        if kind in {"f", "g"} and not isinstance(artifact, Mapping):
            raise TypeError(f"family={self.name!r} {kind} artifact must be a mapping")
        if kind == "g" and isinstance(artifact, Mapping):
            if str(artifact.get("g_mode") or "").strip().lower() == "fixed":
                self._configure_g_execution(artifact)
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

    def _shared_g_contract_payload(self) -> dict[str, Any]:
        """Serializable proof obligation for the runtime parameterization."""

        assert self._model is not None
        g = getattr(self._model, "g")
        return {
            "architecture_version": _SHARED_G_ARCHITECTURE_VERSION,
            "registered_module": "g",
            "single_registered_g": True,
            "parameter_identity_shared_across_call_domains": True,
            "leaf_call": "g(raw_embedding, empty)",
            "merge_call": "g(left_state, right_state)",
            "input_channels": int(_SHARED_G_INPUT_CHANNELS),
            "input_channel_order": [
                "left_content",
                "right_content",
                "left_present",
                "right_present",
            ],
            "empty_right_encoding": "zero_content_with_zero_occupancy",
            "merge_right_encoding": "state_content_with_unit_occupancy",
            "initial_residual_base": "occupancy_masked_mean",
            "g_parameter_count": int(sum(p.numel() for p in g.parameters())),
        }

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
            "g_mode": self._g_execution_mode,
            "iteration": int(iteration),
            "n_train": int(n_train),
            "optimizer_step_count": int(self._last_optimizer_step_count),
            "changed_parameter_tensor_count": int(self._last_parameter_change_count),
            "architecture_version": _SHARED_G_ARCHITECTURE_VERSION,
            "g_role_activity": dict(self._last_g_role_activity),
            "shared_g_optimizer_eligible": bool(
                self._last_g_role_activity["shared_g_optimizer_eligible"]
            ),
            "leaf_domain_gradient_path_present": bool(
                self._last_g_role_activity["leaf_domain_gradient_path_present"]
            ),
            "merge_domain_gradient_path_present": bool(
                self._last_g_role_activity["merge_domain_gradient_path_present"]
            ),
            "fg_parameter_partition": "f_readout__single_shared_g_v3",
            "shared_g_contract": self._shared_g_contract_payload(),
            "loss": loss,
            "training_loss": str(self.config.training_loss),
            "training_loss_definition": training_loss_definition(self.config.training_loss),
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
                "n_trees": int(n_train),
                "n_root_rows": int(self._root_supervision_count),
                "n_leaf_rows": int((self._node_supervision_counts or {}).get("n_leaf_rows", 0)),
                "n_merge_rows": int((self._node_supervision_counts or {}).get("n_merge_rows", 0)),
                "law_source": self._last_law_source,
            },
            "objective_executed": self._objective is not None,
            "objective": self._objective.to_dict() if self._objective is not None else None,
            "objective_components": self._last_objective_components,
            "law_state_supervision": self._law_state_supervision,
            "task_readout": self._task_readout_payload(),
            "output_dim": int(self._output_dim or 1),
            "target_key": self.config.target_key,
            "target_vector_key": self.config.target_vector_key,
            "target_names": list(self.config.target_names or ()),
            "target_oracle_ids": list(self.config.target_oracle_ids or ()),
            "target_schema": _target_schema_payload(self.config),
            "warmstart_compatibility": self._warmstart_compatibility,
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
        if self.config.law_state_root_readout == "first_coordinate":
            if int(self._output_dim) != 1:
                raise ValueError(
                    "law_state_root_readout='first_coordinate' requires a scalar "
                    f"task target (got output_dim={int(self._output_dim)})"
                )
            if bool(self.config.normalize_targets) and (
                self.config.target_min is None or self.config.target_max is None
            ):
                raise ValueError(
                    "law_state_root_readout='first_coordinate' with target "
                    "normalization requires explicit target_min and target_max; "
                    "coordinate zero must use that same normalized task scale"
                )
        if self._model is not None:
            return
        _validate_operator_kind(self.operator_kind, family_name=self.name)
        self._model = _UnifiedGTreeModel(
            operator_kind=self.operator_kind,
            config=self.config,
            torch=self._torch,
            output_dim=int(self._output_dim),
        ).to(self._device)
        self._model.set_g_mode(self._g_execution_mode)

    def _task_readout_payload(self) -> dict[str, Any]:
        mode = self.config.law_state_root_readout
        if mode == "first_coordinate":
            return {
                "kind": "fixed_state_coordinate",
                "state_coordinate": 0,
                "output_dim": 1,
                "learned_readout_trainable": False,
                "training_scale": "configured_normalized_task_coordinate",
                "public_prediction_clamped_to_target_bounds": True,
            }
        payload = {
            "kind": "learned_readout",
            "output_dim": int(self._output_dim or 1),
            "learned_readout_trainable": True,
        }
        schema = _target_schema_payload(self.config)
        if schema is not None:
            payload["target_schema"] = schema
        return payload

    def _normalized_targets(self, y: Any, *, root_observed: Any | None = None) -> Any:
        observed = (
            self._torch.ones(int(y.shape[0]), dtype=self._torch.bool, device=y.device)
            if root_observed is None
            else root_observed.to(device=y.device, dtype=self._torch.bool).reshape(-1)
        )
        observed_y = y[observed]
        if not bool(self.config.normalize_targets):
            self._target_center = self._torch.zeros(
                (int(y.shape[1]),), dtype=y.dtype, device=y.device
            )
            self._target_scale = self._torch.ones(
                (int(y.shape[1]),), dtype=y.dtype, device=y.device
            )
            return y
        if bool(getattr(self._model, "bounded_output", False)):
            # A sigmoid-bounded head predicts in [0, 1]: min-max normalize onto
            # that range (the TT convention) instead of z-scoring, which would
            # put targets outside the head's codomain.
            lo = (
                float(self.config.target_min)
                if self.config.target_min is not None
                else self._observed_target_bound(observed_y, kind="minimum")
            )
            hi = (
                float(self.config.target_max)
                if self.config.target_max is not None
                else self._observed_target_bound(observed_y, kind="maximum")
            )
            span = max(hi - lo, 1.0e-6)
            center = self._torch.full((int(y.shape[1]),), lo, dtype=y.dtype, device=y.device)
            scale = self._torch.full((int(y.shape[1]),), span, dtype=y.dtype, device=y.device)
            self._target_center = center.detach()
            self._target_scale = scale.detach()
            return (y - center) / scale
        if int(observed_y.shape[0]) <= 0:
            raise ValueError(
                "root_observed_doc_ids selected no root labels, so target "
                "normalization needs fixed target_min/target_max or "
                "normalize_targets=False"
            )
        center = observed_y.mean(dim=0)
        scale = observed_y.std(dim=0, unbiased=False).clamp_min(1.0e-6)
        self._target_center = center.detach()
        self._target_scale = scale.detach()
        return (y - center) / scale

    @staticmethod
    def _observed_target_bound(values: Any, *, kind: str) -> float:
        if int(values.shape[0]) <= 0:
            raise ValueError(
                "root_observed_doc_ids selected no root labels, so target "
                "normalization needs fixed target_min/target_max or "
                "normalize_targets=False"
            )
        if kind == "minimum":
            return float(values.min().detach().cpu())
        return float(values.max().detach().cpu())

    def _denormalized_predictions(self, y: Any) -> Any:
        if self._target_center is None or self._target_scale is None:
            return y
        return y * self._target_scale.to(device=y.device, dtype=y.dtype) + self._target_center.to(
            device=y.device, dtype=y.dtype
        )

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
        root_observed: Any | None = None,
    ) -> float | None:
        assert self._model is not None
        self._last_objective_components = None
        self._last_g_role_activity = {
            "shared_g_optimizer_eligible": bool(train_g),
            "leaf_domain_gradient_path_present": bool(train_g),
            "merge_domain_gradient_path_present": False,
        }
        self._last_optimizer_step_count = 0
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
        if root_observed is None:
            root_observed = self._torch.ones(n, dtype=self._torch.bool, device=self._device)
        else:
            root_observed = root_observed.to(device=self._device, dtype=self._torch.bool).reshape(
                -1
            )
        if int(root_observed.shape[0]) != n:
            raise ValueError("root-observation mask must have one row per training tree")
        # The law signal enters training in one of two mutually exclusive ways:
        # a configured ObjectiveSpec drives the convex assembly on every step,
        # while the legacy additive knob applies to g-steps only.
        law_in_loss = bool(self._objective_law_shares) or (
            self._objective is None
            and train_g
            and float(self.config.numeric_transition_state_weight) > 0.0
        )
        metadata_state_targets = (
            metadata_law_state_targets(trees, self.config, torch=self._torch, device=self._device)
            if law_in_loss and trees is not None and self.config.law_state_target_key
            else None
        )
        numeric_state_targets = (
            _numeric_transition_state_targets(
                trees, self.config, torch=self._torch, device=self._device
            )
            if law_in_loss and trees is not None and metadata_state_targets is None
            else None
        )
        state_targets = metadata_state_targets or numeric_state_targets
        if metadata_state_targets is not None:
            n_leaf = 0
            n_merge = 0
            for target in metadata_state_targets:
                n_nodes = int(target.observed.shape[0])
                leaf_count = (n_nodes + 1) // 2
                n_leaf += int(target.observed[:leaf_count].sum().detach().cpu())
                n_merge += int(target.observed[leaf_count:].sum().detach().cpu())
            self._law_state_supervision = {
                "target_key": str(self.config.law_state_target_key),
                "target_dim": int(self.config.law_state_target_dim or 0),
                "n_trees": int(len(metadata_state_targets)),
                "n_leaf_rows": int(n_leaf),
                "n_merge_rows_including_root": int(n_merge),
                "leaf_propensity": float(self.config.law_state_leaf_propensity),
                "merge_propensity": float(self.config.law_state_merge_propensity),
                "evidence_kind": "task_supplied_exact_state_witness",
                "vector_loss": str(self.config.training_loss),
                "vector_loss_definition": training_loss_definition(self.config.training_loss),
                "root_merge_is_separate_from_root_task_loss": True,
                "c2_status": "not_applicable_one_pass_no_recompressor",
                "c3_scope": "observed_realized_canonical_merges_not_universal_closure",
            }
            # Keep the long-standing compact supervision counters useful to
            # callers while the richer state payload states the root policy.
            self._node_supervision_counts = {
                "n_trees": int(len(metadata_state_targets)),
                "n_leaf_rows": int(n_leaf),
                "n_merge_rows": int(n_merge),
            }
        else:
            self._law_state_supervision = None
        # With an ObjectiveSpec, law channels prefer an explicit task-supplied
        # witness, then numeric transition states, then per-node score targets.
        # Node terms thereby fold into the convex corrected term — never a
        # third additive slot.
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
            f"metadata_state:{self.config.law_state_target_key}"
            if metadata_state_targets is not None
            else (
                "numeric_transition"
                if (law_in_loss and numeric_state_targets is not None)
                else ("node_targets" if node_law is not None else None)
            )
        )
        need_trace = state_targets is not None or node_law is not None or node_weighted
        if not bool(root_observed.any()) and node_supervision is None and state_targets is None:
            raise ValueError(
                f"family={self.name!r} received no observed root or local targets; "
                "nothing anchors the loss"
            )
        merge_domain_gradient_path = self._g_merge_domain_gradient_path_present(
            lengths=lengths,
            root_observed=root_observed,
            node_supervision=node_supervision,
            law_targets=state_targets if state_targets is not None else node_law,
            node_weighted=node_weighted,
            train_g=train_g,
        )
        self._last_g_role_activity = {
            "shared_g_optimizer_eligible": bool(train_g),
            "leaf_domain_gradient_path_present": bool(train_g),
            "merge_domain_gradient_path_present": bool(merge_domain_gradient_path),
        }
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
                    if metadata_state_targets is not None:
                        last_loss = self._metadata_law_micro_batch_step(
                            x,
                            lengths,
                            y,
                            idx,
                            opt=opt,
                            params=params,
                            metadata_state_targets=metadata_state_targets,
                            rollup_weights=rollup_weights,
                            root_observed=root_observed,
                            train_f=train_f,
                            train_g=train_g,
                        )
                        continue
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
                        root_observed=root_observed,
                        train_f=train_f,
                        train_g=train_g,
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
                    if metadata_state_targets is not None:
                        law_rows = metadata_law_state_rows(
                            traces,
                            [metadata_state_targets[int(i)] for i in idx_list],
                            training_loss=self.config.training_loss,
                            torch=self._torch,
                            device=self._device,
                            dtype=pred.dtype,
                        )
                    elif numeric_state_targets is not None:
                        law_rows = dense_law_loss_rows(
                            _numeric_transition_law_rows(
                                traces,
                                [numeric_state_targets[int(i)] for i in idx_list],
                                training_loss=self.config.training_loss,
                                torch=self._torch,
                                device=self._device,
                                dtype=pred.dtype,
                            )
                        )
                    else:
                        assert node_supervision is not None
                        node_rows = self._node_supervision_rows(
                            traces,
                            [node_supervision[int(i)] for i in idx_list],
                            dtype=pred.dtype,
                        )
                        if node_law is not None:
                            law_rows = dense_law_loss_rows(node_rows)
                root_loss, root_count = self._masked_root_loss(pred, y[idx], root_observed[idx])
                root_dependency_count = self._root_parameter_dependency_count(
                    lengths=lengths[idx],
                    observed=root_observed[idx],
                    train_f=train_f,
                    train_g=train_g,
                )
                if node_weighted:
                    loss = self._node_weighted_loss(root_loss, node_rows, root_count=root_count)
                    has_supervision = self._node_weighted_batch_is_active(
                        root_dependency_count=root_dependency_count,
                        node_rows=node_rows,
                        train_f=train_f,
                        train_g=train_g,
                    )
                else:
                    loss = self._assemble_loss(root_loss, law_rows)
                    has_supervision = self._assembled_batch_is_active(
                        root_dependency_count=root_dependency_count,
                        law_rows=law_rows,
                        train_f=train_f,
                        train_g=train_g,
                    )
                last_loss = float(loss.detach().cpu())
                if not has_supervision or not bool(loss.requires_grad):
                    # The batch's loss is constant w.r.t. this side's params
                    # or has no observed target. It must skip ``opt.step()``:
                    # AdamW would otherwise apply decoupled weight decay even
                    # though that batch carries no supervision.
                    continue
                opt.zero_grad(set_to_none=True)
                loss.backward()
                clip = self.config.grad_clip_norm
                if clip is not None and float(clip) > 0.0:
                    self._torch.nn.utils.clip_grad_norm_(params, float(clip))
                opt.step()
                self._last_optimizer_step_count += 1
        self._set_trainable(train_f=True, train_g=True)
        return last_loss

    def _metadata_law_micro_batch_step(
        self,
        x: Any,
        lengths: Any,
        y: Any,
        idx: Any,
        *,
        opt: Any,
        params: list[Any],
        metadata_state_targets: Sequence[Any],
        rollup_weights: Any | None,
        root_observed: Any,
        train_f: bool,
        train_g: bool,
    ) -> float:
        """Exact activation microbatching for a root + exact-state objective.

        ``idx`` is one optimizer/outer batch.  Global root/C1/C3 denominators
        are computed once from its persistent masks. Each activation microbatch
        contributes only its numerator divided by that global denominator, so
        the accumulated gradient is exactly the single-pass outer-batch
        gradient. A microbatch may contain none of a channel; the outer batch
        must contain every positively weighted declared channel.
        """

        if self._objective is None or not self._objective_law_shares:
            raise ValueError("metadata law microbatching requires a law-bearing ObjectiveSpec")
        torch = self._torch
        idx_list = [int(value) for value in idx.detach().cpu().tolist()]
        selected_targets = [metadata_state_targets[value] for value in idx_list]
        design = metadata_law_state_design_rows(selected_targets, torch=torch, device=self._device)
        if design is None:
            raise ValueError("metadata law outer batch has no state-design rows")
        channel_masks = {
            LawKind.C1_LEAF.value: design.is_leaf,
            LawKind.C3_MERGE.value: ~design.is_leaf,
        }
        law_denominators = {
            name: float(self._law_channel_denominator(design, mask).detach().cpu())
            for name, mask in channel_masks.items()
        }
        root_count = int(root_observed[idx].sum().detach().cpu())
        if float(self._objective.root_share) > 0.0 and root_count <= 0:
            raise ValueError(
                "metadata law outer batch has no observed roots but root_share > 0; "
                "use an outer batch spanning the complete crossed allocation"
            )
        for name, share in self._objective_law_shares.items():
            if float(share) > 0.0 and law_denominators.get(name, 0.0) <= 0.0:
                raise ValueError(
                    f"metadata law outer batch has no observed {name!r} rows but its "
                    "ObjectiveSpec share is positive; enlarge/stratify the outer batch"
                )

        root_dependency = self._root_parameter_dependency_count(
            lengths=lengths[idx],
            observed=root_observed[idx],
            train_f=train_f,
            train_g=train_g,
        )
        observed = design.observed.to(dtype=torch.bool)
        leaf_active = bool(
            (train_f or train_g)
            and float(self._objective_law_shares.get(LawKind.C1_LEAF.value, 0.0)) > 0.0
            and (observed & design.is_leaf).any()
        )
        merge_active = bool(
            (train_f or train_g)
            and float(self._objective_law_shares.get(LawKind.C3_MERGE.value, 0.0)) > 0.0
            and (observed & ~design.is_leaf).any()
        )
        if not (
            (root_dependency > 0 and float(self._objective.root_share) > 0.0)
            or leaf_active
            or merge_active
        ):
            # A side with no parameter-dependent root or law channel must not
            # receive AdamW decay/momentum updates.
            return 0.0

        opt.zero_grad(set_to_none=True)
        micro = max(1, int(self.config.micro_batch_size or 1))
        component_numerators: dict[str, float] = {
            "root": 0.0,
            LawKind.C1_LEAF.value: 0.0,
            LawKind.C3_MERGE.value: 0.0,
        }
        total = 0.0
        any_grad = False
        for start in range(0, int(idx.shape[0]), micro):
            cidx = idx[start : start + micro]
            if rollup_weights is not None:
                pred, traces = self._model.forward_rollup(
                    x[cidx], lengths[cidx], rollup_weights[cidx], collect_trace=True
                )
            else:
                pred, traces = self._model.forward_with_trace(x[cidx], lengths[cidx])
            chunk_indices = [int(value) for value in cidx.detach().cpu().tolist()]
            law_rows = metadata_law_state_rows(
                traces,
                [metadata_state_targets[value] for value in chunk_indices],
                training_loss=self.config.training_loss,
                torch=torch,
                device=self._device,
                dtype=pred.dtype,
            )
            if law_rows is None:
                raise ValueError("metadata law microbatch produced no trace rows")
            chunk_loss = pred.sum() * 0.0
            root_mask = root_observed[cidx].to(device=pred.device, dtype=torch.bool)
            if root_count > 0 and float(self._objective.root_share) > 0.0:
                root_rows = per_row_training_loss(
                    pred,
                    y[cidx],
                    training_loss=self.config.training_loss,
                )
                root_num = root_rows[root_mask].sum()
                chunk_loss = chunk_loss + (
                    float(self._objective.root_share) * root_num / float(root_count)
                )
                component_numerators["root"] += float(root_num.detach().cpu())
            for name, share in self._objective_law_shares.items():
                if float(share) <= 0.0:
                    continue
                mask = law_rows.is_leaf if name == LawKind.C1_LEAF.value else ~law_rows.is_leaf
                chunk_den = self._law_channel_denominator(law_rows, mask)
                if float(chunk_den.detach().cpu()) <= 0.0:
                    continue
                channel_mean = self._canonical_law_channel_mean(law_rows, mask)
                channel_num = channel_mean * chunk_den
                chunk_loss = chunk_loss + (
                    float(share) * channel_num / float(law_denominators[name])
                )
                component_numerators[name] += float(channel_num.detach().cpu())
            total += float(chunk_loss.detach().cpu())
            if bool(chunk_loss.requires_grad):
                any_grad = True
                chunk_loss.backward()
        if any_grad:
            clip = self.config.grad_clip_norm
            if clip is not None and float(clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(clip))
            opt.step()
            self._last_optimizer_step_count += 1

        root_value = component_numerators["root"] / float(root_count) if root_count > 0 else 0.0
        law_values = {
            name: (
                component_numerators[name] / float(law_denominators[name])
                if law_denominators.get(name, 0.0) > 0.0
                else 0.0
            )
            for name in (LawKind.C1_LEAF.value, LawKind.C3_MERGE.value)
        }
        combined = float(self._objective.root_share) * root_value + sum(
            float(share) * law_values[name] for name, share in self._objective_law_shares.items()
        )
        self._last_objective_components = {
            "scope": "pre_update_last_outer_batch",
            "combined": float(combined),
            "root": float(root_value),
            "c1": float(law_values[LawKind.C1_LEAF.value]),
            "c3": float(law_values[LawKind.C3_MERGE.value]),
            "root_rows": int(root_count),
            "c1_effective_weight": float(law_denominators[LawKind.C1_LEAF.value]),
            "c3_effective_weight": float(law_denominators[LawKind.C3_MERGE.value]),
            "micro_batch_size": int(micro),
            "outer_batch_size": int(idx.shape[0]),
            "exact_gradient_accumulation": True,
        }
        return float(total)

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
        root_observed: Any,
        train_f: bool,
        train_g: bool,
    ) -> float:
        """One optimizer step over ``idx`` in memory-bounded chunks — exact.

        The chunk losses are scaled so their SUM equals the full-batch loss
        (plain root point loss, or the weighted node-mean with its denominator
        computed over the whole batch up front), so accumulated gradients are
        identical to a single-pass batch and only peak activation memory
        changes.
        """

        torch = self._torch
        micro = max(1, int(self.config.micro_batch_size or 1))
        batch_n = int(idx.shape[0])
        root_w = float(self.config.root_weight)
        leaf_w = float(self.config.leaf_weight)
        merge_w = float(self.config.merge_weight)
        idx_list = idx.detach().cpu().tolist()
        batch_root_observed = root_observed[idx].to(dtype=torch.bool)
        root_count = int(batch_root_observed.sum().detach().cpu())

        if not self._micro_batch_depends_on_trainable_side(
            indices=idx_list,
            lengths=lengths,
            root_observed=root_observed,
            node_supervision=node_supervision,
            train_f=train_f,
            train_g=train_g,
        ):
            # A loss may still report ``requires_grad=True`` because trace
            # tensors concatenate states. Calling AdamW.step() without a
            # parameter-dependent channel would apply
            # decay/momentum without any supervision signal.
            return 0.0

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
            denom = root_w * float(root_count) + float(weight_sum.detach().cpu())
        elif node_supervision is None:
            denom = float(root_count)

        active_tree_count: int | None = None
        if node_supervision is not None and self.config.per_tree_loss_lambda is not None:
            active_tree_count = self._active_tree_count(
                idx_list,
                root_observed=root_observed,
                node_supervision=node_supervision,
            )

        if (active_tree_count is not None and active_tree_count <= 0) or (
            active_tree_count is None and float(denom or 0.0) <= 0.0
        ):
            return 0.0

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
                    root_observed=root_observed[cidx],
                    active_tree_count=int(active_tree_count or 1),
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
                # Numerator contribution: root_w * Σ_tree point loss +
                # Σ row_w · node loss, over this chunk only.
                per_root = per_row_training_loss(
                    pred,
                    y[cidx],
                    training_loss=self.config.training_loss,
                )
                chunk_root_mask = root_observed[cidx].to(device=pred.device, dtype=pred.dtype)
                num = root_w * (per_root * chunk_root_mask).sum()
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
                per_root = per_row_training_loss(
                    pred,
                    y[cidx],
                    training_loss=self.config.training_loss,
                )
                chunk_root_mask = root_observed[cidx].to(device=pred.device, dtype=pred.dtype)
                chunk_loss = (per_root * chunk_root_mask).sum() / float(denom or 1.0)
            total += float(chunk_loss.detach().cpu())
            if bool(chunk_loss.requires_grad):
                any_grad = True
                chunk_loss.backward()
        if any_grad:
            clip = self.config.grad_clip_norm
            if clip is not None and float(clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(params, float(clip))
            opt.step()
            self._last_optimizer_step_count += 1
        return total

    def _micro_batch_depends_on_trainable_side(
        self,
        *,
        indices: Sequence[int],
        lengths: Any,
        root_observed: Any,
        node_supervision: Sequence[tuple[Any, Any]] | None,
        train_f: bool,
        train_g: bool,
    ) -> bool:
        """Preflight one accumulated optimizer batch for parameter dependence.

        ``micro_batch_size`` changes only activation chunking: AdamW steps once
        per outer batch.  We therefore inspect the complete outer batch and
        skip it only when every weighted term is constant with respect to the
        side currently being trained.
        """

        torch = self._torch
        index_tensor = torch.tensor(
            [int(index) for index in indices], dtype=torch.long, device=lengths.device
        )
        selected_observed = root_observed[index_tensor]
        selected_lengths = lengths[index_tensor]
        if node_supervision is None:
            root_coefficient = 1.0
        elif self.config.per_tree_loss_lambda is not None:
            root_coefficient = 1.0 - float(self.config.per_tree_loss_lambda)
        else:
            root_coefficient = float(self.config.root_weight)
        if (
            root_coefficient > 0.0
            and self._root_parameter_dependency_count(
                lengths=selected_lengths,
                observed=selected_observed,
                train_f=train_f,
                train_g=train_g,
            )
            > 0
        ):
            return True
        if node_supervision is None:
            return False

        local_scale = (
            float(self.config.per_tree_loss_lambda)
            if self.config.per_tree_loss_lambda is not None
            else 1.0
        )
        if local_scale <= 0.0:
            return False
        leaf_weight = float(self.config.leaf_weight)
        merge_weight = float(self.config.merge_weight)
        for index in indices:
            _targets, mask = node_supervision[int(index)]
            n_nodes = int(mask.shape[0])
            leaf_count = (n_nodes + 1) // 2
            if (train_f or train_g) and leaf_weight > 0.0 and bool(mask[:leaf_count].any()):
                return True
            if (train_f or train_g) and merge_weight > 0.0 and bool(mask[leaf_count:].any()):
                return True
        return False

    def _per_tree_convex_loss(
        self,
        pred: Any,
        y: Any,
        traces: Sequence[Any],
        selected: Sequence[tuple[Any, Any]],
        *,
        root_observed: Any,
        active_tree_count: int,
    ) -> Any:
        """TT-ladder loss shape: mean over trees of a per-tree convex split.

        Per tree: ``(1-λ)·root_loss + λ·(leaf/merge-weighted node mean)``; a
        tree without observed node rows contributes ``(1-λ)·root_loss`` when
        its root is observed; a local-only tree retains its node term. Every
        tree with at least one active term weighs equally regardless of node
        count, so chunked accumulation is exact under any chunking.
        """

        torch = self._torch
        lam = float(self.config.per_tree_loss_lambda or 0.0)
        leaf_w = float(self.config.leaf_weight)
        merge_w = float(self.config.merge_weight)
        read = getattr(self._model, "_read", None) or self._model.readout
        total = None
        for i, (trace, (target, mask)) in enumerate(zip(traces, selected)):
            zero = pred[i].sum() * 0.0
            root_point_loss = per_row_training_loss(
                pred[i],
                y[i],
                training_loss=self.config.training_loss,
            )
            n_nodes = int(trace.shape[0])
            leaf_count = (n_nodes + 1) // 2
            preds = read(trace)
            losses = per_row_training_loss(
                preds,
                target.to(dtype=preds.dtype),
                training_loss=self.config.training_loss,
            )
            is_leaf = torch.arange(n_nodes, device=losses.device) < leaf_count
            row_w = torch.where(
                is_leaf,
                torch.full_like(losses, leaf_w),
                torch.full_like(losses, merge_w),
            )
            keep = mask & (row_w > 0.0)
            tree_loss = (1.0 - lam) * root_point_loss if bool(root_observed[i]) else zero
            if bool(keep.any()):
                node_mean = (row_w[keep] * losses[keep]).sum() / row_w[keep].sum()
                tree_loss = tree_loss + lam * node_mean
            total = tree_loss if total is None else total + tree_loss
        assert total is not None
        return total / float(max(1, int(active_tree_count)))

    def _active_tree_count(
        self,
        indices: Sequence[int],
        *,
        root_observed: Any,
        node_supervision: Sequence[tuple[Any, Any]],
    ) -> int:
        """Count trees contributing a nonzero root or local mixture term."""

        lam = float(self.config.per_tree_loss_lambda or 0.0)
        leaf_w = float(self.config.leaf_weight)
        merge_w = float(self.config.merge_weight)
        active = 0
        for index in indices:
            root_active = bool(root_observed[int(index)]) and (1.0 - lam) > 0.0
            _target, mask = node_supervision[int(index)]
            n_nodes = int(mask.shape[0])
            leaf_count = (n_nodes + 1) // 2
            leaf_seen = lam > 0.0 and bool(mask[:leaf_count].any()) and leaf_w > 0.0
            merge_seen = lam > 0.0 and bool(mask[leaf_count:].any()) and merge_w > 0.0
            active += int(root_active or leaf_seen or merge_seen)
        return int(active)

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

        if self._objective_law_shares and self.config.law_state_target_key:
            # The exact witness lives in hidden trace coordinates and has its
            # own width. Do not also read scalar node labels through the task
            # head merely because the ObjectiveSpec activates C1/C3.
            self._node_supervision_counts = None
            return None
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
                # A singleton has no distinct non-root node: its leaf-built
                # state is also its root state. The root term can anchor
                # f(g(X)), while an explicit C1 row can additionally supervise
                # g's leaf role. An explicit empty unit mask is also
                # the intentional L=0 crossed-budget cell; it falls back to
                # observed roots instead of being mistaken for missing data.
                # Otherwise error when nodes exist (or nothing anchors the
                # loss at all).
                from treepo.tree import tree_leaves

                has_supervisable_nodes = any(
                    len(tuple(tree_leaves(tree) or ())) >= 2 for tree in trees
                )
                configured_units = getattr(self.config, "supervised_node_units", None)
                intentional_zero_budget = configured_units is not None and not tuple(
                    configured_units
                )
                observed_root_anchor = (
                    float(self.config.root_weight) > 0.0 and int(self._root_supervision_count) > 0
                )
                if not (intentional_zero_budget and observed_root_anchor) and (
                    has_supervisable_nodes or not observed_root_anchor
                ):
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
        losses = per_row_training_loss(
            preds,
            targets,
            training_loss=self.config.training_loss,
        )
        depths = torch.cat(depth_chunks)
        is_leaf = torch.cat(leaf_chunks)
        return losses[mask], depths[mask], is_leaf[mask]

    def _masked_root_loss(self, predictions: Any, targets: Any, observed: Any) -> tuple[Any, int]:
        """Mean configured root point loss over observed roots, plus its row count."""

        mask = observed.to(device=predictions.device, dtype=self._torch.bool).reshape(-1)
        count = int(mask.sum().detach().cpu())
        if count <= 0:
            return predictions.sum() * 0.0, 0
        per_root = per_row_training_loss(
            predictions,
            targets,
            training_loss=self.config.training_loss,
        )
        return per_root[mask].mean(), count

    def _root_parameter_dependency_count(
        self,
        *,
        lengths: Any,
        observed: Any,
        train_f: bool,
        train_g: bool,
    ) -> int:
        """Count observed roots that depend on the side being optimized.

        Every root depends on f's readout and on g's leaf-build role, including
        a singleton tree. Recursive trees additionally exercise g's merge role.
        Constant loss must not authorize an optimizer step.
        """

        mask = observed.to(device=lengths.device, dtype=self._torch.bool).reshape(-1)
        if train_f:
            return int(mask.sum().detach().cpu())
        if train_g:
            return int(mask.sum().detach().cpu())
        return 0

    def _node_weighted_loss(self, root_loss: Any, node_rows: Any | None, *, root_count: int) -> Any:
        """Weighted node-mean loss: the TT ladder's root/leaf/merge weighting.

        Every supervised row — each batch item's root plus every observed
        leaf/merge node — enters one weighted mean under the config's
        ``root_weight`` / ``leaf_weight`` / ``merge_weight``. With
        leaf/merge weights at 0 this reduces exactly to the historical root
        root point loss; convexity holds because weights normalize to one.
        """

        torch = self._torch
        root_w = float(self.config.root_weight)
        leaf_w = float(self.config.leaf_weight)
        merge_w = float(self.config.merge_weight)
        root_num = root_w * float(root_count) * root_loss
        root_den = root_w * float(root_count)
        if node_rows is None:
            return root_loss if root_den > 0.0 else root_loss * 0.0
        losses, _depths, is_leaf = node_rows
        row_weights = torch.where(
            is_leaf,
            torch.full_like(losses, leaf_w),
            torch.full_like(losses, merge_w),
        )
        keep = row_weights > 0.0
        node_num = (row_weights * losses * keep.to(losses.dtype)).sum()
        node_den = (row_weights * keep.to(losses.dtype)).sum()
        denom = root_den + node_den
        if float(denom.detach().cpu()) <= 0.0:
            return root_loss * 0.0
        return (root_num + node_num) / denom

    def _node_weighted_batch_is_active(
        self,
        *,
        root_dependency_count: int,
        node_rows: Any | None,
        train_f: bool,
        train_g: bool,
    ) -> bool:
        """Whether a pooled root/node batch can update the trainable side."""

        if int(root_dependency_count) > 0 and float(self.config.root_weight) > 0.0:
            return True
        if node_rows is None:
            return False
        _losses, _depths, is_leaf = node_rows
        leaf_active = (
            bool(train_f or train_g)
            and float(self.config.leaf_weight) > 0.0
            and bool(is_leaf.any())
        )
        merge_active = (
            bool(train_f or train_g)
            and float(self.config.merge_weight) > 0.0
            and bool((~is_leaf).any())
        )
        return bool(leaf_active or merge_active)

    def _assembled_batch_is_active(
        self,
        *,
        root_dependency_count: int,
        law_rows: Any | None,
        train_f: bool,
        train_g: bool,
    ) -> bool:
        """Whether the root/law objective can update the trainable side."""

        if self._objective is None:
            root_active = int(root_dependency_count) > 0
            law_active = False
            if law_rows is not None and float(self.config.numeric_transition_state_weight) > 0.0:
                rows = self._coerce_law_loss_rows(law_rows)
                assert rows is not None
                seen = rows.observed.to(dtype=self._torch.bool)
                law_active = bool((train_f or train_g) and (seen & rows.is_leaf).any()) or bool(
                    (train_f or train_g) and (seen & ~rows.is_leaf).any()
                )
            return bool(root_active or law_active)
        root_active = int(root_dependency_count) > 0 and float(self._objective.root_share) > 0.0
        law_active = False
        if law_rows is not None and self._objective_law_shares:
            rows = self._coerce_law_loss_rows(law_rows)
            assert rows is not None
            seen = rows.observed.to(dtype=self._torch.bool)
            leaf_share = float(self._objective_law_shares.get(LawKind.C1_LEAF.value, 0.0))
            merge_share = float(self._objective_law_shares.get(LawKind.C3_MERGE.value, 0.0))
            law_active = bool(
                (train_f or train_g) and leaf_share > 0.0 and (seen & rows.is_leaf).any()
            ) or bool((train_f or train_g) and merge_share > 0.0 and (seen & ~rows.is_leaf).any())
        return bool(root_active or law_active)

    def _assemble_loss(self, root_loss: Any, law_rows: Any | None) -> Any:
        """Combine root and law terms under the executed objective.

        ObjectiveSpec path: ``root_share * root + sum_c share_c * law_c`` with
        each channel routed through the canonical depth-discounted objective.
        Legacy path: ``root + numeric_transition_state_weight * node_mean`` —
        the law term is the unweighted node mean, the sampling contract's
        audit estimand.
        """
        if self._objective is None:
            if law_rows is None:
                return root_loss
            rows = self._coerce_law_loss_rows(law_rows)
            assert rows is not None
            proxy = rows.losses
            return root_loss + float(self.config.numeric_transition_state_weight) * proxy.mean()
        loss = float(self._objective.root_share) * root_loss
        if not self._objective_law_shares:
            self._last_objective_components = {
                "scope": "pre_update_last_outer_batch",
                "combined": float(loss.detach().cpu()),
                "root": float(root_loss.detach().cpu()),
                "c1": None,
                "c3": None,
            }
            return loss
        if law_rows is None:
            raise ValueError(
                f"family={self.name!r} objective declares a local-law weight but "
                "the training batch produced no law rows"
            )
        rows = self._coerce_law_loss_rows(law_rows)
        assert rows is not None
        channel_masks = {
            LawKind.C1_LEAF.value: rows.is_leaf,
            LawKind.C3_MERGE.value: ~rows.is_leaf,
        }
        component_values: dict[str, float] = {}
        for name, share in self._objective_law_shares.items():
            mask = channel_masks[name]
            denominator = self._law_channel_denominator(rows, mask)
            if float(denominator.detach().cpu()) <= 0.0:
                raise ValueError(
                    f"family={self.name!r} objective weights law channel {name!r} "
                    "but the training batch has no observed effective rows; use "
                    "exact outer-batch microaccumulation, enlarge/stratify the "
                    "batch, or set its component weight to 0"
                )
            channel = self._canonical_law_channel_mean(rows, mask)
            loss = loss + float(share) * channel
            component_values[name] = float(channel.detach().cpu())
        self._last_objective_components = {
            "scope": "pre_update_last_outer_batch",
            "combined": float(loss.detach().cpu()),
            "root": float(root_loss.detach().cpu()),
            "c1": component_values.get(LawKind.C1_LEAF.value),
            "c3": component_values.get(LawKind.C3_MERGE.value),
            "exact_gradient_accumulation": False,
        }
        return loss

    @staticmethod
    def _coerce_law_loss_rows(law_rows: Any | None) -> LawLossRows | None:
        if law_rows is None or isinstance(law_rows, LawLossRows):
            return law_rows
        return dense_law_loss_rows(law_rows)

    def _law_effective_weights(self, rows: LawLossRows) -> Any:
        torch = self._torch
        depths = rows.depths.to(device=rows.losses.device, dtype=torch.float32)
        gamma = torch.full_like(depths, float(self._objective_gamma_depth))
        weights = torch.pow(gamma, depths).to(dtype=rows.losses.dtype)
        weights = weights * rows.node_weights.to(device=weights.device, dtype=weights.dtype)
        if str(rows.objective_mode) == "sampled_ipw":
            observed = rows.observed.to(device=weights.device, dtype=weights.dtype)
            propensity = rows.propensity.to(device=weights.device, dtype=weights.dtype).clamp(
                min=1.0e-12, max=1.0
            )
            weights = weights * observed / propensity
        return weights

    def _law_channel_denominator(self, rows: LawLossRows, mask: Any) -> Any:
        weights = self._law_effective_weights(rows)
        return weights[mask.to(device=weights.device, dtype=self._torch.bool)].sum()

    def _canonical_law_channel_mean(self, rows: LawLossRows, mask: Any) -> Any:
        # Imported lazily to keep ``import treepo.methods.fno`` torch-light.
        from treepo.training.local_law import local_law_objective_from_losses

        keep = mask.to(device=rows.losses.device, dtype=self._torch.bool)
        losses = rows.losses[keep]
        return local_law_objective_from_losses(
            proxy_loss=losses,
            oracle_loss=losses,
            observed=rows.observed[keep],
            propensity=rows.propensity[keep],
            depths=rows.depths[keep],
            node_weights=rows.node_weights[keep],
            gamma_depth=float(self._objective_gamma_depth),
            objective_mode=str(rows.objective_mode),
        )

    def _g_merge_domain_gradient_path_present(
        self,
        *,
        lengths: Any,
        root_observed: Any,
        node_supervision: Sequence[tuple[Any, Any]] | None,
        law_targets: Sequence[Any] | None,
        node_weighted: bool,
        train_g: bool,
    ) -> bool:
        """Whether the loss graph reaches shared g through a merge call.

        A recursive root loss qualifies because it backpropagates through merge
        calls, but it is not thereby a C3 supervision row or C3 evidence.
        """

        if not train_g:
            return False
        recursive = lengths.to(dtype=self._torch.long).reshape(-1) > 1
        if not bool(recursive.any().detach().cpu()):
            return False
        root_uses_merge = str(self.config.root_readout) != "leaf_mean"
        recursive_root = bool(
            (root_observed.to(dtype=self._torch.bool).reshape(-1) & recursive).any().detach().cpu()
        )
        if node_weighted:
            root_active = (
                root_uses_merge and float(self.config.root_weight) > 0.0 and recursive_root
            )
            merge_active = float(
                self.config.merge_weight
            ) > 0.0 and self._has_observed_merge_targets(node_supervision)
        elif self._objective is None:
            root_active = root_uses_merge and recursive_root
            merge_active = float(
                self.config.numeric_transition_state_weight
            ) > 0.0 and self._has_observed_merge_targets(law_targets)
        else:
            root_active = (
                root_uses_merge and float(self._objective.root_share) > 0.0 and recursive_root
            )
            merge_active = float(
                self._objective_law_shares.get(LawKind.C3_MERGE.value, 0.0)
            ) > 0.0 and self._has_observed_merge_targets(law_targets)
        return bool(root_active or merge_active)

    def _has_observed_merge_targets(self, rows: Sequence[Any] | None) -> bool:
        if not rows:
            return False
        for row in rows:
            observed = getattr(row, "observed", None)
            if observed is None and isinstance(row, (tuple, list)) and len(row) >= 2:
                observed = row[1]
            if observed is None:
                continue
            count = int(observed.shape[0]) if hasattr(observed, "shape") else len(observed)
            leaf_count = (count + 1) // 2
            tail = observed[leaf_count:]
            if hasattr(tail, "any"):
                if bool(tail.any().detach().cpu()):
                    return True
            elif any(bool(value) for value in tail):
                return True
        return False

    def _set_trainable(self, *, train_f: bool, train_g: bool) -> None:
        """Train the paper-aligned f/readout or g/state-operator side.

        ``f`` owns only the state readout. ``g`` is one registered module
        reused at leaves and merges; there are no role-specific parameter sets
        to freeze. A singleton g update therefore changes the same parameters
        that a future recursive reduction will call.
        """

        assert self._model is not None
        for param in self._model.parameters():
            param.requires_grad = False
        f_names = getattr(self._model, "f_module_names", ("readout",))
        g_names = getattr(self._model, "g_module_names", ("g",))
        names: list[str] = []
        if train_f:
            names.extend(f_names)
        if train_g:
            names.extend(g_names)
        if self.config.law_state_root_readout == "first_coordinate":
            names = [name for name in names if name != "readout"]
        for name in names:
            module = getattr(self._model, name)
            for param in module.parameters():
                param.requires_grad = True

    def _encode_trees(self, trees: Sequence[Any]) -> tuple[Any, Any]:
        cache_key = _tree_sequence_cache_key(
            trees, dim=int(self.config.embedding_dim), device=str(self._device)
        )
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
