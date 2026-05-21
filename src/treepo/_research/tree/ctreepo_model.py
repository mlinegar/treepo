"""
CTreePO model: learned mergeable sketches over multilingual embeddings.

Learns a compact sketch (default 32-dim) from Qwen3-Embedding-8B vectors
(4096-dim) that captures political position. The sketch merges bottom-up
through the document's tree structure (same topology as the text merge) and
predicts target scores (RILE, etc.) via linear readout heads.

Architecture:
    LeafProjector:  embedding (4096) -> sketch (d)
    GatedMerge:     (sketch_L, sketch_R) -> sketch_parent
    ReadoutHead:    sketch -> scalar score

The GatedMerge uses a soft gate + residual, making it approximately
associative when the gate is near 0.5. Any regularizers in this module are
proxy-only heuristics; they are not Lean local-law certificates.

Requires: torch >= 2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    raise ImportError(
        "PyTorch is required for CTreePO model. "
        "Install with: pip install torch>=2.0.0"
    )

from treepo._research.core.ops_checks import (
    EvidenceStatus,
    LawCapabilityReport,
    LawKind,
    OperatorCapabilityReport,
)
from treepo._research.tree.tree_model_v2 import (
    TREE_MODEL_VERSION_LEGACY,
    TREE_MODEL_VERSION_V2,
    TreeModelV2View,
    normalize_tree_model_version,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CTreePOConfig:
    """Architecture and training hyperparameters for CTreePO."""

    embedding_dim: int = 4096
    sketch_dim: int = 32
    hidden_dim: int = 256
    merge_type: str = "gated"  # "gated" | "mlp" | "avg"
    head_names: tuple = ("rile",)

    # Target scale: RILE is [-100, +100], internally normalized to [0, 1].
    target_min: float = -100.0
    target_max: float = 100.0

    # Score-fiber factorization (optional).  When set, the model gains a phi
    # projector mapping sketch → [score | fiber | aux].
    phi_dim: int = 0            # 0 = disabled
    phi_score_dim: int = 1
    phi_fiber_dim: int = 0      # computed as phi_dim - phi_score_dim if 0
    phi_hidden_dim: int = 128
    tree_model_version: str = "legacy"


DEFAULT_CTREEPO_TRAINER_MODEL_VERSION: str = TREE_MODEL_VERSION_V2


def infer_ctreepo_tree_model_version_from_state_dict(
    state_dict: Mapping[str, Any] | None,
) -> str:
    """Infer the tree model version from checkpoint parameter names."""
    if not state_dict:
        return DEFAULT_CTREEPO_TRAINER_MODEL_VERSION
    if any(str(key).startswith("phi_projector.") for key in state_dict):
        return TREE_MODEL_VERSION_V2
    return TREE_MODEL_VERSION_LEGACY


def _infer_ctreepo_architecture_from_state_dict(
    state_dict: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    if not state_dict:
        return {}
    inferred: Dict[str, Any] = {}
    leaf_in = state_dict.get("leaf_projector.net.0.weight")
    leaf_out = state_dict.get("leaf_projector.net.3.weight")
    if isinstance(leaf_in, torch.Tensor):
        inferred["embedding_dim"] = int(leaf_in.shape[1])
        inferred["hidden_dim"] = int(leaf_in.shape[0])
    if isinstance(leaf_out, torch.Tensor):
        inferred["sketch_dim"] = int(leaf_out.shape[0])

    if "merge_module.cross.weight" in state_dict or "merge_module.fuse.weight" in state_dict:
        inferred["merge_type"] = "bilinear"
    elif "merge_module.residual.0.weight" in state_dict and "merge_module.norm.weight" in state_dict:
        inferred["merge_type"] = "residual_gated"
    elif "merge_module.net.0.weight" in state_dict:
        inferred["merge_type"] = "mlp"
    elif "merge_module.gate.weight" in state_dict:
        inferred["merge_type"] = "gated"
    else:
        inferred["merge_type"] = "avg"

    head_names = sorted(
        {
            str(key).split(".", 2)[1]
            for key in state_dict
            if str(key).startswith("heads.") and str(key).count(".") >= 2
        }
    )
    if head_names:
        inferred["head_names"] = tuple(head_names)

    phi_hidden = state_dict.get("phi_projector.net.1.weight")
    phi_out = state_dict.get("phi_projector.net.3.weight")
    if isinstance(phi_out, torch.Tensor):
        inferred["phi_dim"] = int(phi_out.shape[0])
        inferred["phi_score_dim"] = 1
        inferred["phi_fiber_dim"] = max(0, int(phi_out.shape[0]) - 1)
    if isinstance(phi_hidden, torch.Tensor):
        inferred["phi_hidden_dim"] = int(phi_hidden.shape[0])
    return inferred


def ctreepo_config_from_mapping(
    payload: Mapping[str, Any] | None = None,
    *,
    embedding_dim: int | None = None,
    checkpoint_state_dict: Mapping[str, Any] | None = None,
    tree_model_version: str | None = None,
) -> CTreePOConfig:
    """Build a normalized config for the centralized CTreePO trainer surface."""
    mapping = dict(payload or {})
    inferred = _infer_ctreepo_architecture_from_state_dict(checkpoint_state_dict)
    resolved_tree_model_version = normalize_tree_model_version(
        tree_model_version
        or mapping.get("tree_model_version")
        or infer_ctreepo_tree_model_version_from_state_dict(checkpoint_state_dict)
        or DEFAULT_CTREEPO_TRAINER_MODEL_VERSION
    )
    return CTreePOConfig(
        embedding_dim=int(
            embedding_dim
            if embedding_dim is not None
            else mapping.get(
                "embedding_dim",
                inferred.get("embedding_dim", CTreePOConfig.embedding_dim),
            )
        ),
        sketch_dim=int(mapping.get("sketch_dim", inferred.get("sketch_dim", CTreePOConfig.sketch_dim))),
        hidden_dim=int(mapping.get("hidden_dim", inferred.get("hidden_dim", CTreePOConfig.hidden_dim))),
        merge_type=str(mapping.get("merge_type", inferred.get("merge_type", CTreePOConfig.merge_type))),
        head_names=tuple(mapping.get("head_names", inferred.get("head_names", CTreePOConfig.head_names))),
        target_min=float(mapping.get("target_min", CTreePOConfig.target_min)),
        target_max=float(mapping.get("target_max", CTreePOConfig.target_max)),
        phi_dim=int(mapping.get("phi_dim", inferred.get("phi_dim", CTreePOConfig.phi_dim))),
        phi_score_dim=int(
            mapping.get("phi_score_dim", inferred.get("phi_score_dim", CTreePOConfig.phi_score_dim))
        ),
        phi_fiber_dim=int(
            mapping.get("phi_fiber_dim", inferred.get("phi_fiber_dim", CTreePOConfig.phi_fiber_dim))
        ),
        phi_hidden_dim=int(
            mapping.get("phi_hidden_dim", inferred.get("phi_hidden_dim", CTreePOConfig.phi_hidden_dim))
        ),
        tree_model_version=str(resolved_tree_model_version),
    )


def load_ctreepo_model_checkpoint(
    checkpoint_path: Any,
    *,
    config_overrides: Mapping[str, Any] | None = None,
    map_location: Any = "cpu",
    embedding_dim: int | None = None,
    tree_model_version: str | None = None,
    strict: bool = True,
) -> Tuple["CTreePOModel", CTreePOConfig]:
    """Load a checkpoint with centralized config/version resolution."""
    state_dict = torch.load(checkpoint_path, map_location=map_location, weights_only=True)
    if not isinstance(state_dict, Mapping):
        raise TypeError(
            f"expected checkpoint state_dict mapping, got {type(state_dict).__name__}"
        )
    config = ctreepo_config_from_mapping(
        config_overrides,
        embedding_dim=embedding_dim,
        checkpoint_state_dict=state_dict,
        tree_model_version=tree_model_version,
    )
    model = CTreePOModel(config)
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    return model, config


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------


class LeafProjector(nn.Module):
    """Projects high-dim embeddings to compact sketch space.

    Linear(embedding_dim, hidden_dim) -> LayerNorm -> ReLU -> Linear(hidden_dim, sketch_dim)
    """

    def __init__(self, embedding_dim: int, sketch_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sketch_dim),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        return self.net(embedding)


class GatedMerge(nn.Module):
    """Gated combination of two child sketches.

    gate = sigmoid(W_gate @ [left; right] + b_gate)
    merged = gate * left + (1 - gate) * right + residual([left; right])

    The gate provides soft attention to whichever child carries more signal.
    The residual connection allows learning corrections beyond weighted average.
    Approximately associative when gate ~ 0.5.
    """

    def __init__(self, sketch_dim: int):
        super().__init__()
        self.gate = nn.Linear(2 * sketch_dim, sketch_dim)
        self.residual = nn.Linear(2 * sketch_dim, sketch_dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([left, right], dim=-1)
        g = torch.sigmoid(self.gate(cat))
        return g * left + (1 - g) * right + self.residual(cat)


class MLPMerge(nn.Module):
    """MLP merge (same structure as learned_sketch.py).

    cat(left, right) -> Linear(2d, hidden) -> ReLU -> Linear(hidden, d)
    """

    def __init__(self, sketch_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * sketch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sketch_dim),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([left, right], dim=-1))


class AvgMerge(nn.Module):
    """Simple average merge (baseline)."""

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return 0.5 * (left + right)


class ResidualGatedMerge(nn.Module):
    """More expressive gated merge with residual MLP correction."""

    def __init__(self, sketch_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.gate = nn.Linear(2 * sketch_dim, sketch_dim)
        self.residual = nn.Sequential(
            nn.Linear(2 * sketch_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, sketch_dim),
        )
        self.norm = nn.LayerNorm(sketch_dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        cat = torch.cat([left, right], dim=-1)
        g = torch.sigmoid(self.gate(cat))
        mixed = g * left + (1 - g) * right
        return self.norm(mixed + self.residual(cat))


class BilinearMerge(nn.Module):
    """Merge with explicit pairwise interactions between child sketches."""

    def __init__(self, sketch_dim: int):
        super().__init__()
        self.cross = nn.Bilinear(sketch_dim, sketch_dim, sketch_dim)
        self.fuse = nn.Linear(3 * sketch_dim, sketch_dim)
        self.norm = nn.LayerNorm(sketch_dim)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        cross = self.cross(left, right)
        cat = torch.cat([left, right, cross], dim=-1)
        return self.norm(self.fuse(cat))


class ReadoutHead(nn.Module):
    """Linear readout from sketch to scalar, mapped to target scale."""

    def __init__(self, sketch_dim: int, target_min: float = -100.0, target_max: float = 100.0):
        super().__init__()
        self.linear = nn.Linear(sketch_dim, 1)
        self.target_min = target_min
        self.target_max = target_max

    def forward(self, sketch: torch.Tensor) -> torch.Tensor:
        """Returns scalar prediction on [target_min, target_max] scale."""
        raw = torch.sigmoid(self.linear(sketch))  # [0, 1]
        return self.target_min + raw * (self.target_max - self.target_min)

    def forward_normalized(self, sketch: torch.Tensor) -> torch.Tensor:
        """Returns prediction on [0, 1] scale (for loss computation)."""
        return torch.sigmoid(self.linear(sketch))


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


_MERGE_CLASSES = {
    "gated": GatedMerge,
    "mlp": MLPMerge,
    "avg": AvgMerge,
    "residual_gated": ResidualGatedMerge,
    "bilinear": BilinearMerge,
}


class CTreePOModel(nn.Module):
    """Complete proxy-only CTreePO model: project -> merge -> readout.

    Usage:
        model = CTreePOModel(CTreePOConfig())
        leaf_sketch = model.encode_leaf(embedding_tensor)
        parent_sketch = model.merge(left_sketch, right_sketch)
        rile_pred = model.predict(root_sketch, "rile")
    """

    def __init__(self, config: CTreePOConfig):
        super().__init__()
        self.config = config
        self.evidence_status = EvidenceStatus.PROXY_ONLY
        self.default_head = str(config.head_names[0] if config.head_names else "rile")
        self.tree_model_version = normalize_tree_model_version(
            getattr(config, "tree_model_version", "legacy")
        )

        self.leaf_projector = LeafProjector(
            config.embedding_dim, config.sketch_dim, config.hidden_dim
        )

        merge_cls = _MERGE_CLASSES.get(config.merge_type)
        if merge_cls is None:
            raise ValueError(
                f"Unknown merge_type={config.merge_type!r}, "
                f"expected one of {list(_MERGE_CLASSES.keys())}"
            )
        if config.merge_type == "avg":
            self.merge_module = merge_cls()
        elif config.merge_type == "mlp":
            self.merge_module = merge_cls(
                config.sketch_dim,
                hidden_dim=max(64, int(config.hidden_dim // 2)),
            )
        elif config.merge_type == "residual_gated":
            self.merge_module = merge_cls(
                config.sketch_dim,
                hidden_dim=max(64, int(config.hidden_dim)),
            )
        else:
            self.merge_module = merge_cls(config.sketch_dim)

        self.heads = nn.ModuleDict()
        for name in config.head_names:
            self.heads[name] = ReadoutHead(
                config.sketch_dim, config.target_min, config.target_max
            )

        # Optional phi projector for score-fiber factorization.
        self.phi_projector: Optional[nn.Module] = None
        effective_phi_dim = int(config.phi_dim)
        if self.tree_model_version == TREE_MODEL_VERSION_V2 and effective_phi_dim <= 0:
            effective_phi_dim = max(16, int(config.sketch_dim))
        if effective_phi_dim > 0:
            from treepo._research.tree.core_model import PhiProjector, ScoreFiberConfig

            score_dim = max(1, min(int(config.phi_score_dim), int(effective_phi_dim)))
            remaining_dim = max(0, int(effective_phi_dim) - int(score_dim))
            fiber_dim = (
                min(int(config.phi_fiber_dim), remaining_dim)
                if int(config.phi_fiber_dim) > 0
                else remaining_dim
            )
            phi_cfg = ScoreFiberConfig(
                phi_dim=int(effective_phi_dim),
                score_dim=int(score_dim),
                fiber_dim=fiber_dim,
                aux_dim=int(effective_phi_dim) - int(score_dim) - int(fiber_dim),
                hidden_dim=config.phi_hidden_dim,
            )
            self.phi_projector = PhiProjector(config.sketch_dim, phi_cfg)

    @property
    def state_dim(self) -> int:
        return int(self.config.sketch_dim)

    @property
    def leaf_state_dim(self) -> int:
        return int(self.config.sketch_dim)

    @property
    def uses_tree_model_v2(self) -> bool:
        return self.tree_model_version == TREE_MODEL_VERSION_V2

    def encode_leaf(self, embedding: torch.Tensor) -> torch.Tensor:
        """Project an embedding vector to sketch space."""
        return self.leaf_projector(embedding)

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Merge two child sketches into a parent sketch."""
        return self.merge_module(left, right)

    def predict(self, sketch: torch.Tensor, head: str = "rile") -> torch.Tensor:
        """Predict target score on original scale from sketch."""
        return self.heads[head](sketch)

    def predict_normalized(self, sketch: torch.Tensor, head: str = "rile") -> torch.Tensor:
        """Predict target score on [0, 1] scale from sketch."""
        return self.heads[head].forward_normalized(sketch)

    def predict_confidence(self, sketch: torch.Tensor, head: str = "rile") -> torch.Tensor:
        """Confidence proxy in [0,1] based on distance from center uncertainty band."""
        pred_norm = self.predict_normalized(sketch, head=head)
        return 1.0 - 2.0 * torch.abs(pred_norm - 0.5)

    def predict_interval(
        self,
        sketch: torch.Tensor,
        head: str = "rile",
        *,
        z_score: float = 1.96,
        min_std: float = 0.5,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mean, lower, upper, std) on target scale.

        Uncertainty is a heuristic interval derived from Bernoulli variance on
        normalized predictions. This gives a conservative, architecture-agnostic
        uncertainty proxy without changing checkpoint format.
        """
        mean = self.predict(sketch, head=head)
        pred_norm = torch.clamp(self.predict_normalized(sketch, head=head), min=1e-6, max=1.0 - 1e-6)
        span = float(self.config.target_max - self.config.target_min)
        std_norm = torch.sqrt(pred_norm * (1.0 - pred_norm))
        std = torch.clamp(std_norm * span, min=float(min_std))
        lower = torch.clamp(mean - float(z_score) * std, min=self.config.target_min, max=self.config.target_max)
        upper = torch.clamp(mean + float(z_score) * std, min=self.config.target_min, max=self.config.target_max)
        return mean, lower, upper, std

    # ------------------------------------------------------------------
    # Batched variants — same underlying modules, explicit batch contract
    # ------------------------------------------------------------------

    def encode_leaf_batch(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Encode a batch of leaf embeddings.

        Args:
            embeddings: ``(N, embedding_dim)`` tensor of leaf embeddings.

        Returns:
            ``(N, sketch_dim)`` tensor of leaf sketches.
        """
        return self.leaf_projector(embeddings)

    def encode_leaves(
        self,
        *,
        embeddings: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[Sequence[int]]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if token_ids is not None:
            raise ValueError("CTreePOModel uses embedding leaves, not token-id leaves")
        if embeddings is None:
            raise ValueError("CTreePOModel requires embeddings")
        if device is not None:
            embeddings = embeddings.to(device)
        return self.encode_leaf_batch(embeddings)

    def merge_batch(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Merge batched child sketches.

        Args:
            left: ``(N, sketch_dim)`` left child sketches.
            right: ``(N, sketch_dim)`` right child sketches.

        Returns:
            ``(N, sketch_dim)`` merged parent sketches.
        """
        return self.merge_module(left, right)

    def predict_batch(self, sketches: torch.Tensor, head: str = "rile") -> torch.Tensor:
        """Predict target scores from a batch of sketches.

        Args:
            sketches: ``(N, sketch_dim)`` tensor.
            head: Which readout head to use.

        Returns:
            ``(N, 1)`` predictions on ``[target_min, target_max]`` scale.
        """
        return self.heads[head](sketches)

    def predict_normalized_batch(self, sketches: torch.Tensor, head: str = "rile") -> torch.Tensor:
        """Predict normalized scores from a batch of sketches.

        Args:
            sketches: ``(N, sketch_dim)`` tensor.
            head: Which readout head to use.

        Returns:
            ``(N, 1)`` predictions on ``[0, 1]`` scale.
        """
        return self.heads[head].forward_normalized(sketches)

    def predict_confidence_batch(self, sketches: torch.Tensor, head: str = "rile") -> torch.Tensor:
        """Confidence proxy for a batch of sketches.

        Args:
            sketches: ``(N, sketch_dim)`` tensor.
            head: Which readout head to use.

        Returns:
            ``(N, 1)`` confidence values in ``[0, 1]``.
        """
        pred_norm = self.predict_normalized_batch(sketches, head=head)
        return 1.0 - 2.0 * torch.abs(pred_norm - 0.5)

    # ------------------------------------------------------------------
    # Phi / score-fiber methods (active when phi_dim > 0 in config)
    # ------------------------------------------------------------------

    def phi(self, sketch: torch.Tensor) -> Optional[torch.Tensor]:
        """Project sketch to theorem features, or None if phi not configured."""
        if self.phi_projector is None:
            return None
        return self.phi_projector(sketch)

    def phi_batch(self, sketches: torch.Tensor) -> Optional[torch.Tensor]:
        """Batched phi projection. ``(N, sketch_dim) → (N, phi_dim)``."""
        if self.phi_projector is None:
            return None
        return self.phi_projector(sketches)

    def phi_score(self, sketch: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract score slice from phi, or None."""
        if self.phi_projector is None:
            return None
        return self.phi_projector.score(self.phi_projector(sketch))

    def phi_fiber(self, sketch: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract fiber slice from phi, or None."""
        if self.phi_projector is None:
            return None
        return self.phi_projector.fiber(self.phi_projector(sketch))

    @property
    def has_phi(self) -> bool:
        """Whether this model has a phi projector configured."""
        return self.phi_projector is not None

    def as_tree_model_v2(self) -> TreeModelV2View:
        """Expose the shared V2 tree-model surface without changing checkpoints."""
        return TreeModelV2View(
            self,
            leaf_input_kind="embeddings",
            tree_model_version=self.tree_model_version,
        )

    def capability_report(self) -> OperatorCapabilityReport:
        """Structured local-law capability report for the core architecture."""
        return OperatorCapabilityReport(
            operator_name="ctreepo",
            evidence_status=self.evidence_status,
            latent_mergeability_enforced=False,
            tree_nesting_supported=True,
            theorem_domain_decode_available=False,
            theorem_domain_reencode_available=False,
            exact_reduction_supported=False,
            leaf_law=LawCapabilityReport(
                law_kind=LawKind.L1_LEAF,
                available=False,
                evidence_status=self.evidence_status,
                exact=False,
                notes="No theorem-domain summary/decode path is exposed by the base model.",
            ),
            merge_law=LawCapabilityReport(
                law_kind=LawKind.L2_MERGE,
                available=False,
                evidence_status=self.evidence_status,
                exact=False,
                notes="Recursive latent merges are supported, but not certified against theorem-domain spans.",
            ),
            idempotence_law=LawCapabilityReport(
                law_kind=LawKind.L3_IDEMPOTENCE,
                available=False,
                evidence_status=self.evidence_status,
                exact=False,
                notes="The base architecture has no decode/re-encode loop for theorem-backed re-summary checks.",
            ),
            notes=(
                "Tree nesting is available in latent space through encode_leaf/merge.",
                "Regularizers in this module remain proxy-only until a supplied theorem-domain codec/certificate is attached.",
            ),
        )

    def local_law_capabilities(self) -> Dict[str, object]:
        """Backwards-compatible dictionary view of the capability report."""
        return self.capability_report().to_dict()


# ---------------------------------------------------------------------------
# Loss utilities
# ---------------------------------------------------------------------------


def normalize_target(value: float, target_min: float = -100.0, target_max: float = 100.0) -> float:
    """Map target value from [target_min, target_max] to [0, 1]."""
    span = target_max - target_min
    if span == 0:
        return 0.5
    return (value - target_min) / span


def denormalize_prediction(value: float, target_min: float = -100.0, target_max: float = 100.0) -> float:
    """Map prediction from [0, 1] back to [target_min, target_max]."""
    return target_min + value * (target_max - target_min)


def associativity_penalty(
    model: CTreePOModel,
    sketches: Sequence[torch.Tensor],
    n_triplets: int = 4,
) -> torch.Tensor:
    """Proxy-only merge associativity regularizer over random triplets.

    This improves empirical stability but is not a Lean law witness.
    """
    if len(sketches) < 3:
        return torch.tensor(0.0)

    n = len(sketches)
    penalty = torch.tensor(0.0)
    count = 0

    for _ in range(min(n_triplets, n * (n - 1) * (n - 2) // 6)):
        indices = torch.randperm(n)[:3]
        a, b, c = sketches[indices[0]], sketches[indices[1]], sketches[indices[2]]

        left_first = model.merge(model.merge(a, b), c)
        right_first = model.merge(a, model.merge(b, c))
        penalty = penalty + ((left_first - right_first) ** 2).sum()
        count += 1

    return penalty / max(count, 1)


def readout_aggregation_penalty(
    model: CTreePOModel,
    parent_sketch: torch.Tensor,
    left_sketch: torch.Tensor,
    right_sketch: torch.Tensor,
    left_weight: float,
    head: str = "rile",
) -> torch.Tensor:
    """Proxy-only penalty: parent readout tracks weighted child readouts.

    left_weight = len(left_text) / (len(left_text) + len(right_text))
    """
    parent_pred = model.predict_normalized(parent_sketch, head)
    left_pred = model.predict_normalized(left_sketch, head)
    right_pred = model.predict_normalized(right_sketch, head)
    expected = left_weight * left_pred + (1 - left_weight) * right_pred
    return ((parent_pred - expected) ** 2).sum()


def consistency_penalty(
    model: CTreePOModel,
    parent_sketch: torch.Tensor,
    left_sketch: torch.Tensor,
    right_sketch: torch.Tensor,
    left_weight: float,
    head: str = "rile",
) -> torch.Tensor:
    """Deprecated alias for ``readout_aggregation_penalty``."""
    return readout_aggregation_penalty(
        model,
        parent_sketch,
        left_sketch,
        right_sketch,
        left_weight,
        head=head,
    )


def contrastive_loss(
    sketches: List[torch.Tensor],
    targets: List[float],
    tau: float = 0.1,
    similarity_threshold: float = 10.0,
) -> torch.Tensor:
    """Cross-language contrastive loss.

    Documents with similar RILE (|rile_i - rile_j| < threshold) are positives.
    """
    n = len(sketches)
    if n < 2:
        return torch.tensor(0.0)

    mat = torch.stack(sketches, dim=0)  # (n, d)
    mat = F.normalize(mat, dim=1)
    sims = mat @ mat.T / tau  # (n, n)

    loss = torch.tensor(0.0)
    count = 0

    for i in range(n):
        # Find positives: similar RILE
        positives = [
            j for j in range(n)
            if j != i and abs(targets[i] - targets[j]) < similarity_threshold
        ]
        if not positives:
            continue

        for j in positives:
            # InfoNCE: -log(exp(sim(i,j)) / sum_k exp(sim(i,k)))
            numerator = sims[i, j]
            denominator = torch.logsumexp(
                torch.cat([sims[i, :i], sims[i, i + 1:]]), dim=0
            )
            loss = loss - (numerator - denominator)
            count += 1

    return loss / max(count, 1)
