"""
Shared tree-neural core: encoder-agnostic merge, phi projection, and task heads.

This module defines the common architectural components shared between:
- CTreePOModel (real-document path: pre-computed embeddings)
- FNOCountSketch (simulation path: raw token sequences)

The key abstraction is ``TreeNeuralCore``, which owns the merge operator,
optional phi projector (with score-fiber factorization), and task heads.
It does NOT own the encoder backend — that is composed externally by the
specific model class (CTreePOModel, FNOCountSketch, etc.).

Architecture:

    [Encoder backend]  →  state (B, state_dim)
           ↓
    [TreeNeuralCore]
      ├── merge(left, right) → parent_state
      ├── phi(state) → [score | fiber | aux]   (optional)
      └── heads[name](state) → scalar prediction

Score-fiber factorization:
    The phi projector maps states to a structured theorem-feature vector.
    The score slice carries task-relevant scalar information (RILE, count).
    The fiber slice carries learned structural/categorical embeddings.
    Both share a common backbone — gradients flow through the full phi.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol, Sequence, runtime_checkable

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError(
        "PyTorch is required for tree neural core. "
        "Install with: pip install torch>=2.0.0"
    )


# ---------------------------------------------------------------------------
# Encoder backend protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class EncoderBackend(Protocol):
    """Protocol for leaf-span encoding backends.

    Both pre-computed embedding projection and raw-token FNO encoding
    satisfy this protocol.  The output must be a ``(B, state_dim)`` tensor
    with ``requires_grad=True`` when the model is in training mode.
    """

    @property
    def state_dim(self) -> int:
        """Dimension of the output state vector."""
        ...

    @property
    def leaf_state_dim(self) -> int:
        """Compatibility alias for code using the V2 naming."""
        ...

    def encode(
        self,
        *,
        embeddings: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[Sequence[int]]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Encode leaf inputs to state vectors.

        Exactly one of *embeddings* or *token_ids* should be provided.

        Args:
            embeddings: ``(B, embedding_dim)`` pre-computed vectors.
            token_ids: Sequence of B token-ID sequences (variable length).
            device: Target device for the output tensor.

        Returns:
            ``(B, state_dim)`` state tensor.
        """
        ...


# ---------------------------------------------------------------------------
# Concrete encoder: embedding projector (for real documents)
# ---------------------------------------------------------------------------


class EmbeddingProjectorBackend(nn.Module):
    """Encoder backend that projects pre-computed embeddings to state space.

    Wraps a two-layer MLP: Linear → LayerNorm → ReLU → Linear.
    This is the encoder used by CTreePOModel for real documents.
    """

    def __init__(
        self,
        embedding_dim: int,
        state_dim: int,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self._state_dim = state_dim
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def leaf_state_dim(self) -> int:
        return self._state_dim

    def encode(
        self,
        *,
        embeddings: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[Sequence[int]]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if embeddings is None:
            raise ValueError("EmbeddingProjectorBackend requires embeddings")
        if device is not None:
            embeddings = embeddings.to(device)
        return self.net(embeddings)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Direct forward for nn.Module compatibility."""
        return self.net(embeddings)


class TokenSequenceEncoderBackend(nn.Module):
    """Adapter backend for token-sequence encoders already owned elsewhere.

    This wraps an existing ``encode_token_batch`` style callable so the leaf
    encoder can plug into the shared ``TreeModelV2`` surface without forcing the
    caller to rebuild its token encoder stack.
    """

    def __init__(
        self,
        state_dim: int,
        encode_tokens_batch: Callable[
            [Sequence[Sequence[int]], torch.device],
            torch.Tensor,
        ],
    ):
        super().__init__()
        self._state_dim = int(state_dim)
        self._encode_tokens_batch = encode_tokens_batch

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def leaf_state_dim(self) -> int:
        return self._state_dim

    def encode(
        self,
        *,
        embeddings: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[Sequence[int]]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if token_ids is None:
            raise ValueError("TokenSequenceEncoderBackend requires token_ids")
        if device is None:
            device = torch.device("cpu")
        token_id_batch = [tuple(int(token) for token in sequence) for sequence in token_ids]
        return self._encode_tokens_batch(token_id_batch, device=device)


# ---------------------------------------------------------------------------
# Score-fiber factorization config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScoreFiberConfig:
    """Configuration for score-fiber factorization of the phi embedding.

    The phi vector is split into three contiguous slices:
      phi = [score | fiber | aux]

    - **score**: Task-relevant scalar (RILE for real docs, count for Markov).
    - **fiber**: Learned structural/categorical embedding for contrastive
      supervision (C2 law).
    - **aux**: Optional extra dimensions.

    ``score_dim + fiber_dim + aux_dim`` must equal ``phi_dim``.
    """

    phi_dim: int = 48
    score_dim: int = 1
    fiber_dim: int = 47
    aux_dim: int = 0
    hidden_dim: int = 128

    def __post_init__(self) -> None:
        if self.score_dim + self.fiber_dim + self.aux_dim != self.phi_dim:
            raise ValueError(
                f"score_dim ({self.score_dim}) + fiber_dim ({self.fiber_dim}) "
                f"+ aux_dim ({self.aux_dim}) != phi_dim ({self.phi_dim})"
            )


# ---------------------------------------------------------------------------
# Phi projector
# ---------------------------------------------------------------------------


class PhiProjector(nn.Module):
    """Projects state vectors to structured theorem-feature embeddings.

    Architecture: LayerNorm → Linear → SiLU → Linear

    The output phi vector is factorized into score, fiber, and aux slices
    via simple tensor indexing (NOT separate learned projections).  This
    means gradients flow through the shared backbone regardless of which
    slice a loss term targets.
    """

    def __init__(self, state_dim: int, config: ScoreFiberConfig):
        super().__init__()
        self.config = config
        self._score_slice = slice(0, config.score_dim)
        self._fiber_slice = slice(config.score_dim, config.score_dim + config.fiber_dim)
        self._aux_slice = slice(
            config.score_dim + config.fiber_dim, config.phi_dim
        )

        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.phi_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Project state to phi. Shape: ``(..., state_dim) → (..., phi_dim)``."""
        return self.net(state)

    def score(self, phi: torch.Tensor) -> torch.Tensor:
        """Extract score slice. Shape: ``(..., phi_dim) → (..., score_dim)``."""
        return phi[..., self._score_slice]

    def fiber(self, phi: torch.Tensor) -> torch.Tensor:
        """Extract fiber slice. Shape: ``(..., phi_dim) → (..., fiber_dim)``."""
        return phi[..., self._fiber_slice]

    def aux(self, phi: torch.Tensor) -> torch.Tensor:
        """Extract aux slice. Shape: ``(..., phi_dim) → (..., aux_dim)``."""
        return phi[..., self._aux_slice]

    def fiber_aux(self, phi: torch.Tensor) -> torch.Tensor:
        """Extract fiber + aux slices concatenated."""
        if self.config.aux_dim > 0:
            return phi[..., self._fiber_slice.start :]
        return self.fiber(phi)


# ---------------------------------------------------------------------------
# Merge modules (re-exported from ctreepo_model for convenience)
# ---------------------------------------------------------------------------

# The merge modules (GatedMerge, MLPMerge, etc.) remain in ctreepo_model.py
# to avoid circular imports.  They satisfy the merge protocol:
#   forward(left: Tensor, right: Tensor) -> Tensor


# ---------------------------------------------------------------------------
# Task heads
# ---------------------------------------------------------------------------


class ScalarReadoutHead(nn.Module):
    """Readout from state to a scalar, mapped to a target scale.

    Identical to the existing ReadoutHead in ctreepo_model.py but
    accepts input from either raw state or phi score slice.
    """

    def __init__(
        self,
        input_dim: int,
        target_min: float = -100.0,
        target_max: float = 100.0,
    ):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        self.target_min = target_min
        self.target_max = target_max

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Predict on ``[target_min, target_max]`` scale."""
        raw = torch.sigmoid(self.linear(features))
        return self.target_min + raw * (self.target_max - self.target_min)

    def forward_normalized(self, features: torch.Tensor) -> torch.Tensor:
        """Predict on ``[0, 1]`` scale (for loss computation)."""
        return torch.sigmoid(self.linear(features))


# ---------------------------------------------------------------------------
# Core config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeNeuralCoreConfig:
    """Configuration for the shared tree-neural core."""

    state_dim: int = 32
    merge_type: str = "gated"
    head_names: tuple = ("rile",)

    # Score-fiber factorization (optional)
    phi_config: Optional[ScoreFiberConfig] = None

    # Target scale
    target_min: float = -100.0
    target_max: float = 100.0


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


class TreeNeuralCore(nn.Module):
    """Shared core: encoder-agnostic merge + optional phi + readout heads.

    This module does NOT own the encoder backend.  The encoder is composed
    externally by the specific model class (CTreePOModel, etc.).

    Usage::

        core = TreeNeuralCore(config)

        # Leaf encoding done by external encoder
        leaf_states = encoder.encode(embeddings=emb_batch)

        # Core handles merge + readout
        parent = core.merge(left_states, right_states)
        prediction = core.predict(root_state, head="rile")

        # Optional theorem features
        phi = core.phi(root_state)  # None if phi_config not set
    """

    def __init__(self, config: TreeNeuralCoreConfig):
        super().__init__()
        self.config = config

        # Import merge classes here to avoid circular import
        from treepo._research.tree.ctreepo_model import _MERGE_CLASSES

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
                config.state_dim,
                hidden_dim=max(64, config.state_dim * 2),
            )
        elif config.merge_type == "residual_gated":
            self.merge_module = merge_cls(
                config.state_dim,
                hidden_dim=max(64, config.state_dim * 4),
            )
        else:
            self.merge_module = merge_cls(config.state_dim)

        # Optional phi projector
        self.phi_projector: Optional[PhiProjector] = None
        if config.phi_config is not None:
            self.phi_projector = PhiProjector(config.state_dim, config.phi_config)

        # Task heads (read from state, not phi)
        self.heads = nn.ModuleDict()
        for name in config.head_names:
            self.heads[name] = ScalarReadoutHead(
                config.state_dim, config.target_min, config.target_max,
            )

    # ------------------------------------------------------------------
    # Forward methods
    # ------------------------------------------------------------------

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Merge two child states into a parent state.

        Supports both single ``(state_dim,)`` and batched ``(B, state_dim)``
        inputs.
        """
        return self.merge_module(left, right)

    def merge_batch(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        """Batched merge. ``(N, state_dim) × 2 → (N, state_dim)``."""
        return self.merge_module(left, right)

    def phi(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """Project state to theorem features, or None if not configured."""
        if self.phi_projector is None:
            return None
        return self.phi_projector(state)

    def phi_score(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract score slice from phi, or None."""
        if self.phi_projector is None:
            return None
        return self.phi_projector.score(self.phi_projector(state))

    def phi_fiber(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        """Extract fiber slice from phi, or None."""
        if self.phi_projector is None:
            return None
        return self.phi_projector.fiber(self.phi_projector(state))

    def predict(self, state: torch.Tensor, head: str = "rile") -> torch.Tensor:
        """Predict target score from state."""
        return self.heads[head](state)

    def predict_normalized(self, state: torch.Tensor, head: str = "rile") -> torch.Tensor:
        """Predict on [0, 1] scale."""
        return self.heads[head].forward_normalized(state)

    def predict_batch(self, states: torch.Tensor, head: str = "rile") -> torch.Tensor:
        """Batched predict. ``(N, state_dim) → (N, 1)``."""
        return self.heads[head](states)

    def predict_normalized_batch(self, states: torch.Tensor, head: str = "rile") -> torch.Tensor:
        """Batched normalized predict. ``(N, state_dim) → (N, 1)``."""
        return self.heads[head].forward_normalized(states)

    def predict_confidence(self, state: torch.Tensor, head: str = "rile") -> torch.Tensor:
        """Confidence proxy in [0, 1]."""
        pred_norm = self.predict_normalized(state, head=head)
        return 1.0 - 2.0 * torch.abs(pred_norm - 0.5)

    def predict_confidence_batch(self, states: torch.Tensor, head: str = "rile") -> torch.Tensor:
        """Batched confidence proxy."""
        pred_norm = self.predict_normalized_batch(states, head=head)
        return 1.0 - 2.0 * torch.abs(pred_norm - 0.5)


# V2 naming aliases.  Keep the original names for backward compatibility.
TreeStateCoreConfig = TreeNeuralCoreConfig
TreeStateCore = TreeNeuralCore
