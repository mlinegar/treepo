"""Shared TreeModel V2 surface for encoder/core composition and wrappers."""

from __future__ import annotations

from typing import Literal, Optional, Protocol, Sequence, runtime_checkable

import torch
import torch.nn as nn

from treepo._research.tree.core_model import (
    EncoderBackend,
    TreeStateCore,
)


TreeModelVersion = Literal["legacy", "v2", "unified_g"]
TREE_MODEL_VERSION_LEGACY: TreeModelVersion = "legacy"
TREE_MODEL_VERSION_V2: TreeModelVersion = "v2"
TREE_MODEL_VERSION_UNIFIED_G: TreeModelVersion = "unified_g"
VALID_TREE_MODEL_VERSIONS: tuple[TreeModelVersion, ...] = (
    TREE_MODEL_VERSION_LEGACY,
    TREE_MODEL_VERSION_V2,
    TREE_MODEL_VERSION_UNIFIED_G,
)


def normalize_tree_model_version(value: Optional[str]) -> TreeModelVersion:
    normalized = str(value or TREE_MODEL_VERSION_LEGACY).strip().lower()
    if normalized in {"tree_model_v2", "shared_core_v2", "shared_tree_v2"}:
        normalized = TREE_MODEL_VERSION_V2
    if normalized in {"", "legacy", "v1"}:
        return TREE_MODEL_VERSION_LEGACY
    if normalized == TREE_MODEL_VERSION_V2:
        return TREE_MODEL_VERSION_V2
    if normalized in {"unified_g", "unified-g"}:
        return TREE_MODEL_VERSION_UNIFIED_G
    raise ValueError(
        f"unsupported tree_model_version={value!r}; expected one of {VALID_TREE_MODEL_VERSIONS}"
    )


@runtime_checkable
class TreeModelProtocol(Protocol):
    tree_model_version: str
    default_head: str

    @property
    def state_dim(self) -> int:
        ...

    @property
    def leaf_state_dim(self) -> int:
        ...

    @property
    def has_phi(self) -> bool:
        ...

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        ...

    def merge_batch(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        ...

    def predict(self, state: torch.Tensor, head: str = "rile") -> torch.Tensor:
        ...

    def predict_batch(self, states: torch.Tensor, head: str = "rile") -> torch.Tensor:
        ...

    def predict_normalized(self, state: torch.Tensor, head: str = "rile") -> torch.Tensor:
        ...

    def predict_normalized_batch(self, states: torch.Tensor, head: str = "rile") -> torch.Tensor:
        ...

    def predict_confidence(self, state: torch.Tensor, head: str = "rile") -> torch.Tensor:
        ...

    def predict_confidence_batch(self, states: torch.Tensor, head: str = "rile") -> torch.Tensor:
        ...

    def phi(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        ...

    def phi_batch(self, states: torch.Tensor) -> Optional[torch.Tensor]:
        ...

    def phi_score(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        ...

    def phi_fiber(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        ...


class TreeModelV2(nn.Module):
    """Composable V2 tree model: leaf encoder + shared tree state core."""

    def __init__(
        self,
        *,
        encoder_backend: EncoderBackend,
        state_core: TreeStateCore,
        default_head: str = "rile",
        tree_model_version: str = TREE_MODEL_VERSION_V2,
    ) -> None:
        super().__init__()
        self.encoder_backend = encoder_backend
        self.state_core = state_core
        self.default_head = str(default_head or "rile")
        self.tree_model_version = normalize_tree_model_version(tree_model_version)

    @property
    def state_dim(self) -> int:
        return int(self.state_core.config.state_dim)

    @property
    def leaf_state_dim(self) -> int:
        return int(self.encoder_backend.leaf_state_dim)

    @property
    def has_phi(self) -> bool:
        return self.state_core.phi_projector is not None

    def encode_leaves(
        self,
        *,
        embeddings: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[Sequence[int]]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        return self.encoder_backend.encode(
            embeddings=embeddings,
            token_ids=token_ids,
            device=device,
        )

    def encode_leaf_batch(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.encode_leaves(embeddings=embeddings)

    def encode_leaf_tokens_batch(
        self,
        token_id_batch: Sequence[Sequence[int]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return self.encode_leaves(token_ids=token_id_batch, device=device)

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.state_core.merge(left, right)

    def merge_batch(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.state_core.merge_batch(left, right)

    def phi(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        return self.state_core.phi(state)

    def phi_batch(self, states: torch.Tensor) -> Optional[torch.Tensor]:
        return self.state_core.phi(states)

    def phi_score(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        return self.state_core.phi_score(state)

    def phi_fiber(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        return self.state_core.phi_fiber(state)

    def predict(self, state: torch.Tensor, head: str = "rile") -> torch.Tensor:
        return self.state_core.predict(state, head=head)

    def predict_batch(self, states: torch.Tensor, head: str = "rile") -> torch.Tensor:
        return self.state_core.predict_batch(states, head=head)

    def predict_normalized(self, state: torch.Tensor, head: str = "rile") -> torch.Tensor:
        return self.state_core.predict_normalized(state, head=head)

    def predict_normalized_batch(self, states: torch.Tensor, head: str = "rile") -> torch.Tensor:
        return self.state_core.predict_normalized_batch(states, head=head)

    def predict_confidence(self, state: torch.Tensor, head: str = "rile") -> torch.Tensor:
        return self.state_core.predict_confidence(state, head=head)

    def predict_confidence_batch(self, states: torch.Tensor, head: str = "rile") -> torch.Tensor:
        return self.state_core.predict_confidence_batch(states, head=head)


class TreeModelV2View:
    """Non-owning V2 wrapper over existing tree models with legacy module names."""

    def __init__(
        self,
        model: TreeModelProtocol,
        *,
        leaf_input_kind: Literal["embeddings", "token_ids"] = "embeddings",
        tree_model_version: Optional[str] = None,
    ) -> None:
        self.model = model
        self.leaf_input_kind = str(leaf_input_kind)
        self.tree_model_version = normalize_tree_model_version(
            tree_model_version or getattr(model, "tree_model_version", TREE_MODEL_VERSION_LEGACY)
        )
        self.default_head = str(getattr(model, "default_head", "rile") or "rile")

    @property
    def state_dim(self) -> int:
        model_config = getattr(self.model, "config", None)
        return int(getattr(self.model, "state_dim", getattr(model_config, "sketch_dim", 0)))

    @property
    def leaf_state_dim(self) -> int:
        return int(getattr(self.model, "leaf_state_dim", self.state_dim))

    @property
    def has_phi(self) -> bool:
        return bool(getattr(self.model, "has_phi", False))

    def encode_leaves(
        self,
        *,
        embeddings: Optional[torch.Tensor] = None,
        token_ids: Optional[Sequence[Sequence[int]]] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if self.leaf_input_kind == "embeddings":
            if embeddings is None:
                raise ValueError("embedding-backed TreeModelV2View requires embeddings")
            return self.model.encode_leaf_batch(embeddings)
        if token_ids is None:
            raise ValueError("token-backed TreeModelV2View requires token_ids")
        if device is None:
            device = torch.device("cpu")
        return self.model.encode_leaf_tokens_batch(token_ids, device=device)

    def encode_leaf_batch(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.encode_leaves(embeddings=embeddings)

    def encode_leaf_tokens_batch(
        self,
        token_id_batch: Sequence[Sequence[int]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return self.encode_leaves(token_ids=token_id_batch, device=device)

    def merge(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.model.merge(left, right)

    def merge_batch(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.model.merge_batch(left, right)

    def phi(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        return self.model.phi(state)

    def phi_batch(self, states: torch.Tensor) -> Optional[torch.Tensor]:
        return self.model.phi_batch(states)

    def phi_score(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        return self.model.phi_score(state)

    def phi_fiber(self, state: torch.Tensor) -> Optional[torch.Tensor]:
        return self.model.phi_fiber(state)

    def predict(self, state: torch.Tensor, head: Optional[str] = None) -> torch.Tensor:
        return self.model.predict(state, head=head or self.default_head)

    def predict_batch(self, states: torch.Tensor, head: Optional[str] = None) -> torch.Tensor:
        return self.model.predict_batch(states, head=head or self.default_head)

    def predict_normalized(self, state: torch.Tensor, head: Optional[str] = None) -> torch.Tensor:
        return self.model.predict_normalized(state, head=head or self.default_head)

    def predict_normalized_batch(
        self,
        states: torch.Tensor,
        head: Optional[str] = None,
    ) -> torch.Tensor:
        return self.model.predict_normalized_batch(states, head=head or self.default_head)

    def predict_confidence(self, state: torch.Tensor, head: Optional[str] = None) -> torch.Tensor:
        return self.model.predict_confidence(state, head=head or self.default_head)

    def predict_confidence_batch(
        self,
        states: torch.Tensor,
        head: Optional[str] = None,
    ) -> torch.Tensor:
        return self.model.predict_confidence_batch(states, head=head or self.default_head)
