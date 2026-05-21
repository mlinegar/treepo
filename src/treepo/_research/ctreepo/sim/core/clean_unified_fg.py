"""Clean f/g composition model for the C-TreePO unified-g pattern.

Three learned components, each its own named submodule:

- ``leaf_encoder``: tokens -> state. Preprocessing that turns raw input
  sequences into states. Includes the FNO + pool. Not part of the f/g
  algebra; it just builds the leaf states that g/f then consume.
- ``g``: (state, state) -> state. The merge / compose operator. Pure
  state-space, no FNO (no spatial structure to exploit at merge time).
- ``f``: state -> prediction. The scorer / readout. Applicable at ANY
  node in the tree -- leaves OR internal merges OR the root -- because
  it just operates on a state.

For the current contextual-sufficiency interpretation, ``g`` should be read
as the learned state map: it is good when the state it produces preserves all
downstream contextual responses that ``f`` may need. In the operator variant
below, the same ``g`` is used for leaf reads, ``g(embed(x), null)``, and for
internal merges, ``g(left_state, right_state)``. This follows the NASS/SSS/SSNL
line of learned sufficient statistics while keeping the Lean-backed condition
deterministic: no bad state collisions under two-sided contexts.

Design intent (vs ``FNOCountSketch`` in ``markov_neural_operator_baselines.py``):
- One file, no surface modes, no opaque-carrier paths, no telemetry knobs.
- ``leaf_encoder``, ``g``, ``f`` are exposed as named submodules -- swap
  any one out by replacing the attribute.
- ``forward_doc`` returns a ``TreeForwardOutput`` with ``f`` already
  applied to every node (leaf, merge, root).
- Local laws (C1 monotonicity, C2 leaf calibration, C3 smoothness) are
  expressed as supervision-sparsity patterns over the tree, not as
  separate model heads. The trainer decides which subset of (root, leaves,
  merges) to score against observed labels.

The count target is the number of regime change-points in the doc covered by
a given subtree node. Predictions are normalized by ``target_scale`` during
training; the public ``predict_count`` method returns un-normalized counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from treepo._research.ctreepo.sim.core.fno_doc_baselines import (
    HAS_NEURAL_OPERATOR,
    apply_fno_token_encoder,
    _NeuralOpFNO,
    _INSTALL_MSG,
)
from treepo._research.ctreepo.sim.core.markov_local_laws import (
    MARKOV_COUNT_SKETCH_LAW_SET_ID,
)


@dataclass(frozen=True)
class TreeForwardOutput:
    """Per-node count predictions for a single doc.

    Predictions are *normalized* (divided by target_scale). To get
    un-normalized counts call ``model.predict_count(state)`` or multiply
    by ``model.target_scale``.

    Attributes:
        leaf_states: list of length n_leaves. Each is a (state_dim,) tensor.
        merge_states: list of length n_leaves - 1 (for a balanced binary
            tree on n_leaves). Last entry is the root.
        leaf_counts_norm: (n_leaves,) tensor of normalized count predictions
            at each leaf.
        merge_counts_norm: (n_leaves - 1,) tensor of normalized count
            predictions at each merge node.
        root_state: convenience alias for merge_states[-1] when n_leaves > 1,
            or leaf_states[0] when n_leaves == 1.
        root_count_norm: convenience alias for the root's normalized count.
    """

    leaf_states: List[torch.Tensor]
    merge_states: List[torch.Tensor]
    leaf_counts_norm: torch.Tensor
    merge_counts_norm: torch.Tensor
    root_state: torch.Tensor
    root_count_norm: torch.Tensor


class CleanLeafEncoder(nn.Module):
    """Leaf encoder: tokens -> state.

    Three steps, no extras:
      1. ``token_embedding`` (nn.Embedding) - vocab to fno_width channels
      2. neuralop ``FNO`` (1D, n_modes Fourier modes, n_layers layers)
      3. masked pool over the token dimension

    The output state has shape (B, fno_width); we don't add any
    pool-to-state projection. To keep the model "official", the model's
    ``state_dim`` is required to equal ``fno_width`` -- no width adapter.
    Override the FNO directly (``self.fno``) or pre-pool projection
    upstream if you need a different shape.

    Calls ``apply_fno_token_encoder`` for one source of truth on the
    embed+FNO+pool sequence shared with the rest of the codebase.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        fno_width: int,
        fno_n_modes: int,
        fno_n_layers: int,
        pooling_mode: str = "sum",
    ) -> None:
        super().__init__()
        if _NeuralOpFNO is None or not HAS_NEURAL_OPERATOR:
            raise ImportError(_INSTALL_MSG)
        if pooling_mode not in {"mean", "sum"}:
            raise ValueError(f"pooling_mode must be 'mean' or 'sum'; got {pooling_mode!r}")
        self.pad_id = int(vocab_size)
        self.fno_width = int(fno_width)
        self.pooling_mode = str(pooling_mode)
        self.token_embedding = nn.Embedding(
            int(vocab_size) + 1, int(fno_width), padding_idx=self.pad_id
        )
        self.fno = _NeuralOpFNO(
            n_modes=(int(fno_n_modes),),
            in_channels=int(fno_width),
            out_channels=int(fno_width),
            hidden_channels=int(fno_width),
            n_layers=int(fno_n_layers),
        )

    @property
    def state_dim(self) -> int:
        return self.fno_width

    def forward(
        self, tokens: torch.Tensor, *, token_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be (B, L); got shape {tuple(tokens.shape)}")
        if token_mask is None:
            token_mask = tokens.ne(self.pad_id).to(dtype=torch.float32)
        _x, pooled = apply_fno_token_encoder(
            tokens,
            token_mask=token_mask,
            token_embedding=self.token_embedding,
            fno=self.fno,
            pooling_mode=self.pooling_mode,
        )
        return pooled


class CleanMergeG(nn.Module):
    """g: (left_state, right_state) -> merged_state.

    A bare ``nn.Linear(2*state_dim, state_dim)`` over concat(left, right).
    No activation, no LayerNorm, no MLP stack. The whole point is that g
    should be the simplest possible learned merge so we can attribute
    behavior to the operator and the surrounding composition rather than
    to a custom architecture choice baked into g.

    If the linear g doesn't have enough capacity for a problem, swap
    ``self.linear`` for something with the same input/output signature
    (an MLP, a standard PyTorch module, an FNO over a length-2 sequence,
    etc.) instead of layering activations onto this baseline.
    """

    def __init__(self, *, state_dim: int) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.linear = nn.Linear(2 * int(state_dim), int(state_dim))

    def forward(
        self, left_state: torch.Tensor, right_state: torch.Tensor
    ) -> torch.Tensor:
        return self.linear(torch.cat([left_state, right_state], dim=-1))


class CleanScorerF(nn.Module):
    """f: state -> prediction (normalized scalar count).

    A bare ``nn.Linear(state_dim, 1)`` over the state. No activation,
    no MLP stack. f is applied at every node in the tree (leaves,
    internal merges, root); composes cleanly with anything that
    produces a state (the leaf encoder OR g).

    If the linear f doesn't have enough capacity, swap ``self.linear``
    for something with the same signature.
    """

    def __init__(self, *, state_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(state_dim), 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.linear(state).squeeze(-1)


class CleanUnifiedFG(nn.Module):
    """Compose a leaf encoder + learned g + learned f into a balanced binary tree.

    Three named submodules:
        - ``leaf_encoder``: tokens -> state (preprocessing for raw inputs)
        - ``g``: (state, state) -> state (merge / compose operator)
        - ``f``: state -> prediction (scorer applied at every node)

    Public API:
        - ``leaf_encoder(tokens, token_mask=None) -> state`` — encode raw leaves
        - ``g(left_state, right_state) -> merged_state`` — compose two states
        - ``f(state) -> normalized_count`` — score any state (leaf or merge)
        - ``predict_count(state) -> count`` — un-normalized count = f(state) * target_scale
        - ``forward_doc(leaf_token_sequences, leaf_mask=None) -> TreeForwardOutput``
            Encode all leaves, build a balanced binary tree via g, score every
            node with f, return all per-node predictions.

    Tree shape (balanced binary, parents-after-children):
        - n_leaves leaves at depth d_max
        - n_leaves - 1 merges total
        - For odd-count layers, the leftover state passes through unchanged.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        target_scale: float,
        fno_width: int = 128,
        fno_n_modes: int = 8,
        fno_n_layers: int = 4,
        pooling_mode: str = "sum",
    ) -> None:
        super().__init__()
        self.target_scale = float(target_scale)
        self.leaf_encoder = CleanLeafEncoder(
            vocab_size=vocab_size,
            fno_width=fno_width,
            fno_n_modes=fno_n_modes,
            fno_n_layers=fno_n_layers,
            pooling_mode=pooling_mode,
        )
        # state_dim = fno_width by construction (leaf_encoder has no
        # pool-to-state projection layer)
        self.state_dim = self.leaf_encoder.state_dim
        self.g = CleanMergeG(state_dim=self.state_dim)
        self.f = CleanScorerF(state_dim=self.state_dim)

    def predict_count(self, state: torch.Tensor) -> torch.Tensor:
        return self.f(state) * self.target_scale

    def forward_doc(
        self,
        leaf_tokens: torch.Tensor,
        *,
        leaf_mask: torch.Tensor | None = None,
    ) -> TreeForwardOutput:
        """Run the full f/g composition for a single doc's leaves.

        Args:
            leaf_tokens: (n_leaves, L) tensor — tokens for each leaf in
                left-to-right reading order.
            leaf_mask: (n_leaves, L) tensor — 1 for valid tokens, 0 for
                padding. If None, derived from ``tokens != pad_id``.
        """
        if leaf_tokens.ndim != 2:
            raise ValueError(
                f"leaf_tokens must be (n_leaves, L); got shape {tuple(leaf_tokens.shape)}"
            )
        n_leaves = int(leaf_tokens.shape[0])
        if n_leaves == 0:
            raise ValueError("need at least one leaf")

        # Encode each leaf to a state (batched)
        leaf_states_batch = self.leaf_encoder(leaf_tokens, token_mask=leaf_mask)
        # (n_leaves, state_dim)
        leaf_states = [leaf_states_batch[i] for i in range(n_leaves)]
        # Apply f to each leaf state (the scorer applies at every node)
        leaf_counts_norm = self.f(leaf_states_batch)

        # Balanced binary tree composition via g
        merge_states: List[torch.Tensor] = []
        cur = list(leaf_states)
        while len(cur) > 1:
            nxt: List[torch.Tensor] = []
            pair_count = len(cur) // 2
            if pair_count > 0:
                left = torch.stack(cur[: 2 * pair_count : 2], dim=0)
                right = torch.stack(cur[1 : 2 * pair_count : 2], dim=0)
                merged_batch = self.g(left, right)  # (pair_count, state_dim)
                for k in range(pair_count):
                    merge_states.append(merged_batch[k])
                    nxt.append(merged_batch[k])
            if len(cur) % 2 == 1:
                # Leftover passes through unchanged (no g call)
                nxt.append(cur[-1])
            cur = nxt

        # Apply f to every internal merge state too
        if merge_states:
            merge_states_batch = torch.stack(merge_states, dim=0)
            merge_counts_norm = self.f(merge_states_batch)
        else:
            merge_counts_norm = leaf_counts_norm.new_empty((0,))

        root_state = cur[0]
        if merge_states:
            root_count_norm = merge_counts_norm[-1]
        else:
            root_count_norm = leaf_counts_norm[0]

        return TreeForwardOutput(
            leaf_states=leaf_states,
            merge_states=merge_states,
            leaf_counts_norm=leaf_counts_norm,
            merge_counts_norm=merge_counts_norm,
            root_state=root_state,
            root_count_norm=root_count_norm,
        )

    def forward(
        self,
        leaf_tokens: torch.Tensor,
        *,
        leaf_mask: torch.Tensor | None = None,
    ) -> TreeForwardOutput:
        return self.forward_doc(leaf_tokens, leaf_mask=leaf_mask)


# ---------------------------------------------------------------------------
# Operator variants: state is a discretized FUNCTION (B, C, L) instead of a
# vector (B, C). g and f are then both true neural operators -- thin wrappers
# around `neuralop.FNO`. Use these when you want g and f to be operators in
# the operator-learning sense, not just neural nets.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TreeForwardOutputNO:
    """Per-node count predictions for a single doc (operator variant).

    Shape contract: ``leaf_states[i].shape == (channels, length)`` -- a
    discretized function on a 1D domain. Same for ``merge_states``.
    Predictions are still scalar normalized counts at each node.
    """

    leaf_states: List[torch.Tensor]      # each (C, L)
    merge_states: List[torch.Tensor]     # each (C, L)
    leaf_counts_norm: torch.Tensor       # (n_leaves,)
    merge_counts_norm: torch.Tensor      # (n_leaves - 1,)
    root_state: torch.Tensor             # (C, L)
    root_count_norm: torch.Tensor        # scalar


class CleanUnifiedG(nn.Module):
    """g (operator variant): the SAME FNO applied at leaves AND merges.

    The public call interface mirrors the algebra:

      - ``g(content)`` means ``g(content, null)`` for leaf reads.
      - ``g(left_state, right_state)`` means the merge ``g(left, right)``.

    Internally both calls are lowered to the same pair-shaped FNO input
    ``(B, 2C, L) -> (B, C, L)``:

      - merge: ``cat([left_state, right_state], dim=channels)``
      - leaf: ``cat([content, zeros], dim=channels)``

    A pre-built pair-shaped tensor is still accepted for compatibility, but
    the reference lane should prefer the direct ``g(x)`` / ``g(x, y)`` forms so
    the surface API stays as close as possible to ``f(state)``.
    """

    def __init__(
        self,
        *,
        channels: int,
        n_modes: int,
        n_layers: int,
    ) -> None:
        super().__init__()
        if _NeuralOpFNO is None or not HAS_NEURAL_OPERATOR:
            raise ImportError(_INSTALL_MSG)
        self.channels = int(channels)
        self.fno = _NeuralOpFNO(
            n_modes=(int(n_modes),),
            in_channels=2 * int(channels),
            out_channels=int(channels),
            hidden_channels=int(channels),
            n_layers=int(n_layers),
        )

    def forward(
        self,
        left_or_pair: torch.Tensor,
        right_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply g to either a leaf content function or a pair of states.

        Args:
            left_or_pair: Either ``(B, C, L)`` leaf content / left state or a
                compatibility ``(B, 2C, L)`` pre-built pair tensor.
            right_state: Optional ``(B, C, L)`` right state. If omitted and
                ``left_or_pair`` has ``C`` channels, the right side is null.
        """
        if left_or_pair.ndim != 3:
            raise ValueError(
                "g input must be (B, C, L) or (B, 2C, L); "
                f"got shape {tuple(left_or_pair.shape)}"
            )
        if right_state is not None:
            if right_state.ndim != 3:
                raise ValueError(
                    f"right_state must be (B, C, L); got shape {tuple(right_state.shape)}"
                )
            if tuple(left_or_pair.shape) != tuple(right_state.shape):
                raise ValueError(
                    "left and right states must have the same shape; "
                    f"got {tuple(left_or_pair.shape)} and {tuple(right_state.shape)}"
                )
            if int(left_or_pair.shape[-2]) != self.channels:
                raise ValueError(
                    f"left/right states must have {self.channels} channels; "
                    f"got {int(left_or_pair.shape[-2])}"
                )
            pair_input = torch.cat([left_or_pair, right_state], dim=-2)
            return self.fno(pair_input)

        n_channels = int(left_or_pair.shape[-2])
        if n_channels == self.channels:
            null_state = torch.zeros_like(left_or_pair)
            pair_input = torch.cat([left_or_pair, null_state], dim=-2)
        elif n_channels == 2 * self.channels:
            pair_input = left_or_pair
        else:
            raise ValueError(
                "g input must have C channels for g(x, null) or 2C channels "
                f"for a pre-built pair; got {n_channels}, expected "
                f"{self.channels} or {2 * self.channels}"
            )
        return self.fno(pair_input)

    def merge(
        self, left_state: torch.Tensor, right_state: torch.Tensor
    ) -> torch.Tensor:
        """Compatibility alias for ``g(left_state, right_state)``."""
        return self.forward(left_state, right_state)

    def encode_leaf(
        self, embedded_tokens: torch.Tensor
    ) -> torch.Tensor:
        """Compatibility alias for ``g(embedded_tokens)``.

        A leaf is the base case of the tree. We feed it through the same
        g as merges by treating it as ``g(content, null)``. The "null half"
        is plain zeros so g can learn to ignore it (or use it as a marker
        that this is a leaf, if useful).
        """
        return self.forward(embedded_tokens)


class CleanLeafTokenEmbedding(nn.Module):
    """Just an ``nn.Embedding`` -- token ids to (B, C, L) functions.

    Kept as its own tiny module so the model has a clean named submodule
    for the leaf-specific preprocessing step. The real heavy lifting at
    leaves happens inside g (via ``g.encode_leaf``).
    """

    def __init__(self, *, vocab_size: int, channels: int) -> None:
        super().__init__()
        self.pad_id = int(vocab_size)
        self.channels = int(channels)
        self.embedding = nn.Embedding(
            int(vocab_size) + 1, int(channels), padding_idx=self.pad_id
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be (B, L); got shape {tuple(tokens.shape)}")
        emb = self.embedding(tokens)              # (B, L, C)
        return emb.permute(0, 2, 1).contiguous()  # (B, C, L)


class CleanScorerFNO(nn.Module):
    """f (operator variant): score a function-valued state.

    Pipeline: FNO -> masked pool -> Linear to scalar. The FNO call is what
    makes f an operator (rather than a pure pointwise scorer). Pool + Linear
    is the standard scalar-readout step needed because the prediction
    target is a single number per node.
    """

    def __init__(
        self,
        *,
        channels: int,
        n_modes: int,
        n_layers: int,
        pooling_mode: str = "sum",
    ) -> None:
        super().__init__()
        if _NeuralOpFNO is None or not HAS_NEURAL_OPERATOR:
            raise ImportError(_INSTALL_MSG)
        if pooling_mode not in {"mean", "sum"}:
            raise ValueError(f"pooling_mode must be 'mean' or 'sum'; got {pooling_mode!r}")
        self.channels = int(channels)
        self.pooling_mode = str(pooling_mode)
        self.fno = _NeuralOpFNO(
            n_modes=(int(n_modes),),
            in_channels=int(channels),
            out_channels=int(channels),
            hidden_channels=int(channels),
            n_layers=int(n_layers),
        )
        self.linear = nn.Linear(int(channels), 1)

    def forward(
        self, state: torch.Tensor, *, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # state: (B, C, L). Apply FNO -> pool over L -> linear to scalar.
        x = self.fno(state)
        if mask is None:
            mask = state.new_ones((int(x.shape[0]), 1, int(x.shape[-1])))
        else:
            mask = mask.unsqueeze(1)
        pooled = (x * mask).sum(dim=-1)           # (B, C)
        if self.pooling_mode == "mean":
            pooled = pooled / mask.sum(dim=-1).clamp(min=1)
        return self.linear(pooled).squeeze(-1)    # (B,)


class CleanUnifiedNO(nn.Module):
    """Pure neural-operator composition with a SHARED g.

    Three named submodules:
        - ``token_embedding``: token ids -> ``(B, C, L)``
        - ``g``: the SAME FNO applied at leaves AND merges. Leaves call
          ``g(content)`` for ``g(content, null)``; merges call ``g(left, right)``.
        - ``f``: ``(B, C, L) -> scalar`` -- FNO + pool + Linear scalar
          readout, applied at every node.

    Key property: leaves and merges share the SAME g parameters. This is
    "unified-g" in the operator sense -- g is one learned function applied
    at every node in the tree.

    Statistical interpretation: ``g(embed(x), null)`` is the learned
    contextual sufficient statistic for span ``x``. It should preserve the
    empirical response signature
    ``[fstar(left_i * x * right_i)]_i`` across sampled two-sided contexts.
    ``f`` is the downstream readout on this state. The Markov
    ``(count, first, last)`` sketch is a validation witness for this condition,
    not a hard-coded definition of the learned state.

    Minimal contract:
        ``z_x = g(embed(x), null)``
        ``z_y = g(embed(y), null)``
        ``z_xy = g(z_x, z_y)``
        ``score = f(z_xy)``

    ``FNOCountSketch(tree_model_version="unified_g")`` is production
    infrastructure, not this minimal reference lane: it shares a downstream
    summary encoder but prepares leaf and merge summaries through different
    learned paths. This class is the narrow contract target for review.
    """

    minimal_unified_gf_contract = (
        "z_x=g(embed(x),null); z_y=g(embed(y),null); "
        "z_xy=g(z_x,z_y); score=f(z_xy)"
    )

    def __init__(
        self,
        *,
        vocab_size: int,
        target_scale: float,
        channels: int = 64,
        g_n_modes: int = 16,
        g_n_layers: int = 4,
        scorer_n_modes: int = 8,
        scorer_n_layers: int = 2,
        pooling_mode: str = "sum",
    ) -> None:
        super().__init__()
        self.target_scale = float(target_scale)
        self.channels = int(channels)
        self.token_embedding = CleanLeafTokenEmbedding(
            vocab_size=vocab_size,
            channels=channels,
        )
        self.g = CleanUnifiedG(
            channels=channels,
            n_modes=g_n_modes,
            n_layers=g_n_layers,
        )
        self.f = CleanScorerFNO(
            channels=channels,
            n_modes=scorer_n_modes,
            n_layers=scorer_n_layers,
            pooling_mode=pooling_mode,
        )

    def predict_count(self, state: torch.Tensor) -> torch.Tensor:
        return self.f(state.unsqueeze(0)).squeeze(0) * self.target_scale

    def _encode_leaf_states_via_g(self, leaf_tokens: torch.Tensor) -> torch.Tensor:
        """Token ids -> embedded content -> shared-g leaf states.

        This is the only learned leaf-read path in the minimal reference lane.
        The token embedding is a trivial input adapter; all state construction
        flows through ``g(content, null)``.
        """
        embedded = self.token_embedding(leaf_tokens)
        return self.g(embedded)

    def _merge_state_batch_via_g(
        self,
        left_states: torch.Tensor,
        right_states: torch.Tensor,
    ) -> torch.Tensor:
        """Merge child states through the same shared g used at leaves."""
        return self.g(left_states, right_states)

    def _score_states_via_f(self, states: torch.Tensor) -> torch.Tensor:
        """Score leaf, merge, or root states through f only."""
        return self.f(states)

    def forward_doc(self, leaf_tokens: torch.Tensor) -> TreeForwardOutputNO:
        """Run the full operator composition for a single doc's leaves.

        At leaves: embed tokens -> (B, C, L), then call ``g(content)`` which
        means ``g(content, null)``. At merges: pair-up two children and call
        ``g(left, right)``. Both calls run the same shared FNO parameters.

        Args:
            leaf_tokens: (n_leaves, L) tensor of token ids.
        """
        if leaf_tokens.ndim != 2:
            raise ValueError(
                f"leaf_tokens must be (n_leaves, L); got {tuple(leaf_tokens.shape)}"
            )
        n_leaves = int(leaf_tokens.shape[0])
        if n_leaves == 0:
            raise ValueError("need at least one leaf")

        # Leaves go through g(content, null). There is no separate learned
        # leaf encoder beyond the token embedding adapter.
        leaf_states_batch = self._encode_leaf_states_via_g(leaf_tokens)
        leaf_states = [leaf_states_batch[i] for i in range(n_leaves)]
        leaf_counts_norm = self._score_states_via_f(leaf_states_batch)

        merge_states: List[torch.Tensor] = []
        cur = list(leaf_states)
        while len(cur) > 1:
            nxt: List[torch.Tensor] = []
            pair_count = len(cur) // 2
            if pair_count > 0:
                left = torch.stack(cur[: 2 * pair_count : 2], dim=0)   # (P, C, L)
                right = torch.stack(cur[1 : 2 * pair_count : 2], dim=0)
                # Same g, just called with both child states for merging.
                merged_batch = self._merge_state_batch_via_g(left, right)
                for k in range(pair_count):
                    merge_states.append(merged_batch[k])
                    nxt.append(merged_batch[k])
            if len(cur) % 2 == 1:
                nxt.append(cur[-1])
            cur = nxt

        if merge_states:
            merge_states_batch = torch.stack(merge_states, dim=0)
            merge_counts_norm = self._score_states_via_f(merge_states_batch)
        else:
            merge_counts_norm = leaf_counts_norm.new_empty((0,))

        root_state = cur[0]
        if merge_states:
            root_count_norm = merge_counts_norm[-1]
        else:
            root_count_norm = leaf_counts_norm[0]

        return TreeForwardOutputNO(
            leaf_states=leaf_states,
            merge_states=merge_states,
            leaf_counts_norm=leaf_counts_norm,
            merge_counts_norm=merge_counts_norm,
            root_state=root_state,
            root_count_norm=root_count_norm,
        )

    def forward(self, leaf_tokens: torch.Tensor) -> TreeForwardOutputNO:
        return self.forward_doc(leaf_tokens)


# ---------------------------------------------------------------------------
# Exact-zero Markov controls. These are deliberately simple and deterministic:
# the state is the sufficient sketch `(count, first, last)` rather than a
# generic learned vector. The join table is a learnable parameter initialized
# to the exact boundary-count rule, so it can be trained/perturbed in structural
# probes while the default witness reaches numerical zero immediately.
# ---------------------------------------------------------------------------


def _straight_through_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def _straight_through_one_hot(probs: torch.Tensor) -> torch.Tensor:
    idx = torch.argmax(probs, dim=-1)
    hard = F.one_hot(idx, num_classes=int(probs.shape[-1])).to(dtype=probs.dtype)
    return probs + (hard - probs).detach()


class MarkovSketchLeafEncoder(nn.Module):
    """Exact Markov leaf encoder for the sufficient sketch.

    Tokens are mapped to regimes via ``block_by_token``. The emitted state is
    ``[count / target_scale, first_regime_one_hot, last_regime_one_hot]``.
    The count is the deterministic number of adjacent regime transitions in
    the leaf, not a learned scalar head.
    """

    def __init__(
        self,
        *,
        block_by_token: Sequence[int],
        target_scale: float,
        vocab_size: int | None = None,
        n_regimes: int | None = None,
        count_discretization: str = "st_round",
    ) -> None:
        super().__init__()
        blocks = [int(x) for x in block_by_token]
        if not blocks:
            raise ValueError("block_by_token must be non-empty")
        resolved_vocab = len(blocks) if vocab_size is None else int(vocab_size)
        if resolved_vocab <= 0:
            raise ValueError("vocab_size must be positive")
        if len(blocks) < resolved_vocab:
            raise ValueError(
                "block_by_token must define a regime for every non-pad token; "
                f"got {len(blocks)} entries for vocab_size={resolved_vocab}"
            )
        resolved_regimes = int(max(blocks[:resolved_vocab])) + 1 if n_regimes is None else int(n_regimes)
        if resolved_regimes <= 0:
            raise ValueError("n_regimes must be positive")
        if any(block < 0 or block >= resolved_regimes for block in blocks[:resolved_vocab]):
            raise ValueError("block_by_token contains regime ids outside [0, n_regimes)")
        if count_discretization not in {"none", "round", "st_round"}:
            raise ValueError(
                "count_discretization must be 'none', 'round', or 'st_round'; "
                f"got {count_discretization!r}"
            )
        self.vocab_size = int(resolved_vocab)
        self.pad_id = int(resolved_vocab)
        self.n_regimes = int(resolved_regimes)
        self.target_scale = float(target_scale)
        self.count_discretization = str(count_discretization)
        mapping = torch.zeros(self.vocab_size + 1, dtype=torch.long)
        mapping[: self.vocab_size] = torch.as_tensor(
            blocks[: self.vocab_size],
            dtype=torch.long,
        )
        self.register_buffer("token_to_regime", mapping, persistent=False)

    @property
    def state_dim(self) -> int:
        return 1 + 2 * self.n_regimes

    def _discretize_count(self, count: torch.Tensor) -> torch.Tensor:
        if self.count_discretization == "none":
            return count
        if self.count_discretization == "round":
            return torch.round(count)
        return _straight_through_round(count)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2:
            raise ValueError(f"tokens must be (B, L); got shape {tuple(tokens.shape)}")
        if tokens.numel() and (int(tokens.min()) < 0 or int(tokens.max()) > self.pad_id):
            raise ValueError(
                f"tokens must be in [0, {self.pad_id}] where {self.pad_id} is pad_id"
            )
        valid = tokens.ne(self.pad_id)
        regime_ids = self.token_to_regime[tokens.to(dtype=torch.long)]
        regime_oh = F.one_hot(
            regime_ids,
            num_classes=self.n_regimes,
        ).to(dtype=torch.float32)
        if int(tokens.shape[1]) <= 1:
            count = tokens.new_zeros((int(tokens.shape[0]),), dtype=torch.float32)
        else:
            pair_valid = (valid[:, 1:] & valid[:, :-1]).to(dtype=regime_oh.dtype)
            same_regime = (regime_oh[:, 1:] * regime_oh[:, :-1]).sum(dim=-1)
            count = ((1.0 - same_regime) * pair_valid).sum(dim=-1)
        count = self._discretize_count(count)
        count_norm = count / float(self.target_scale)

        lengths = valid.to(dtype=torch.long).sum(dim=1)
        first_idx = torch.argmax(valid.to(dtype=torch.long), dim=1)
        last_idx = torch.clamp(lengths - 1, min=0)
        row_idx = torch.arange(int(tokens.shape[0]), device=tokens.device)
        first_oh = regime_oh[row_idx, first_idx]
        last_oh = regime_oh[row_idx, last_idx]
        return torch.cat([count_norm.unsqueeze(-1), first_oh, last_oh], dim=-1)


class MarkovSketchMergeG(nn.Module):
    """Merge two canonical Markov sketch states.

    The join contribution is ``1[first_left != last_right]`` expressed as a
    learnable table initialized to the exact deterministic rule.
    """

    def __init__(
        self,
        *,
        n_regimes: int,
        target_scale: float,
        learnable_join: bool = True,
        count_discretization: str = "st_round",
        canonicalize_endpoints: bool = True,
    ) -> None:
        super().__init__()
        if int(n_regimes) <= 0:
            raise ValueError("n_regimes must be positive")
        if count_discretization not in {"none", "round", "st_round"}:
            raise ValueError(
                "count_discretization must be 'none', 'round', or 'st_round'; "
                f"got {count_discretization!r}"
            )
        self.n_regimes = int(n_regimes)
        self.target_scale = float(target_scale)
        self.count_discretization = str(count_discretization)
        self.canonicalize_endpoints = bool(canonicalize_endpoints)
        exact_join = torch.ones(self.n_regimes, self.n_regimes, dtype=torch.float32)
        exact_join.fill_diagonal_(0.0)
        self.join_table = nn.Parameter(exact_join, requires_grad=bool(learnable_join))

    @property
    def state_dim(self) -> int:
        return 1 + 2 * self.n_regimes

    def _split_state(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state.ndim != 2 or int(state.shape[-1]) != self.state_dim:
            raise ValueError(
                f"state must be (B, {self.state_dim}); got shape {tuple(state.shape)}"
            )
        count = state[:, 0]
        first = state[:, 1 : 1 + self.n_regimes]
        last = state[:, 1 + self.n_regimes : 1 + 2 * self.n_regimes]
        if self.canonicalize_endpoints:
            first = _straight_through_one_hot(first)
            last = _straight_through_one_hot(last)
        return count, first, last

    def _discretize_count_norm(self, count_norm: torch.Tensor) -> torch.Tensor:
        if self.count_discretization == "none":
            return count_norm
        raw = count_norm * float(self.target_scale)
        if self.count_discretization == "round":
            raw = torch.round(raw)
        else:
            raw = _straight_through_round(raw)
        return raw / float(self.target_scale)

    def forward(
        self,
        left_state: torch.Tensor,
        right_state: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(left_state.shape) != tuple(right_state.shape):
            raise ValueError(
                "left and right states must have the same shape; "
                f"got {tuple(left_state.shape)} and {tuple(right_state.shape)}"
            )
        left_count, left_first, left_last = self._split_state(left_state)
        right_count, right_first, right_last = self._split_state(right_state)
        join_raw = (
            left_last.unsqueeze(-1)
            * right_first.unsqueeze(-2)
            * self.join_table.unsqueeze(0)
        ).sum(dim=(-2, -1))
        count_norm = left_count + right_count + join_raw / float(self.target_scale)
        count_norm = self._discretize_count_norm(count_norm)
        return torch.cat(
            [count_norm.unsqueeze(-1), left_first, right_last],
            dim=-1,
        )


class MarkovSketchScorerF(nn.Module):
    """Read the normalized count slot from a canonical Markov sketch state."""

    def __init__(self, *, n_regimes: int) -> None:
        super().__init__()
        if int(n_regimes) <= 0:
            raise ValueError("n_regimes must be positive")
        self.n_regimes = int(n_regimes)
        self.state_dim = 1 + 2 * self.n_regimes

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            if int(state.shape[-1]) != self.state_dim:
                raise ValueError(
                    f"state must have {self.state_dim} features; got shape {tuple(state.shape)}"
                )
            return state[0]
        if state.ndim != 2 or int(state.shape[-1]) != self.state_dim:
            raise ValueError(
                f"state must be (B, {self.state_dim}); got shape {tuple(state.shape)}"
            )
        return state[:, 0]


class LearnedMarkovSketchLeafEncoder(nn.Module):
    """Learned projection over exact Markov sketch-shaped leaf features.

    The input adapter is the theorem-domain sketch. The learnable part is a
    linear map initialized to identity so dense exact local-law training starts
    at the witness and can still be perturbed/optimized in ablations.
    """

    def __init__(
        self,
        *,
        block_by_token: Sequence[int],
        target_scale: float,
        vocab_size: int | None = None,
        n_regimes: int | None = None,
        count_discretization: str = "st_round",
    ) -> None:
        super().__init__()
        self.exact_encoder = MarkovSketchLeafEncoder(
            block_by_token=block_by_token,
            target_scale=target_scale,
            vocab_size=vocab_size,
            n_regimes=n_regimes,
            count_discretization=count_discretization,
        )
        self.projection = nn.Linear(self.exact_encoder.state_dim, self.exact_encoder.state_dim)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(self.exact_encoder.state_dim))
            self.projection.bias.zero_()

    @property
    def state_dim(self) -> int:
        return self.exact_encoder.state_dim

    @property
    def n_regimes(self) -> int:
        return self.exact_encoder.n_regimes

    @property
    def pad_id(self) -> int:
        return self.exact_encoder.pad_id

    def exact_state(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.exact_encoder(tokens)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(self.exact_encoder(tokens))


class ExactZeroMarkovFG(nn.Module):
    """Exact-zero f/g lane for the simple Markov sufficient-statistic setting."""

    def __init__(
        self,
        *,
        block_by_token: Sequence[int],
        target_scale: float,
        vocab_size: int | None = None,
        n_regimes: int | None = None,
        learnable_join: bool = True,
        count_discretization: str = "st_round",
    ) -> None:
        super().__init__()
        self.target_scale = float(target_scale)
        self.leaf_encoder = MarkovSketchLeafEncoder(
            block_by_token=block_by_token,
            target_scale=target_scale,
            vocab_size=vocab_size,
            n_regimes=n_regimes,
            count_discretization=count_discretization,
        )
        self.n_regimes = int(self.leaf_encoder.n_regimes)
        self.state_dim = int(self.leaf_encoder.state_dim)
        self.g = MarkovSketchMergeG(
            n_regimes=self.n_regimes,
            target_scale=target_scale,
            learnable_join=learnable_join,
            count_discretization=count_discretization,
        )
        self.f = MarkovSketchScorerF(n_regimes=self.n_regimes)

    def predict_count(self, state: torch.Tensor) -> torch.Tensor:
        return self.f(state) * float(self.target_scale)

    def forward_doc(self, leaf_tokens: torch.Tensor) -> TreeForwardOutput:
        if leaf_tokens.ndim != 2:
            raise ValueError(
                f"leaf_tokens must be (n_leaves, L); got shape {tuple(leaf_tokens.shape)}"
            )
        n_leaves = int(leaf_tokens.shape[0])
        if n_leaves == 0:
            raise ValueError("need at least one leaf")

        leaf_states_batch = self.leaf_encoder(leaf_tokens)
        leaf_states = [leaf_states_batch[i] for i in range(n_leaves)]
        leaf_counts_norm = self.f(leaf_states_batch)

        merge_states: List[torch.Tensor] = []
        cur = list(leaf_states)
        while len(cur) > 1:
            nxt: List[torch.Tensor] = []
            pair_count = len(cur) // 2
            if pair_count > 0:
                left = torch.stack(cur[: 2 * pair_count : 2], dim=0)
                right = torch.stack(cur[1 : 2 * pair_count : 2], dim=0)
                merged_batch = self.g(left, right)
                for k in range(pair_count):
                    merge_states.append(merged_batch[k])
                    nxt.append(merged_batch[k])
            if len(cur) % 2 == 1:
                nxt.append(cur[-1])
            cur = nxt

        if merge_states:
            merge_states_batch = torch.stack(merge_states, dim=0)
            merge_counts_norm = self.f(merge_states_batch)
        else:
            merge_counts_norm = leaf_counts_norm.new_empty((0,))

        root_state = cur[0]
        root_count_norm = merge_counts_norm[-1] if merge_states else leaf_counts_norm[0]
        return TreeForwardOutput(
            leaf_states=leaf_states,
            merge_states=merge_states,
            leaf_counts_norm=leaf_counts_norm,
            merge_counts_norm=merge_counts_norm,
            root_state=root_state,
            root_count_norm=root_count_norm,
        )

    def forward(self, leaf_tokens: torch.Tensor) -> TreeForwardOutput:
        return self.forward_doc(leaf_tokens)


class LearnedLocalLawMarkovFG(nn.Module):
    """Repo-owned learned Markov local-law lane.

    This keeps the public f/g shape of ``ExactZeroMarkovFG`` but makes the leaf
    state projection and merge join table trainable. The default initialization
    is the exact Markov count-sketch witness.
    """

    law_set_id = MARKOV_COUNT_SKETCH_LAW_SET_ID

    def __init__(
        self,
        *,
        block_by_token: Sequence[int],
        target_scale: float,
        vocab_size: int | None = None,
        n_regimes: int | None = None,
        count_discretization: str = "st_round",
    ) -> None:
        super().__init__()
        self.target_scale = float(target_scale)
        self.leaf_encoder = LearnedMarkovSketchLeafEncoder(
            block_by_token=block_by_token,
            target_scale=target_scale,
            vocab_size=vocab_size,
            n_regimes=n_regimes,
            count_discretization=count_discretization,
        )
        self.n_regimes = int(self.leaf_encoder.n_regimes)
        self.state_dim = int(self.leaf_encoder.state_dim)
        self.g = MarkovSketchMergeG(
            n_regimes=self.n_regimes,
            target_scale=target_scale,
            learnable_join=True,
            count_discretization=count_discretization,
        )
        self.f = MarkovSketchScorerF(n_regimes=self.n_regimes)

    def predict_count(self, state: torch.Tensor) -> torch.Tensor:
        return self.f(state) * float(self.target_scale)

    def exact_leaf_state(self, leaf_tokens: torch.Tensor) -> torch.Tensor:
        return self.leaf_encoder.exact_state(leaf_tokens)

    def forward_doc(self, leaf_tokens: torch.Tensor) -> TreeForwardOutput:
        if leaf_tokens.ndim != 2:
            raise ValueError(
                f"leaf_tokens must be (n_leaves, L); got shape {tuple(leaf_tokens.shape)}"
            )
        n_leaves = int(leaf_tokens.shape[0])
        if n_leaves == 0:
            raise ValueError("need at least one leaf")

        leaf_states_batch = self.leaf_encoder(leaf_tokens)
        leaf_states = [leaf_states_batch[i] for i in range(n_leaves)]
        leaf_counts_norm = self.f(leaf_states_batch)

        merge_states: List[torch.Tensor] = []
        cur = list(leaf_states)
        while len(cur) > 1:
            nxt: List[torch.Tensor] = []
            pair_count = len(cur) // 2
            if pair_count > 0:
                left = torch.stack(cur[: 2 * pair_count : 2], dim=0)
                right = torch.stack(cur[1 : 2 * pair_count : 2], dim=0)
                merged_batch = self.g(left, right)
                for k in range(pair_count):
                    merge_states.append(merged_batch[k])
                    nxt.append(merged_batch[k])
            if len(cur) % 2 == 1:
                nxt.append(cur[-1])
            cur = nxt

        if merge_states:
            merge_states_batch = torch.stack(merge_states, dim=0)
            merge_counts_norm = self.f(merge_states_batch)
        else:
            merge_counts_norm = leaf_counts_norm.new_empty((0,))

        root_state = cur[0]
        root_count_norm = merge_counts_norm[-1] if merge_states else leaf_counts_norm[0]
        return TreeForwardOutput(
            leaf_states=leaf_states,
            merge_states=merge_states,
            leaf_counts_norm=leaf_counts_norm,
            merge_counts_norm=merge_counts_norm,
            root_state=root_state,
            root_count_norm=root_count_norm,
        )

    def forward(self, leaf_tokens: torch.Tensor) -> TreeForwardOutput:
        return self.forward_doc(leaf_tokens)


# ---------------------------------------------------------------------------
# Loss helpers — clean, IPW-corrected MSE on observed labels at each tree
# position. The "local laws" (C1/C2/C3) are supervision-sparsity patterns:
# the trainer decides which subset of (root, leaves, merges) to observe and
# at what label rates; this module provides the loss building blocks.
# ---------------------------------------------------------------------------


def _normalized_label(count: torch.Tensor, target_scale: float) -> torch.Tensor:
    return count.to(dtype=torch.float32) / float(target_scale)


def root_mse_loss(
    output: TreeForwardOutput,
    *,
    root_count: torch.Tensor,
    target_scale: float,
) -> torch.Tensor:
    """MSE on the root's normalized count prediction."""
    target = _normalized_label(root_count, target_scale)
    return (output.root_count_norm - target.reshape(())) ** 2


def leaf_mse_loss(
    output: TreeForwardOutput,
    *,
    leaf_counts: torch.Tensor,
    leaf_observed: torch.Tensor | None = None,
    target_scale: float,
) -> torch.Tensor:
    """Mean MSE over observed leaves' normalized count predictions.

    Args:
        leaf_counts: (n_leaves,) tensor of true per-leaf counts.
        leaf_observed: (n_leaves,) bool/0-1 tensor — 1 if this leaf's label
            is observed in the current minibatch (C2 sparsity). Defaults to
            "all observed".
    """
    targets = _normalized_label(leaf_counts, target_scale)
    sq = (output.leaf_counts_norm - targets) ** 2
    if leaf_observed is None:
        return sq.mean()
    mask = leaf_observed.to(dtype=sq.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (sq * mask).sum() / denom


def merge_mse_loss(
    output: TreeForwardOutput,
    *,
    merge_counts: torch.Tensor,
    merge_observed: torch.Tensor | None = None,
    target_scale: float,
) -> torch.Tensor:
    """Mean MSE over observed internal-merge nodes' normalized count predictions."""
    if output.merge_counts_norm.numel() == 0:
        return output.leaf_counts_norm.new_zeros(())
    targets = _normalized_label(merge_counts, target_scale)
    sq = (output.merge_counts_norm - targets) ** 2
    if merge_observed is None:
        return sq.mean()
    mask = merge_observed.to(dtype=sq.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (sq * mask).sum() / denom
