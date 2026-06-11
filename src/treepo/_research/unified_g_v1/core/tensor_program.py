from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
except ImportError:
    raise ImportError(
        "PyTorch is required for tensor Unified-G programs. "
        "Install with: uv sync --extra torch"
    )

from treepo._research.unified_g_v1.core.program import UnifiedFGProgram, UnifiedGContract, UnifiedGSurface
from treepo._research.unified_g_v1.core.specs import build_embedding_sequence_fno_program_spec
from treepo._research.unified_g_v1.dimension_guards import promote_dim


# Repo-wide invariant: the FNO-style operator's output (state_dim) must be at
# least FNO_HEAD_MIN_RATIO × its input (summary_dim). This ensures the
# summarizer g can worst-case concatenate two leaves rather than compress —
# for a learned g to match classical mergeable-sketch behavior it needs room
# to carry both inputs through the merge.
FNO_HEAD_MIN_RATIO: int = 2


def _resolve_fno_state_dim(
    *,
    summary_dim: int,
    state_dim: int | None,
    context: str,
) -> int:
    """Resolve `state_dim >= FNO_HEAD_MIN_RATIO * summary_dim` with a warning."""

    min_state_dim = int(FNO_HEAD_MIN_RATIO) * int(summary_dim)
    return promote_dim(
        name="state_dim",
        requested=state_dim,
        default=min_state_dim,
        minimum=min_state_dim,
        context=context,
        reason=(
            f"FNO-head ratio must be >= {FNO_HEAD_MIN_RATIO} * summary_dim "
            "so g can carry both child summaries"
        ),
    )


def resolve_operator_head_width(
    embedding_dim: int,
    *,
    head_width: int | None = None,
) -> int:
    """Resolve the operator head width to roughly 1x-2x the embedding width."""

    embedding_width = int(embedding_dim)
    if embedding_width <= 0:
        raise ValueError("embedding_dim must be positive")
    lower = embedding_width
    upper = 2 * embedding_width
    if head_width is None:
        return max(lower, min(upper, int(round(1.5 * embedding_width))))
    resolved = int(head_width)
    if resolved < lower:
        return promote_dim(
            name="head_width",
            requested=resolved,
            default=max(lower, min(upper, int(round(1.5 * embedding_width)))),
            minimum=lower,
            context="resolve_operator_head_width",
            reason="operator head width must be at least 1x the embedding width",
        )
    if resolved > upper:
        raise ValueError(
            "embedding operator head_width must stay within 1x-2x the embedding width; "
            f"got head_width={resolved} for embedding_dim={embedding_width}"
        )
    return resolved


class EmbeddingLeafAdapter(nn.Module):
    """Projects ordered embedding sequences onto the shared g-input surface."""

    def __init__(
        self,
        embedding_dim: int,
        summary_dim: int,
        hidden_dim: int,
        n_modes: int | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.summary_dim = int(summary_dim)
        self.hidden_dim = int(hidden_dim)
        self.token_proj = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.summary_dim),
        )
        max_modes = max(2, (self.summary_dim // 2) + 1)
        resolved_modes = int(n_modes or max(4, self.summary_dim // 8))
        self.n_modes = max(2, min(max_modes, resolved_modes))
        self.spectral_weight_real = nn.Parameter(
            torch.randn(self.summary_dim, self.n_modes) * 0.02
        )
        self.spectral_weight_imag = nn.Parameter(
            torch.randn(self.summary_dim, self.n_modes) * 0.02
        )
        self.output_norm = nn.LayerNorm(self.summary_dim)

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(embedding, dtype=torch.float32)
        if tensor.ndim == 1:
            tensor = tensor.reshape(1, 1, -1)
        elif tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim != 3:
            raise ValueError("embedding leaf input must be rank-1, rank-2, or rank-3")

        projected = self.token_proj(tensor)
        channels_first = projected.transpose(-1, -2)
        freq = torch.fft.rfft(channels_first, dim=-1)
        active_modes = max(1, min(int(self.n_modes), int(freq.shape[-1])))
        weights = torch.complex(
            self.spectral_weight_real[:, :active_modes],
            self.spectral_weight_imag[:, :active_modes],
        )
        if active_modes < int(freq.shape[-1]):
            mixed = torch.cat(
                [
                    freq[..., :active_modes] * weights.unsqueeze(0),
                    freq[..., active_modes:],
                ],
                dim=-1,
            )
        else:
            mixed = freq[..., :active_modes] * weights.unsqueeze(0)
        recovered = torch.fft.irfft(mixed, n=channels_first.shape[-1], dim=-1)
        pooled = recovered.mean(dim=-1)
        return self.output_norm(pooled)


class StatePairMergeAdapter(nn.Module):
    """Learns the merge-side g input from two child embedding states."""

    def __init__(
        self,
        state_dim: int,
        summary_dim: int,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * int(state_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(summary_dim)),
        )

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([left, right], dim=-1))


class SharedTensorG(nn.Module):
    """Single learned operator-style embedding-tree summary function g."""

    def __init__(
        self,
        summary_dim: int,
        state_dim: int,
        hidden_dim: int,
        n_modes: int | None = None,
    ) -> None:
        super().__init__()
        self.summary_dim = int(summary_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.summary_dim),
            nn.Linear(self.summary_dim, self.hidden_dim),
        )
        max_modes = max(2, (self.hidden_dim // 2) + 1)
        resolved_modes = int(n_modes or max(4, self.hidden_dim // 8))
        self.n_modes = max(2, min(max_modes, resolved_modes))
        self.spectral_weight_real = nn.Parameter(torch.randn(self.n_modes) * 0.02)
        self.spectral_weight_imag = nn.Parameter(torch.randn(self.n_modes) * 0.02)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.state_dim),
        )

    def forward(self, summary_surface: torch.Tensor) -> torch.Tensor:
        signal = self.input_proj(summary_surface)
        freq = torch.fft.rfft(signal, dim=-1)
        weights = torch.complex(self.spectral_weight_real, self.spectral_weight_imag)
        active_modes = max(1, min(int(self.n_modes), int(freq.shape[-1])))
        if active_modes < int(freq.shape[-1]):
            mixed = torch.cat(
                [
                    freq[..., :active_modes] * weights[:active_modes],
                    freq[..., active_modes:],
                ],
                dim=-1,
            )
        else:
            mixed = freq[..., :active_modes] * weights[:active_modes]
        recovered = torch.fft.irfft(mixed, n=self.hidden_dim, dim=-1)
        return self.output_proj(signal + recovered)


class NeuralOperatorFHead(nn.Module):
    """Operator-style readout over embedding channels.

    This is a lane-local 1D Fourier head: it treats the state vector as a field
    over embedding channels, mixes low-frequency modes, and projects to the task
    output. It is intentionally the embedding-side ``f`` counterpart to the
    Markov neural-operator route.
    """

    def __init__(
        self,
        state_dim: int,
        embedding_dim: int,
        *,
        head_width: int | None = None,
        output_dim: int = 1,
        n_modes: int | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.state_dim = int(state_dim)
        self.head_width = resolve_operator_head_width(
            int(embedding_dim),
            head_width=head_width,
        )
        max_modes = max(2, (self.head_width // 2) + 1)
        resolved_modes = int(n_modes or max(4, self.head_width // 8))
        self.n_modes = max(2, min(max_modes, resolved_modes))
        self.input_proj = nn.Sequential(
            nn.LayerNorm(self.state_dim),
            nn.Linear(self.state_dim, self.head_width),
        )
        self.pre_mix_norm = nn.LayerNorm(self.head_width)
        self.spectral_weight_real = nn.Parameter(torch.randn(self.n_modes) * 0.02)
        self.spectral_weight_imag = nn.Parameter(torch.randn(self.n_modes) * 0.02)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(self.head_width),
            nn.SiLU(),
            nn.Linear(self.head_width, int(output_dim)),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        signal = self.input_proj(state)
        normalized = self.pre_mix_norm(signal)
        freq = torch.fft.rfft(normalized, dim=-1)
        weights = torch.complex(self.spectral_weight_real, self.spectral_weight_imag)
        active_modes = max(1, min(int(self.n_modes), int(freq.shape[-1])))
        if active_modes < int(freq.shape[-1]):
            mixed = torch.cat(
                [
                    freq[..., :active_modes] * weights[:active_modes],
                    freq[..., active_modes:],
                ],
                dim=-1,
            )
        else:
            mixed = freq[..., :active_modes] * weights[:active_modes]
        recovered = torch.fft.irfft(mixed, n=self.head_width, dim=-1)
        return self.output_proj(signal + recovered)


@dataclass
class EmbeddingOperatorModules:
    leaf_adapter: EmbeddingLeafAdapter
    merge_adapter: StatePairMergeAdapter
    g: SharedTensorG
    f: NeuralOperatorFHead


EmbeddingOperatorUnifiedFGProgram = UnifiedFGProgram[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    str,
]


def build_embedding_operator_unified_fg_program(
    *,
    embedding_dim: int,
    summary_dim: int | None = None,
    state_dim: int | None = None,
    adapter_hidden_dim: int | None = None,
    g_hidden_dim: int | None = None,
    head_width: int | None = None,
    operator_modes: int | None = None,
    output_dim: int = 1,
) -> EmbeddingOperatorUnifiedFGProgram:
    """Build the embedding-sequence/operator unified-f/g backend.

    Semantics:
    - leaves provide ordered embedding sequences
    - the learned operator ``g`` summarizes both leaves and merges
    - the operator-style readout ``f`` performs the final task mapping
    """

    resolved_embedding_dim = int(embedding_dim)
    resolved_summary_dim = int(summary_dim or resolved_embedding_dim)
    # Repo-wide invariant: the FNO head (state_dim, output of g) must be at
    # least 2 × summary_dim so the summarizer can worst-case concatenate two
    # leaves rather than compress. When state_dim is unset, default to 2 ×
    # summary_dim. When explicitly set below 2 × summary_dim, validate below.
    resolved_state_dim = _resolve_fno_state_dim(
        summary_dim=resolved_summary_dim,
        state_dim=state_dim,
        context="build_embedding_operator_unified_fg_program",
    )
    adapter_hidden_floor = max(resolved_summary_dim, resolved_embedding_dim)
    resolved_adapter_hidden_dim = promote_dim(
        name="adapter_hidden_dim",
        requested=adapter_hidden_dim,
        default=adapter_hidden_floor,
        minimum=adapter_hidden_floor,
        context="build_embedding_operator_unified_fg_program",
        reason="leaf/merge adapters must have at least 1x input feature width",
    )
    g_hidden_floor = max(2 * resolved_state_dim, resolved_summary_dim)
    resolved_g_hidden_dim = promote_dim(
        name="g_hidden_dim",
        requested=g_hidden_dim,
        default=g_hidden_floor,
        minimum=g_hidden_floor,
        context="build_embedding_operator_unified_fg_program",
        reason="g hidden width must cover the concatenated 2*state_dim merge input",
    )
    resolved_head_width = resolve_operator_head_width(
        resolved_embedding_dim,
        head_width=head_width,
    )

    modules = EmbeddingOperatorModules(
        leaf_adapter=EmbeddingLeafAdapter(
            embedding_dim=resolved_embedding_dim,
            summary_dim=resolved_summary_dim,
            hidden_dim=resolved_adapter_hidden_dim,
            n_modes=operator_modes,
        ),
        merge_adapter=StatePairMergeAdapter(
            state_dim=resolved_state_dim,
            summary_dim=resolved_summary_dim,
            hidden_dim=resolved_adapter_hidden_dim,
        ),
        g=SharedTensorG(
            summary_dim=resolved_summary_dim,
            state_dim=resolved_state_dim,
            hidden_dim=resolved_g_hidden_dim,
            n_modes=operator_modes,
        ),
        f=NeuralOperatorFHead(
            state_dim=resolved_state_dim,
            embedding_dim=resolved_embedding_dim,
            head_width=resolved_head_width,
            output_dim=int(output_dim),
            n_modes=operator_modes,
        ),
    )
    program_spec = build_embedding_sequence_fno_program_spec(
        feature_dim=resolved_embedding_dim,
        tokenizer_or_adapter_id="embedding_sequence",
        operator_width=resolved_head_width,
        operator_modes=int(modules.f.n_modes),
    )
    contract = UnifiedGContract(
        name="embedding_sequence_fno_unified_fg",
        surface=UnifiedGSurface(
            raw_input_kind="ordered_embedding_sequence",
            g_input_kind="embedding_sequence_summary_surface",
            state_kind="embedding_sequence_tree_state",
            output_kind="operator_prediction",
            task_spec_kind="head_name",
            backend_family=str(program_spec.program_family),
            shared_g=True,
            shared_f=True,
        ),
        leaf_adapter_name="ordered_embedding_sequence_adapter",
        merge_adapter_name="embedding_state_pair_merge_adapter",
        g_name="shared_embedding_sequence_fno_g",
        f_name="embedding_sequence_fno_head",
        notes=(
            "Embedding-side backend: leaves are ordered embedding sequences; g and f "
            "are both operator-style modules over the embedding space."
        ),
        program_spec=program_spec,
        extra={
            "approach_kind": "embedding_operator",
            "embedding_dim": resolved_embedding_dim,
            "summary_dim": resolved_summary_dim,
            "state_dim": resolved_state_dim,
            "adapter_hidden_dim": resolved_adapter_hidden_dim,
            "g_hidden_dim": resolved_g_hidden_dim,
            "g_input_dim": 2 * resolved_state_dim,
            "operator_head_width": resolved_head_width,
            "operator_modes": int(modules.f.n_modes),
            "operator_impl": "torch_fft_1d",
            "space_kind": "embedding_sequence",
        },
    )
    return UnifiedFGProgram(
        contract=contract,
        leaf_adapter=lambda embedding, _task_spec=None: modules.leaf_adapter(embedding),
        merge_adapter=lambda left, right, _task_spec=None: modules.merge_adapter(left, right),
        g=lambda summary_surface, _task_spec=None: modules.g(summary_surface),
        f=lambda state, _task_spec=None: modules.f(state),
        runtime=modules,
    )


def build_embedding_unified_g_program(
    *,
    embedding_dim: int,
    summary_dim: int | None = None,
    state_dim: int | None = None,
    adapter_hidden_dim: int | None = None,
    g_hidden_dim: int | None = None,
    head_width: int | None = None,
    operator_modes: int | None = None,
    output_dim: int = 1,
) -> EmbeddingOperatorUnifiedFGProgram:
    """Backward-compatible alias for the embedding/operator unified-f/g backend."""

    return build_embedding_operator_unified_fg_program(
        embedding_dim=int(embedding_dim),
        summary_dim=summary_dim,
        state_dim=state_dim,
        adapter_hidden_dim=adapter_hidden_dim,
        g_hidden_dim=g_hidden_dim,
        head_width=head_width,
        operator_modes=operator_modes,
        output_dim=int(output_dim),
    )
