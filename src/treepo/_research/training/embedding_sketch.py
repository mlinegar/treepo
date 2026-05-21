"""
Mergeable learned sketch over multilingual embeddings.

This implements a simple, *strictly mergeable* DeepSets-style sketch:

  state(text) = ( sum_i φ(e_i), count )
  merge(a, b) = ( a.sum_phi + b.sum_phi, a.count + b.count )

where e_i are window embeddings for a long document and φ is a small MLP.

Downstream tasks (e.g., RILE regression) use a readout head on the merged state.

Why this structure:
  - window pooling is a special case (φ = identity, sum+count -> mean)
  - mergeability lets us aggregate representations along a tree without
    re-embedding or re-processing all raw text at every merge.

This module is currently proxy-only for Lean local-law purposes: it exposes a
mergeable latent state plus scalar readouts, but not the decode/re-summary
interface required by the sketch theorems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn

from treepo._research.core.ops_checks import (
    EvidenceStatus,
    LawCapabilityReport,
    LawKind,
    OperatorCapabilityReport,
)


@dataclass(frozen=True)
class EmbeddingSketchConfig:
    embedding_dim: int
    state_dim: int = 64
    phi_hidden_dim: int = 256
    readout_hidden_dim: int = 128
    dropout: float = 0.0

    include_meta: bool = True
    meta_hidden_dim: int = 128
    use_count_feature: bool = True
    include_retrieval_features: bool = False
    retrieval_feature_dim: int = 6
    include_delta_head: bool = False


@dataclass(frozen=True)
class SketchState:
    """Mergeable sketch state."""

    sum_phi: torch.Tensor  # (..., state_dim)
    count: torch.Tensor  # (...) float or int

    def merge(self, other: "SketchState") -> "SketchState":
        return SketchState(
            sum_phi=self.sum_phi + other.sum_phi,
            count=self.count + other.count,
        )


class MergeableEmbeddingSketch(nn.Module):
    """Learned mergeable sketch with optional metadata conditioning."""

    def __init__(self, config: EmbeddingSketchConfig):
        super().__init__()
        self.config = config
        self.evidence_status = EvidenceStatus.PROXY_ONLY

        d = int(config.embedding_dim)
        m = int(config.state_dim)
        phi_h = int(config.phi_hidden_dim)
        ro_h = int(config.readout_hidden_dim)
        drop = float(config.dropout)

        self.phi = nn.Sequential(
            nn.Linear(d, phi_h),
            nn.ReLU(),
            nn.Dropout(p=drop) if drop > 0 else nn.Identity(),
            nn.Linear(phi_h, m),
        )

        self.meta_proj: nn.Module
        if config.include_meta:
            meta_h = int(config.meta_hidden_dim)
            self.meta_proj = nn.Sequential(
                nn.Linear(d, meta_h),
                nn.ReLU(),
                nn.Dropout(p=drop) if drop > 0 else nn.Identity(),
                nn.Linear(meta_h, m),
            )
        else:
            self.meta_proj = nn.Identity()

        readout_in = m
        if config.use_count_feature:
            readout_in += 1
        if config.include_meta:
            readout_in += m
        if config.include_retrieval_features:
            readout_in += int(config.retrieval_feature_dim)

        self.readout_rile = nn.Sequential(
            nn.Linear(readout_in, ro_h),
            nn.ReLU(),
            nn.Dropout(p=drop) if drop > 0 else nn.Identity(),
            nn.Linear(ro_h, 1),
        )
        self.readout_delta: Optional[nn.Module]
        if config.include_delta_head:
            self.readout_delta = nn.Sequential(
                nn.Linear(readout_in, ro_h),
                nn.ReLU(),
                nn.Dropout(p=drop) if drop > 0 else nn.Identity(),
                nn.Linear(ro_h, 1),
            )
        else:
            self.readout_delta = None

    def encode_windows(
        self,
        window_embeddings: torch.Tensor,
        *,
        counts: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> SketchState:
        """
        Encode a padded batch of windows into a mergeable state.

        Args:
            window_embeddings: [B, W, D]
            counts: Optional [B] number of valid windows per row.
            mask: Optional [B, W] boolean mask (True=valid).
        """
        if window_embeddings.ndim != 3:
            raise ValueError(f"window_embeddings must be [B,W,D], got shape {tuple(window_embeddings.shape)}")

        bsz, w, d = window_embeddings.shape
        flat = window_embeddings.reshape(bsz * w, d)
        phi_flat = self.phi(flat).reshape(bsz, w, -1)  # [B,W,M]

        if mask is None:
            if counts is None:
                mask = torch.ones((bsz, w), dtype=torch.bool, device=window_embeddings.device)
                counts = torch.full((bsz,), float(w), device=window_embeddings.device)
            else:
                counts = counts.to(window_embeddings.device)
                idx = torch.arange(w, device=window_embeddings.device).unsqueeze(0).expand(bsz, w)
                mask = idx < counts.unsqueeze(1)

        mask_f = mask.to(dtype=phi_flat.dtype).unsqueeze(-1)
        phi_masked = phi_flat * mask_f
        sum_phi = phi_masked.sum(dim=1)
        count = mask_f.squeeze(-1).sum(dim=1)
        return SketchState(sum_phi=sum_phi, count=count)

    def predict_from_state(
        self,
        state: SketchState,
        *,
        meta_embeddings: Optional[torch.Tensor] = None,
        retrieval_features: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Optional[torch.Tensor]]:
        """Predict normalized score in [0,1] and optional delta in [-1,1]."""
        sum_phi = state.sum_phi
        count = state.count
        denom = torch.clamp(count, min=1.0).unsqueeze(-1)
        avg_phi = sum_phi / denom

        feats = [avg_phi]
        if self.config.use_count_feature:
            feats.append(torch.log1p(torch.clamp(count, min=0.0)).unsqueeze(-1))

        if self.config.include_meta:
            if meta_embeddings is None:
                meta_feat = torch.zeros_like(avg_phi)
            else:
                meta_feat = self.meta_proj(meta_embeddings.to(avg_phi.device))
            feats.append(meta_feat)

        if self.config.include_retrieval_features:
            rdim = int(self.config.retrieval_feature_dim)
            if retrieval_features is None:
                ret_feat = torch.zeros((avg_phi.shape[0], rdim), dtype=avg_phi.dtype, device=avg_phi.device)
            else:
                ret_feat = retrieval_features.to(device=avg_phi.device, dtype=avg_phi.dtype)
                if ret_feat.ndim == 1:
                    ret_feat = ret_feat.unsqueeze(0)
                if ret_feat.shape[-1] < rdim:
                    pad = torch.zeros(
                        (ret_feat.shape[0], rdim - int(ret_feat.shape[-1])),
                        dtype=ret_feat.dtype,
                        device=ret_feat.device,
                    )
                    ret_feat = torch.cat([ret_feat, pad], dim=-1)
                elif ret_feat.shape[-1] > rdim:
                    ret_feat = ret_feat[:, :rdim]
                if ret_feat.shape[0] != avg_phi.shape[0]:
                    if ret_feat.shape[0] == 1:
                        ret_feat = ret_feat.expand(avg_phi.shape[0], -1)
                    else:
                        raise ValueError(
                            f"retrieval_features batch mismatch: got {ret_feat.shape[0]}, expected {avg_phi.shape[0]}"
                        )
            feats.append(ret_feat)

        x = torch.cat(feats, dim=-1)
        rile_logit = self.readout_rile(x).squeeze(-1)
        rile = torch.sigmoid(rile_logit)

        delta: Optional[torch.Tensor] = None
        if self.readout_delta is not None:
            delta = torch.tanh(self.readout_delta(x).squeeze(-1))

        if return_dict:
            return {"rile": rile, "delta": delta}
        return rile

    def forward(
        self,
        window_embeddings: torch.Tensor,
        *,
        counts: Optional[torch.Tensor] = None,
        meta_embeddings: Optional[torch.Tensor] = None,
        retrieval_features: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, Optional[torch.Tensor]]:
        state = self.encode_windows(window_embeddings, counts=counts)
        return self.predict_from_state(
            state,
            meta_embeddings=meta_embeddings,
            retrieval_features=retrieval_features,
            return_dict=return_dict,
        )

    def local_law_capabilities(self) -> Dict[str, object]:
        """
        Describe which local-law properties this proxy model actually supports.

        The latent state is exactly mergeable by construction, but the model does
        not expose a theorem-domain decode/re-summary path, so C1/C2/C3-style
        span-oracle supervision is not available in the core architecture alone.
        """
        return self.capability_report().to_dict()

    def capability_report(self) -> OperatorCapabilityReport:
        """Structured local-law capability report for the core architecture."""
        return OperatorCapabilityReport(
            operator_name="mergeable_embedding_sketch",
            evidence_status=self.evidence_status,
            latent_mergeability_enforced=True,
            tree_nesting_supported=True,
            theorem_domain_decode_available=False,
            theorem_domain_reencode_available=False,
            exact_reduction_supported=True,
            leaf_law=LawCapabilityReport(
                law_kind=LawKind.L1_LEAF,
                available=False,
                evidence_status=self.evidence_status,
                exact=False,
                notes="No decoded theorem-domain summary is exposed for leaf preservation checks.",
            ),
            merge_law=LawCapabilityReport(
                law_kind=LawKind.L2_MERGE,
                available=False,
                evidence_status=self.evidence_status,
                exact=False,
                notes="Latent-state merge is exact, but theorem-domain span preservation is not yet exposed.",
            ),
            idempotence_law=LawCapabilityReport(
                law_kind=LawKind.L3_IDEMPOTENCE,
                available=False,
                evidence_status=self.evidence_status,
                exact=False,
                notes="A theorem-domain decode/re-encode path is required before L3-style re-summary checks are meaningful.",
            ),
            notes=(
                "The latent state (sum_phi, count) is exactly mergeable by construction.",
                "The model becomes theorem-backed only after attaching a supplied theorem-domain codec/certificate.",
            ),
        )


@torch.no_grad()
def merge_prediction_consistency(
    model: MergeableEmbeddingSketch,
    window_embeddings: torch.Tensor,
    *,
    counts: Optional[torch.Tensor] = None,
    meta_embeddings: Optional[torch.Tensor] = None,
    retrieval_features: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """
    Measure the exact mergeability that *is* enforced by this architecture.

    We compare predictions from:
      1) the full encoded state, and
      2) a split-half encode followed by exact state merge.
    """
    if window_embeddings.ndim != 3:
        raise ValueError(f"window_embeddings must be [B,W,D], got {tuple(window_embeddings.shape)}")

    bsz, width, _dim = window_embeddings.shape
    if counts is None:
        counts = torch.full((bsz,), int(width), dtype=torch.int64, device=window_embeddings.device)
    else:
        counts = counts.to(device=window_embeddings.device, dtype=torch.int64)

    split = max(1, int(width // 2))
    left_counts = torch.clamp(counts, min=0, max=split)
    right_counts = torch.clamp(counts - split, min=0)

    full_state = model.encode_windows(window_embeddings, counts=counts)
    left_state = model.encode_windows(window_embeddings[:, :split, :], counts=left_counts)
    right_state = model.encode_windows(window_embeddings[:, split:, :], counts=right_counts)
    merged_state = left_state.merge(right_state)

    full_out = model.predict_from_state(
        full_state,
        meta_embeddings=meta_embeddings,
        retrieval_features=retrieval_features,
    )
    merged_out = model.predict_from_state(
        merged_state,
        meta_embeddings=meta_embeddings,
        retrieval_features=retrieval_features,
    )
    if isinstance(full_out, dict) or isinstance(merged_out, dict):
        raise ValueError("merge_prediction_consistency expects scalar rile predictions, not dict outputs")

    diffs = torch.abs(full_out - merged_out).detach().cpu()
    return {
        "prediction_mae": float(diffs.mean().item()) if diffs.numel() else 0.0,
        "prediction_max_abs": float(diffs.max().item()) if diffs.numel() else 0.0,
        "n_samples": float(diffs.numel()),
    }
