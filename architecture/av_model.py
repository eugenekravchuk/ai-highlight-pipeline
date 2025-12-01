"""Audio-visual highlight detector with unimodal and cross-modal attention."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _invert_mask(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return ~mask


class TemporalSelfAttention(nn.Module):
    """Sequence-wise self-attention block with residual connection."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        key_padding = _invert_mask(mask)
        out, _ = self.attn(self.norm(x), self.norm(x), self.norm(x), key_padding_mask=key_padding, need_weights=False)
        out = self.drop(out)
        return out + x


class CrossAttention(nn.Module):
    """Cross-modal attention from query stream to context stream."""

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(dim)
        self.k_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        query_mask: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        key_padding = _invert_mask(context_mask)
        out, _ = self.attn(
            self.q_norm(query),
            self.k_norm(context),
            self.k_norm(context),
            key_padding_mask=key_padding,
            need_weights=False,
        )
        out = self.drop(out)
        return out + query


class ScoreRegressor(nn.Module):
    """Maps fused clip features to highlight logits."""

    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        return self.net(fused)


class AVHighlightDetector(nn.Module):
    """Audio-visual highlight scoring head inspired by the WACV'25 paper."""

    def __init__(
        self,
        audio_dim: int,
        visual_dim: int,
        model_dim: int = 512,
        num_heads: int = 4,
        dropout: float = 0.1,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.audio_in = nn.Linear(audio_dim, model_dim)
        self.visual_in = nn.Linear(visual_dim, model_dim)

        self.audio_self = TemporalSelfAttention(model_dim, num_heads, dropout)
        self.visual_self = TemporalSelfAttention(model_dim, num_heads, dropout)

        self.audio_cross = CrossAttention(model_dim, num_heads, dropout)
        self.visual_cross = CrossAttention(model_dim, num_heads, dropout)

        fusion_dim = model_dim * 4
        self.regressor = ScoreRegressor(fusion_dim, hidden_dim=hidden_dim, dropout=dropout)

    def forward(
        self,
        audio_feats: torch.Tensor,
        visual_feats: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return per-clip logits (B, T)."""

        if audio_feats.shape[:2] != visual_feats.shape[:2]:
            raise ValueError("Audio and visual tensors must share B and T dimensions")

        audio = self.audio_in(audio_feats)
        visual = self.visual_in(visual_feats)

        audio_self = self.audio_self(audio, mask)
        visual_self = self.visual_self(visual, mask)

        audio_cross = self.audio_cross(audio_self, visual_self, mask, mask)
        visual_cross = self.visual_cross(visual_self, audio_self, mask, mask)

        fused = torch.cat([audio_self, visual_self, audio_cross, visual_cross], dim=-1)
        logits = self.regressor(fused).squeeze(-1)
        return logits

    @torch.no_grad()
    def predict_scores(
        self,
        audio_feats: torch.Tensor,
        visual_feats: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = self.forward(audio_feats, visual_feats, mask)
        return torch.sigmoid(logits)


def masked_bce_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy averaged over valid (mask==1) positions."""

    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    mask = mask.float()
    loss = loss * mask
    denom = mask.sum().clamp(min=1.0)
    return loss.sum() / denom
