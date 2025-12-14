# classifier_highlight.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class HighlightClassifier(nn.Module):
    def __init__(self, D: int, hidden: int = 256, p: float = 0.5):
        super().__init__()
        self.ln1   = nn.LayerNorm(D)
        self.drop1 = nn.Dropout(p)
        self.W1    = nn.Linear(D, hidden)

        self.ln2   = nn.LayerNorm(hidden)
        self.drop2 = nn.Dropout(p)
        self.W2    = nn.Linear(hidden, 1)

    def _forward_2d(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h = self.drop1(h)
        h = self.W1(h)
        h = F.relu(h)
        h = self.ln2(h)
        h = self.drop2(h)
        h = self.W2(h)
        logits = h.squeeze(-1)
        return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            return self._forward_2d(x)
        elif x.dim() == 3:
            B, T, D = x.shape
            x_flat = x.view(B * T, D)
            logits_flat = self._forward_2d(x_flat)
            logits = logits_flat.view(B, T)
            return logits
        else:
            raise ValueError(f"Expected x with shape (B, D) or (B, T, D), got {x.shape}")

    @torch.no_grad()
    def scores(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)
