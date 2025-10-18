# classifier_audio_only.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioClassifier(nn.Module):
    """
    Simple classifier for audio embeddings (after self-attention)
    Input:
      x: (B, D)  # one embedding vector per clip
    Output:
      logits: (B,)   # before sigmoid
      prob: (B,)     # after sigmoid (optional)
    """
    def __init__(self, D: int, hidden: int = 256, p: float = 0.5):
        super().__init__()
        self.ln1   = nn.LayerNorm(D)
        self.drop1 = nn.Dropout(p)
        self.W1    = nn.Linear(D, hidden)

        self.ln2   = nn.LayerNorm(hidden)
        self.drop2 = nn.Dropout(p)
        self.W2    = nn.Linear(hidden, 1)

    def forward(self, x):
        """
        x: (B, D)
        """
        h = self.W1(self.drop1(self.ln1(x)))  # (B, hidden)
        h = F.relu(h)
        h = self.W2(self.drop2(self.ln2(h)))  # (B, 1)
        logits = h.squeeze(-1)                # (B,)
        return logits

    @torch.no_grad()
    def scores(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits)          # highlight probability (B,)
