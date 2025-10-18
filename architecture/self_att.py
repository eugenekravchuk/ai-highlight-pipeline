import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):

    def __init__(self, D, d):
        super().__init__()
        self.D = D
        self.d = d

        self.Wq = nn.Linear(D, d, bias=False)
        self.Wk = nn.Linear(D, d, bias=False)

        self.Wg = nn.Linear(D, d, bias=False)
        self.out_proj = nn.Linear(d, D, bias=False)

        self.Wm = nn.Linear(D, 1, bias=True)

        self.c = nn.Parameter(torch.tensor(1.0 / math.sqrt(d), dtype=torch.float32))

    def forward(self, x, mask=None):

        single_sample = False

        if x.dim() == 2:  # (T, D) -> treat as batch size 1
            x = x.unsqueeze(0)
            single_sample = True

        if x.dim() != 3:
            raise ValueError("x must have shape (B, T, D) or (T, D)")

        _, _, D = x.shape
        assert D == self.D, f"expected input D={self.D}, got {D}"

        q = self.Wq(x)
        k = self.Wk(x)
        g = self.Wg(x)
        m = self.Wm(x)

        q = q - q.mean(dim=1, keepdim=True)
        k = k - k.mean(dim=1, keepdim=True)

        logits = self.c * torch.matmul(q, k.transpose(-2, -1))

        if mask is not None:
            mask2d = mask.unsqueeze(1).expand_as(logits)
            logits = logits.masked_fill(~mask2d, float("-inf"))
    
        alpha = torch.softmax(logits, dim=-1) + torch.softmax(m.squeeze(-1).unsqueeze(1), dim=-1)

        attn_out = torch.matmul(alpha, g)

        out = self.out_proj(attn_out)
        out = out + x

        if single_sample:
            return out.squeeze(0) 
        return out
