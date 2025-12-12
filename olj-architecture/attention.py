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

        if x.dim() == 2:
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
            mask = mask.to(torch.bool)
            mask2d = mask.unsqueeze(1).expand_as(logits)
            logits = logits.masked_fill(~mask2d, float("-inf"))
            m_scores = m.squeeze(-1).masked_fill(~mask, float("-inf"))
        else:
            m_scores = m.squeeze(-1)

        alpha = torch.softmax(logits, dim=-1) + torch.softmax(m_scores.unsqueeze(1), dim=-1)

        attn_out = torch.matmul(alpha, g)

        out = self.out_proj(attn_out)
        out = out + x

        if single_sample:
            return out.squeeze(0) 
        return out

class BimodalSelfAttention(nn.Module):

    def __init__(self, D, d):
        super().__init__()
        self.D = D
        self.d = d

        self.Wq_v2a = nn.Linear(D, d, bias=False)
        self.Wk_v2a = nn.Linear(D, d, bias=False)
        self.Wg_v2a = nn.Linear(D, d, bias=False)
        self.Wm_v2a = nn.Linear(D, 1, bias=True)
        self.out_v2a = nn.Linear(d, D, bias=False)

        self.Wq_a2v = nn.Linear(D, d, bias=False)
        self.Wk_a2v = nn.Linear(D, d, bias=False)
        self.Wg_a2v = nn.Linear(D, d, bias=False)
        self.Wm_a2v = nn.Linear(D, 1, bias=True)
        self.out_a2v = nn.Linear(d, D, bias=False)

        self.na = nn.Parameter(torch.zeros(D))
        self.nv = nn.Parameter(torch.zeros(D))

        self.c_v2a = nn.Parameter(torch.tensor(1.0 / math.sqrt(d), dtype=torch.float32))
        self.c_a2v = nn.Parameter(torch.tensor(1.0 / math.sqrt(d), dtype=torch.float32))

    def _cross_attention_with_noise(self, q_input, keys_input, Wq, Wk, Wg, Wm, out_proj, noise_param, c, keys_mask=None):
        B, T_q, D = q_input.shape
        _, T_k, _ = keys_input.shape
        device = q_input.device

        q = Wq(q_input)
        k = Wk(keys_input)
        g = Wg(keys_input)
        m = Wm(keys_input)

        knoise = Wk(noise_param.unsqueeze(0))
        gnoise = Wg(noise_param.unsqueeze(0))
        mnoise = Wm(noise_param.unsqueeze(0))

        q = q - q.mean(dim=1, keepdim=True)
        k = k - k.mean(dim=1, keepdim=True)

        sim_logits = c * torch.matmul(q, k.transpose(-2, -1))
        sim_noise = c * torch.matmul(q, knoise.transpose(-2, -1))

        sim_with_noise = torch.cat([sim_logits, sim_noise], dim=-1)

        if keys_mask is not None:
            noise_mask = torch.ones((B, 1), dtype=torch.bool, device=device)
            keys_mask_with_noise = torch.cat([keys_mask, noise_mask], dim=1)
        else:
            keys_mask_with_noise = torch.ones((B, T_k + 1), dtype=torch.bool, device=device)

        mask2d = keys_mask_with_noise.unsqueeze(1).expand_as(sim_with_noise)
        sim_with_noise = sim_with_noise.masked_fill(~mask2d, float("-inf"))

        m_logits = m.squeeze(-1).unsqueeze(1).expand(-1, T_q, -1)
        mnoise_expand = mnoise.view(1, 1, 1).expand(B, T_q, 1)
        m_with_noise = torch.cat([m_logits, mnoise_expand], dim=-1)

        m_with_noise = m_with_noise.masked_fill(~mask2d, float("-inf"))

        alpha = torch.softmax(sim_with_noise, dim=-1) + torch.softmax(m_with_noise, dim=-1)

        g_noise_expanded = gnoise.unsqueeze(0).expand(B, -1, -1)
        g_with_noise = torch.cat([g, g_noise_expanded], dim=1)

        attn_out = torch.matmul(alpha, g_with_noise)
        out = out_proj(attn_out)

        if T_k == T_q:
            residual = keys_input
        else:
            residual = keys_input.mean(dim=1, keepdim=True).expand(-1, T_q, -1)

        out = out + residual

        return out

    def forward(self, v_self, a_self, mask_v=None, mask_a=None):

        v2a = self._cross_attention_with_noise(
            q_input=v_self,
            keys_input=a_self,
            Wq=self.Wq_v2a, Wk=self.Wk_v2a, Wg=self.Wg_v2a, Wm=self.Wm_v2a,
            out_proj=self.out_v2a,
            noise_param=self.na,
            c=self.c_v2a,
            keys_mask=mask_a
        )
        a2v = self._cross_attention_with_noise(
            q_input=a_self,
            keys_input=v_self,
            Wq=self.Wq_a2v, Wk=self.Wk_a2v, Wg=self.Wg_a2v, Wm=self.Wm_a2v,
            out_proj=self.out_a2v,
            noise_param=self.nv,
            c=self.c_a2v,
            keys_mask=mask_v
        )

        return v2a, a2v
