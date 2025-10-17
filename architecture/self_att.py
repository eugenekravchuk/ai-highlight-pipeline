import torch
import torch.nn as nn
import math


class SelfAttention(nn.Module):
    def __init__(self, D, d):
        super().__init__()
        self.Wq = nn.Linear(D, d, bias=False)
        self.Wk = nn.Linear(D, d, bias=False)
        self.Wg = nn.Linear(D, d, bias=False)
        self.Wm = nn.Linear(D, 1, bias=True)
        self.fc = nn.Linear(D, 1)
        self.c = nn.Parameter(torch.tensor(1.0 / math.sqrt(d)))

    def _mean_projection(self, projected_vecs):

        vec_sz = len(projected_vecs)
        vec_sum = torch.tensor.zero(projected_vecs[0].shape())

        for vec in projected_vecs:
            vec_sum += vec
        
        mean_vec = vec_sum / vec_sz

        return mean_vec


    def forward(self, x):
        q_hat = self.Wq(x)
        k_hat = self.Wk(x)

        q = q_hat - self._mean_projection(q_hat)
        k = k_hat - self._mean_projection(k_hat)

        m = self.Wm(x)

        omega = torch.softmax(self.c * q.T @ k) + torch.softmax(m)

        g = self.Wg(x)

        v_v = omega @ g + x

        return v_v


