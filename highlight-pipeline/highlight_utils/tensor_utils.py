import torch

def to_padded_batch(emb_list, device):
    tensors = [torch.as_tensor(e, dtype=torch.float32, device=device) for e in emb_list]
    lens = [t.shape[0] for t in tensors]
    if not lens:
        raise ValueError("Empty emb_list")
    D = tensors[0].shape[1]
    T_max = max(lens)
    B = len(tensors)

    x = torch.zeros((B, T_max, D), dtype=torch.float32, device=device)
    m = torch.zeros((B, T_max), dtype=torch.bool, device=device)

    for i, t in enumerate(tensors):
        L = t.shape[0]
        x[i, :L] = t
        m[i, :L] = True

    return x, m, lens

def unpad_to_lists(x, lens):
    return [x[i, :L].detach().cpu() for i, L in enumerate(lens)]
