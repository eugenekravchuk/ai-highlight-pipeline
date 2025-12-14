import numpy as np

def smooth_moving_average(x, win=7):
    x = np.asarray(x, dtype=np.float32)
    if win <= 1 or x.size == 0:
        return x
    win = int(win)
    if win % 2 == 0:
        win += 1
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(win, dtype=np.float32) / win
    return np.convolve(xp, kernel, mode="valid").astype(np.float32)

def merge_ranges(ranges, gap=0.0):
    if not ranges:
        return []
    ranges = sorted([(float(a), float(b)) for a, b in ranges], key=lambda x: x[0])
    out = [ranges[0]]
    for a, b in ranges[1:]:
        la, lb = out[-1]
        if a <= lb + gap:
            out[-1] = (la, max(lb, b))
        else:
            out.append((a, b))
    return out

def clip_ranges_to_budget(ranges, budget_s):
    if budget_s <= 0:
        return []
    out, acc = [], 0.0
    for a, b in ranges:
        dur = b - a
        if acc + dur <= budget_s:
            out.append((a, b))
            acc += dur
        else:
            remain = budget_s - acc
            if remain > 0:
                out.append((a, a + remain))
            break
    return out

def pick_peaks_nms(scores, seg_dur, min_sep_s=30.0, max_peaks=50, score_floor=None):
    s = np.asarray(scores, dtype=np.float32)
    T = s.size
    if T == 0:
        return []

    if score_floor is None:
        score_floor = float(np.quantile(s, 0.70))

    candidates = np.where(s >= score_floor)[0]
    if candidates.size == 0:
        candidates = np.arange(T)

    cand_sorted = candidates[np.argsort(s[candidates])[::-1]]

    min_sep_frames = max(1, int(round(min_sep_s / max(seg_dur, 1e-6))))
    chosen = []
    blocked = np.zeros(T, dtype=bool)

    for idx in cand_sorted:
        if blocked[idx]:
            continue
        chosen.append(int(idx))
        if len(chosen) >= max_peaks:
            break
        lo = max(0, idx - min_sep_frames)
        hi = min(T, idx + min_sep_frames + 1)
        blocked[lo:hi] = True

    chosen.sort()
    return chosen

def peaks_to_ranges(peaks, seg_dur, pre_s=12.0, post_s=18.0, clip_dur=None):
    ranges = []
    for p in peaks:
        t = p * seg_dur
        a = max(0.0, t - pre_s)
        b = t + post_s
        if clip_dur is not None:
            b = min(float(clip_dur), b)
        if b > a:
            ranges.append((a, b))
    return ranges

def build_highlight_ranges_from_scores(
    scores,
    seg_dur,
    clip_dur,
    budget_s=240.0,
    smooth_win=7,
    min_sep_s=30.0,
    pre_s=12.0,
    post_s=18.0,
    merge_gap_s=2.0,
    score_floor_q=0.75,
    exclude_start_s=30.0,
    exclude_end_s=0.0,
    abs_score_floor=None,
):
    s = np.asarray(scores, dtype=np.float32).reshape(-1)
    if s.size == 0 or seg_dur <= 0:
        return []

    s_sm = smooth_moving_average(s, win=smooth_win)

    if exclude_start_s > 0:
        k0 = int(np.ceil(exclude_start_s / seg_dur))
        s_sm[:k0] = -1e9
    if exclude_end_s > 0:
        k1 = int(np.floor(exclude_end_s / seg_dur))
        if k1 > 0:
            s_sm[-k1:] = -1e9

    valid = s_sm > -1e8
    q_floor = float(np.quantile(s_sm[valid], score_floor_q)) if np.any(valid) else -1e9
    score_floor = max(q_floor, float(abs_score_floor)) if abs_score_floor is not None else q_floor

    peaks = pick_peaks_nms(
        s_sm, seg_dur=seg_dur, min_sep_s=min_sep_s, max_peaks=200, score_floor=score_floor
    )
    ranges = peaks_to_ranges(peaks, seg_dur, pre_s=pre_s, post_s=post_s, clip_dur=clip_dur)
    ranges = merge_ranges(ranges, gap=merge_gap_s)

    if not ranges:
        return []

    range_scored = []
    for a, b in ranges:
        i0 = int(max(0, np.floor(a / seg_dur)))
        i1 = int(min(len(s_sm), np.ceil(b / seg_dur)))
        if i1 > i0:
            range_scored.append(((a, b), float(s_sm[i0:i1].max())))
    range_scored.sort(key=lambda x: x[1], reverse=True)

    picked, total = [], 0.0
    for (a, b), _ in range_scored:
        if total >= budget_s:
            break
        dur = b - a
        picked.append((a, b))
        total += dur

    picked = merge_ranges(picked, gap=merge_gap_s)
    picked = picked[:12]
    picked = sorted(picked, key=lambda x: x[0])
    picked = clip_ranges_to_budget(picked, budget_s)
    return picked

def time_ranges_to_segment_labels(time_ranges, T, seg_dur):
    labels = np.zeros(T, dtype=np.float32)
    for a, b in time_ranges:
        i0 = max(0, int(np.floor(a / seg_dur)))
        i1 = min(T, int(np.ceil(b / seg_dur)))
        if i1 > i0:
            labels[i0:i1] = 1.0
    return labels

def split_scores_by_embeddings(scores_flat, embeddings_list):
    out, idx = [], 0
    for emb in embeddings_list:
        L = emb.shape[0]
        out.append(np.array(scores_flat[idx:idx + L], dtype=np.float32))
        idx += L
    if idx != len(scores_flat):
        raise ValueError(f"Used {idx} scores, but have {len(scores_flat)}")
    return out
