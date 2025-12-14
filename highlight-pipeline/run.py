#!/usr/bin/env python3
import argparse
from pathlib import Path
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from moviepy import VideoFileClip

from core import (
    list_files_oswalk,
    preprocess_audio_paths,
    preprocess_video_paths,
    get_video_embeddings_list,
    get_audio_embeddings_list,
    get_pseudo_highlight_scores,
    SelfAttention,
    BimodalSelfAttention,
    HighlightClassifier,
)

from highlight_utils import (
    build_highlight_ranges_from_scores,
    time_ranges_to_segment_labels,
    split_scores_by_embeddings,
    to_padded_batch,
    render_global_highlights,
    split_match_to_fixed_segments,
    RenderConfig,
)

from scripts.generate_voiceover import create_voiceover_for_match


def make_pseudo_labels(video_paths, pseudo_scores_per_video, seg_s: float):
    """
    Returns: pseudo_labels_per_video: list[np.ndarray (T,)]
    """
    pseudo_labels_per_video = []

    for i, (scores, vpath) in enumerate(zip(pseudo_scores_per_video, video_paths)):
        scores = np.asarray(scores, dtype=np.float32)
        T = len(scores)
        if T == 0:
            pseudo_labels_per_video.append(np.zeros((0,), dtype=np.float32))
            continue

        with VideoFileClip(vpath) as clip:
            clip_dur = float(clip.duration or 0.0)

        if clip_dur <= 0.0:
            pseudo_labels_per_video.append(np.zeros((T,), dtype=np.float32))
            print(f"Video {i}: invalid duration, labels=all-zeros")
            continue

        seg_dur = clip_dur / T

        budget_s = min(300.0, max(20.0, 0.15 * clip_dur))
        pre_s    = min(15.0, 0.06 * clip_dur)
        post_s   = min(20.0, 0.08 * clip_dur)
        min_sep  = min(40.0, max(5.0, 0.12 * clip_dur))

        time_ranges = build_highlight_ranges_from_scores(
            scores=scores,
            seg_dur=seg_dur,
            clip_dur=clip_dur,
            budget_s=budget_s,
            smooth_win=7,
            min_sep_s=min_sep,
            pre_s=pre_s,
            post_s=post_s,
            merge_gap_s=1.0,
            score_floor_q=0.85,
        )

        labels = time_ranges_to_segment_labels(time_ranges, T=T, seg_dur=seg_dur)
        pseudo_labels_per_video.append(labels)

        print(
            f"Video {i}: {int(labels.sum())}/{T} positives | "
            f"{sum(b-a for a,b in time_ranges):.1f}s selected | clip_dur={clip_dur:.1f}s"
        )

    return pseudo_labels_per_video


def compute_pos_weight(pseudo_labels_per_video, device: str):
    if len(pseudo_labels_per_video) == 0:
        return torch.tensor([1.0], device=device)

    all_labels = np.concatenate(pseudo_labels_per_video) if pseudo_labels_per_video else np.array([], dtype=np.float32)
    pos_fraction = float(all_labels.mean()) if all_labels.size else 0.0

    if pos_fraction > 0.0:
        w_p = (1.0 - pos_fraction) / max(pos_fraction, 1e-6)
    else:
        w_p = 1.0

    print(f"Global positive fraction: {pos_fraction:.4f}, pos_weight={w_p:.2f}")
    return torch.tensor([w_p], device=device)


def train_model(
    video_embeddings_list,
    audio_embeddings_list,
    pseudo_labels_per_video,
    device: str,
    d_self=128,
    D_common=256,
    d_bimodal=128,
    hidden=256,
    dropout=0.5,
    lr=1e-4,
    num_epochs=50,
    batch_size=8,
    thr=0.5,
):
    D_v = int(video_embeddings_list[0].shape[-1])
    D_a = int(audio_embeddings_list[0].shape[-1])

    video_self_att = SelfAttention(D_v, d_self).to(device)
    audio_self_att = SelfAttention(D_a, d_self).to(device)

    proj_v = nn.Linear(D_v, D_common).to(device)
    proj_a = nn.Linear(D_a, D_common).to(device)

    bimodal_att = BimodalSelfAttention(D_common, d_bimodal).to(device)
    head = HighlightClassifier(D=D_common, hidden=hidden, p=dropout).to(device)

    alpha_logits = nn.Parameter(torch.zeros(4, device=device))

    params = (
        list(video_self_att.parameters())
        + list(audio_self_att.parameters())
        + list(proj_v.parameters())
        + list(proj_a.parameters())
        + list(bimodal_att.parameters())
        + list(head.parameters())
        + [alpha_logits]
    )
    optimizer = optim.Adam(params, lr=lr)

    pos_weight = compute_pos_weight(pseudo_labels_per_video, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    use_amp = (device == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    amp_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    num_videos = len(video_embeddings_list)

    for epoch in range(num_epochs):
        video_self_att.train()
        audio_self_att.train()
        proj_v.train()
        proj_a.train()
        bimodal_att.train()
        head.train()

        indices = np.random.permutation(num_videos)

        epoch_loss = 0.0
        n_batches = 0

        tp = fp = fn = tn = 0

        for start in range(0, num_videos, batch_size):
            batch_idxs = indices[start : start + batch_size]

            v_list = [video_embeddings_list[i] for i in batch_idxs]
            a_list = [audio_embeddings_list[i] for i in batch_idxs]
            y_list = [pseudo_labels_per_video[i] for i in batch_idxs]

            v_batch, v_mask, v_lens = to_padded_batch(v_list, device)
            a_batch, a_mask, a_lens = to_padded_batch(a_list, device)
            if v_lens != a_lens:
                raise RuntimeError("Audio/video segment counts per clip must match")

            B, T_max, _ = v_batch.shape

            y_batch = torch.zeros((B, T_max), dtype=torch.float32, device=device)
            for bi, L in enumerate(v_lens):
                y_batch[bi, :L] = torch.as_tensor(y_list[bi], dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx():
                v_self = video_self_att(v_batch, mask=v_mask)
                a_self = audio_self_att(a_batch, mask=a_mask)

                v_common = proj_v(v_self)
                a_common = proj_a(a_self)

                v2a, a2v = bimodal_att(
                    v_self=v_common,
                    a_self=a_common,
                    mask_v=v_mask,
                    mask_a=a_mask,
                )

                alpha = F.softmax(alpha_logits, dim=0).view(4, 1, 1, 1)
                comps = torch.stack([v_common, v2a, a2v, a_common], dim=0)
                z = (alpha * comps).sum(dim=0)

                logits = head(z)
                loss = criterion(logits[v_mask], y_batch[v_mask])

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
                optimizer.step()

            epoch_loss += float(loss.detach().cpu())
            n_batches += 1

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                probs_m = probs[v_mask].detach().float().cpu().numpy()
                y_m = y_batch[v_mask].detach().float().cpu().numpy()

                y_pred = (probs_m >= thr).astype(np.int32)
                y_true = (y_m >= 0.5).astype(np.int32)

                tp += int(((y_pred == 1) & (y_true == 1)).sum())
                fp += int(((y_pred == 1) & (y_true == 0)).sum())
                fn += int(((y_pred == 0) & (y_true == 1)).sum())
                tn += int(((y_pred == 0) & (y_true == 0)).sum())

        avg_loss = epoch_loss / max(n_batches, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)

        alpha_now = F.softmax(alpha_logits, dim=0).detach().cpu().numpy()
        print(
            f"Epoch {epoch+1:03d}/{num_epochs} | loss={avg_loss:.6f} | "
            f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} | alpha={alpha_now}"
        )

    return dict(
        video_self_att=video_self_att,
        audio_self_att=audio_self_att,
        proj_v=proj_v,
        proj_a=proj_a,
        bimodal_att=bimodal_att,
        head=head,
        alpha_logits=alpha_logits,
    )


@torch.no_grad()
def infer_probs_per_video(video_emb, audio_emb, device: str, model_dict):
    video_self_att = model_dict["video_self_att"]
    audio_self_att = model_dict["audio_self_att"]
    proj_v = model_dict["proj_v"]
    proj_a = model_dict["proj_a"]
    bimodal_att = model_dict["bimodal_att"]
    head = model_dict["head"]
    alpha_logits = model_dict["alpha_logits"]

    alpha = F.softmax(alpha_logits, dim=0).view(4, 1, 1, 1)

    video_self_att.eval()
    audio_self_att.eval()
    proj_v.eval()
    proj_a.eval()
    bimodal_att.eval()
    head.eval()

    v_batch, v_mask, v_lens = to_padded_batch([video_emb], device=device)
    a_batch, a_mask, a_lens = to_padded_batch([audio_emb], device=device)
    if v_lens != a_lens:
        raise RuntimeError("Audio/video segment counts per clip must match")

    v_self = video_self_att(v_batch, mask=v_mask)
    a_self = audio_self_att(a_batch, mask=a_mask)

    v_common = proj_v(v_self)
    a_common = proj_a(a_self)

    v2a, a2v = bimodal_att(
        v_self=v_common,
        a_self=a_common,
        mask_v=v_mask,
        mask_a=a_mask,
    )

    comps = torch.stack([v_common, v2a, a2v, a_common], dim=0)
    z = (alpha * comps).sum(dim=0)

    logits = head(z)
    probs = torch.sigmoid(logits).masked_fill(~a_mask, 0.0)

    p = probs[0, : v_lens[0]].detach().cpu().numpy().astype(np.float32)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--match_video", required=True, help="Path to full match video")
    ap.add_argument("--out_root", default="./output/match_processed", help="Where to write chunks")
    ap.add_argument("--seg_s", type=int, default=20, help="Chunk seconds for splitting match")
    ap.add_argument("--global_budget_s", type=float, default=240.0, help="Total highlight duration budget (seconds)")
    ap.add_argument("--pre_roll_s", type=float, default=0.0, help="Pre-roll seconds for renderer")

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)

    ap.add_argument("--make_voiceover", action="store_true")
    ap.add_argument("--home_team", default="Home")
    ap.add_argument("--away_team", default="Away")
    ap.add_argument("--voiceover_target_min", type=float, default=3.0)
    ap.add_argument("--voiceover_work_dir", default="output/voiceover")

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    video_root, audio_root = split_match_to_fixed_segments(
        args.match_video, out_root=args.out_root, seg_s=args.seg_s
    )

    video_paths = list_files_oswalk(video_root, ext_filter={".mp4", ".avi", ".mov"})
    audio_paths = list_files_oswalk(audio_root, ext_filter={".wav", ".mp3", ".flac"})
    video_paths.sort()
    audio_paths.sort()

    if len(video_paths) != len(audio_paths):
        raise RuntimeError(f"Mismatch between #videos ({len(video_paths)}) and #audios ({len(audio_paths)})")
    print(f"Found {len(video_paths)} video/audio pairs")

    video_segments_list, audios_duration, n_segments_list = preprocess_video_paths(video_paths)
    audio_segments_list = preprocess_audio_paths(audio_paths, audios_duration, n_segments_list)

    print("Extracting video embeddings...")
    video_embeddings_list = get_video_embeddings_list(video_segments_list, device=device)

    print("Extracting audio embeddings...")
    audio_embeddings_list = get_audio_embeddings_list(audio_segments_list, device=device)

    print("Computing pseudo-highlight scores (PHS)...")
    pseudo_scores_flat = get_pseudo_highlight_scores(
        audio_embeddings_list=audio_embeddings_list,
        video_embeddings_list=video_embeddings_list,
    )

    pseudo_scores_per_video = split_scores_by_embeddings(
        pseudo_scores_flat,
        video_embeddings_list,
    )

    print("Building pseudo-labels from scores...")
    pseudo_labels_per_video = make_pseudo_labels(
        video_paths=video_paths,
        pseudo_scores_per_video=pseudo_scores_per_video,
        seg_s=float(args.seg_s),
    )

    print("Training model...")
    model = train_model(
        video_embeddings_list=video_embeddings_list,
        audio_embeddings_list=audio_embeddings_list,
        pseudo_labels_per_video=pseudo_labels_per_video,
        device=device,
        lr=float(args.lr),
        num_epochs=int(args.epochs),
        batch_size=int(args.batch_size),
    )

    print("Running inference...")
    pred_probs_per_video = []
    for i in range(len(video_paths)):
        p = infer_probs_per_video(video_embeddings_list[i], audio_embeddings_list[i], device=device, model_dict=model)
        pred_probs_per_video.append(p)
        if i < 3:
            print(f"Example probs {i}: shape={p.shape} min/max={float(p.min()):.4f}/{float(p.max()):.4f}")

    cfg = RenderConfig(
        global_budget_s=float(args.global_budget_s),
        pre_roll_s=float(args.pre_roll_s),
    )

    final_out, per_video_outs = render_global_highlights(
        video_paths=video_paths,
        audio_paths=audio_paths,
        pred_probs_per_video=pred_probs_per_video,
        out_root="output",
        cfg=cfg,
    )

    print("Final:", final_out)
    print("Per-video:", len(per_video_outs))

    if args.make_voiceover:
        res = create_voiceover_for_match(
            video_path=str(final_out),
            home_team=args.home_team,
            away_team=args.away_team,
            target_min=float(args.voiceover_target_min),
            work_dir=args.voiceover_work_dir,
        )
        print("Voiceover result:", res)

if __name__ == "__main__":
    main()
