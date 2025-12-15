#!/usr/bin/env python3
# scripts/full_pipeline.py
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import argparse
import subprocess
from pathlib import Path
from contextlib import nullcontext

# ---- Make repo root importable (so `core`, `highlight_utils`, etc. work) ----
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from moviepy import VideoFileClip
from torchvision import transforms
from torchvision.models.video import r3d_18, R3D_18_Weights

from panns_inference import AudioTagging

from core.attention import SelfAttention, BimodalSelfAttention
from core.classifier import HighlightClassifier
from core.phs_creation import get_pseudo_highlight_scores

from highlight_utils.media_split import split_match_to_fixed_segments
from highlight_utils.render_highlights import render_global_highlights, RenderConfig
from highlight_utils.ranges import (
    build_highlight_ranges_from_scores,
    time_ranges_to_segment_labels,
    split_scores_by_embeddings,
)
from highlight_utils.tensor_utils import to_padded_batch

from generate_voiceover import create_voiceover_for_match


# =========================
# Utils
# =========================

def check_tool(name: str):
    try:
        subprocess.run([name, "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        raise RuntimeError(f"Required tool not found or not working: {name}. Error: {e}")


def list_files_oswalk(root: str, followlinks: bool = False, ext_filter: set | None = None):
    result = []
    for dirpath, _, filenames in os.walk(root, followlinks=followlinks):
        for fn in filenames:
            if ext_filter:
                if not any(fn.lower().endswith(e) for e in ext_filter):
                    continue
            result.append(os.path.join(dirpath, fn))
    return result


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def ffprobe_duration(path: str | Path) -> float:
    """Return duration seconds using ffprobe."""
    check_tool("ffprobe")
    path = str(path)
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path
    ]
    out = subprocess.check_output(cmd, text=True).strip()
    try:
        return float(out)
    except Exception:
        return 0.0


def ffmpeg_decode_audio_mono(path: str, sr: int = 32000) -> np.ndarray:
    """
    Decode any audio file using ffmpeg to mono float32 [-1, 1].
    Avoids librosa/numba completely.
    """
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(path),
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(int(sr)),
        "pipe:1",
    ]
    raw = subprocess.check_output(cmd)
    audio = np.frombuffer(raw, dtype=np.float32)
    return audio


# =========================
# Video preprocessing + embeddings
# =========================

CLIP_DURATION_FRAMES = 16
DEFAULT_MEAN = [0.43216, 0.394666, 0.37645]
DEFAULT_STD  = [0.22803, 0.22145, 0.216989]

default_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 171)),
    transforms.CenterCrop((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
])


def preprocess_video(video_path: str, fps: float | None = None, segment_frames: int = CLIP_DURATION_FRAMES):
    """
    Returns:
      segments: np.ndarray (n_segments, 3, segment_frames, 112, 112)
      n_segments: int
      segment_duration: float seconds per segment (segment_frames / fps)
    """
    clip = VideoFileClip(video_path)
    file_fps = float(clip.fps or 25.0)
    fps = float(fps or file_fps)

    frames = []
    try:
        for fr in clip.iter_frames(fps=fps, dtype="uint8"):
            frames.append(default_preprocess(fr))
    finally:
        try:
            clip.reader.close()
        except Exception:
            pass
        try:
            if clip.audio is not None:
                clip.audio.reader.close_proc()
        except Exception:
            pass

    if len(frames) == 0:
        seg_dur = segment_frames / fps
        empty = np.zeros((0, 3, segment_frames, 112, 112), dtype=np.float32)
        return empty, 0, seg_dur

    frames = torch.stack(frames, dim=0)  # (T, 3, H, W)
    T = frames.shape[0]
    n_segments = (T + segment_frames - 1) // segment_frames

    pad_T = n_segments * segment_frames - T
    if pad_T > 0:
        pad_frame = torch.zeros_like(frames[0:1])
        frames = torch.cat([frames, pad_frame.repeat(pad_T, 1, 1, 1)], dim=0)

    frames = frames.view(n_segments, segment_frames, *frames.shape[1:])  # (n, seg_frames, 3, H, W)
    frames = frames.permute(0, 2, 1, 3, 4).contiguous()  # (n, 3, seg_frames, H, W)

    seg_dur = segment_frames / fps
    return frames.numpy().astype(np.float32), int(n_segments), float(seg_dur)


def preprocess_video_paths(video_paths: list[str]):
    video_segments_list = []
    seg_durations = []
    n_segments_list = []
    for vp in video_paths:
        segs, n, seg_dur = preprocess_video(vp)
        video_segments_list.append(segs)
        seg_durations.append(seg_dur)
        n_segments_list.append(n)
    return video_segments_list, seg_durations, n_segments_list


def _amp_context(device: str):
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def get_video_embeddings_list(segments_list, device: str = "cpu", chunk_size: int = 2):
    weights = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    embeddings_list = []
    for segments in segments_list:
        segments_t = torch.as_tensor(segments, dtype=torch.float32)
        B = segments_t.shape[0]
        emb_chunks = []
        for i in range(0, B, chunk_size):
            chunk = segments_t[i:i + chunk_size].to(device=device, dtype=torch.float32)
            with _amp_context(device):
                out = model(chunk)
            emb_chunks.append(out.detach().cpu())
            del chunk, out
            if device.startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        embedding = torch.cat(emb_chunks, dim=0) if emb_chunks else torch.zeros((0, 512))
        embeddings_list.append(embedding)
    return embeddings_list


# =========================
# Audio preprocessing + embeddings (NO librosa)
# =========================

def split_audio_into_segments(audio_1d: np.ndarray, sr: int, num_segments: int, segment_duration: float):
    """
    Returns padded array (num_segments, max_len) like original pipeline.
    """
    audio_1d = np.squeeze(audio_1d)
    total_samples = int(audio_1d.shape[0])

    seg_samples = int(round(segment_duration * sr))
    seg_samples = max(seg_samples, 1)

    segments = []
    idx = 0
    count = 0
    while idx < total_samples and count < num_segments:
        if count == num_segments - 1:
            segments.append(audio_1d[idx:total_samples])
            break
        next_idx = min(idx + seg_samples, total_samples)
        segments.append(audio_1d[idx:next_idx])
        idx = next_idx
        count += 1

    # pad missing segments
    while len(segments) < num_segments:
        segments.append(np.zeros(0, dtype=audio_1d.dtype))

    max_len = max((s.shape[0] for s in segments), default=0)
    out = np.zeros((len(segments), max_len), dtype=np.float32)
    for i, s in enumerate(segments):
        if s.size:
            out[i, :s.shape[0]] = s.astype(np.float32)
    return out


def preprocess_audio_paths_ffmpeg(audio_paths: list[str], seg_durations: list[float], n_segments_list: list[int], sr: int = 32000):
    audio_segments_list = []
    for ap, seg_dur, nseg in zip(audio_paths, seg_durations, n_segments_list):
        audio = ffmpeg_decode_audio_mono(ap, sr=sr)
        segs = split_audio_into_segments(audio, sr=sr, num_segments=int(nseg), segment_duration=float(seg_dur))
        audio_segments_list.append(segs)
    return audio_segments_list


@torch.no_grad()
def get_audio_embeddings_list(segments_list, device: str = "cpu", model_path: str | None = None):
    """
    PANNs AudioTagging expects: (n_segments, n_samples) float32
    """
    if model_path is None:
        model_path = str(ROOT / "architecture" / "models" / "Cnn14_mAP=0.431.pth")

    at = AudioTagging(checkpoint_path=model_path, device=device)

    embeddings_list = []
    for segments in segments_list:
        _, embedding = at.inference(segments)
        embeddings_list.append(torch.as_tensor(embedding).detach().cpu())
    return embeddings_list


# =========================
# Pseudo labels + train/infer
# =========================

def make_pseudo_labels(video_paths, pseudo_scores_per_video, seg_s: float):
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
            batch_idxs = indices[start: start + batch_size]
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


# =========================
# Cache + checkpoint IO
# =========================

def save_embeddings_cache_npz(path: Path, video_emb_list, audio_emb_list):
    ensure_dir(path.parent)
    np.savez_compressed(
        str(path),
        video=np.array([v.numpy() if isinstance(v, torch.Tensor) else v for v in video_emb_list], dtype=object),
        audio=np.array([a.numpy() if isinstance(a, torch.Tensor) else a for a in audio_emb_list], dtype=object),
    )


def load_embeddings_cache_npz(path: Path):
    z = np.load(str(path), allow_pickle=True)
    video = [torch.as_tensor(v) for v in list(z["video"])]
    audio = [torch.as_tensor(a) for a in list(z["audio"])]
    return video, audio


def save_checkpoint_pt(path: Path, model_dict: dict, device: str):
    ensure_dir(path.parent)
    payload = {
        "video_self_att": model_dict["video_self_att"].state_dict(),
        "audio_self_att": model_dict["audio_self_att"].state_dict(),
        "proj_v": model_dict["proj_v"].state_dict(),
        "proj_a": model_dict["proj_a"].state_dict(),
        "bimodal_att": model_dict["bimodal_att"].state_dict(),
        "head": model_dict["head"].state_dict(),
        "alpha_logits": model_dict["alpha_logits"].detach().cpu(),
    }
    torch.save(payload, str(path))


def load_checkpoint_pt(path: Path, device: str, d_self=128, D_common=256, d_bimodal=128, hidden=256, dropout=0.5,
                      D_v: int | None = None, D_a: int | None = None):
    ckpt = torch.load(str(path), map_location=device)

    if D_v is None or D_a is None:
        raise ValueError("Need D_v and D_a to rebuild model modules for loading.")

    video_self_att = SelfAttention(D_v, d_self).to(device)
    audio_self_att = SelfAttention(D_a, d_self).to(device)
    proj_v = nn.Linear(D_v, D_common).to(device)
    proj_a = nn.Linear(D_a, D_common).to(device)
    bimodal_att = BimodalSelfAttention(D_common, d_bimodal).to(device)
    head = HighlightClassifier(D=D_common, hidden=hidden, p=dropout).to(device)
    alpha_logits = nn.Parameter(torch.zeros(4, device=device))

    video_self_att.load_state_dict(ckpt["video_self_att"], strict=True)
    audio_self_att.load_state_dict(ckpt["audio_self_att"], strict=True)
    proj_v.load_state_dict(ckpt["proj_v"], strict=True)
    proj_a.load_state_dict(ckpt["proj_a"], strict=True)
    bimodal_att.load_state_dict(ckpt["bimodal_att"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    if "alpha_logits" in ckpt:
        alpha_logits.data.copy_(ckpt["alpha_logits"].to(device))

    return dict(
        video_self_att=video_self_att,
        audio_self_att=audio_self_att,
        proj_v=proj_v,
        proj_a=proj_a,
        bimodal_att=bimodal_att,
        head=head,
        alpha_logits=alpha_logits,
    )


# =========================
# Final assembly: posters + voiceover + music + transitions
# =========================

def make_still_video_with_music(
    img_path: str,
    music_mp3: str,
    out_path: str,
    duration_s: float,
    fps: int = 25,
    music_volume: float = 0.75,
    fade_in: float = 0.6,
    fade_out: float = 0.8,
):
    """
    Creates a still video from poster + music bed (looped) with fade-in/out.
    """
    check_tool("ffmpeg")
    dur = float(duration_s)
    fade_out_start = max(0.0, dur - float(fade_out))

    vf = "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2"
    af = (
        f"volume={music_volume},"
        f"afade=t=in:st=0:d={float(fade_in)},"
        f"afade=t=out:st={fade_out_start:.3f}:d={float(fade_out)}"
    )

    cmd = [
        "ffmpeg", "-y",
        "-loop", "1",
        "-i", img_path,
        "-stream_loop", "-1",
        "-i", music_mp3,
        "-t", f"{dur:.3f}",
        "-vf", vf,
        "-filter:a", af,
        "-r", str(int(fps)),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)


def mix_voiceover_and_music_ducked(
    voiceover_mp3: str,
    music_mp3: str,
    out_aac: str,
    target_duration_s: float,
    music_volume: float = 0.05,   # <-- вдвічі тихіше
    fade_in: float = 0.35,
    fade_out: float = 0.55,
):
    """
    Mix voiceover + bg music with ducking (sidechaincompress) + fades.
    Output AAC trimmed to target_duration_s.
    """
    check_tool("ffmpeg")
    dur = float(target_duration_s)
    fade_out_start = max(0.0, dur - float(fade_out))

    # Explanation:
    # - Music loops and is trimmed to dur
    # - Music volume lowered (0.5)
    # - sidechaincompress reduces music while voice is present
    # - amix combines voice + ducked music
    # - loudnorm to keep stable output loudness
    fc = (
        f"[0:a]aresample=48000,apad=pad_dur={dur},atrim=0:{dur}[vo];"
        f"[1:a]aresample=48000,volume={music_volume},"
        f"afade=t=in:st=0:d={fade_in},"
        f"afade=t=out:st={fade_out_start:.3f}:d={fade_out},"
        f"atrim=0:{dur}[m];"
        f"[m][vo]sidechaincompress=threshold=0.08:ratio=10:attack=5:release=200[mduck];"
        f"[vo][mduck]amix=inputs=2:duration=longest:dropout_transition=0,"
        f"alimiter=limit=0.95[a]"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", voiceover_mp3,
        "-stream_loop", "-1",
        "-i", music_mp3,
        "-t", f"{dur:.3f}",
        "-filter_complex", fc,
        "-map", "[a]",
        "-c:a", "aac",
        "-b:a", "192k",
        out_aac,
    ]
    subprocess.run(cmd, check=True)


def attach_audio_replace(video_in: str, audio_in: str, out_path: str):
    cmd = [
        "ffmpeg", "-y",
        "-i", video_in,
        "-i", audio_in,
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)


# =========================
# Transitions helpers
# =========================

def get_duration(path: str) -> float:
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    out = subprocess.check_output(cmd).decode().strip()
    return float(out)


def compute_xfade_offsets(
    intro_mp4: str,
    highlights_mp4: str,
    outro_mp4: str,
    transition_s: float = 0.7,
):
    intro_d = get_duration(intro_mp4)
    highlights_d = get_duration(highlights_mp4)
    outro_d = get_duration(outro_mp4)

    offset_1 = max(0.0, intro_d - transition_s)
    offset_2 = offset_1 + highlights_d - transition_s

    return {
        "intro_d": round(intro_d, 3),
        "highlights_d": round(highlights_d, 3),
        "outro_d": round(outro_d, 3),
        "offset_1": round(offset_1, 3),
        "offset_2": round(offset_2, 3),
    }

def assemble_with_transitions(
    intro_mp4: str,
    highlights_mp4: str,
    outro_mp4: str,
    out_mp4: str,
    transition_s: float = 0.7,
    fps: int = 25,
    audio_sr: int = 48000,
):
    check_tool("ffmpeg")
    check_tool("ffprobe")

    info = compute_xfade_offsets(intro_mp4, highlights_mp4, outro_mp4, transition_s=transition_s)
    o1 = info["offset_1"]
    o2 = info["offset_2"]

    # ВАЖЛИВО: вирівнюємо timebase для xfade + audio pts для acrossfade
    fc = (
        f"[0:v]fps={fps},settb=AVTB,format=yuv420p[v0];"
        f"[1:v]fps={fps},settb=AVTB,format=yuv420p[v1];"
        f"[2:v]fps={fps},settb=AVTB,format=yuv420p[v2];"
        f"[0:a]aresample={audio_sr},asetpts=N/SR/TB[a0];"
        f"[1:a]aresample={audio_sr},asetpts=N/SR/TB[a1];"
        f"[2:a]aresample={audio_sr},asetpts=N/SR/TB[a2];"
        f"[v0][v1]xfade=transition=fade:duration={transition_s}:offset={o1}[v01];"
        f"[v01][v2]xfade=transition=fade:duration={transition_s}:offset={o2}[v];"
        f"[a0][a1]acrossfade=d={transition_s}:c1=tri:c2=tri[a01];"
        f"[a01][a2]acrossfade=d={transition_s}:c1=tri:c2=tri[a]"
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", intro_mp4,
        "-i", highlights_mp4,
        "-i", outro_mp4,
        "-filter_complex", fc,
        "-map", "[v]",
        "-map", "[a]",
        "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k",
        out_mp4,
    ]
    subprocess.run(cmd, check=True)

    print(f"[TRANSITIONS] intro={info['intro_d']}s highlights={info['highlights_d']}s outro={info['outro_d']}s")
    print(f"[TRANSITIONS] offsets: o1={o1}s o2={o2}s (transition={transition_s}s)")


# =========================
# Main
# =========================

def main():
    check_tool("ffmpeg")

    ap = argparse.ArgumentParser()
    ap.add_argument("--match_video", required=True, help="Path to full match video")
    ap.add_argument("--music", required=True, help="Background music mp3")

    # ⬇️ було required=True, але в тебе є генерація всередині — робимо опціональним
    ap.add_argument("--voiceover_mp3", default="", help="Your voiceover mp3 (if missing -> generated inside pipeline)")

    ap.add_argument("--voiceover_text", default="", help="Text for TTS voiceover. If provided and voiceover_mp3 missing -> generate it.")
    ap.add_argument("--voice", default="uk-UA-PolinaNeural", help="TTS voice name (Edge TTS).")
    ap.add_argument("--voiceover_rate", default="+0%", help="TTS speaking rate, e.g. +10% or -10% (Edge TTS).")
    ap.add_argument("--voiceover_volume", default="+0dB", help="TTS volume, e.g. +0dB, -3dB (Edge TTS).")

    ap.add_argument("--intro_img", required=True, help="Intro poster image (png/jpg)")
    ap.add_argument("--outro_img", required=True, help="Outro poster image (png/jpg)")

    ap.add_argument("--out_root", default="./output/match_processed", help="Where to write chunks")
    ap.add_argument("--seg_s", type=int, default=20, help="Chunk seconds for splitting match")

    ap.add_argument("--global_budget_s", type=float, default=240.0, help="Total highlight duration budget (seconds)")
    ap.add_argument("--pre_roll_s", type=float, default=0.0, help="Pre-roll seconds for renderer")

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)

    ap.add_argument("--checkpoint", default="", help="Path to .pt checkpoint. If exists -> skip training.")
    ap.add_argument("--save_checkpoint", default="", help="Optional path to save trained checkpoint (.pt).")
    ap.add_argument("--embeddings_cache", default="", help="Path to .npz embeddings cache. If exists -> reuse, else save.")

    ap.add_argument("--panns_model", default="", help="Optional PANNs model path (.pth).")
    ap.add_argument("--device", default="", help="Force device: cpu/cuda. Default: auto.")

    # ⬇️ НОВЕ: керування інтро/аутро + переходами
    ap.add_argument("--intro_duration", type=float, default=2.8)
    ap.add_argument("--outro_duration", type=float, default=5.5)   # <-- довше outro за замовчуванням
    ap.add_argument("--transition", default="fade")               # fade, slideleft, wiperight, circlecrop, etc.
    ap.add_argument("--transition_duration", type=float, default=0.7)

    args = ap.parse_args()

    # 0) Ensure voiceover exists (use existing or generate)
    voiceover_path = Path(args.voiceover_mp3) if args.voiceover_mp3 else Path("")
    if (not voiceover_path) or (not voiceover_path.exists()):
        print("[VOICEOVER] voiceover mp3 not found -> generating inside pipeline...")

        artifacts = create_voiceover_for_match(
            video_path=args.match_video,
            home_team=getattr(args, "home_team", ""),
            away_team=getattr(args, "away_team", ""),
            season=getattr(args, "season", 2024),
            date_str=getattr(args, "date_str", "2024-10-26"),
            target_min=3.0,
            work_dir=str(Path(args.out_root) / "_voiceover_work"),
            sr=16000,
        )
        voiceover_path = Path(artifacts["voice_mp3"])
        print(f"[VOICEOVER] Generated: {voiceover_path}")
    else:
        print(f"[VOICEOVER] Using existing: {voiceover_path}")

    device = args.device.strip().lower() if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # 1) Split match to fixed segments (video-only + audio-only)
    video_root, audio_root = split_match_to_fixed_segments(
        args.match_video, out_root=args.out_root, seg_s=args.seg_s
    )

    video_paths = list_files_oswalk(str(video_root), ext_filter={".mp4", ".avi", ".mov"})
    audio_paths = list_files_oswalk(str(audio_root), ext_filter={".wav", ".mp3", ".flac"})
    video_paths.sort()
    audio_paths.sort()

    if len(video_paths) != len(audio_paths):
        raise RuntimeError(f"Mismatch between #videos ({len(video_paths)}) and #audios ({len(audio_paths)})")
    print(f"Found {len(video_paths)} video/audio pairs")

    # 2) Embeddings: load cache OR compute
    cache_path = Path(args.embeddings_cache) if args.embeddings_cache else None
    if cache_path and cache_path.exists():
        print(f"Loading embeddings cache: {cache_path}")
        video_embeddings_list, audio_embeddings_list = load_embeddings_cache_npz(cache_path)
    else:
        print("Preprocessing video segments...")
        video_segments_list, seg_durations, n_segments_list = preprocess_video_paths(video_paths)

        print("Preprocessing audio segments via ffmpeg (no librosa)...")
        audio_segments_list = preprocess_audio_paths_ffmpeg(audio_paths, seg_durations, n_segments_list, sr=32000)

        print("Extracting video embeddings...")
        video_embeddings_list = get_video_embeddings_list(video_segments_list, device=device)

        print("Extracting audio embeddings (PANNs)...")
        panns_path = args.panns_model.strip() or None
        audio_embeddings_list = get_audio_embeddings_list(audio_segments_list, device=device, model_path=panns_path)

        if cache_path:
            print(f"Saving embeddings cache: {cache_path}")
            save_embeddings_cache_npz(cache_path, video_embeddings_list, audio_embeddings_list)

    # 3) PHS -> pseudo labels
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

    # 4) Model: load checkpoint OR train
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None

    D_v = int(video_embeddings_list[0].shape[-1]) if len(video_embeddings_list) else 512
    D_a = int(audio_embeddings_list[0].shape[-1]) if len(audio_embeddings_list) else 2048

    if ckpt_path and ckpt_path.exists():
        print(f"Loading checkpoint (skip training): {ckpt_path}")
        model = load_checkpoint_pt(ckpt_path, device=device, D_v=D_v, D_a=D_a)
    else:
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
        if args.save_checkpoint:
            save_path = Path(args.save_checkpoint)
            print(f"Saving checkpoint: {save_path}")
            save_checkpoint_pt(save_path, model, device=device)

    # 5) Inference
    print("Running inference...")
    pred_probs_per_video = []
    for i in range(len(video_paths)):
        p = infer_probs_per_video(video_embeddings_list[i], audio_embeddings_list[i], device=device, model_dict=model)
        pred_probs_per_video.append(p)
        if i < 3:
            print(f"Example probs {i}: shape={p.shape} min/max={float(p.min()):.4f}/{float(p.max()):.4f}")

    # 6) Render highlights
    cfg = RenderConfig(
        global_budget_s=float(args.global_budget_s),
        pre_roll_s=float(args.pre_roll_s),
    )

    final_highlights_path, _ = render_global_highlights(
        video_paths=video_paths,
        audio_paths=audio_paths,
        pred_probs_per_video=pred_probs_per_video,
        out_root="output",
        cfg=cfg,
    )

    final_highlights_path = Path(final_highlights_path)
    print("Highlights video:", final_highlights_path)

    # 7) Assemble final with posters + voiceover + music + transitions
    work = Path("output") / "final_assembly"
    ensure_dir(work)

    intro_vid = work / "intro.mp4"
    outro_vid = work / "outro.mp4"
    mix_aac = work / "mix.aac"
    highlights_with_audio = work / "highlights_with_vo_music.mp4"
    final_out = work / "FINAL.mp4"

    # intro/outro WITH MUSIC + fades
    print("Making intro/outro videos with music + fades...")
    make_still_video_with_music(
        img_path=args.intro_img,
        music_mp3=args.music,
        out_path=str(intro_vid),
        duration_s=float(args.intro_duration),
        music_volume=0.75,
        fade_in=0.6,
        fade_out=0.8,
    )
    make_still_video_with_music(
        img_path=args.outro_img,
        music_mp3=args.music,
        out_path=str(outro_vid),
        duration_s=float(args.outro_duration),
        music_volume=0.75,
        fade_in=0.6,
        fade_out=1.2,
    )

    # Highlights: VO + music (half volume) + ducking + fades
    print("Mixing voiceover + music (music half volume + ducking + fades)...")
    highlights_dur = ffprobe_duration(final_highlights_path)
    if highlights_dur <= 0:
        raise RuntimeError("Could not read highlights duration (ffprobe failed).")

    mix_voiceover_and_music_ducked(
        voiceover_mp3=str(voiceover_path),
        music_mp3=args.music,
        out_aac=str(mix_aac),
        target_duration_s=highlights_dur,
        music_volume=0.05,   # <-- вдвічі тихіше
        fade_in=0.15,
        fade_out=0.15,
    )

    print("Attaching mixed audio to highlights (replace original)...")
    attach_audio_replace(str(final_highlights_path), str(mix_aac), str(highlights_with_audio))

    print("Assembling with transitions (xfade + acrossfade)...")
    assemble_with_transitions(
        str(intro_vid),
        str(highlights_with_audio),
        str(outro_vid),
        str(final_out),
        transition_s=0.7,
        fps=25,
        audio_sr=48000,
    )

    print("\n✅ DONE")
    print("Final output:", final_out)


if __name__ == "__main__":
    main()
