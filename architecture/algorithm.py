import os
import numpy as np
import librosa
import torch
from panns_inference import AudioTagging

### our files ###
from phs import get_pseudo_highlight_scores
from visual_features import get_visual_embeddings_list
from av_model import AVHighlightDetector

CLIP_DURATION = 2
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi"}

def split_audio_into_segments(audio, sr, segment_duration=15, pad=True):
    audio_1d = np.squeeze(audio)
    if audio_1d.ndim != 1:
        raise ValueError("audio повинен бути 1D або (1, N)")

    segment_samples = int(segment_duration * sr)
    total_samples = len(audio_1d)

    if pad:
        num_segments = (total_samples + segment_samples - 1) // segment_samples
        segments = np.zeros((num_segments, segment_samples), dtype=audio_1d.dtype)
        for i in range(num_segments):
            start = i * segment_samples
            end = start + segment_samples
            seg = audio_1d[start:end]
            segments[i, :len(seg)] = seg
    else:
        segments = []
        for start in range(0, total_samples, segment_samples):
            end = start + segment_samples
            segments.append(audio_1d[start:end])
        segments = np.array(segments, dtype=object)

    return segments

def preprocess_audio(audio_path):
    (audio, sr) = librosa.core.load(audio_path, sr=32000, mono=True)
    audio = audio[None, :]

    segments = split_audio_into_segments(audio, sr, segment_duration=CLIP_DURATION, pad=True)

    return segments

def preprocess_audio_paths(audio_path_list):

    audio_list = []

    for audio_path in audio_path_list:
        audio_segments = preprocess_audio(audio_path)
        audio_list.append(audio_segments)
    
    return audio_list

def get_embeddings_list(segments_list, model_path, device='cuda'):

    at = AudioTagging(checkpoint_path=model_path, device=device)

    embeddings_list = []

    for segments in segments_list:
        (_, embedding) = at.inference(segments)
        embeddings_list.append(embedding)
    
    return embeddings_list


def list_files_oswalk(root: str, followlinks: bool = False, ext_filter: set | None = None):
    result = []
    for dirpath, _, filenames in os.walk(root, followlinks=followlinks):
        for fn in filenames:
            if ext_filter:
                if not any(fn.lower().endswith(e) for e in ext_filter):
                    continue
            full = os.path.join(dirpath, fn)
            result.append(full)
    return result


def predict_highlights(
    audio_embeddings_list,
    visual_embeddings_list,
    *,
    device: str = 'cuda',
    checkpoint_path: str | None = None,
):
    if visual_embeddings_list is None:
        raise ValueError("Visual embeddings are required for the AV model")

    if len(audio_embeddings_list) != len(visual_embeddings_list):
        raise ValueError("Mismatch between audio and visual samples")

    clip_tensors = [torch.as_tensor(e, dtype=torch.float32) for e in audio_embeddings_list]
    visual_tensors = [torch.as_tensor(v, dtype=torch.float32) for v in visual_embeddings_list]

    B = len(clip_tensors)
    audio_dim = clip_tensors[0].shape[-1]
    visual_dim = visual_tensors[0].shape[-1]
    lengths = [t.shape[0] for t in clip_tensors]
    T_max = max(lengths)

    audio_pad = torch.zeros((B, T_max, audio_dim), dtype=torch.float32)
    visual_pad = torch.zeros((B, T_max, visual_dim), dtype=torch.float32)
    mask = torch.zeros((B, T_max), dtype=torch.bool)

    for i, (a, v) in enumerate(zip(clip_tensors, visual_tensors)):
        T = a.shape[0]
        audio_pad[i, :T] = a
        visual_pad[i, :T] = v
        mask[i, :T] = True

    audio_pad = audio_pad.to(device)
    visual_pad = visual_pad.to(device)
    mask = mask.to(device)

    model = AVHighlightDetector(audio_dim=audio_dim, visual_dim=visual_dim).to(device)
    if checkpoint_path and os.path.isfile(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state.get('model_state', state))

    model.eval()
    with torch.no_grad():
        logits = model(audio_pad, visual_pad, mask)
        probs = torch.sigmoid(logits)

    probs_per_clip = [probs[i, :lengths[i]].detach().cpu().numpy() for i in range(B)]
    return probs_per_clip

if __name__ == '__main__':

    device = 'cuda'
    audio_dir = 'audios'
    video_dir = os.getenv('VIDEO_DIR', './videos_pipeline/downloads')

    audio_paths = sorted(list_files_oswalk(audio_dir))
    if not audio_paths:
        raise RuntimeError(f"No audio files found in {audio_dir}")

    segments_list = preprocess_audio_paths(audio_paths)

    model_path = './architecture/models/Cnn14_mAP=0.431.pth'
    embeddings_list = get_embeddings_list(segments_list, model_path, device)

    # Visual pathway -------------------------------------------------
    video_paths = []
    if os.path.isdir(video_dir):
        video_paths = [p for p in list_files_oswalk(video_dir, ext_filter=VIDEO_EXTENSIONS)]
        video_paths.sort()

    if video_paths:
        if len(video_paths) != len(embeddings_list):
            min_len = min(len(video_paths), len(embeddings_list))
            print(f"Warning: {len(video_paths)} videos for {len(embeddings_list)} audio items. Truncating to {min_len} pairs.")
            video_paths = video_paths[:min_len]
            embeddings_list = embeddings_list[:min_len]
            segments_list = segments_list[:min_len]
            audio_paths = audio_paths[:min_len]

        clip_counts = [emb.shape[0] for emb in embeddings_list]
        visual_embeddings_list = get_visual_embeddings_list(
            video_paths,
            clip_counts,
            clip_duration=CLIP_DURATION,
            target_fps=16.0,
            device=device,
            backbone=os.getenv('VISUAL_BACKBONE', 'resnet34'),
            cache_dir='./output/visual_embeddings',
        )
    else:
        raise RuntimeError(f"No videos found in {video_dir}; cannot run audio-visual model")

    # Pseudo-highlights ---------------------------------------------
    phs_result = get_pseudo_highlight_scores(embeddings_list, visual_embeddings_list)
    av_scores = phs_result.av_scores
    print(f"Pseudo-categories discovered: K={phs_result.model.best_k}")
    if av_scores:
        sample_vid = next(iter(av_scores))
        print(f"Sample AV recurrence scores for clip {sample_vid}: {av_scores[sample_vid][:5]}")

    print("Predicting highlights with AV head…")
    probs = predict_highlights(
        embeddings_list,
        visual_embeddings_list,
        device=device,
        checkpoint_path=os.getenv('AV_CHECKPOINT', './checkpoints/best_model.pth'),
    )
    for idx, clip_probs in enumerate(probs[:3]):
        print(f"Clip {idx}: {clip_probs[:8]}")
