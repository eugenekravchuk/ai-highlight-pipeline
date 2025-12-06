import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
from sklearn.cluster import KMeans
import torch
import time
import math
import cv2

from torchvision import transforms
from phs import get_pseudo_highlight_scores
from self_att import SelfAttention
from classifier import AudioClassifier
from moviepy import VideoFileClip
from torchvision.models.video import r3d_18, R3D_18_Weights
import torch.nn.functional as F

CLIP_DURATION_FRAMES = 20

DEFAULT_MEAN = [0.43216, 0.394666, 0.37645]
DEFAULT_STD  = [0.22803, 0.22145, 0.216989]

default_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 171)),
    transforms.CenterCrop((112, 112)),
    transforms.ToTensor(),                   # -> (C, H, W) float32 [0,1]
    transforms.Normalize(mean=DEFAULT_MEAN, std=DEFAULT_STD),
])

device = "cuda" if torch.cuda.is_available() else "cpu"


def preprocess_video(video,
                fps = None,
                segment_frames = CLIP_DURATION_FRAMES) -> np.ndarray:

    clip = VideoFileClip(video)
    file_fps = clip.fps or fps or 25.0

    if fps is None:
        fps = file_fps

    frames = []

    for fr in clip.iter_frames(fps=fps, dtype="uint8"):

        f = fr.copy()
        f = torch.as_tensor(f)
        f = f.permute(2, 0, 1)  
        if f.max() > 2.5:
            f = f / 255.0
        f = default_preprocess(f)
        f = f.permute(1, 2, 0)
        f = np.asarray(f)

        frames.append(f)

    try:
        clip.reader.close()
    except Exception:
        pass
    try:
        if clip.audio is not None:
            clip.audio.reader.close_proc()
    except Exception:
        pass

    frames = np.stack(frames, axis=0)

    T = frames.shape[0]

    n_segments = (T + segment_frames - 1) // segment_frames
    out_shape = (n_segments, segment_frames) + frames.shape[1:]
    out = np.zeros(out_shape, dtype=frames.dtype)

    segment_duration = fps / segment_frames

    for i in range(n_segments):
        s = i * segment_frames
        e = s + segment_frames
        seg = frames[s:e]
        out[i, :len(seg)] = seg

    out = out.transpose(0, 4, 1, 2, 3) 
    return out, n_segments, segment_duration


def split_audio_into_segments(audio, sr, num_segments=None, segment_duration=None):

    audio_1d = np.squeeze(audio)

    total_samples = audio_1d.shape[0]

    seg_samples = int(round(segment_duration * sr))

    segments = []
    idx = 0
    count = 0

    while idx < total_samples and (num_segments is None or count < num_segments):

        if num_segments is not None and count == num_segments - 1:
            segments.append(audio_1d[idx:total_samples])
            idx = total_samples
            count += 1
            break

        next_idx = min(idx + seg_samples, total_samples)
        segments.append(audio_1d[idx:next_idx])
        idx = next_idx
        count += 1

    if num_segments is not None and len(segments) < num_segments:

        missing = num_segments - len(segments)

        for _ in range(missing):
            segments.append(np.zeros(0, dtype=audio_1d.dtype))

    n_segments = num_segments if num_segments is not None else len(segments)

    if len(segments) < n_segments:
        for _ in range(n_segments - len(segments)):
            segments.append(np.zeros(0, dtype=audio_1d.dtype))

    max_len = max((s.shape[0] for s in segments), default=0)

    out = np.zeros((len(segments), max_len), dtype=audio_1d.dtype)

    for i, s in enumerate(segments):
        out[i, : s.shape[0]] = s

    return out


def preprocess_audio(audio_path, sr=32000, num_segments=None, segment_duration=None):

    audio, _ = librosa.core.load(audio_path, sr=sr, mono=True)

    return split_audio_into_segments(audio, sr=sr, num_segments=num_segments, 
                                    segment_duration=segment_duration)


def preprocess_video_paths(video_path_list):
    
    video_list = []
    audios_duration = []
    n_segments_list = []
    
    for video_path in video_path_list:
        video_segments, n_segments, audio_duration = preprocess_video(video_path)
        video_list.append(video_segments)
        audios_duration.append(audio_duration)
        n_segments_list.append(n_segments)
    
    return video_list, audios_duration, n_segments_list

def preprocess_audio_paths(audio_path_list, audios_duration, n_segments_list):
    
    audio_list = []
    
    for audio_path, audio_duration, n_segments in zip(audio_path_list, audios_duration, n_segments_list):
        media_segments = preprocess_audio(audio_path, segment_duration=audio_duration, num_segments=n_segments)
        audio_list.append(media_segments)

    return audio_list

def get_audio_embeddings_list(segments_list, model_path = None, device='cuda'):

    if model_path == None:
        model_path = './architecture/models/Cnn14_mAP=0.431.pth'

    at = AudioTagging(checkpoint_path=model_path, device=device)

    embeddings_list = []

    for segments in segments_list:
        (_, embedding) = at.inference(segments)
        embeddings_list.append(embedding)
    
    return embeddings_list

def get_video_embeddings_list(
    segments_list,
    device='cuda',
    chunk_size=2
):

    weights = R3D_18_Weights.KINETICS400_V1
    model = r3d_18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)

    embeddings_list = []

    with torch.no_grad():
        for segments in segments_list:

            segments = torch.as_tensor(segments, dtype=torch.float32)

            B = segments.shape[0]
            emb_chunks = []

            for i in range(0, B, chunk_size):
                chunk = segments[i:i + chunk_size]

                chunk = chunk.to(device=device, dtype=torch.float32)

                with torch.torch.amp.autocast("cuda"):
                    out = model(chunk)

                emb_chunks.append(out.detach().cpu())

                del chunk, out
                torch.cuda.empty_cache()

            embedding = torch.cat(emb_chunks, dim=0)
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


def classify_embeddings(embeddings_list, device='cuda', d=128, hidden=256, dropout=0.5):

    clip_tensors = [torch.as_tensor(e, dtype=torch.float32) for e in embeddings_list]
    B = len(clip_tensors)
    D = clip_tensors[0].shape[-1]
    T_lens = [e.shape[0] for e in clip_tensors]
    T_max = max(T_lens)

    x = torch.zeros((B, T_max, D), dtype=torch.float32)
    mask = torch.zeros((B, T_max), dtype=torch.bool)
    for i, e in enumerate(clip_tensors):
        t = e.shape[0]
        x[i, :t] = e
        mask[i, :t] = True

    x = x.to(device)
    mask = mask.to(device)

    att = SelfAttention(D, d).to(device)
    att.eval()
    with torch.no_grad():
        att_out = att(x, mask=mask)          # (B, T_max, D)

    clf = AudioClassifier(D=D, hidden=hidden, p=dropout).to(device)
    clf.eval()
    with torch.no_grad():
        flat = att_out.reshape(B * T_max, D) # (B*T_max, D)
        flat_logits = clf(flat)              # (B*T_max,)
        flat_probs = torch.sigmoid(flat_logits)
        probs = flat_probs.reshape(B, T_max)   # (B, T_max)
        logits = flat_logits.reshape(B, T_max) # (B, T_max)

    # ---- Unpad back to lists ----
    probs_per_clip = []
    logits_per_clip = []
    for i, t in enumerate(T_lens):
        probs_per_clip.append(probs[i, :t].detach().cpu().numpy())
        logits_per_clip.append(logits[i, :t].detach().cpu().numpy())

    return probs_per_clip, logits_per_clip

if __name__ == '__main__':

    device = 'cuda'

    audio_path = 'data/audios'
    video_path = 'data/videos'

    video_paths = list_files_oswalk(video_path)
    audio_paths = list_files_oswalk(audio_path)

    video_segments_list, audios_duration, n_segments_list = preprocess_video_paths(video_paths) # always goes before audio
    audio_segments_list = preprocess_audio_paths(audio_paths, audios_duration, n_segments_list)

    video_embeddings_list = get_video_embeddings_list(video_segments_list, device = device)
    audio_embeddings_list = get_audio_embeddings_list(audio_segments_list, device = device)

    class_aph_dct = get_pseudo_highlight_scores(audio_embeddings_list, video_embeddings_list)

    raise ValueError

    d1 = 128
    D1 = len(audio_embeddings_list[0][0])
    audio_embeddings_list = torch.tensor(audio_embeddings_list)

    d2 = 128
    D2 = len(video_embeddings_list[0][0])
    video_embeddings_list = torch.tensor(video_embeddings_list)


    # no classifier part
    audio_model = SelfAttention(D1, d1)
    video_model = SelfAttention(D2, d2)
    
    res = audio_model.forward(audio_embeddings_list)
    print(class_aph_dct)

    print("Predicting highlights…")
    # with classifier part
    probs, logits = classify_embeddings(audio_embeddings_list, device='cpu')
    print("Predicted highlight probabilities:", probs)
