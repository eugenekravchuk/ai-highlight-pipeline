import os
import numpy as np
import librosa
from panns_inference import AudioTagging, labels
import torch
import cv2
from contextlib import nullcontext

from torchvision import transforms
from attention import SelfAttention
from classifier import HighlightClassifier
from moviepy import VideoFileClip
from torchvision.models.video import r3d_18, R3D_18_Weights
import torch.nn.functional as nn

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

device = "cuda" if torch.cuda.is_available() else "cpu"

###
### Preprocessing
###

def split_audio_into_segments(audio, sr, num_segments=None, segment_duration=None):

    audio_1d = np.squeeze(audio)

    total_samples = audio_1d.shape[0]

    if segment_duration is None or segment_duration <= 0:
        raise ValueError("segment_duration must be provided and > 0 for audio splitting")

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

def preprocess_video(video_path, fps=None, segment_frames=CLIP_DURATION_FRAMES):
    clip = VideoFileClip(video_path)

    file_fps = clip.fps or 25.0
    fps = fps or file_fps

    frames = []
    try:
        for fr in clip.iter_frames(fps=fps, dtype="uint8"):
            img_t = default_preprocess(fr)
            frames.append(img_t)
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
        segment_duration = segment_frames / fps
        empty = np.zeros((0, 3, segment_frames, 112, 112), dtype=np.float32)
        return empty, 0, segment_duration

    frames = torch.stack(frames, dim=0)
    T = frames.shape[0]

    n_segments = (T + segment_frames - 1) // segment_frames

    pad_T = n_segments * segment_frames - T
    if pad_T > 0:
        pad_frame = torch.zeros_like(frames[0:1])
        frames = torch.cat([frames, pad_frame.repeat(pad_T, 1, 1, 1)], dim=0)

    frames = frames.view(n_segments, segment_frames, *frames.shape[1:])

    frames = frames.permute(0, 2, 1, 3, 4).contiguous()

    segment_duration = segment_frames / fps
    return frames.numpy(), n_segments, segment_duration

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

###
### Getting embedings
### 

def get_audio_embeddings_list(segments_list, model_path = None, device='cuda'):

    if model_path == None:
        model_path = './architecture/models/Cnn14_mAP=0.431.pth'

    at = AudioTagging(checkpoint_path=model_path, device=device)

    embeddings_list = []

    for segments in segments_list:
        (_, embedding) = at.inference(segments)
        embeddings_list.append(embedding)
    
    return embeddings_list

def _amp_context(device: str):
    if device.startswith('cuda') and torch.cuda.is_available():
        return torch.cuda.amp.autocast(dtype=torch.float16)
    return nullcontext()

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

                with _amp_context(device):
                    out = model(chunk)

                emb_chunks.append(out.detach().cpu())

                del chunk, out
                if device.startswith('cuda') and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            embedding = torch.cat(emb_chunks, dim=0)
            embeddings_list.append(embedding)

    return embeddings_list

###
### Classification 
###

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

    clf = HighlightClassifier(D=D, hidden=hidden, p=dropout).to(device)
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

### 
### Utilities
### 

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