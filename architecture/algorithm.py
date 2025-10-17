import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
from sklearn.cluster import KMeans
import torch

### our files ###
from phs import get_pseudo_highlight_scores
from self_att import SelfAttention


CLIP_DURATION = 2


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

def get_embeddings_list(segments_list, device):

    at = AudioTagging(checkpoint_path="/home/mpiuser/panns_data/Cnn14_mAP=0.431.pth", device=device)

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

if __name__ == '__main__':

    device = 'cuda'
    dir_path = 'audios'

    audio_paths = list_files_oswalk(dir_path)

    segments_list = preprocess_audio_paths(audio_paths)

    embeddings_list = get_embeddings_list(segments_list, device)

    class_aph_dct = get_pseudo_highlight_scores(embeddings_list)

    d = 128
    D = len(embeddings_list[0][0])

    embeddings_list = torch.tensor(embeddings_list)

    model = SelfAttention(D, d)

    res = model.forward(embeddings_list)


