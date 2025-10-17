import os
import matplotlib.pyplot as plt
import numpy as np
import librosa
import panns_inference
from panns_inference import AudioTagging, SoundEventDetection, labels
import time
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn


CLIP_DURATION = 2


def audio_features_mean(features):
    return sum(features)/features.shape[0]

def get_feature_vec_mean(features_list):

    feature_means = []

    for feature in features_list:
        feature_mean = audio_features_mean(feature)
        feature_means.append(feature_mean)

    return feature_means

def get_segments_means(embeddings_list):

    segments_means = []

    for embeddings in embeddings_list:
        feature_vec_mean = get_feature_vec_mean(embeddings)
        segments_means.append(feature_vec_mean)
    
    return segments_means


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

def reduce_dimentionality(data):
    reducer = umap.UMAP(n_neighbors=15,
                    min_dist=0.1,      # how tightly points are packed
                    n_components=10,    # output dim (2D)
                    metric='euclidean',
                    random_state=42)

    embedding = reducer.fit_transform(data)

    return embedding

def get_labels(features, k):

    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(features)

    return labels

def select_optimal_k(features, k_min=2, k_max=10):
    scores = {}
    for k in range(k_min, k_max + 1):
        labels = get_labels(features, k)

        if k > 1:
            sc = silhouette_score(features, labels)
        else:
            sc = -1

        scores[k] = sc

    best_k = max(scores, key=lambda k: scores[k])

    return best_k

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


def get_class_clips(embeddings_list, labels):
    labels_dct = {}

    for i, label_i in enumerate(labels):
        clips_i = np.asarray(embeddings_list[i])

        if clips_i.ndim == 1:
            clips_i = clips_i.reshape(1, -1)

        if label_i in labels_dct:
            labels_dct[label_i] = np.vstack((labels_dct[label_i], clips_i))
        else:
            labels_dct[label_i] = clips_i

    return labels_dct

def aph(clips):

    A = np.vstack(clips)

    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)

    cos_sim = A_norm @ A_norm.T

    return cos_sim.mean(axis=1)

def get_class_aph(labels_clips_dct):

    class_aph_dct = {}

    for class_i, clips_i in labels_clips_dct.items():
        class_aph_dct[class_i] = aph(clips_i)

    return class_aph_dct

def get_pseudo_highlight_scores(embeddings_list):

    clips_feature_means = get_segments_means(embeddings_list)

    reduced_features = reduce_dimentionality(clips_feature_means)

    best_k = select_optimal_k(reduced_features)

    labels = get_labels(reduced_features, best_k)

    labels_clips_dct = get_class_clips(embeddings_list, labels)

    class_aph_dct = get_class_aph(labels_clips_dct)

    return class_aph_dct

def attention(x):

    D = len(x[0][0])

    mha = nn.MultiheadAttention(embed_dim=D, num_heads=1, dropout=0.1, batch_first=True)

    out, _ = mha(x, x, x, attn_mask=None, key_padding_mask=None)

    return out

if __name__ == '__main__':

    device = 'cuda'
    dir_path = 'audios'

    audio_paths = list_files_oswalk(dir_path)

    segments_list = preprocess_audio_paths(audio_paths)

    embeddings_list = get_embeddings_list(segments_list, device)

    embeddings_list = torch.tensor(embeddings_list)

    updated_embeddings = attention(embeddings_list)

    class_aph_dct = get_pseudo_highlight_scores(embeddings_list)
