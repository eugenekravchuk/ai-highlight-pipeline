import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import torch

def _ensure_vector(embedding):
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim > 1:
        return arr.reshape(-1)
    return arr

def get_feature_vec_mean(embeddings):
    if embeddings is None:
        return np.zeros((1,), dtype=np.float32)
    if isinstance(embeddings, torch.Tensor):
        if embeddings.numel() == 0:
            return np.zeros((1,), dtype=np.float32)
        arr = embeddings.detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.array(embeddings, dtype=np.float32)
    if arr.size == 0:
        return np.zeros((1,), dtype=np.float32)
    return arr.mean(axis=0)

def get_segments_means(embeddings_list):
    segments_means = []
    for embeddings in embeddings_list:
        feature_vec_mean = get_feature_vec_mean(embeddings)
        segments_means.append(feature_vec_mean)
    return np.array(segments_means)

def reduce_dimentionality(data):
    data = np.asarray(data)
    n_samples, n_features = data.shape
    if n_samples <= 10:
        print(f"[PHS] Warning: n_samples ({n_samples}) <= 10. Skipping UMAP reduction.")
        return data
    n_components = min(10, n_features, n_samples - 2)
    n_neighbors = min(15, n_samples - 1)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.0,
        n_components=n_components,
        metric='euclidean',
        random_state=42,
    )
    embedding = reducer.fit_transform(data)
    return embedding

def get_labels(features, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(features)
    return labels

def select_optimal_k(features, k_min=4, k_max=15):
    features = np.asarray(features)
    n_samples = features.shape[0]
    k_max_eff = min(k_max, n_samples - 1)
    k_min_eff = min(k_min, k_max_eff)
    if k_min_eff >= k_max_eff:
        return k_min_eff
    scores = {}
    print(f"[PHS] Searching optimal K in range [{k_min_eff}, {k_max_eff}]...")
    for k in range(k_min_eff, k_max_eff + 1):
        labels = get_labels(features, k)
        sc = silhouette_score(features, labels)
        scores[k] = sc
    best_k = max(scores, key=lambda k: scores[k])
    print(f"[PHS] Selected optimal K={best_k}")
    return best_k

def get_class_clips(indexed_embeddings_list, labels):
    labels_dct = {}
    for i, label_i in enumerate(labels):
        clips_i = indexed_embeddings_list[i]
        if not clips_i:
            continue
        labels_dct.setdefault(label_i, []).extend(clips_i)
    return labels_dct

def aph(clips):
    if not clips:
        return np.array([])
    A = np.vstack([_ensure_vector(c) for c in clips]).astype(np.float32)
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    A_norm = A / norms
    cos_sim = A_norm @ A_norm.T
    return cos_sim.mean(axis=1)

def get_clips_aph(labels_clips_dct):
    class_aph = []
    for label, clips_i in labels_clips_dct.items():
        if not clips_i:
            continue
        indexes = [item[0] for item in clips_i]
        clip_vectors = [item[1] for item in clips_i]
        aph_scores = aph(clip_vectors)
        class_aph.extend(zip(indexes, aph_scores))
    return class_aph

def get_indexed_embeddings(embeddings_list):
    counter = 0
    indexed_embeddings_list = []
    for video in embeddings_list:
        indexed_clips = []
        for clip in video:
            indexed_clips.append((counter, _ensure_vector(clip)))
            counter += 1
        indexed_embeddings_list.append(indexed_clips)
    return indexed_embeddings_list

def sort_clips_aph(clips_aph):
    sorted_aph_lst = sorted(clips_aph, key=lambda x: x[0])
    return np.array([float(score) for _, score in sorted_aph_lst], dtype=np.float32)

def concatenate_embeddings(audio_embeddings, video_embeddings):
    return np.concatenate((audio_embeddings, video_embeddings), axis=1)

def fuse_audio_video_aph(audio_aph, video_aph):
    dict_a = dict(audio_aph)
    dict_b = dict(video_aph)
    all_indices = set(dict_a.keys()) | set(dict_b.keys())
    fused_aph = []
    for idx in all_indices:
        s_a = dict_a.get(idx, 0.0)
        s_v = dict_b.get(idx, 0.0)
        avg_score = (s_a + s_v) / 2.0
        fused_aph.append((idx, avg_score))
    return fused_aph

def visualize_clustering(features, labels, title="Video Clusters (2D Projection)"):
    print("[Vis] Projecting features to 2D for visualization...")
    reducer_2d = umap.UMAP(
        n_components=2, 
        n_neighbors=15, 
        min_dist=0.1, 
        random_state=42
    )
    embedding_2d = reducer_2d.fit_transform(features)
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for k, color in zip(unique_labels, colors):
        mask = (labels == k)
        plt.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            label=f"Cluster {k}",
            color=color,
            alpha=0.7,
            s=50
        )
    plt.title(title, fontsize=14)
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend(title="Pseudo-Categories", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

def get_pseudo_highlight_scores(audio_embeddings_list, video_embeddings_list):
    clips_audio_feature_means = get_segments_means(audio_embeddings_list)
    clips_video_feature_means = get_segments_means(video_embeddings_list)
    video_level_features = concatenate_embeddings(clips_audio_feature_means, clips_video_feature_means)
    reduced_features = reduce_dimentionality(video_level_features)
    best_k = select_optimal_k(reduced_features, k_min=4, k_max=15)
    labels = get_labels(reduced_features, best_k)
    visualize_clustering(video_level_features, labels)
    index_audio_embeddings_list = get_indexed_embeddings(audio_embeddings_list)
    audio_labels_clips_dct = get_class_clips(index_audio_embeddings_list, labels)
    clips_audio_aph = get_clips_aph(audio_labels_clips_dct)
    index_video_embeddings_list = get_indexed_embeddings(video_embeddings_list)
    video_labels_clips_dct = get_class_clips(index_video_embeddings_list, labels)
    clips_video_aph = get_clips_aph(video_labels_clips_dct)
    clips_aph = fuse_audio_video_aph(clips_audio_aph, clips_video_aph)
    sorted_clips_aph = sort_clips_aph(clips_aph)
    return sorted_clips_aph
