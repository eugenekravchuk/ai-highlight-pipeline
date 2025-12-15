import os
os.environ["NUMBA_DISABLE_JIT"] = "1"
try:
    import umap.umap_ as umap
    _HAS_UMAP = True
except Exception as e:
    print("[PHS] UMAP disabled:", e)
    umap = None
    _HAS_UMAP = False

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def _ensure_vector(embedding):
    arr = np.asarray(embedding, dtype=np.float32)
    if arr.ndim > 1:
        return arr.reshape(-1)
    return arr

def _l2_normalize(A: np.ndarray, eps=1e-8) -> np.ndarray:
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    return A / np.maximum(norms, eps)

def _knn_mean_sim(query_norm: np.ndarray, bank_norm: np.ndarray, k: int) -> np.ndarray:
    S = query_norm @ bank_norm.T
    k_eff = min(k, S.shape[1])
    topk = np.partition(S, -k_eff, axis=1)[:, -k_eff:]
    return topk.mean(axis=1).astype(np.float32)

def _within_cluster_knn(A_norm: np.ndarray, k: int) -> np.ndarray:
    S = A_norm @ A_norm.T
    np.fill_diagonal(S, -np.inf)
    k_eff = min(k, S.shape[1] - 1) if S.shape[1] > 1 else 1
    if k_eff <= 0:
        return np.zeros((A_norm.shape[0],), dtype=np.float32)
    topk = np.partition(S, -k_eff, axis=1)[:, -k_eff:]
    return topk.mean(axis=1).astype(np.float32)

def _build_global_bank(indexed_embeddings_list, max_bank=50000, seed=42):
    rng = np.random.default_rng(seed)
    flat = []
    for vid in indexed_embeddings_list:
        flat.extend(vid)
    if not flat:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    if len(flat) > max_bank:
        flat = rng.choice(flat, size=max_bank, replace=False).tolist()

    bank_ids = np.array([idx for idx, _ in flat], dtype=np.int64)
    bank = np.vstack([_ensure_vector(v) for _, v in flat]).astype(np.float32)
    bank_norm = _l2_normalize(bank)
    return bank_norm, bank_ids

def _zscore_dict(d: dict, eps: float = 1e-6) -> dict:
    if not d:
        return {}
    vals = np.asarray(list(d.values()), dtype=np.float32)
    mu = float(vals.mean())
    sd = float(vals.std())
    if sd < eps:
        sd = eps
    return {k: (float(v) - mu) / sd for k, v in d.items()}

def distinctiveness_scores(indexed_embeddings_list, labels, k_within=20, k_global=20, max_bank=50000, seed=42):
    bank_norm, bank_ids = _build_global_bank(indexed_embeddings_list, max_bank=max_bank, seed=seed)

    labels_dct = {}
    for vid_i, lab in enumerate(labels):
        clips = indexed_embeddings_list[vid_i]
        if clips:
            labels_dct.setdefault(lab, []).extend(clips)

    out = []
    for lab, clips in labels_dct.items():
        if not clips:
            continue

        ids = np.array([idx for idx, _ in clips], dtype=np.int64)
        A = np.vstack([_ensure_vector(v) for _, v in clips]).astype(np.float32)
        A_norm = _l2_normalize(A)

        within = _within_cluster_knn(A_norm, k=k_within)

        if bank_norm.shape[0] == 0:
            global_score = np.zeros_like(within)
        else:
            global_score = _knn_mean_sim(A_norm, bank_norm, k=k_global)

        score = (within - global_score).astype(np.float32)
        out.extend(list(zip(ids.tolist(), score.tolist())))

    return out

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
        print(f"[PHS] Warning: n_samples ({n_samples}) <= 10. Skipping reduction.")
        return data

    if _HAS_UMAP:
        n_components = min(10, n_features, n_samples - 2)
        n_neighbors = min(15, n_samples - 1)
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.0,
            n_components=n_components,
            metric="cosine",
            random_state=42,
        )
        return reducer.fit_transform(data)

    print("[PHS] Using identity fallback (no UMAP).")
    return data


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

def aph_knn(clips, k=20):
    if not clips:
        return np.array([])
    A = np.vstack([_ensure_vector(c) for c in clips]).astype(np.float32)
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-8)

    S = A @ A.T
    np.fill_diagonal(S, -np.inf)

    k_eff = min(k, S.shape[1] - 1)
    topk = np.partition(S, -k_eff, axis=1)[:, -k_eff:]
    return topk.mean(axis=1)

def get_clips_aph(labels_clips_dct):
    class_aph = []
    for label, clips_i in labels_clips_dct.items():
        if not clips_i:
            continue
        indexes = [item[0] for item in clips_i]
        clip_vectors = [item[1] for item in clips_i]
        aph_scores = aph_knn(clip_vectors)
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

def get_pseudo_highlight_scores(audio_embeddings_list, video_embeddings_list,
                               k_within=20, k_global=20, max_bank=50000,
                               w_audio=0.7, w_video=0.3, seed=42):
    clips_audio_feature_means = get_segments_means(audio_embeddings_list)
    clips_video_feature_means = get_segments_means(video_embeddings_list)
    video_level_features = concatenate_embeddings(clips_audio_feature_means, clips_video_feature_means)

    scaler = StandardScaler()
    video_level_features_scaled = scaler.fit_transform(video_level_features)
    reduced_features = reduce_dimentionality(video_level_features_scaled)
    best_k = select_optimal_k(reduced_features, k_min=4, k_max=15)
    labels = get_labels(reduced_features, best_k)

    if umap is not None:
        visualize_clustering(video_level_features_scaled, labels)
    else:
        print("[Vis] Skipping visualization (UMAP not available).")


    index_audio_embeddings_list = get_indexed_embeddings(audio_embeddings_list)
    index_video_embeddings_list = get_indexed_embeddings(video_embeddings_list)

    audio_dist = distinctiveness_scores(
        index_audio_embeddings_list, labels,
        k_within=k_within, k_global=k_global, max_bank=max_bank, seed=seed
    )
    video_dist = distinctiveness_scores(
        index_video_embeddings_list, labels,
        k_within=k_within, k_global=k_global, max_bank=max_bank, seed=seed
    )

    dict_a = dict(audio_dist)
    dict_v = dict(video_dist)

    dict_a = _zscore_dict(dict_a)
    dict_v = _zscore_dict(dict_v)

    all_idx = sorted(set(dict_a.keys()) | set(dict_v.keys()))

    fused = []
    for idx in all_idx:
        s_a = dict_a.get(idx, 0.0)
        s_v = dict_v.get(idx, 0.0)
        fused.append((idx, w_audio * s_a + w_video * s_v))

    fused_sorted = sorted(fused, key=lambda x: x[0])
    return np.array([s for _, s in fused_sorted], dtype=np.float32)