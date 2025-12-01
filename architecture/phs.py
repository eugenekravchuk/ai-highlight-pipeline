
"""Pseudo-category discovery and pseudo-highlight generation utilities."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


DEFAULT_REDUCER_KWARGS = dict(
    n_neighbors=15,
    min_dist=0.1,
    n_components=10,
    metric="euclidean",
    random_state=42,
)


@dataclass
class PseudoCategoryModel:
    """Container for pseudo-category clustering artifacts."""

    reducer: umap.UMAP
    clusterer: KMeans
    reduced_features: np.ndarray
    labels: np.ndarray
    best_k: int
    silhouette_scores: Dict[int, float]


def _ensure_2d(array: np.ndarray) -> np.ndarray:
    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError("Expected 2D array for clip embeddings")
    return arr


def _mean_feature(matrix: np.ndarray) -> np.ndarray:
    return matrix.mean(axis=0)


def compute_video_level_features(
    audio_embeddings_list: Sequence[np.ndarray],
    visual_embeddings_list: Optional[Sequence[np.ndarray]] = None,
) -> np.ndarray:
    """Concatenate mean audio+visual descriptors per video."""

    video_features: List[np.ndarray] = []
    num_videos = len(audio_embeddings_list)
    visual_embeddings_list = visual_embeddings_list or [None] * num_videos

    if len(visual_embeddings_list) != num_videos:
        raise ValueError("Audio and visual lists must have the same length")

    for audio_emb, visual_emb in zip(audio_embeddings_list, visual_embeddings_list):
        audio_vec = _mean_feature(_ensure_2d(audio_emb))
        if visual_emb is not None:
            visual_vec = _mean_feature(_ensure_2d(visual_emb))
            combined = np.concatenate([audio_vec, visual_vec])
        else:
            combined = audio_vec
        video_features.append(combined)

    return np.vstack(video_features)


def _fit_reducer(features: np.ndarray, reducer_kwargs: Optional[Dict] = None) -> Tuple[umap.UMAP, np.ndarray]:
    kwargs = dict(DEFAULT_REDUCER_KWARGS)
    if reducer_kwargs:
        kwargs.update(reducer_kwargs)
    reducer = umap.UMAP(**kwargs)
    reduced = reducer.fit_transform(features)
    return reducer, reduced


def _search_best_k(features: np.ndarray, k_min: int, k_max: int, random_state: int) -> Tuple[int, Dict[int, float], KMeans, np.ndarray]:
    scores: Dict[int, float] = {}
    best_model: Optional[KMeans] = None
    best_labels: Optional[np.ndarray] = None
    best_score = -np.inf
    num_samples = features.shape[0]

    for k in range(k_min, k_max + 1):
        if k <= 1 or k >= num_samples:
            continue
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init="auto")
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        scores[k] = score
        if score > best_score:
            best_score = score
            best_model = kmeans
            best_labels = labels

    if best_model is None or best_labels is None:
        # Fallback to k=1 when dataset is too small for silhouette
        kmeans = KMeans(n_clusters=min(max(1, k_min), num_samples), random_state=random_state, n_init="auto")
        best_labels = kmeans.fit_predict(features)
        best_model = kmeans
        scores[kmeans.n_clusters] = -1.0

    best_k = best_model.n_clusters
    return best_k, scores, best_model, best_labels


def build_pseudo_category_model(
    audio_embeddings_list: Sequence[np.ndarray],
    visual_embeddings_list: Optional[Sequence[np.ndarray]] = None,
    k_range: Tuple[int, int] = (4, 15),
    reducer_kwargs: Optional[Dict] = None,
    random_state: int = 42,
) -> PseudoCategoryModel:
    """Generate pseudo-categories following the paper's Sec. 3.1."""

    features = compute_video_level_features(audio_embeddings_list, visual_embeddings_list)
    reducer, reduced = _fit_reducer(features, reducer_kwargs)
    k_min, k_max = k_range
    best_k, scores, clusterer, labels = _search_best_k(reduced, k_min, k_max, random_state)
    return PseudoCategoryModel(
        reducer=reducer,
        clusterer=clusterer,
        reduced_features=reduced,
        labels=labels,
        best_k=best_k,
        silhouette_scores=scores,
    )


def _cosine_similarity_matrix(features: np.ndarray) -> np.ndarray:
    eps = 1e-8
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    normalized = features / norms
    return normalized @ normalized.T


def _aggregate_cluster_scores(
    embeddings_list: Sequence[np.ndarray], labels: np.ndarray
) -> Dict[int, np.ndarray]:
    per_video_scores: Dict[int, np.ndarray] = {}
    for cluster_id in np.unique(labels):
        video_indices = np.where(labels == cluster_id)[0]
        if not len(video_indices):
            continue

        stacks: List[np.ndarray] = []
        spans: List[Tuple[int, int, int]] = []
        cursor = 0
        for vid in video_indices:
            clips = _ensure_2d(embeddings_list[vid])
            stacks.append(clips)
            num_clips = clips.shape[0]
            spans.append((vid, cursor, cursor + num_clips))
            cursor += num_clips

        all_clips = np.vstack(stacks)
        sim = _cosine_similarity_matrix(all_clips)
        recurrence = sim.mean(axis=1)

        for vid, start, end in spans:
            per_video_scores[vid] = recurrence[start:end]

    return per_video_scores


def combine_modalities(
    audio_scores: Dict[int, np.ndarray],
    visual_scores: Optional[Dict[int, np.ndarray]] = None,
    audio_weight: float = 0.5,
) -> Dict[int, np.ndarray]:
    """Average audio and visual recurrence scores."""

    combined: Dict[int, np.ndarray] = {}
    if visual_scores is None:
        return dict(audio_scores)

    for vid, a_scores in audio_scores.items():
        v_scores = visual_scores.get(vid)
        if v_scores is None:
            combined[vid] = a_scores
            continue
        if len(v_scores) != len(a_scores):
            raise ValueError("Audio and visual clip counts must match per video")
        aw = np.clip(audio_weight, 0.0, 1.0)
        combined[vid] = aw * a_scores + (1.0 - aw) * v_scores
    return combined


def scores_to_pseudo_labels(
    scores: Dict[int, np.ndarray],
    top_ratio: float = 0.5,
) -> Dict[int, np.ndarray]:
    """Convert clip scores to binary pseudo labels (top t%)."""

    labels: Dict[int, np.ndarray] = {}
    ratio = np.clip(top_ratio, 0.05, 0.95)
    for vid, clip_scores in scores.items():
        num_clips = len(clip_scores)
        keep = max(1, int(round(num_clips * ratio)))
        idx = np.argpartition(-clip_scores, keep - 1)[:keep]
        mask = np.zeros(num_clips, dtype=np.float32)
        mask[idx] = 1.0
        labels[vid] = mask
    return labels


@dataclass
class PseudoHighlightResult:
    model: PseudoCategoryModel
    audio_scores: Dict[int, np.ndarray]
    visual_scores: Optional[Dict[int, np.ndarray]]
    av_scores: Dict[int, np.ndarray]
    pseudo_labels: Dict[int, np.ndarray]
    top_indices: Dict[int, np.ndarray]


def get_pseudo_highlight_scores(
    audio_embeddings_list: Sequence[np.ndarray],
    visual_embeddings_list: Optional[Sequence[np.ndarray]] = None,
    k_range: Tuple[int, int] = (4, 15),
    top_ratio: float = 0.5,
    reducer_kwargs: Optional[Dict] = None,
    random_state: int = 42,
) -> PseudoHighlightResult:
    """End-to-end pseudo-category discovery and scoring."""

    model = build_pseudo_category_model(
        audio_embeddings_list,
        visual_embeddings_list,
        k_range=k_range,
        reducer_kwargs=reducer_kwargs,
        random_state=random_state,
    )

    audio_scores = _aggregate_cluster_scores(audio_embeddings_list, model.labels)
    visual_scores = None
    if visual_embeddings_list is not None:
        visual_scores = _aggregate_cluster_scores(visual_embeddings_list, model.labels)

    av_scores = combine_modalities(audio_scores, visual_scores)
    pseudo_labels = scores_to_pseudo_labels(av_scores, top_ratio=top_ratio)
    top_indices = {
        vid: np.flatnonzero(mask)
        for vid, mask in pseudo_labels.items()
    }

    return PseudoHighlightResult(
        model=model,
        audio_scores=audio_scores,
        visual_scores=visual_scores,
        av_scores=av_scores,
        pseudo_labels=pseudo_labels,
        top_indices=top_indices,
    )


def get_pseudo_highlight_labels_dict(
    audio_embeddings_list: Sequence[np.ndarray],
    visual_embeddings_list: Optional[Sequence[np.ndarray]] = None,
    **kwargs,
) -> Dict[int, np.ndarray]:
    """Convenience wrapper returning only clip-level pseudo labels."""

    result = get_pseudo_highlight_scores(
        audio_embeddings_list,
        visual_embeddings_list=visual_embeddings_list,
        **kwargs,
    )
    return result.pseudo_labels