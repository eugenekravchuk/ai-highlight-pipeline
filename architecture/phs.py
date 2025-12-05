import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def get_feature_vec_mean(embeddings):
    return sum(embeddings)/len(embeddings)

def get_segments_means(embeddings_list):

    segments_means = []

    for embeddings in embeddings_list:
        feature_vec_mean = get_feature_vec_mean(embeddings)
        segments_means.append(feature_vec_mean)
    
    return segments_means

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

def get_class_clips(embeddings_list, labels):
    labels_dct = {}

    for i, label_i in enumerate(labels):
        clips_i = np.asarray(embeddings_list[i])

        if clips_i.ndim == 1:
            clips_i = clips_i.reshape(1, -1)

        if label_i in labels_dct:
            labels_dct[label_i] += [clips_i]
        else:
            labels_dct[label_i] = [clips_i]

    return labels_dct

def aph(clips):

    A = np.vstack(clips)

    A_norm = A / np.linalg.norm(A, axis=1, keepdims=True)

    cos_sim = A_norm @ A_norm.T

    return cos_sim.mean(axis=1)

def get_clips_aph(labels_clips_dct):

    class_aph = []

    for _, clips_i in labels_clips_dct.items():

        unlabeled_clips_i = np.array([clip_emb for clips_j in clips_i  for clip_k in clips_j for clip in clip_k for _, clip_emb in clip.items()])

        indexes = np.array([index for clips_j in clips_i  for clip_k in clips_j for clip in clip_k for index, _ in clip.items()])

        aph_clips_i = aph(unlabeled_clips_i)

        class_aph += [[index, aph_clip_score] for index, aph_clip_score in zip(indexes, aph_clips_i)]

    return class_aph

def get_indexed_embeddings(embeddings_list):

    counter = 0

    indexed_embeddings_list = []

    for video in embeddings_list:

        indexed_clips = []

        for clip in video:
            indexed_clips.append({counter : clip})

            counter += 1
        
        indexed_embeddings_list.append(indexed_clips)
    
    return indexed_embeddings_list

def sort_clips_aph(clips_aph):
    sorted_aph_lst = sorted(clips_aph, key = lambda x: x[0])

    return [aph_tuple[1].item() for aph_tuple in sorted_aph_lst]

def conratenate_embeddings(audio_embeddings, video_embeddings):
    return np.concatenate((audio_embeddings, video_embeddings), axis=1)

def fuse_audio_video_aph(audio_aph, video_aph):
    fused_aph = []

    for (index_a, aph_a), (index_v, aph_v) in zip(audio_aph, video_aph):
        assert index_a == index_v, "Indexes do not match!"
        combined_aph = (aph_a + aph_v) / 2
        fused_aph.append((index_a, combined_aph))
    
    return fused_aph

def get_pseudo_highlight_scores(audio_embeddings_list, video_embeddings_list):

    clips_audio_feature_means = get_segments_means(audio_embeddings_list)
    clips_video_feature_means = get_segments_means(video_embeddings_list)
    
    clips_feature_means = conratenate_embeddings(clips_audio_feature_means, clips_video_feature_means)

    reduced_features = reduce_dimentionality(clips_feature_means)

    best_k = select_optimal_k(reduced_features)

    labels = get_labels(reduced_features, best_k)

    index_audio_embeddings_list = get_indexed_embeddings(audio_embeddings_list)
    audio_labels_clips_dct = get_class_clips(index_audio_embeddings_list, labels)
    clips_audio_aph = get_clips_aph(audio_labels_clips_dct)

    index_video_embeddings_list = get_indexed_embeddings(video_embeddings_list)
    video_labels_clips_dct = get_class_clips(index_video_embeddings_list, labels)
    clips_video_aph = get_clips_aph(video_labels_clips_dct)

    clips_aph = fuse_audio_video_aph(clips_audio_aph, clips_video_aph)

    sorted_clips_aph = sort_clips_aph(clips_aph)

    return sorted_clips_aph
