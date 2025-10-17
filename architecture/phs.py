import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


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
