from algorithm import *
from phs import *

def plot_umap_2d(emb, labels=None, title='UMAP 2D', s=10, alpha=0.8):
    plt.figure(figsize=(8,6))
    if labels is None:
        plt.scatter(emb[:,0], emb[:,1], s=s, alpha=alpha)
    else:
        unique = np.unique(labels)
        for u in unique:
            mask = labels == u
            plt.scatter(emb[mask,0], emb[mask,1], s=s, alpha=alpha, label=str(u))
        plt.legend(markerscale=2, fontsize='small', bbox_to_anchor=(1.05,1), loc='upper left')
    plt.xlabel('UMAP-1'); plt.ylabel('UMAP-2'); plt.title(title)
    plt.tight_layout()
    plt.show()


def visualize_clustering(embeddings_list):

    clips_feature_means = get_segments_means(embeddings_list)

    reduced_features = reduce_dimentionality(clips_feature_means)

    best_k = select_optimal_k(reduced_features)

    labels = get_labels(reduced_features, best_k)

    plot_umap_2d(reduced_features, labels)

if __name__ == '__main__':

    device = 'cuda'
    dir_path = 'audios'

    audio_paths = list_files_oswalk(dir_path)

    segments_list = preprocess_audio_paths(audio_paths)

    model_path = './architecture/models/Cnn14_mAP=0.431.pth'
    embeddings_list = get_embeddings_list(segments_list, model_path, device)

    # no classifier part
    class_aph_dct = visualize_clustering(embeddings_list)