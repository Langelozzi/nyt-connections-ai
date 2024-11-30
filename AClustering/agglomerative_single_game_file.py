from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import numpy as np
from itertools import combinations


def normalize_embeddings(embeddings):
    """Normalize embeddings to unit vectors for cosine similarity."""
    return normalize(embeddings)


def perform_clustering(embeddings, num_initial_clusters):
    """Perform Agglomerative Clustering with more initial clusters."""
    normalized_embeddings = normalize_embeddings(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=num_initial_clusters, metric="cosine", linkage="complete"
    )
    return clustering.fit_predict(normalized_embeddings)


def enforce_equal_sized_clusters(words, embeddings, num_clusters, group_size):
    """Force clustering into equal-sized groups."""
    # Start with a larger number of clusters
    initial_cluster_labels = perform_clustering(embeddings, num_clusters * 2)

    # Group words by initial cluster
    clusters = {}
    for word, label in zip(words, initial_cluster_labels):
        clusters.setdefault(label, []).append(word)

    # Flatten and sort words by similarity
    flat_words = [(word, label) for label, group in clusters.items() for word in group]
    flat_words.sort(key=lambda x: x[1])  # Sort by cluster label

    # Redistribute into equal-sized groups
    grouped_words = []
    for i in range(0, len(flat_words), group_size):
        group = [word for word, _ in flat_words[i:i + group_size]]
        grouped_words.append(group)

    return grouped_words


def main():
    # Words and embeddings
    words = [
        "SHOW", "HANDLE", "BELIEVE", "MOUNT", "BLUFF", "SHAM",
        "ACCEPT", "TRUST", "FRONT", "BRACKET", "BUY", "STAND",
        "FIFTH", "PINT", "BASE", "LITER"
    ]

    model_path = "../GoogleNews-vectors-negative300.bin"  # Update with your file's actual path
    num_clusters = 4
    group_size = 4

    print("Loading word embeddings...")
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Word embeddings loaded successfully.")

    # Filter valid words and embeddings
    valid_words = [word for word in words if word in word_vectors]
    embeddings = np.array([word_vectors[word] for word in valid_words])

    # Enforce equal-sized clusters
    final_groups = enforce_equal_sized_clusters(valid_words, embeddings, num_clusters, group_size)

    # Display the groups
    print("\nClusters:")
    for i, group in enumerate(final_groups, 1):
        print(f"Group {i}: {group}")


if __name__ == "__main__":
    main()
