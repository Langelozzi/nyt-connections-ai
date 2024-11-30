from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.spatial.distance import cosine


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


def compute_confidence_scores(words, embeddings, cluster_labels, num_clusters):
    """Compute confidence scores based on similarity to cluster centroids."""
    cluster_centroids = np.zeros((num_clusters, embeddings.shape[1]))
    cluster_sizes = np.zeros(num_clusters)

    # Calculate cluster centroids
    for idx, label in enumerate(cluster_labels):
        cluster_centroids[label] += embeddings[idx]
        cluster_sizes[label] += 1
    cluster_centroids /= cluster_sizes[:, np.newaxis]

    # Compute cosine similarity of each word to its cluster centroid
    confidence_scores = []
    for idx, label in enumerate(cluster_labels):
        similarity = 1 - cosine(embeddings[idx], cluster_centroids[label])
        confidence_scores.append(similarity)

    return confidence_scores


def enforce_equal_sized_clusters(words, embeddings, num_clusters, group_size):
    """Force clustering into equal-sized groups and return confidence scores."""
    # Perform clustering
    cluster_labels = perform_clustering(embeddings, num_clusters * 2)

    # Compute confidence scores
    confidence_scores = compute_confidence_scores(words, embeddings, cluster_labels, num_clusters * 2)

    # Group words by initial cluster
    clusters = {}
    for word, label, score in zip(words, cluster_labels, confidence_scores):
        clusters.setdefault(label, []).append((word, score))

    # Flatten and sort words by similarity
    flat_words = [(word, score, label) for label, group in clusters.items() for word, score in group]
    flat_words.sort(key=lambda x: x[2])  # Sort by cluster label

    # Redistribute into equal-sized groups
    grouped_words = []
    for i in range(0, len(flat_words), group_size):
        group = [{"word": word, "confidence": score} for word, score, _ in flat_words[i:i + group_size]]
        grouped_words.append(group)

    return grouped_words


def main():
    # Words for clustering
    words = [
        "SHOW", "HANDLE", "BELIEVE", "MOUNT", "BLUFF", "SHAM",
        "ACCEPT", "TRUST", "FRONT", "BRACKET", "BUY", "STAND",
        "FIFTH", "PINT", "BASE", "LITER"
    ]

    num_clusters = 4
    group_size = 4

    print("Loading Sentence-BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight Sentence-BERT model
    print("Model loaded successfully.")

    # Generate contextual embeddings
    print("Generating embeddings...")
    embeddings = model.encode(words)
    print("Embeddings generated.")

    # Enforce equal-sized clusters and get confidence scores
    final_groups = enforce_equal_sized_clusters(words, embeddings, num_clusters, group_size)

    # Display the groups with confidence scores
    print("\nClusters with Confidence Scores:")
    for i, group in enumerate(final_groups, 1):
        print(f"Group {i}:")
        for item in group:
            print(f"  Word: {item['word']}, Confidence: {item['confidence']:.4f}")


if __name__ == "__main__":
    main()
