from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import pandas as pd
from AClust_connections_evaluator import ConnectionsEvaluator
from scipy.spatial.distance import cosine
import numpy as np

def normalize_embeddings(embeddings):
    """Normalize embeddings to unit vectors for cosine similarity."""
    return normalize(embeddings)


def perform_clustering(embeddings, num_clusters):
    """Perform Agglomerative Clustering."""
    normalized_embeddings = normalize_embeddings(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=num_clusters, metric="cosine", linkage="complete"
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


def cluster_words_with_scores(words, embeddings, num_clusters, group_size=4):
    """Cluster words into groups of exactly `group_size` words with confidence scores."""
    cluster_labels = perform_clustering(embeddings, num_clusters)
    confidence_scores = compute_confidence_scores(words, embeddings, cluster_labels, num_clusters)

    # Group words by cluster labels with confidence scores
    clusters = {}
    for word, label, score in zip(words, cluster_labels, confidence_scores):
        clusters.setdefault(label, []).append((word, score))

    # Sort clusters by average confidence scores
    sorted_clusters = sorted(
        clusters.values(),
        key=lambda group: np.mean([score for _, score in group])
    )  # Sort by ascending confidence (lowest first)

    # Flatten and redistribute into equal-sized groups
    flat_words = [(word, score) for group in sorted_clusters for word, score in group]
    groups = []
    for i in range(0, len(flat_words), group_size):
        group = flat_words[i:i + group_size]
        if len(group) == group_size:
            groups.append(group)

    return groups


def bert_agglomerative_predictor_low_confidence(input_words: list[str]) -> list[list[str]]:
    """Predict groups of 4 words using BERT Agglomerative Clustering (low confidence guessing)."""
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load BERT model
    embeddings = model.encode(input_words)  # Generate embeddings
    remaining_words = input_words.copy()
    remaining_embeddings = embeddings.copy()
    num_clusters = 4
    solved_groups = []

    while len(remaining_words) > 4:
        # Cluster remaining words with confidence scores
        clusters = cluster_words_with_scores(remaining_words, remaining_embeddings, num_clusters)

        # Select the guess with the lowest confidence (first cluster in sorted list)
        best_guess = [word for word, _ in clusters[0]]
        solved_groups.append(best_guess)

        # Remove guessed words
        remaining_words = [word for word in remaining_words if word not in best_guess]
        remaining_embeddings = model.encode(remaining_words)

        # Decrease the number of clusters
        num_clusters = max(1, len(remaining_words) // 4)

    # Final group (remaining 4 words)
    if remaining_words:
        solved_groups.append(remaining_words)

    return solved_groups

def main():
    # Load dataset
    data_path = "../data/connection_answers_aggregate.csv"
    connections_answers = pd.read_csv(data_path)
    connections_answers = connections_answers[400:500]

    # Define predictor function (low confidence version)
    predictor_func = bert_agglomerative_predictor_low_confidence

    # Evaluate performance
    evaluator = ConnectionsEvaluator(predictor_func)
    accuracy, connections_made = evaluator.evaluate(connections_answers)

    # Calculate the total possible connections
    total_connections = 4 * len(connections_answers)

    # Calculate the percentage of correct connections
    connections_percent = (connections_made / total_connections) * 100

    # Print results
    print(f"Number of connections made: {connections_made}/{total_connections}")
    print(f"Percentage of connections made: {connections_percent:.2f}%")
    print(f"Win percent (all correct groups): {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
