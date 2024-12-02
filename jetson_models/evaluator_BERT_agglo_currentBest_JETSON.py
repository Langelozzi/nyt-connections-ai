import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
from AClust_connections_evaluator import ConnectionsEvaluator
from typing import List


def normalize_embeddings(embeddings):
    """Normalize embeddings to unit vectors for cosine similarity."""
    return normalize(embeddings)


def perform_clustering(embeddings, num_clusters, metric, linkage):
    """Perform Agglomerative Clustering with specified metric and linkage."""
    normalized_embeddings = normalize(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=num_clusters, metric=metric, linkage=linkage
    )
    return clustering.fit_predict(normalized_embeddings)


def cluster_words(words, embeddings, num_clusters, metric, linkage, group_size=4):
    """Cluster words into groups of exactly `group_size` words."""
    cluster_labels = perform_clustering(embeddings, num_clusters, metric, linkage)

    # Group words by cluster labels
    clusters = {}
    for word, label in zip(words, cluster_labels):
        clusters.setdefault(label, []).append(word)

    # Flatten all clusters into a single list of words
    all_words = [word for group in clusters.values() for word in group]

    # Force equal-sized groups of `group_size`
    groups = []
    for i in range(0, len(all_words), group_size):
        group = all_words[i:i + group_size]
        if len(group) == group_size:
            groups.append(group)

    return groups


def bert_agglomerative_predictor(input_words: List[str], metric: str, linkage: str) -> List[List[str]]:
    """Predict groups of 4 words using BERT Agglomerative Clustering."""
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Load BERT model
    embeddings = model.encode(input_words)  # Generate embeddings
    remaining_words = input_words.copy()
    remaining_embeddings = embeddings.copy()
    num_clusters = 4
    solved_groups = []

    while len(remaining_words) > 4:
        # Cluster remaining words
        groups = cluster_words(remaining_words, remaining_embeddings, num_clusters, metric, linkage)

        # Select the best guess (first cluster)
        best_guess = groups[0]
        solved_groups.append(best_guess)

        # Remove guessed words from remaining words
        remaining_words = [word for word in remaining_words if word not in best_guess]
        remaining_embeddings = model.encode(remaining_words)

        # Decrease the number of clusters for the next round
        num_clusters = max(1, len(remaining_words) // 4)  # Ensure clusters can still be formed

    # Final group (remaining 4 words)
    if remaining_words:
        solved_groups.append(remaining_words)

    return solved_groups


def main():
    # Load dataset
    data_path = "../data/connection_answers_aggregate.csv"
    connections_answers = pd.read_csv(data_path)
    connections_answers = connections_answers[400:425]

    # Define a single metric and linkage to use
    metric = 'cosine'  # Specify the metric you want to use
    linkage = 'single'  # Specify the linkage you want to use

    print(f"Evaluating with metric='{metric}' and linkage='{linkage}'")

    # Define predictor function with the specified parameters
    def predictor_func(input_words):
        return bert_agglomerative_predictor(input_words, metric, linkage)

    # Evaluate performance
    evaluator = ConnectionsEvaluator(predictor_func)
    accuracy, connections_made = evaluator.evaluate(connections_answers)

    # Calculate the total possible connections
    total_connections = 4 * len(connections_answers)

    # Calculate the percentage of correct connections
    connections_percent = (connections_made / total_connections) * 100

    # Print results for this configuration
    print(f"Metric: {metric}, Linkage: {linkage}")
    print(f"Number of connections made: {connections_made}/{total_connections}")
    print(f"Percentage of connections made: {connections_percent:.2f}%")
    print(f"Win percent (all correct groups): {accuracy * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
