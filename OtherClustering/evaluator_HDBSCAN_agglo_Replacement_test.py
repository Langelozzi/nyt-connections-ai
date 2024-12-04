import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
import hdbscan
from AClust_connections_evaluator import ConnectionsEvaluator


def normalize_embeddings(embeddings):
    """Normalize embeddings to unit vectors for cosine similarity."""
    return normalize(embeddings)


def calculate_average_embedding(word_vectors):
    """Calculate the average embedding for all words in the Word2Vec model."""
    all_embeddings = word_vectors.vectors  # Access all embeddings
    return np.mean(all_embeddings, axis=0)  # Compute the mean embedding


def cluster_words_hdbscan(words, embeddings, min_cluster_size, group_size=4):
    """
    Cluster words using HDBSCAN into groups of `group_size`.
    Ensures clusters are exactly `group_size` words.
    """
    # Run HDBSCAN clustering
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
    cluster_labels = clusterer.fit_predict(embeddings)

    # Group words by cluster labels
    clusters = {}
    for word, label in zip(words, cluster_labels):
        if label != -1:  # Ignore noise points
            clusters.setdefault(label, []).append(word)

    # Flatten all clusters into a single list of words
    all_words = [word for group in clusters.values() for word in group]

    # Force equal-sized groups of `group_size`
    groups = []
    current_group = []

    for word in all_words:
        current_group.append(word)
        if len(current_group) == group_size:
            groups.append(current_group)
            current_group = []

    # Handle leftover words by redistributing them
    if current_group:
        print(f"Leftover words: {current_group}. Redistributing to existing groups.")
        for i, word in enumerate(current_group):
            groups[i % len(groups)].append(word)

    # Ensure all groups are exactly `group_size`
    return [group[:group_size] for group in groups if len(group) >= group_size]



def word2vec_hdbscan_predictor(input_words: list[str], word_vectors, min_cluster_size=4) -> list[list[str]]:
    """Predict groups of 4 words using Word2Vec and HDBSCAN."""

    # Calculate the average embedding for missing words
    average_embedding = calculate_average_embedding(word_vectors)

    # Generate embeddings for input words
    embeddings = []
    valid_words = []

    for word in input_words:
        if word in word_vectors:
            embeddings.append(word_vectors[word])  # Use Word2Vec embedding
        else:
            print(f"Word '{word}' not found in Word2Vec. Using fallback embedding.")
            embeddings.append(average_embedding)
        valid_words.append(word)

    embeddings = np.array(embeddings)

    # Cluster words using HDBSCAN
    groups = cluster_words_hdbscan(valid_words, embeddings, min_cluster_size=min_cluster_size)

    return groups


def main():
    # Load dataset
    data_path = "../data/connection_answers_aggregate.csv"
    connections_answers = pd.read_csv(data_path)
    connections_answers = connections_answers[:522]

    # Load pre-trained Word2Vec embeddings
    model_path = "../GoogleNews-vectors-negative300.bin"
    print("Loading Word2Vec model...")
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Word2Vec model loaded successfully.")

    print("Evaluating with HDBSCAN clustering...")

    # Define predictor function
    def predictor_func(input_words):
        return word2vec_hdbscan_predictor(input_words, word_vectors, min_cluster_size=4)

    # Evaluate performance
    evaluator = ConnectionsEvaluator(predictor_func)
    accuracy, connections_made = evaluator.evaluate(connections_answers)

    # Calculate statistics
    total_connections = 4 * len(connections_answers)
    connections_percent = (connections_made / total_connections) * 100

    # Print results
    print(f"Number of connections made: {connections_made}/{total_connections}")
    print(f"Percentage of connections made: {connections_percent:.2f}%")
    print(f"Win percent (all correct groups): {accuracy * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    main()
