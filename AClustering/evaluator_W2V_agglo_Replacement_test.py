import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from AClust_connections_evaluator import ConnectionsEvaluator


def normalize_embeddings(embeddings):
    """Normalize embeddings to unit vectors for cosine similarity."""
    return normalize(embeddings)


def calculate_average_embedding(word_vectors):
    """Calculate the average embedding for all words in the Word2Vec model."""
    all_embeddings = word_vectors.vectors  # Access all embeddings
    return np.mean(all_embeddings, axis=0)  # Compute the mean embedding


def compute_phrase_embedding(phrase, word_vectors, fallback_embedding):
    """Compute an embedding for a multi-term phrase by averaging its component word embeddings."""
    words = phrase.split()  # Split the phrase into individual words
    embeddings = []

    for word in words:
        if word in word_vectors:
            embeddings.append(word_vectors[word])  # Use Word2Vec embedding
        else:
            print(f"Word '{word}' not found in Word2Vec. Using fallback embedding.")
            embeddings.append(fallback_embedding)  # Use fallback for OOV words

    # Return the average embedding for the phrase
    if embeddings:
        return np.mean(embeddings, axis=0)  # Aggregate embeddings (e.g., mean)

    return fallback_embedding  # Fallback if none of the words are found


def perform_clustering(embeddings, num_clusters, metric, linkage):
    """Perform Agglomerative Clustering wit specified metric and linkage."""
    normalized_embeddings = normalize_embeddings(embeddings)
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

def word2vec_agglomerative_predictor(input_words: list[str], word_vectors, metric: str, linkage: str) -> list[list[str]]:
    """Predict groups of 4 words using Word2Vec Agglomerative Clustering."""

    # Calculate the average embedding for missing words
    average_embedding = calculate_average_embedding(word_vectors)

    # Generate embeddings for input words
    embeddings = []
    valid_words = []

    for word in input_words:
        if word in word_vectors:
            embeddings.append(word_vectors[word])  # Use the word's embedding directly
        elif "-" in word:  # Handle hyphenated words
            hyphen_removed = word.replace("-", "")
            if hyphen_removed in word_vectors:
                print(f"Word '{word}' not found. Using hyphen-removed version: '{hyphen_removed}'.")
                embeddings.append(word_vectors[hyphen_removed])
            else:
                print(f"Hyphen-removed word '{hyphen_removed}' not found. Using fallback embedding.")
                embeddings.append(average_embedding)
        elif " " in word:  # If the word is a phrase
            print(f"Phrase '{word}' not found in embeddings, computing phrase embedding.")
            embeddings.append(compute_phrase_embedding(word, word_vectors, average_embedding))
        else:
            print(f"Word '{word}' not found in embeddings, assigning average embedding.")
            embeddings.append(average_embedding)
        valid_words.append(word)

    remaining_words = valid_words.copy()
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
        remaining_embeddings = np.array(
            [embedding for word, embedding in zip(valid_words, embeddings) if word in remaining_words]
        )

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
    connections_answers = connections_answers[:522]

    # Path to pre-trained Google News Word2Vec embeddings
    model_path = "../GoogleNews-vectors-negative300.bin"

    # Load the Word2Vec model once
    print("Loading Word2Vec model...")
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Model loaded successfully.")

    # Define a single metric and linkage to use
    metric = 'cosine'  # Specify the metric you want to use
    linkage = 'single'  # Specify the linkage you want to use

    print(f"Evaluating with metric='{metric}' and linkage='{linkage}'")

    # Define predictor function with the loaded Word2Vec model
    def predictor_func(input_words):
        return word2vec_agglomerative_predictor(input_words, word_vectors, metric, linkage)

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
