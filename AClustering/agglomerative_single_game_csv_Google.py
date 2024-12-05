import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine
import numpy as np
from gensim.models import KeyedVectors

def normalize_embeddings(embeddings):
    """Normalize embeddings to unit vectors for cosine similarity."""
    return normalize(embeddings)

def perform_clustering(embeddings, num_initial_clusters):
    """Perform Agglomerative Clustering with more initial clusters."""
    normalized_embeddings = normalize_embeddings(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=num_initial_clusters, metric="cosine", linkage="average"
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
    cluster_labels = perform_clustering(embeddings, num_clusters * 2)
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

def process_game_row(row, word_vectors, num_clusters, group_size):
    """Process a single game row and predict clusters."""
    words = row["Puzzle"].split(", ")
    print(f"Words in puzzle: {words}")

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = []
    for word in words:
        if word in word_vectors:
            embeddings.append(word_vectors[word])
        else:
            print(f"Word '{word}' not found in Word2Vec model, skipping.")
            embeddings.append(np.zeros(word_vectors.vector_size))  # Fallback for missing words
    embeddings = np.array(embeddings)

    # Enforce equal-sized clusters
    print("Clustering words...")
    final_groups = enforce_equal_sized_clusters(words, embeddings, num_clusters, group_size)

    return final_groups

def main():
    # Load CSV data
    data_path = "../data/connection_answers_aggregate.csv"
    data = pd.read_csv(data_path)

    # Select a single row (example index 20)
    single_row = data.iloc[400]

    # Load the Google News Word2Vec model
    print("Loading Google News Word2Vec model...")
    model_path = "../GoogleNews-vectors-negative300.bin"
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Model loaded successfully.")

    # Define clustering parameters
    num_clusters = 4
    group_size = 4

    # Process the selected game row
    final_groups = process_game_row(single_row, word_vectors, num_clusters, group_size)

    # Display the groups
    print("\nPredicted Groups:")
    for i, group in enumerate(final_groups, 1):
        print(f"Group {i}:")
        for item in group:
            print(f"  Word: {item['word']}, Confidence: {item['confidence']:.4f}")

if __name__ == "__main__":
    main()
