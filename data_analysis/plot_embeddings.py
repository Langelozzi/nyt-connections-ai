import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist

def load_word2vec_embeddings(model_path):
    """Load pre-trained Word2Vec embeddings."""
    print("Loading Word2Vec model...")
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Model loaded successfully.")
    return word_vectors

def compute_word_embedding(word, word_vectors, fallback_embedding):
    """Get the embedding for a word or use a fallback if not found."""
    if word in word_vectors:
        return word_vectors[word]
    else:
        return fallback_embedding

def compute_group_distance(group, word_vectors):
    """Compute the mean pairwise cosine distance for a group."""
    fallback_embedding = np.mean(word_vectors.vectors, axis=0)  # Fallback to mean embedding
    embeddings = [compute_word_embedding(word, word_vectors, fallback_embedding) for word in group]
    pairwise_distances = pdist(embeddings, metric='cosine')  # Use cosine distance
    return np.mean(pairwise_distances)

def load_puzzle_data(csv_path):
    """Load the Connections puzzle data."""
    data = pd.read_csv(csv_path)
    data['Answer'] = data['Answer'].apply(eval)  # Convert string to list
    return data

def process_all_groups(data, word_vectors):
    """Calculate the mean pairwise distances for all answer groups."""
    distances = []

    for index, row in data.iterrows():
        answers = row['Answer']
        for group in answers:
            group_words = group.split(', ')
            distance = compute_group_distance(group_words, word_vectors)
            distances.append(distance)

    return distances

def plot_distance_distribution(distances):
    """Plot a histogram of distances."""
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=30, color='blue', alpha=0.7, edgecolor='black')
    plt.title("Distribution of Mean Pairwise Cosine Distances for Answer Groups", fontsize=14)
    plt.xlabel("Mean Pairwise Cosine Distance", fontsize=12)
    plt.ylabel("Frequency (Number of Groups)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()

def main():
    # Path to Google News Word2Vec model
    model_path = "../GoogleNews-vectors-negative300.bin"

    # Load Word2Vec model
    word_vectors = load_word2vec_embeddings(model_path)

    # Path to CSV data
    csv_path = "../data/connection_answers_aggregate.csv"
    data = load_puzzle_data(csv_path)

    data_path = "../data/connection_answers_aggregate.csv"
    connections_answers = pd.read_csv(data_path)
    connections_answers = connections_answers[:520]

    print("Processing puzzle data...")
    distances = process_all_groups(data, word_vectors)

    print(f"Processed {len(distances)} groups.")
    print("Plotting distance distribution...")
    plot_distance_distribution(distances)

if __name__ == "__main__":
    main()
