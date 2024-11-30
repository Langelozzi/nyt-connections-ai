import pandas as pd
from itertools import combinations
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import numpy as np

def generate_embeddings(words):
    """Generate BERT embeddings for a list of words."""
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight Sentence-BERT model
    embeddings = model.encode(words)
    return embeddings

def calculate_pairwise_differences(embeddings):
    """Calculate pairwise differences (distances) between embeddings."""
    return cdist(embeddings, embeddings, metric='euclidean')  # Use Euclidean distance

def find_optimal_groups(words, embeddings, num_clusters):
    """
    Find optimal groups of 4 words that minimize intra-group differences.

    Args:
        words (list): List of words to cluster.
        embeddings (ndarray): BERT embeddings for the words.
        num_clusters (int): Number of groups to form.

    Returns:
        list: Optimal groups of words.
    """
    group_size = 4
    n = len(words)

    # Ensure there are enough words for the requested clusters
    assert n == num_clusters * group_size, "Number of words must match the number of groups * group size."

    # Calculate pairwise differences
    pairwise_differences = calculate_pairwise_differences(embeddings)

    # Generate all possible combinations of groups
    all_combinations = list(combinations(range(n), group_size))

    # Minimize total intra-group differences
    best_groups = None
    lowest_total_difference = float('inf')

    for comb in combinations(all_combinations, num_clusters):
        # Ensure groups are disjoint
        indices = set(idx for group in comb for idx in group)
        if len(indices) != n:
            continue

        # Calculate total intra-group difference
        total_difference = sum(
            pairwise_differences[i, j]
            for group in comb
            for i, j in combinations(group, 2)
        )

        # Update if this grouping is better
        if total_difference < lowest_total_difference:
            lowest_total_difference = total_difference
            best_groups = comb

    # Convert indices to words
    word_groups = [[words[idx] for idx in group] for group in best_groups]

    return word_groups, lowest_total_difference

def main():
    # Load CSV data
    data_path = "../data/connection_answers_aggregate.csv"
    data = pd.read_csv(data_path)

    # Select a single row (example index 406)
    single_row = data.iloc[406]
    words = single_row["Puzzle"].split(", ")

    # Number of clusters
    num_clusters = 4

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = generate_embeddings(words)

    # Find optimal groups
    print("Finding optimal groups...")
    optimal_groups, total_difference = find_optimal_groups(words, embeddings, num_clusters)

    # Display results
    print("\nOptimal Groups:")
    for i, group in enumerate(optimal_groups, 1):
        print(f"Group {i}: {group}")
    print(f"\nTotal Difference: {total_difference:.4f}")

if __name__ == "__main__":
    main()
