import pandas as pd
from itertools import combinations
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cdist
import numpy as np
from AClust_connections_evaluator import ConnectionsEvaluator

# Functions for BERT grouping
def generate_embeddings(words):
    """Generate BERT embeddings for a list of words."""
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight Sentence-BERT model
    embeddings = model.encode(words)
    return embeddings

def calculate_pairwise_differences(embeddings):
    """Calculate pairwise differences (distances) between embeddings."""
    return cdist(embeddings, embeddings, metric='euclidean')  # Use Euclidean distance

def find_best_group_of_4(words, embeddings):
    """
    Find the best group of 4 words that minimizes intra-group differences.

    Args:
        words (list): List of words to cluster.
        embeddings (ndarray): BERT embeddings for the words.

    Returns:
        tuple: Best group of 4 words, their indices, and total difference.
    """
    pairwise_differences = calculate_pairwise_differences(embeddings)

    # Generate all possible combinations of 4 words
    best_group = None
    best_indices = None
    lowest_difference = float('inf')

    if len(words) < 4:
        print(f"Not enough words to form a group: {words}")
        return None, None, None

    for group in combinations(range(len(words)), 4):
        # Calculate total difference within this group
        total_difference = sum(
            pairwise_differences[i, j]
            for i, j in combinations(group, 2)
        )

        # Update best group if this one is better
        if total_difference < lowest_difference:
            lowest_difference = total_difference
            best_group = [words[idx] for idx in group]
            best_indices = group

    return best_group, best_indices, lowest_difference


def find_top_4_groups(words, embeddings, num_groups=4):
    """
    Find the top groups of 4 words minimizing intra-group differences.

    Args:
        words (list): List of words to cluster.
        embeddings (ndarray): BERT embeddings for the words.
        num_groups (int): Number of groups to form.

    Returns:
        list: List of top groups and their differences.
    """
    groups = []
    remaining_words = words.copy()
    remaining_embeddings = embeddings.copy()

    for _ in range(num_groups):
        if len(remaining_words) < 4:
            print(f"Not enough words left to form a group: {remaining_words}")
            break

        best_group, best_indices, total_difference = find_best_group_of_4(remaining_words, remaining_embeddings)

        if best_group is None:
            print(f"Failed to find a valid group in: {remaining_words}")
            break

        groups.append((best_group, total_difference))

        # Remove selected words and their embeddings
        remaining_indices = [i for i in range(len(remaining_words)) if i not in best_indices]
        remaining_words = [remaining_words[i] for i in remaining_indices]
        remaining_embeddings = remaining_embeddings[remaining_indices]

    return groups


# Predictor function
def bert_predictor(input_words: list[str]) -> list[list[str]]:
    if len(input_words) < 4:
        print(f"Insufficient words for grouping: {input_words}")
        return []

    embeddings = generate_embeddings(input_words)
    top_groups = find_top_4_groups(input_words, embeddings, num_groups=4)

    # Ensure all groups are valid and handle None gracefully
    return [group for group, _ in top_groups if group is not None and len(group) == 4]


# Main evaluation script
def main():
    # Load dataset
    data_path = "../data/connection_answers_aggregate.csv"
    connections_answers = pd.read_csv(data_path)

    # connections_answers = connections_answers[120:130]

    # Define predictor function
    predictor_func = bert_predictor

    # Evaluate performance
    evaluator = ConnectionsEvaluator(predictor_func)
    accuracy, connections_made = evaluator.evaluate(connections_answers)

    # Print results
    print(f"Number of connections made: {connections_made}/{4 * len(connections_answers)}")
    print(f"Win percent: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
