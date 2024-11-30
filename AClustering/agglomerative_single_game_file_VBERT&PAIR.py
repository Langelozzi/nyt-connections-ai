from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

def normalize_embeddings(embeddings):
    """Normalize embeddings to unit vectors for cosine similarity."""
    return normalize(embeddings)

def calculate_similarity_matrix(embeddings):
    """Calculate the pairwise cosine similarity matrix."""
    return cosine_similarity(embeddings)

def group_by_pairwise_similarity(words, similarity_matrix, num_clusters, group_size):
    """
    Group words by pairwise similarity.

    Args:
        words (list): List of words.
        similarity_matrix (ndarray): Pairwise similarity matrix.
        num_clusters (int): Number of clusters to form.
        group_size (int): Number of words per group.

    Returns:
        list: Groups of words based on pairwise similarity.
    """
    # Create a list of all pairwise similarities with their indices
    pairs = [
        (i, j, similarity_matrix[i, j])
        for i in range(len(words))
        for j in range(i + 1, len(words))
    ]
    # Sort pairs by similarity in descending order
    pairs.sort(key=lambda x: x[2], reverse=True)

    # Track assigned words to avoid duplication
    assigned_words = set()
    groups = []

    # Assign strongly similar pairs to groups
    for i, j, _ in pairs:
        if len(groups) == num_clusters:
            break
        if i not in assigned_words and j not in assigned_words:
            # Start a new group with the pair
            group = [i, j]
            assigned_words.update(group)
            groups.append(group)

    # Fill up groups to the desired size
    for group in groups:
        while len(group) < group_size:
            # Find the most similar unassigned word to the group
            candidate = max(
                (k for k in range(len(words)) if k not in assigned_words),
                key=lambda k: sum(similarity_matrix[k, idx] for idx in group),
                default=None,
            )
            if candidate is not None:
                group.append(candidate)
                assigned_words.add(candidate)
            else:
                break

    # Handle remaining unassigned words (if any)
    unassigned_words = [k for k in range(len(words)) if k not in assigned_words]
    for k in unassigned_words:
        # Add to an existing group if possible
        added = False
        for group in groups:
            if len(group) < group_size:
                group.append(k)
                assigned_words.add(k)
                added = True
                break
        # If no room, start a new group
        if not added:
            groups.append([k])

    # Convert groups from indices to words
    word_groups = [[words[idx] for idx in group] for group in groups]

    return word_groups

def process_game_row_with_similarity(row, model, num_clusters, group_size):
    """Process a single game row and predict clusters using pairwise similarity."""
    words = row["Puzzle"].split(", ")
    print(f"Words in puzzle: {words}")

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(words)

    # Normalize embeddings
    normalized_embeddings = normalize_embeddings(embeddings)

    # Calculate pairwise similarity matrix
    print("Calculating similarity matrix...")
    similarity_matrix = calculate_similarity_matrix(normalized_embeddings)

    # Group words by pairwise similarity
    print("Grouping words by pairwise similarity...")
    final_groups = group_by_pairwise_similarity(words, similarity_matrix, num_clusters, group_size)

    return final_groups

def main():
    # Load CSV data
    data_path = "../data/connection_answers_aggregate.csv"
    data = pd.read_csv(data_path)

    # Select a single row (example index 406)
    single_row = data.iloc[406]

    # Load the Sentence-BERT model
    print("Loading Sentence-BERT model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("Model loaded successfully.")

    # Define clustering parameters
    num_clusters = 4
    group_size = 4

    # Process the selected game row using pairwise similarity
    final_groups = process_game_row_with_similarity(single_row, model, num_clusters, group_size)

    # Display the groups
    print("\nPredicted Groups (Pairwise Similarity):")
    for i, group in enumerate(final_groups, 1):
        print(f"Group {i}: {group}")

if __name__ == "__main__":
    main()
