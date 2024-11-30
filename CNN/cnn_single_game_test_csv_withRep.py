import torch
from gensim.models import KeyedVectors
import numpy as np
import random
from CNN.cnn_model import ConnectionsCNN


def cnn_predict_top_groups(row, cnn_model, word_vectors, top_n=4):
    """
    Predict the top N most likely groups of words for a given row using the trained CNN.

    Args:
        row (pd.Series): A single row from the CSV file with 'Puzzle' and 'Answer' columns.
        cnn_model (ConnectionsCNN): Trained CNN model.
        word_vectors (KeyedVectors): Word embeddings model.
        top_n (int): Number of top groups to return.

    Returns:
        list[list[str]]: The top N most likely groups of words.
    """
    # Extract words from the puzzle
    words = row['Puzzle'].lower().split(', ')
    valid_words = [word for word in words if word in word_vectors]

    if len(valid_words) < 4:
        print("Not enough valid words in the vocabulary.")
        return []

    # Randomize the order of valid words
    random.shuffle(valid_words)
    print(f"Randomized order of valid words: {valid_words}")  # Output the randomized order

    # Convert words to embeddings
    embeddings = [word_vectors[word] for word in valid_words]
    print(f"Number of valid embeddings: {len(embeddings)}")
    print(f"Embedding dimensions: {len(embeddings[0]) if embeddings else 'No embeddings'}")
    print("Sample embeddings:")
    for word, embedding in zip(valid_words, embeddings[:5]):  # Print first 5 embeddings
        print(f"Word: {word}, Embedding: {embedding[:10]}...")  # Truncate for brevity

    # Convert embeddings to tensor
    input_tensor = torch.tensor(np.array(embeddings), dtype=torch.float32).unsqueeze(0)
    print(f"Input tensor shape: {input_tensor.shape}")

    # Run the model
    cnn_model.eval()
    with torch.no_grad():
        output = cnn_model(input_tensor)

    # Extract probabilities for groupings
    group_scores = output.squeeze().cpu().numpy()
    group_scores = group_scores[:len(valid_words)]  # Ensure alignment with valid words
    group_scores = group_scores.reshape(len(valid_words), 4)  # Reshape to match [words, groups]

    print(f"Group scores shape: {group_scores.shape}")

    # Rank all word-group combinations
    ranked_indices = np.dstack(np.unravel_index(np.argsort(group_scores.ravel())[::-1], group_scores.shape))
    ranked_indices = ranked_indices.squeeze(0)  # Flatten to usable indices

    print(f"Ranked indices: {ranked_indices[:10]}")  # Debug: Print top 10 ranked indices

    # Collect top N groups allowing overlap but no exact repetition
    predicted_groups = []
    used_combinations = set()  # To ensure unique groups

    for group_num in range(top_n):
        group = []
        group_indices = set()  # Track indices for this group
        for word_idx, group_idx in ranked_indices:
            if len(group) >= 4:  # Stop after forming a group of 4 words
                break
            if word_idx not in group_indices:  # Ensure no repetition within the group
                group.append(valid_words[word_idx])
                group_indices.add(word_idx)

        group = sorted(group)  # Sort to ensure order doesn't affect uniqueness
        group_tuple = tuple(group)
        if group_tuple not in used_combinations:
            predicted_groups.append(group)
            used_combinations.add(group_tuple)
            print(f"Group {group_num + 1}: {group}")  # Debug: Print the group
        else:
            print(f"Skipping duplicate group: {group}")  # Debug: Skipping duplicates

    return predicted_groups


def predict_top_groups(row, model_path, cnn_model_path, top_n=4):
    """
    Use CNN to predict the top N most likely groups of words for a given row.

    Args:
        row (pd.Series): A single row from the CSV file with 'Puzzle' and 'Answer' columns.
        model_path (str): Path to the word2vec model.
        cnn_model_path (str): Path to the trained CNN model.
        top_n (int): Number of top groups to return.

    Returns:
        list[list[str]]: The top N most likely groups of words.
    """
    # Load word embeddings
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)

    # Load CNN model
    cnn_model = ConnectionsCNN()
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=torch.device('cpu')))
    cnn_model.eval()

    # Predict the top N groups
    top_groups = cnn_predict_top_groups(row, cnn_model, word_vectors, top_n)
    return top_groups


# Example usage
if __name__ == "__main__":
    import pandas as pd

    # Paths to resources
    data_path = "../data/connection_answers_aggregate.csv"
    model_path = '../GoogleNews-vectors-negative300.bin'
    cnn_model_path = "cnn_model.pth"

    # Load a single row of data
    data = pd.read_csv(data_path)
    single_row = data.iloc[406]  # Change index as needed

    # Predict top 4 groups for the single game
    top_groups = predict_top_groups(single_row, model_path, cnn_model_path, top_n=4)
    print(f"Top Groups: {top_groups}")
