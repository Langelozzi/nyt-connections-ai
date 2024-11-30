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
    input_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)

    # Run the model
    cnn_model.eval()
    with torch.no_grad():
        output = cnn_model(input_tensor)

    # Extract probabilities for groupings
    group_scores = output.squeeze().cpu().numpy()
    group_scores = group_scores[:len(valid_words)]  # Ensure alignment with valid words
    group_scores = group_scores.reshape(len(valid_words), 4)  # Reshape to match [words, groups]

    # Rank all word-group combinations
    ranked_indices = np.dstack(np.unravel_index(np.argsort(group_scores.ravel())[::-1], group_scores.shape))
    ranked_indices = ranked_indices.squeeze(0)  # Flatten to usable indices

    # Collect top N groups based on ranked scores
    predicted_groups = []
    used_words = set()
    for group_rank in range(top_n):
        group = []
        for word_idx, group_idx in ranked_indices:
            if word_idx in used_words or len(group) >= 4:
                continue
            if np.argmax(group_scores[word_idx]) == group_idx:
                group.append(valid_words[word_idx])
                used_words.add(word_idx)
        if group:
            predicted_groups.append(group)

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
