import torch
import torch.nn as nn
import torch.nn.functional as F
from connections_evaluator import ConnectionsEvaluator
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from word_embeddings_model import load_words
from cnn_model import ConnectionsCNN

# CNN predictor function
def cnn_predictor(input_words: list[str], model, word_vectors, top_n=4) -> list[list[str]]:
    valid_words = [word for word in input_words if word in word_vectors]
    print(f"Valid words: {valid_words}")  # Debugging

    if len(valid_words) < 4:
        print("Not enough valid words in the vocabulary.")
        return []

    # Convert words to embeddings
    embeddings = [word_vectors[word] for word in valid_words]
    print(f"Number of embeddings: {len(embeddings)}")  # Debugging

    input_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)
    print(f"Input tensor shape: {input_tensor.shape}")  # Debugging

    # Run the model
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    print(f"Model output shape: {output.shape}")  # Debugging
    print(f"Model output: {output}")  # Debugging

    # Extract probabilities for groupings
    group_scores = output.squeeze().cpu().numpy()
    print(f"Group scores: {group_scores}")  # Debugging

    valid_indices = len(valid_words)
    group_scores = group_scores.reshape(valid_indices, 4)  # Reshape to match [words, groups]
    ranked_indices = np.dstack(np.unravel_index(np.argsort(group_scores.ravel())[::-1], group_scores.shape))
    ranked_indices = ranked_indices.squeeze(0)  # Reshape to match usable indices

    print(f"Distribution of group scores:\nMin: {group_scores.min()}, Max: {group_scores.max()}")

    print(f"Ranked indices: {ranked_indices}")  # Debugging

    # Convert to groups
    predicted_groups = []
    for i in range(0, len(ranked_indices), 4):
        group_indices = ranked_indices[i:i + 4]
        group_indices = [idx for idx in group_indices if idx[0] < len(valid_words)]  # Check word indices only
        if len(group_indices) < 4:
            continue
        group_words = [valid_words[idx[0]] for idx in group_indices]
        predicted_groups.append(group_words)

    print(f"Predicted groups: {predicted_groups}")  # Debugging
    return predicted_groups



def run_cnn_on_words(word_file: str, model_path: str, cnn_model_path: str):
    """
    Finds top sets of 4 words using CNN on the provided word list.
    """
    # Load word embeddings
    print("Loading word embeddings...")
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Word embeddings loaded successfully.")

    # Load words
    words = load_words(word_file)

    # Load CNN model
    print("Loading CNN model...")
    cnn_model = torch.load(cnn_model_path, map_location=torch.device('cpu'))
    cnn_model.eval()
    print("CNN model loaded successfully.")

    # Run CNN predictor
    print("Predicting groups...")
    top_sets = cnn_predictor(words, cnn_model, word_vectors, top_n=10)

    for i, group in enumerate(top_sets, 1):
        print(f"Set {i}: Words = {group}")


def run_cnn_on_evaluator(test_data_path: str, model_path: str, cnn_model_path: str, num_games: int = 10):
    """
    Evaluates CNN model using ConnectionsEvaluator on the given dataset.
    """
    # Load word embeddings
    print("Loading word embeddings...")
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Word embeddings loaded successfully.")

    # Load CNN model
    print("Loading CNN model...")
    cnn_model = torch.load(cnn_model_path)
    cnn_model.eval()
    print("CNN model loaded successfully.")

    # Define predictor function
    predictor_func = lambda input_words: cnn_predictor(input_words, cnn_model, word_vectors)

    # Create evaluator
    evaluator = ConnectionsEvaluator(predictor_func)

    # Load test data
    print("Loading test dataset...")
    test_data = pd.read_csv(test_data_path)
    print("Test dataset loaded successfully.")

    # Evaluate model
    print("Evaluating model...")
    accuracy, total_connections_made = evaluator.evaluate(test_data[:num_games])
    print(f"Number of connections made: {total_connections_made}/{4 * num_games}")
    print(f"Win percent: {accuracy * 100:.2f}%")


def main():
    """
    Main method for selecting and running the desired functionality.
    """
    # Paths to resources
    word_file = "words.txt"
    test_data_path = "data/connection_answers_aggregate.csv"
    model_path = './GoogleNews-vectors-negative300.bin'
    cnn_model_path = "cnn_model.pth"

    # User input to select functionality
    # print("Select mode of operation:")
    # print("1. Run CNN on words (words.txt)")
    # print("2. Evaluate CNN using ConnectionsEvaluator")
    # mode = input("Enter your choice (1/2): ").strip()

    run_cnn_on_words(word_file, model_path, cnn_model_path)


    # if mode == "1":
    #     print("\nRunning CNN on words...")
    #     run_cnn_on_words(word_file, model_path, cnn_model_path)
    # elif mode == "2":
    #     num_games = int(input("\nEnter number of games to evaluate: "))
    #     print("\nRunning CNN evaluation...")
    #     run_cnn_on_evaluator(test_data_path, model_path, cnn_model_path, num_games)
    # else:
    #     print("Invalid choice. Please select either 1 or 2.")


if __name__ == "__main__":
    main()
