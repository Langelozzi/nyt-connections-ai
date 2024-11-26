import torch
from connections_evaluator import ConnectionsEvaluator
from gensim.models import KeyedVectors
import pandas as pd

# CNN predictor function
def cnn_predictor(input_words: list[str], model, word_vectors, top_n=4) -> list[list[str]]:
    """
    Predicts top groupings using a trained CNN model.

    Args:
    - input_words: List of words from the game.
    - model: Trained CNN model.
    - word_vectors: Pretrained word embedding model.
    - top_n: Number of top group predictions to return.

    Returns:
    - List of predicted groups (each group is a list of 4 words).
    """
    # Filter valid words
    valid_words = [word for word in input_words if word in word_vectors]
    if len(valid_words) < 4:
        print("Not enough valid words in the vocabulary.")
        return []

    # Convert words to embeddings
    embeddings = [word_vectors[word] for word in valid_words]  # Shape: (16, 300)
    input_tensor = torch.tensor(embeddings, dtype=torch.float32).unsqueeze(0)  # Shape: (1, 16, 300)

    # Run the model
    model.eval()
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(1))  # Add channel dimension

    # Extract probabilities for groupings
    group_scores = output.squeeze().cpu().numpy()  # Shape: (16,)
    ranked_indices = group_scores.argsort()[::-1][:top_n * 4]  # Top N groups (4 words each)

    # Convert to groups
    predicted_groups = []
    for i in range(0, len(ranked_indices), 4):
        group_indices = ranked_indices[i:i + 4]
        group_words = [valid_words[idx] for idx in group_indices]
        predicted_groups.append(group_words)

    return predicted_groups


def main():
    # Load the pretrained Word2Vec model
    model_path = './embeddings/GoogleNews-vectors-negative300-500000.bin'
    print("Loading word embeddings...")
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Word embeddings loaded successfully.")

    # Load the trained CNN model
    print("Loading CNN model...")
    cnn_model = torch.load("cnn_model.pth")  # Replace with your actual model path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cnn_model.to(device)
    print("CNN model loaded successfully.")

    # Define the predictor function
    predictor_func = lambda input_words: cnn_predictor(input_words, cnn_model, word_vectors)

    # Create the evaluator
    evaluator = ConnectionsEvaluator(predictor_func)

    # Load the test dataset
    print("Loading test dataset...")
    test_data_path = 'data/connection_answers_aggregate.csv'  # Replace with your test dataset path
    test_data = pd.read_csv(test_data_path)
    print("Test dataset loaded successfully.")

    # Evaluate the model
    print("Evaluating the model...")
    num_games = 10  # Number of games to evaluate
    accuracy, total_connections_made = evaluator.evaluate(test_data[:num_games])
    print(f"Number of connections made: {total_connections_made}/{4 * num_games}")
    print(f"Win percent: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
