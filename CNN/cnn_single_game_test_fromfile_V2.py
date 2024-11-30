import torch
from CNN.cnn_model import ConnectionsCNN
from gensim.models import KeyedVectors
import numpy as np
from word_embeddings_model import load_words


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
    cnn_model = ConnectionsCNN()
    cnn_model.load_state_dict(torch.load(cnn_model_path, map_location=torch.device('cpu')))
    cnn_model.eval()
    print("CNN model loaded successfully.")

    # Run CNN predictor
    print("Predicting groups...")
    top_sets = cnn_predictor(words, cnn_model, word_vectors, top_n=10)

    for i, group in enumerate(top_sets, 1):
        print(f"Set {i}: Words = {group}")


if __name__ == "__main__":
    # Paths to resources
    word_file = "words.txt"
    model_path = '../GoogleNews-vectors-negative300.bin'
    cnn_model_path = "cnn_model.pth"

    run_cnn_on_words(word_file, model_path, cnn_model_path)
