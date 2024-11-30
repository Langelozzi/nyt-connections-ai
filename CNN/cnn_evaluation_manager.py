import pandas as pd
from gensim.models import KeyedVectors
import torch
from cnn_model_evaluation import CNNConnectionsEvaluator
from cnn_model import ConnectionsCNN


def load_cnn_model(model_path, embedding_dim=300):
    """Load the pretrained CNN model."""
    model = ConnectionsCNN(embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def load_word_vectors(model_path):
    """Load pretrained Word2Vec word vectors."""
    print("Loading word vectors... This may take some time.")
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Word vectors loaded successfully.")
    return word_vectors


def main():
    # Define file paths
    word_vectors_path = '../GoogleNews-vectors-negative300.bin'
    cnn_model_path = './cnn_model.pth'
    test_data_path = '../data/connection_answers_aggregate.csv'

    # Load resources
    word_vectors = load_word_vectors(word_vectors_path)
    cnn_model = load_cnn_model(cnn_model_path)

    # Load test data
    test_data = pd.read_csv(test_data_path)
    test_data = test_data[406:407]

    # Evaluate the CNN model
    evaluator = CNNConnectionsEvaluator(cnn_model, word_vectors)
    accuracy, total_connections_made = evaluator.evaluate(test_data)

    # Output the results
    print("=== CNN Evaluation Results ===")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Total connections made: {total_connections_made}")


if __name__ == "__main__":
    main()
