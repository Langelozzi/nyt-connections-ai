from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
import numpy as np


def get_word2vec_embedding(input_text, word_vectors, fallback_embedding):
    """
    Compute the Word2Vec embedding for a word or phrase.

    Args:
        input_text (str): The input text (word or phrase).
        word_vectors (KeyedVectors): Pre-trained Word2Vec model.
        fallback_embedding (numpy.ndarray): Fallback embedding for missing words.

    Returns:
        numpy.ndarray: The embedding for the word or phrase.
    """
    words = input_text.split()  # Split the input into words
    embeddings = []

    for word in words:
        if word in word_vectors:
            embeddings.append(word_vectors[word])  # Use Word2Vec embedding
        else:
            print(f"Word '{word}' not found in Word2Vec. Using fallback embedding.")
            embeddings.append(fallback_embedding)  # Use fallback for OOV words

    # Compute the average embedding for the phrase
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        raise ValueError("No valid words found in input_text.")


def compare_texts_with_word2vec(text1, text2, model_path="../GoogleNews-vectors-negative300.bin"):
    """
    Compare the closeness of two inputs (words or phrases) using Word2Vec embeddings.

    Args:
        text1 (str): The first input text (word or phrase).
        text2 (str): The second input text (word or phrase).
        model_path (str): Path to the pre-trained Word2Vec model.

    Returns:
        float: Cosine similarity between the embeddings of the inputs.
    """
    # Load the pre-trained Word2Vec model
    print("Loading Word2Vec model...")
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Word2Vec model loaded successfully.")

    # Calculate the fallback embedding (average of all Word2Vec vectors)
    fallback_embedding = np.mean(word_vectors.vectors, axis=0)

    # Compute embeddings for each input
    embedding1 = get_word2vec_embedding(text1, word_vectors, fallback_embedding)
    embedding2 = get_word2vec_embedding(text2, word_vectors, fallback_embedding)

    # Compute cosine similarity
    similarity = 1 - cosine(embedding1, embedding2)

    # if "milky way" in word_vectors:
    #     print("'milky way' exists as a single token in the Word2Vec vocabulary.")
    # else:
    #     print("'milky way' does NOT exist as a single token in the Word2Vec vocabulary.")

    return similarity


# Example usage
if __name__ == "__main__":
    # Inputs to compare
    input1 = "flipflop"
    input2 = "sandal"

    # Compare the inputs using Word2Vec
    similarity = compare_texts_with_word2vec(input1, input2)

    print(f"Cosine similarity between '{input1}' and '{input2}': {similarity:.4f}")
