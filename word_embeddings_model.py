from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
import numpy as np


def load_model():
    # Load a pre-trained Word2Vec model (GoogleNews-vectors is common but large)
    # Example: download the model from https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
    print("Loading model... This may take a while.")
    model_path = 'path/to/GoogleNews-vectors-negative300.bin'  # Update this path
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Model loaded successfully.")
    return word_vectors


def get_closest_words(word_list, word_vectors, top_n=4):
    # Filter out words not in the model's vocabulary
    valid_words = [word for word in word_list if word in word_vectors]
    if len(valid_words) < top_n:
        print(f"Not enough valid words in the model's vocabulary (found {len(valid_words)}).")
        return []

    # Compute pairwise cosine similarities
    similarities = {}
    for i, word1 in enumerate(valid_words):
        for j, word2 in enumerate(valid_words):
            if i < j:  # Avoid redundant calculations
                sim = 1 - cosine(word_vectors[word1], word_vectors[word2])
                similarities[(word1, word2)] = sim

    # Sort pairs by similarity in descending order and get top pairs
    closest_pairs = sorted(similarities.items(), key=lambda item: item[1], reverse=True)[:top_n]

    # Flatten the pairs and deduplicate the words to get top unique words
    closest_words = list(set([word for pair, _ in closest_pairs for word in pair]))
    return closest_words[:top_n]


def main():
    # Load the model
    word_vectors = load_model()

    # Take 16 words as input from the user
    words = input("Enter 16 words, separated by commas: ").strip().split(',')
    words = [word.strip().lower() for word in words if word.strip()]

    if len(words) != 16:
        print("Please enter exactly 16 words.")
        return

    # Find and output the 4 closest words
    closest_words = get_closest_words(words, word_vectors)
    if closest_words:
        print("The 4 closest words are:", closest_words)


if __name__ == "__main__":
    main()
