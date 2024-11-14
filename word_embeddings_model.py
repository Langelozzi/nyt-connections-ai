from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from itertools import combinations


def load_model():
    print("Loading model... This may take a while.")
    model_path = './embeddings/GoogleNews-vectors-negative300.bin'  # Update this path
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Model loaded successfully.")
    return word_vectors


def calculate_group_similarity(group, word_vectors):
    # Calculate the sum of cosine similarities for each pair in the group
    similarity_sum = 0
    for word1, word2 in combinations(group, 2):
        sim = 1 - cosine(word_vectors[word1], word_vectors[word2])
        similarity_sum += sim
    return similarity_sum


def get_top_n_sets(word_list, word_vectors, top_n_sets=3, group_size=4):
    # Filter out words not in the model's vocabulary
    valid_words = [word for word in word_list if word in word_vectors]
    if len(valid_words) < group_size:
        print(f"Not enough valid words in the model's vocabulary (found {len(valid_words)}).")
        return []

    # Generate all possible combinations of 4 words and calculate similarity scores
    group_similarities = []
    for group in combinations(valid_words, group_size):
        similarity_sum = calculate_group_similarity(group, word_vectors)
        group_similarities.append((group, similarity_sum))

    # Sort groups by cumulative similarity score in descending order and get the top sets
    top_groups = sorted(group_similarities, key=lambda item: item[1], reverse=True)[:top_n_sets]
    return top_groups


def load_words(filename):
    words = []
    with open(filename, 'r') as file:
        for line in file:
            # Strip whitespace, convert to lowercase, and skip empty lines
            word = line.strip().lower()
            if word:
                words.append(word)
    return words


def main():
    # Load the model
    word_vectors = load_model()

    # Load predefined list of words
    word_file = "words.txt"
    words = load_words(word_file)

    # Find and output the top 3 sets of 4 closest words
    top_sets = get_top_n_sets(words, word_vectors, top_n_sets=5, group_size=4)
    for i, (group, score) in enumerate(top_sets, 1):
        print(f"Set {i}: Words = {group}, Similarity Score = {score:.4f}")


if __name__ == "__main__":
    main()

