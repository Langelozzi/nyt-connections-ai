from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from itertools import combinations
from connections_evaluator import ConnectionsEvaluator
import pandas as pd


def load_model(model_path):
    print("Loading model... This may take a while.")
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Model loaded successfully.")
    return word_vectors


def truncate_model():
    # Path to the full pretrained Word2Vec file
    original_model_path = './embeddings/GoogleNews-vectors-negative300.bin'

    # Limit to the top N words (e.g., 500,000)
    N = 500000

    # Path to save the truncated model
    truncated_model_path = f'./embeddings/GoogleNews-vectors-negative300-{N}.bin'

    # Load the model with the limit
    print(f"Loading the top {N} words from the Word2Vec model...")
    word_vectors = KeyedVectors.load_word2vec_format(original_model_path, binary=True, limit=N)
    print(f"Loaded {len(word_vectors)} words.")

    # Save the truncated model back to disk
    print(f"Saving the truncated model to {truncated_model_path}...")
    word_vectors.save_word2vec_format(truncated_model_path, binary=True)
    print("Truncated model saved successfully.")


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


def predictor(input_words: list[str], word_vectors) -> list[list[str]]:
    top_answers: list[tuple[tuple[str], float]] = get_top_n_sets(input_words, word_vectors, top_n_sets=4, group_size=4)
    return [list(predicted_words) for predicted_words, _ in top_answers]


def build_aggregate_connections_answers(input_file):
    output_file = "data/connection_answers_aggregate.csv"

    df = pd.read_csv(input_file)

    # Group by 'Puzzle' and collect all 'Answer' values into a list
    transformed_df = df.groupby("Puzzle")["Answer"].apply(list).reset_index()

    # Save the transformed DataFrame to a new CSV
    transformed_df.to_csv(output_file, index=False)


def main():
    # Load the model
    model_path = './embeddings/GoogleNews-vectors-negative300-500000.bin'  # Update this path
    word_vectors = load_model(model_path)

    # Load predefined list of words
    word_file = "words.txt"
    words = load_words(word_file)

    # Find and output the top sets of 4 closest words
    # top_sets = get_top_n_sets(words, word_vectors, top_n_sets=10, group_size=4)
    # for i, (group, score) in enumerate(top_sets, 1):
    #     print(f"Set {i}: Words = {group}, Similarity Score = {score:.4f}")

    # Evaluate model
    n = 10
    connections_answers = pd.read_csv('data/connection_answers_aggregate.csv')
    predictor_func = lambda word_input: predictor(word_input, word_vectors)
    evaluator = ConnectionsEvaluator(predictor_func)
    accuracy, connections_made = evaluator.evaluate(connections_answers[:n])
    print(f"Number of connections made: {connections_made}/{4 * n}")
    print(f"Win percent: {accuracy}%")


if __name__ == "__main__":
    main()

