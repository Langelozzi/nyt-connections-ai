import numpy as np
from gensim.models import KeyedVectors
from sklearn.cluster import AgglomerativeClustering


def normalize_embeddings(embeddings):
    """Normalize embeddings to unit vectors for cosine similarity."""
    from sklearn.preprocessing import normalize
    return normalize(embeddings)


def calculate_average_embedding(word_vectors):
    """Calculate the average embedding for all words in the Word2Vec model."""
    all_embeddings = word_vectors.vectors
    return np.mean(all_embeddings, axis=0)


def compute_phrase_embedding(phrase, word_vectors, fallback_embedding):
    """Compute an embedding for a multi-term phrase by averaging its component word embeddings."""
    words = phrase.split()
    embeddings = []

    for word in words:
        if word in word_vectors:
            embeddings.append(word_vectors[word])
        else:
            embeddings.append(fallback_embedding)

    return np.mean(embeddings, axis=0) if embeddings else fallback_embedding


def perform_clustering(embeddings, num_clusters, metric, linkage):
    """Perform Agglomerative Clustering."""
    normalized_embeddings = normalize_embeddings(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=num_clusters, metric=metric, linkage=linkage
    )
    return clustering.fit_predict(normalized_embeddings)


def cluster_words(words, embeddings, num_clusters, metric, linkage, group_size=4):
    """Cluster words into groups of exactly `group_size` words."""
    cluster_labels = perform_clustering(embeddings, num_clusters, metric, linkage)

    # Group words by cluster labels
    clusters = {}
    for word, label in zip(words, cluster_labels):
        clusters.setdefault(label, []).append(word)

    # Flatten and redistribute into equal-sized groups
    all_words = [word for group in clusters.values() for word in group]
    groups = []
    for i in range(0, len(all_words), group_size):
        group = all_words[i:i + group_size]
        if len(group) == group_size:
            groups.append(group)

    return groups


def word2vec_agglomerative_predictor(input_words, word_vectors, metric, linkage):
    """Predict groups of 4 words using Word2Vec Agglomerative Clustering."""
    average_embedding = calculate_average_embedding(word_vectors)

    embeddings = []
    valid_words = []
    for word in input_words:
        if word in word_vectors:
            embeddings.append(word_vectors[word])
        elif "-" in word:
            hyphen_removed = word.replace("-", "")
            embeddings.append(word_vectors[hyphen_removed] if hyphen_removed in word_vectors else average_embedding)
        elif " " in word:
            embeddings.append(compute_phrase_embedding(word, word_vectors, average_embedding))
        else:
            embeddings.append(average_embedding)
        valid_words.append(word)

    embeddings = np.array(embeddings)
    return cluster_words(valid_words, embeddings, num_clusters=4, metric=metric, linkage=linkage)


def evaluate_single_game(input_words, expected_groups, word_vectors, metric="cosine", linkage="single"):
    """Evaluate predictions for a single game."""
    print("Running predictions for a single game...")

    predictions = word2vec_agglomerative_predictor(input_words, word_vectors, metric, linkage)
    correct_count = sum(1 for group in predictions if group in expected_groups)
    incorrect_groups = [group for group in predictions if group not in expected_groups]

    print("\nPredictions:")
    for i, group in enumerate(predictions, start=1):
        print(f"Group {i}: {group}")

    print("\nEvaluation:")
    print(f"Correct groups: {correct_count}/{len(expected_groups)}")
    print(f"Incorrect groups: {len(incorrect_groups)}")
    print(f"Accuracy: {correct_count / len(expected_groups) * 100:.2f}%")


def main():
    # input_words = [
    #     "SHOW", "HANDLE", "BELIEVE", "MOUNT", "BLUFF", "SHAM",
    #     "ACCEPT", "TRUST", "FRONT", "BRACKET", "BUY", "STAND",
    #     "FIFTH", "PINT", "BASE", "LITER"
    # ]

    input_words = ['FIGURE', 'CHARACTER', 'AVATAR',
                   'LOOK', 'GIANT', 'PERSONA', 'APPEAR', 'MONSTER',
                   'LISTEN', 'WITCH', 'TITANIC', 'MAMMOTH', 'SEEM', 'CASTLE',
                   'HUSTLE', 'SOUND']

    # input_words = ['SOUND', 'LOOK', 'APPEAR', 'SEEM', 'AVATAR',
    #                'FIGURE', 'PERSONA', 'CHARACTER', 'TITANIC',
    #                'GIANT', 'MONSTER', 'MAMMOTH', 'LISTEN', 'WITCH',
    #                'HUSTLE', 'CASTLE']


    """
    Answers left: [['sound', 'look', 'appear', 'seem'], ['avatar', 'figure', 'persona', 'character'], ['titanic', 'giant', 'monster', 'mammoth'], ['listen', 'witch', 'hustle', 'castle']]
    Current prediction: ['sound', 'look', 'appear', 'seem']
    Is prediction correct: True
    """

    # expected_groups = [
    #     ["SHOW", "HANDLE", "FRONT", "BRACKET"],
    #     ["BELIEVE", "TRUST", "ACCEPT", "STAND"],
    #     ["BLUFF", "SHAM", "BUY", "MOUNT"],
    #     ["FIFTH", "PINT", "BASE", "LITER"]
    # ]

    expected_groups = [
        ['SOUND', 'LOOK', 'APPEAR', 'SEEM'],
        ['AVATAR', 'FIGURE', 'PERSONA', 'CHARACTER'],
        ['TITANIC', 'GIANT', 'MONSTER', 'MAMMOTH'],
        ['LISTEN', 'WITCH', 'HUSTLE', 'CASTLE']
    ]

    model_path = "../GoogleNews-vectors-negative300.bin"
    print("Loading Word2Vec model...")
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Word2Vec model loaded successfully.")

    evaluate_single_game(input_words, expected_groups, word_vectors)


if __name__ == "__main__":
    main()
