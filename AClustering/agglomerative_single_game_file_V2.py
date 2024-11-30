from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
import numpy as np


def normalize_embeddings(embeddings):
    """Normalize embeddings to unit vectors for cosine similarity."""
    return normalize(embeddings)


def perform_clustering(embeddings, num_initial_clusters):
    """Perform Agglomerative Clustering with more initial clusters."""
    normalized_embeddings = normalize_embeddings(embeddings)
    clustering = AgglomerativeClustering(
        n_clusters=num_initial_clusters, metric="cosine", linkage="complete"
    )
    return clustering.fit_predict(normalized_embeddings)


def compute_average_similarity(group, embeddings):
    """Compute the average cosine similarity for a group of words."""
    if len(group) <= 1:  # Avoid empty or single-member groups
        return 0
    group_embeddings = np.array([embeddings[word] for word in group])
    similarity_matrix = cosine_similarity(group_embeddings)
    avg_similarity = np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)])
    return avg_similarity


def refine_groups(words, initial_cluster_labels, embeddings, num_final_clusters):
    """Refine groups to maximize intra-cluster similarity."""
    # Map words to their initial clusters
    initial_groups = {}
    for word, label in zip(words, initial_cluster_labels):
        initial_groups.setdefault(label, []).append(word)

    # Flatten into a single pool of words
    all_words = [word for group in initial_groups.values() for word in group]

    # Start with empty final groups
    final_groups = [[] for _ in range(num_final_clusters)]

    # Iteratively assign words to groups to maximize average similarity
    for word in all_words:
        best_group_idx = -1
        best_similarity = -1
        for idx in range(num_final_clusters):
            # Temporarily add the word to a group
            temp_group = final_groups[idx] + [word]
            avg_similarity = compute_average_similarity(temp_group, embeddings)
            if avg_similarity > best_similarity:
                best_group_idx = idx
                best_similarity = avg_similarity
        # Assign the word to the best group
        if best_group_idx != -1:
            final_groups[best_group_idx].append(word)

    # Ensure all groups have at least one word (fallback mechanism)
    unassigned_words = [word for word in all_words if word not in sum(final_groups, [])]
    for idx, group in enumerate(final_groups):
        if not group and unassigned_words:
            final_groups[idx].append(unassigned_words.pop(0))

    # Distribute remaining unassigned words evenly across groups
    while unassigned_words:
        for idx in range(num_final_clusters):
            if not unassigned_words:
                break
            final_groups[idx].append(unassigned_words.pop(0))

    # Balance groups to ensure all have the correct number of words
    total_words = len(all_words)
    target_size = total_words // num_final_clusters
    for idx in range(num_final_clusters):
        while len(final_groups[idx]) > target_size:
            for other_idx in range(num_final_clusters):
                if len(final_groups[other_idx]) < target_size:
                    final_groups[other_idx].append(final_groups[idx].pop())
                    break

    return final_groups




def main():
    # Words and embeddings
    words = [
        "SHOW", "HANDLE", "BELIEVE", "MOUNT", "BLUFF", "SHAM",
        "ACCEPT", "TRUST", "FRONT", "BRACKET", "BUY", "STAND",
        "FIFTH", "PINT", "BASE", "LITER"
    ]

    model_path = "../GoogleNews-vectors-negative300.bin"  # Update with your file's actual path
    num_initial_clusters = 8
    num_final_clusters = 4

    print("Loading word embeddings...")
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Word embeddings loaded successfully.")

    # Filter valid words and embeddings
    valid_words = [word for word in words if word in word_vectors]
    embeddings = {word: word_vectors[word] for word in valid_words}

    # Perform initial clustering
    print("Performing initial clustering...")
    initial_cluster_labels = perform_clustering(
        np.array(list(embeddings.values())), num_initial_clusters
    )

    # Refine groups to maximize intra-cluster similarity
    print("Refining groups...")
    final_groups = refine_groups(valid_words, initial_cluster_labels, embeddings, num_final_clusters)

    # Display the refined groups
    print("\nRefined Clusters:")
    for i, group in enumerate(final_groups, 1):
        print(f"Group {i}: {group}")


if __name__ == "__main__":
    main()
