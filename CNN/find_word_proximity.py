from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine

def get_embedding(input_text, word_vectors):
    """
    Compute the embedding for an input that might be a single word or a phrase.

    Args:
        input_text (str): The input text (word or phrase).
        word_vectors (KeyedVectors): Pre-trained word embeddings.

    Returns:
        numpy.ndarray: The embedding for the word or the averaged embedding for the phrase.
    """
    words = input_text.lower().split()

    # If input is a single word and in the vocabulary, return its embedding directly
    if len(words) == 1 and words[0] in word_vectors:
        return word_vectors[words[0]]

    # Otherwise, treat it as a phrase and average the embeddings of its words
    valid_embeddings = [word_vectors[word] for word in words if word in word_vectors]
    if not valid_embeddings:
        return None  # Return None if no valid words are found

    return sum(valid_embeddings) / len(valid_embeddings)


def compare_texts(text1, text2, model_path):
    """
    Compare the closeness of two inputs (words or phrases) using word embeddings.

    Args:
        text1 (str): The first input text (word or phrase).
        text2 (str): The second input text (word or phrase).
        model_path (str): Path to the pre-trained word embeddings.

    Returns:
        float: Cosine similarity between the embeddings of the inputs.
    """
    # Load pre-trained word embeddings
    print("Loading word embeddings...")
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Word embeddings loaded successfully.")

    # Compute embeddings for each input
    embedding1 = get_embedding(text1, word_vectors)
    embedding2 = get_embedding(text2, word_vectors)

    if embedding1 is None or embedding2 is None:
        print(f"One or both inputs have no valid words in the vocabulary: '{text1}', '{text2}'")
        return None

    # Compute cosine similarity
    similarity = 1 - cosine(embedding1, embedding2)

    return similarity


# Example usage
if __name__ == "__main__":
    # Path to pre-trained word embeddings
    model_path = '../GoogleNews-vectors-negative300.bin'  # Update with your actual path

    # Inputs to compare
    input1 = "house"
    input2 = "electronic"

    # Compare the inputs
    similarity = compare_texts(input1, input2, model_path)

    if similarity is not None:
        print(f"Cosine similarity between '{input1}' and '{input2}': {similarity:.4f}")
    else:
        print("Could not compute similarity.")
