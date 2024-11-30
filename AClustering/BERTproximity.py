from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np


def get_bert_embedding(input_text, model):
    """
    Compute the BERT embedding for an input that might be a single word or a phrase.

    Args:
        input_text (str): The input text (word or phrase).
        model (SentenceTransformer): Pre-trained BERT model.

    Returns:
        numpy.ndarray: The embedding for the word or phrase.
    """
    # Use the BERT model to encode the input text
    embedding = model.encode(input_text)
    return embedding


def compare_texts_with_bert(text1, text2, bert_model_name="all-MiniLM-L6-v2"):
    """
    Compare the closeness of two inputs (words or phrases) using BERT embeddings.

    Args:
        text1 (str): The first input text (word or phrase).
        text2 (str): The second input text (word or phrase).
        bert_model_name (str): Name of the pre-trained BERT model to load.

    Returns:
        float: Cosine similarity between the embeddings of the inputs.
    """
    # Load the pre-trained BERT model
    print("Loading BERT model...")
    model = SentenceTransformer(bert_model_name)
    print("BERT model loaded successfully.")

    # Compute embeddings for each input
    embedding1 = get_bert_embedding(text1, model)
    embedding2 = get_bert_embedding(text2, model)

    # Compute cosine similarity
    similarity = 1 - cosine(embedding1, embedding2)

    return similarity


# Example usage
if __name__ == "__main__":
    # Inputs to compare
    input1 = "house"
    input2 = "electronic"

    # Compare the inputs using BERT
    similarity = compare_texts_with_bert(input1, input2)

    print(f"Cosine similarity between '{input1}' and '{input2}': {similarity:.4f}")
