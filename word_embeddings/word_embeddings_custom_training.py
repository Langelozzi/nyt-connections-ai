import ast
import pickle
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec

def load_model(model_path):
    print("Loading model... This may take a while.")
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Model loaded successfully.")

    # Initialize Word2Vec model with a minimal dummy corpus to build vocabulary
    word2vec_model = Word2Vec(vector_size=300, window=5, min_count=1)

    # Create a dummy corpus with words from the pre-trained word vectors
    dummy_corpus = [[word] for word in word_vectors.key_to_index.keys()]
    word2vec_model.build_vocab(dummy_corpus)

    # Now, update the Word2Vec model's vocabulary with the pre-trained model
    word2vec_model.build_vocab([list(word_vectors.key_to_index.keys())], update=True)

    # Copy the vectors into the Word2Vec model
    word2vec_model.wv.vectors = word_vectors.vectors
    word2vec_model.wv.index_to_key = word_vectors.index_to_key
    word2vec_model.wv.key_to_index = word_vectors.key_to_index

    return word2vec_model


def generate_corpus(csv_file, output_file):
    # Load the CSV file
    df = pd.read_csv(csv_file)

    # Extract the 'Answer' column, parse the string lists, and tokenize
    corpus = []
    for answer in df['Answer'][157:]:
        # Use ast.literal_eval to safely convert the string to a list of lists
        list_of_groups = ast.literal_eval(answer)

        # Flatten the list and process the words
        words = []
        for group in list_of_groups:
            # Split by comma, strip spaces and convert to lowercase
            words.extend([word.strip().lower() for word in group.split(',')])

        corpus.append(words)

    # Save the corpus to a file for later use
    with open(output_file, 'wb') as f:
        pickle.dump(corpus, f)

    print(f"Corpus saved to {output_file}")


def load_corpus(file_path):
    with open(file_path, 'rb') as f:
        corpus = pickle.load(f)
    return corpus

def update_training_with_corpus(word2vec_model: Word2Vec, corpus, epochs=10):
    word2vec_model.build_vocab(corpus, update=True)
    word2vec_model.train(
        corpus,
        total_examples=len(corpus),
        epochs=epochs
    )
    return word2vec_model

def save_model_bin(model: Word2Vec, save_path):
    # Save the trained Word2Vec model in binary format
    model.wv.save_word2vec_format(save_path, binary=True)
    print(f"Model saved to {save_path}")

def main():
    csv = '../data/connection_answers_aggregate.csv'
    corpus_file = './connections_corpus.pkl'  # Path to saved corpus
    # generate_corpus(csv, corpus_file)
    corpus = load_corpus(corpus_file)  # Load custom corpus

    pre_trained_model_path = '../embeddings/GoogleNews-vectors-negative300-1000000.bin'  # Path to pre-trained model
    model = load_model(pre_trained_model_path)  # Load pre-trained Word2Vec model

    # Update the model with the custom corpus
    epochs = 1
    custom_trained_model = update_training_with_corpus(model, corpus, epochs=epochs)

    # Save the custom-trained model
    custom_model_path = f'../embeddings/custom-connections-word2vec-model-{epochs}-epoch.bin'
    save_model_bin(custom_trained_model, custom_model_path)

if __name__ == '__main__':
    main()
