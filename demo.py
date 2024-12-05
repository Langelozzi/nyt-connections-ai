from typing import Callable
import pandas as pd
import evaluator_W2V_agglo_JETSON
from gensim.models import KeyedVectors


class Demo:
    def __init__(self, predictor_func: Callable[[list[str]], list[list[str]]]):
        self._predictor_func = predictor_func

    def play_interactive(self):
        print("Enter the 16 words from the puzzle:")
        input_str = input().strip()
        input_words = [word.strip().lower() for word in input_str.split(",")]

        mistakes_left = 4
        words_left = input_words
        correct_guesses = 0

        while mistakes_left > 0 and correct_guesses < 4:
            print("\n----------------------------------")
            print(f"Words remaining: {words_left}")
            print(f"Mistakes remaining: {mistakes_left}")
            print(f"Correct guesses: {correct_guesses}")

            input("\nPress Enter when you're ready for the next guess...")

            predictions = self._predictor_func(words_left)
            current_prediction_index = 0

            while current_prediction_index < len(predictions):
                current_prediction = predictions[current_prediction_index]
                print(f"\nPrediction: {current_prediction}")

                while True:
                    response = input("Was this guess correct? (y/n): ").lower()
                    if response in ["y", "n"]:
                        break
                    print("Please enter 'y' for yes or 'n' for no.")

                if response == "y":
                    words_left = [
                        word for word in words_left if word not in current_prediction
                    ]
                    correct_guesses += 1
                    break
                else:
                    mistakes_left -= 1
                    if mistakes_left == 0:
                        print("\nGame Over! Too many incorrect guesses.")
                        return

                    if current_prediction_index < len(predictions) - 1:
                        current_prediction_index += 1
                        print("\nTrying next most likely prediction...")
                    else:
                        print("\nNo more predictions available for these words.")
                        break

        if correct_guesses == 4:
            print("\nCongratulations! You've solved the puzzle!")
        else:
            print("\nGame Over!")


def main():
    data_path = "data/connection_answers_aggregate.csv"
    connections_answers = pd.read_csv(data_path)
    connections_answers = connections_answers[400:420]

    # Path to pre-trained Google News Word2Vec embeddings
    model_path = "GoogleNews-vectors-negative300.bin"

    # Load the Word2Vec model once
    print("Loading Word2Vec model...")
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print("Model loaded successfully.")

    # Define a single metric and linkage to use
    metric = "cosine"  # Specify the metric you want to use
    linkage = "single"  # Specify the linkage you want to use

    print(f"Evaluating with metric='{metric}' and linkage='{linkage}'")

    def predictor_func(input_words):
        return evaluator_W2V_agglo_JETSON.word2vec_agglomerative_predictor(
            input_words, word_vectors, metric, linkage
        )

    evaluator = Demo(predictor_func)

    evaluator.play_interactive()


if __name__ == "__main__":
    main()
