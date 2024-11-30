import torch
import pandas as pd
import numpy as np
import ast


class CNNConnectionsEvaluator:
    X_NAME = 'Puzzle'
    Y_NAME = 'Answer'

    def __init__(self, cnn_model, word_vectors):
        self.cnn_model = cnn_model
        self.word_vectors = word_vectors
        self.cnn_model.eval()

    def evaluate(self, test_set: pd.DataFrame) -> tuple[float, int]:
        x, y = self.__split_x_y(test_set)

        results = []
        total_connections_made = 0

        for input_words, output_words in zip(x, y):
            formatted_input = self.__csv_input_to_list(input_words)
            actual_output = self.__csv_output_to_list(output_words)

            mistakes_left, answers_missed = self.evaluate_puzzle(formatted_input, actual_output)
            num_connections_made = 4 - len(answers_missed)
            total_connections_made += num_connections_made
            is_correct = mistakes_left > 0

            results.append(is_correct)

        prediction_rate = sum(results) / len(results)
        return prediction_rate, total_connections_made

    def evaluate_puzzle(self, input_words: list[str], actual_outputs: list[list[str]]) -> tuple[int, list[list[str]]]:
        mistakes_left = 4
        answers_left = actual_outputs

        # Shuffle input words
        import random
        shuffled_words = input_words[:]
        random.shuffle(shuffled_words)
        print(f"Shuffled Words: {shuffled_words}")  # Debug shuffled order

        while mistakes_left > 0 and len(answers_left) > 0:
            predictions = self.predict_with_cnn(shuffled_words)

            for prediction in predictions:
                correct = CNNConnectionsEvaluator.__prediction_is_in_answer(prediction, answers_left)
                CNNConnectionsEvaluator.__print_game_status(prediction, correct, answers_left, mistakes_left)

                if not correct:
                    mistakes_left -= 1
                    if mistakes_left <= 0:
                        break
                else:
                    answers_left = CNNConnectionsEvaluator.__remove_sublist(answers_left, prediction)
                    shuffled_words = [word for word in shuffled_words if word not in prediction]
                    break
        return mistakes_left, answers_left

    def pad_inputs(self, input_words, word_vectors, max_length=16):
        """Pads or truncates the input words to ensure a consistent length."""
        embeddings = [
            word_vectors[word] if word in word_vectors else np.zeros(word_vectors.vector_size)
            for word in input_words
        ]
        while len(embeddings) < max_length:  # Pad with zeros
            embeddings.append(np.zeros(word_vectors.vector_size))
        return embeddings[:max_length]  # Trim if necessary

    def predict_with_cnn(self, input_words: list[str]) -> list[list[str]]:
        # Pad inputs to ensure consistent length
        embeddings = self.pad_inputs(input_words, self.word_vectors)  # Use padded inputs
        embeddings = torch.tensor(np.array(embeddings), dtype=torch.float32).unsqueeze(0)

        # Get predictions from the model
        with torch.no_grad():
            outputs = self.cnn_model(embeddings)  # Shape: [1, 16, 4]
            print(f"Outputs: {outputs}")  # Debugging outputs

        # Extract probabilities for groupings
        group_scores = outputs.squeeze(0).sum(dim=0).tolist()  # Sum probabilities for each group
        print(f"Group Probabilities: {group_scores}")  # Debugging group probabilities

        # Rank groups by their aggregate probabilities
        ranked_groups = sorted(enumerate(group_scores), key=lambda x: x[1], reverse=True)

        # Create groups based on top probabilities
        predictions = [[] for _ in range(4)]  # Initialize empty groups
        for word_idx, word in enumerate(input_words):
            word_probabilities = outputs[0, word_idx].tolist()
            # top_group = max(range(4), key=lambda g: word_probabilities[g])  # Find group with max probability
            top_group = np.argmax(word_probabilities)
            predictions[top_group].append(word)  # Add word to predicted group

        # Filter out empty groups
        filtered_predictions = [group for group in predictions if group]
        print(f"Predicted Groups: {filtered_predictions}")  # Debugging predicted groups
        return filtered_predictions

    def __split_x_y(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        return df[self.X_NAME], df[self.Y_NAME]

    @staticmethod
    def __print_game_status(prediction, correct, answers_left, mistakes_left):
        print('----------------------------------')
        print(f'Mistakes left: {mistakes_left}')
        print(f'Answers left: {answers_left}')
        print(f'Current prediction: {prediction}')
        print(f'Is prediction correct: {correct}')
        print('----------------------------------')

    @staticmethod
    def __csv_input_to_list(value: str) -> list[str]:
        return value.lower().split(', ')

    @staticmethod
    def __csv_output_to_list(value: str) -> list[list[str]]:
        return [
            [s.strip().lower() for s in group.split(", ")]
            for group in ast.literal_eval(value)
        ]

    @staticmethod
    def __normalize_list(lst: list[str]) -> tuple[str, ...]:
        return tuple(sorted(s.lower() for s in lst))

    @staticmethod
    def __prediction_is_in_answer(prediction: list[str], answers: list[list[str]]):
        normalized_pred = CNNConnectionsEvaluator.__normalize_list(prediction)
        normalized_answers_set = {CNNConnectionsEvaluator.__normalize_list(sublist) for sublist in answers}
        return normalized_pred in normalized_answers_set

    @staticmethod
    def __remove_sublist(list_of_lists: list[list[str]], sublist: list[str]) -> list[list[str]]:
        normalized_target = CNNConnectionsEvaluator.__normalize_list(sublist)
        filtered_list = [
            sublist for sublist in list_of_lists
            if CNNConnectionsEvaluator.__normalize_list(sublist) != normalized_target
        ]
        return filtered_list
