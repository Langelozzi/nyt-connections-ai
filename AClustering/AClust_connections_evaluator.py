import pandas as pd
import ast
import random
from typing import Callable

class ConnectionsEvaluator:
    X_NAME = 'Puzzle'
    Y_NAME = 'Answer'

    def __init__(self, predictor_func: Callable[[list[str]], list[list[str]]]):
        self._predictor_func = predictor_func
        self.game_count = 0

    def evaluate(self, test_set: pd.DataFrame) -> tuple[float, int]:
        # Currently only evaluating if the top answer is correct
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

            self.game_count += 1
            if self.game_count % 10 == 0:
                print(f"Games completed: {self.game_count}")

        prediction_rate = sum(results) / len(results)
        return prediction_rate, total_connections_made

    def evaluate_puzzle(self, input_words: list[str], actual_outputs: list[list[str]]) -> tuple[int, list[list[str]]]:
        mistakes_left = 4
        randomized_inputs = input_words.copy()
        random.shuffle(randomized_inputs)  # Shuffle the input words
        inputs_left = randomized_inputs
        # inputs_left = input_words
        answers_left = actual_outputs
        while mistakes_left > 0 and len(answers_left) > 0:
            predictions = self._predictor_func(inputs_left)
            print(inputs_left)
            for prediction in predictions:
                correct = ConnectionsEvaluator.__prediction_is_in_answer(prediction, answers_left)
                ConnectionsEvaluator.__print_game_status(prediction, correct, answers_left, mistakes_left)
                if not correct:
                    mistakes_left -= 1
                    continue
                else:
                    answers_left = ConnectionsEvaluator.__remove_sublist(answers_left, prediction)
                    inputs_left = [word for word in inputs_left if word not in prediction]
                    break
        return int(mistakes_left), answers_left


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
        # Normalize list1
        normalized_pred = ConnectionsEvaluator.__normalize_list(prediction)

        # Normalize each sublist in list2 and store in a set for efficient lookup
        normalized_answers_set = {ConnectionsEvaluator.__normalize_list(sublist) for sublist in answers}

        return normalized_pred in normalized_answers_set

    @staticmethod
    def __remove_sublist(list_of_lists: list[list[str]], sublist: list[str]) -> list[list[str]]:
        # Normalize the target list
        normalized_target = ConnectionsEvaluator.__normalize_list(sublist)

        # Create a new list excluding the matching sublist
        filtered_list = [
            sublist for sublist in list_of_lists
            if ConnectionsEvaluator.__normalize_list(sublist) != normalized_target
        ]

        return filtered_list