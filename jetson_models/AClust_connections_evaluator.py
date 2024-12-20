import pandas as pd
import ast
from typing import Callable, List, Tuple

class ConnectionsEvaluator:
    X_NAME = 'Puzzle'
    Y_NAME = 'Answer'

    def __init__(self, predictor_func: Callable[[List[str]], List[List[str]]]):
        self._predictor_func = predictor_func

    def evaluate(self, test_set: pd.DataFrame) -> Tuple[float, int]:
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

        prediction_rate = sum(results) / len(results)
        return prediction_rate, total_connections_made

    def evaluate_puzzle(self, input_words: List[str], actual_outputs: List[List[str]]) -> Tuple[int, List[List[str]]]:
        mistakes_left = 4
        inputs_left = input_words
        answers_left = actual_outputs
        while mistakes_left > 0 and len(answers_left) > 0:
            predictions = self._predictor_func(inputs_left)
            # print(inputs_left)
            for prediction in predictions:
                correct = ConnectionsEvaluator.__prediction_is_in_answer(prediction, answers_left)
                # ConnectionsEvaluator.__print_game_status(prediction, correct, answers_left, mistakes_left)
                if not correct:
                    mistakes_left -= 1
                    continue
                else:
                    answers_left = ConnectionsEvaluator.__remove_sublist(answers_left, prediction)
                    inputs_left = [word for word in inputs_left if word not in prediction]
                    break
        return int(mistakes_left), answers_left


    def __split_x_y(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
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
    def __csv_input_to_list(value: str) -> List[str]:
        return value.lower().split(', ')

    @staticmethod
    def __csv_output_to_list(value: str) -> List[List[str]]:
        return [
            [s.strip().lower() for s in group.split(", ")]
            for group in ast.literal_eval(value)
        ]

    @staticmethod
    def __normalize_list(lst: List[str]) -> Tuple[str, ...]:
        return tuple(sorted(s.lower() for s in lst))

    @staticmethod
    def __prediction_is_in_answer(prediction: List[str], answers: List[List[str]]):
        # Normalize list1
        normalized_pred = ConnectionsEvaluator.__normalize_list(prediction)

        # Normalize each sublist in list2 and store in a set for efficient lookup
        normalized_answers_set = {ConnectionsEvaluator.__normalize_list(sublist) for sublist in answers}

        return normalized_pred in normalized_answers_set

    @staticmethod
    def __remove_sublist(list_of_lists: List[List[str]], sublist: List[str]) -> List[List[str]]:
        # Normalize the target list
        normalized_target = ConnectionsEvaluator.__normalize_list(sublist)

        # Create a new list excluding the matching sublist
        filtered_list = [
            sublist for sublist in list_of_lists
            if ConnectionsEvaluator.__normalize_list(sublist) != normalized_target
        ]

        return filtered_list