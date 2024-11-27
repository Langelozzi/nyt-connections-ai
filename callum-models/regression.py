import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import string

# Load Word2Vec model
model_path = 'GoogleNews-vectors-negative300.bin'  # Update this path
word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Load your CSV data
data = pd.read_csv('Connection_Answers.csv')


# Function to get the word embeddings for a list of words
def get_word_embeddings(words):
    embeddings = []
    for word in words:
        # Handle missing words (those not in the Word2Vec model)
        try:
            embeddings.append(word_vectors[word])
        except KeyError:
            embeddings.append(np.zeros(300))  # If word is not found, use zero vector
    return np.mean(embeddings, axis=0)  # Average the embeddings

# Prepare the dataset
X = []
y = []

for index, row in data.iterrows():
    puzzle = row['Puzzle'].split(';')  # Assuming Puzzle is a comma-separated string of words
    answer = row['Answer'].split(';')  # Assuming Answer is a comma-separated string of answer words

    puzzle = [item.replace(" ", "") for item in puzzle]
    answer = [item.replace(" ", "") for item in answer]

    # Get the embeddings for the puzzle
    puzzle_embedding = get_word_embeddings(puzzle)

    # Get the indices of the answer words in the puzzle
    answer_indices = [puzzle.index(word) for word in answer]

    # Ensure that answer_indices always has 4 values (pad with -1 if fewer)
    while len(answer_indices) < 4:
        answer_indices.append(-1)  # Use -1 to indicate missing values

    X.append(puzzle_embedding) 
    y.append(answer_indices)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Since the output is continuous, we need to round to the nearest integer and map back to indices
y_pred_rounded = np.round(y_pred).astype(int)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred_rounded)
print(f'Mean Squared Error: {mse}')


# Example: predict the answer for a new puzzle
def predict_answer(puzzle):
    puzzle_embedding = get_word_embeddings(puzzle.split(','))
    predicted_indices = model.predict([puzzle_embedding])
    predicted_indices = np.round(predicted_indices).astype(int)

    # Map indices back to the words in the puzzle
    print(predicted_indices)
    answer = [puzzle.split(",")[i] for i in predicted_indices[0]]
    return answer


# Example usage
new_puzzle = "discount,bonus,animal,eloise,forget,equity,club,plaza,ignore,promotion,goldfish,pug,overlook,raise,ritz,turtle"
puzzled = new_puzzle.split(',')

predicted_answer = predict_answer(new_puzzle)
print(f'Predicted answer: {predicted_answer}')
