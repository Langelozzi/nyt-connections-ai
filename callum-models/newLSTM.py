import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Embedding, Bidirectional, TimeDistributed
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

# Load Word2Vec model
model_path = 'GoogleNews-vectors-negative300.bin'  # Update this path
word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)

# Load your CSV data
data = pd.read_csv('Connection_Answers.csv')


# Function to get the word embeddings for a list of words
def get_word_embeddings(words):
    embeddings = []
    for word in words:
        try:
            embeddings.append(word_vectors[word])
        except KeyError:
            embeddings.append(np.zeros(300))  # Use zero vector for unknown words
    return np.array(embeddings)


# Prepare the dataset
X = []
y = []

for _, row in data.iterrows():
    puzzle = row['Puzzle'].split(';')  # Assuming semicolon-separated
    answer = row['Answer'].split(';')  # Assuming semicolon-separated

    puzzle = [word.strip().lower() for word in puzzle]
    answer = [word.strip().lower() for word in answer]

    # Get the embeddings for the puzzle
    puzzle_embedding = get_word_embeddings(puzzle)

    # Create target: binary labels indicating which words are part of the answer
    answer_labels = np.zeros(len(puzzle))
    for word in answer:
        if word in puzzle:
            answer_labels[puzzle.index(word)] = 1  # Mark as 1 if part of the answer

    # Pad or truncate to 16 words
    padded_embeddings = np.pad(puzzle_embedding, ((0, 16 - len(puzzle)), (0, 0)), 'constant')[:16]
    padded_labels = np.pad(answer_labels, (0, 16 - len(answer_labels)), 'constant')[:16]

    X.append(padded_embeddings)
    y.append(padded_labels)

# Convert to numpy arrays
X = np.array(X)  # Shape: (samples, 16, 300)
y = np.array(y)  # Shape: (samples, 16)

# Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Bidirectional(LSTM(128, return_sequences=True), input_shape=(16, 300)),
    Dropout(0.3),
    TimeDistributed(Dense(64, activation='relu')),
    TimeDistributed(Dense(1, activation='sigmoid'))  # Predict binary labels for each word
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Summarize the model
model.summary()

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=32)


# Prediction function
def predict_answer(puzzle):
    puzzle = [word.strip().lower() for word in puzzle.split(',')]
    puzzle_embedding = get_word_embeddings(puzzle)
    padded_embeddings = np.pad(puzzle_embedding, ((0, 16 - len(puzzle)), (0, 0)), 'constant')[:16]
    padded_embeddings = np.expand_dims(padded_embeddings, axis=0)  # Add batch dimension

    predictions = model.predict(padded_embeddings)[0].flatten()  # Shape: (16,)
    predicted_indices = np.argsort(predictions)[-4:][::-1]  # Get top 4 words

    answer = [puzzle[i] for i in predicted_indices if i < len(puzzle)]
    return answer


# Example usage
new_puzzle = "discount,animal,eloise,forget,club,plaza,ignore,goldfish,pug,overlook,ritz,turtle"
predicted_answer = predict_answer(new_puzzle)
print(f'Predicted answer: {predicted_answer}')
