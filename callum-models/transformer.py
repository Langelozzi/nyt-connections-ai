import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Lambda
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import BertTokenizer, TFBertModel

# Load the CSV file
data = pd.read_csv('Connection_Answers.csv')

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = TFBertModel.from_pretrained("bert-base-uncased")

# Function to tokenize input
def tokenize_puzzle(puzzle):
    encoded = tokenizer(
        puzzle,
        padding="max_length",
        truncation=True,
        max_length=16,  # Ensure a fixed size for all inputs
        return_tensors="np",  # Convert directly to NumPy arrays
    )
    return encoded["input_ids"], encoded["attention_mask"]

# Preprocess the dataset
X_input_ids = []
X_attention_masks = []
y = []

for _, row in data.iterrows():
    # Process the puzzle and answer
    puzzle = row['Puzzle'].replace(" ", "").split(";")  # Remove spaces, split words
    puzzle_text = " ".join(puzzle)  # Combine words into a single string
    answer = row['Answer'].replace(" ", "").split(";")

    # Tokenize the puzzle
    input_ids, attention_mask = tokenize_puzzle(puzzle_text)
    X_input_ids.append(input_ids[0])  # Use the first element from the batch
    X_attention_masks.append(attention_mask[0])

    # Create a binary vector for the answer
    mlb = MultiLabelBinarizer(classes=puzzle)  # Binarize based on puzzle words
    y.append(mlb.fit_transform([answer])[0])

# Convert data to NumPy arrays
X_input_ids = np.array(X_input_ids)
X_attention_masks = np.array(X_attention_masks)
y = np.array(y)

# Train-test split
X_train_ids, X_test_ids, X_train_mask, X_test_mask, y_train, y_test = train_test_split(
    X_input_ids, X_attention_masks, y, test_size=0.2, random_state=42
)

# Define the input layers
input_ids = Input(shape=(16,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(16,), dtype=tf.int32, name="attention_mask")

# Wrap BERT model in a Lambda layer
bert_output = Lambda(
    lambda x: bert_model(input_ids=x[0], attention_mask=x[1]).last_hidden_state,
    output_shape=(16, 768),  # Define output shape explicitly
)([input_ids, attention_mask])

# Add classification layers
x = Flatten()(bert_output)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
output = Dense(16, activation="sigmoid")(x)  # Output probabilities for each word

# Build the model
model = Model(inputs=[input_ids, attention_mask], outputs=output)
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Summarize the model
model.summary()

# Train the model
history = model.fit(
    {"input_ids": X_train_ids, "attention_mask": X_train_mask},
    y_train,
    validation_data=({"input_ids": X_test_ids, "attention_mask": X_test_mask}, y_test),
    epochs=15,
    batch_size=16,
)

# Prediction function
def predict_answer(puzzle):
    puzzle = puzzle.replace(" ", "").split(",")  # Process input puzzle
    puzzle_text = " ".join(puzzle)
    input_ids, attention_mask = tokenize_puzzle(puzzle_text)

    # Predict probabilities
    predicted_probs = model.predict({"input_ids": input_ids, "attention_mask": attention_mask})
    predicted_indices = np.argsort(predicted_probs[0])[-4:][::-1]  # Top 4 words
    predicted_words = [puzzle[i] for i in predicted_indices]
    return predicted_words

# Example usage
new_puzzle = "discount,bonus,animal,eloise,forget,equity,club,plaza,ignore,promotion,goldfish,pug,overlook,raise,ritz,turtle"
predicted_answer = predict_answer(new_puzzle)
print(f"Predicted answer: {predicted_answer}")
