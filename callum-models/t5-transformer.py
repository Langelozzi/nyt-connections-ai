import pandas as pd
import numpy as np
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split

# Load your CSV data
data = pd.read_csv('Connection_Answers.csv')

# Initialize the T5 tokenizer and model
tokenizer = T5Tokenizer.from_pretrained("t5-base")
t5_model = T5ForConditionalGeneration.from_pretrained("t5-base")


# Function to convert puzzle and answer into a text format
def prepare_data_for_t5(puzzle, answer):
    input_text = "puzzle: " + " ".join(puzzle)
    target_text = "answer: " + " ".join(answer)
    return input_text, target_text


# Tokenizing the input and target
def tokenize_input_output(input_text, target_text):
    input_enc = tokenizer(input_text, padding="max_length", truncation=True, max_length=16, return_tensors="pt")
    target_enc = tokenizer(target_text, padding="max_length", truncation=True, max_length=16, return_tensors="pt")
    return input_enc, target_enc


# Prepare dataset
inputs = []
targets = []

for _, row in data.iterrows():
    puzzle = row['Puzzle'].split(';')  # Assuming Puzzle is a semicolon-separated string of words
    answer = row['Answer'].split(';')  # Assuming Answer is a semicolon-separated string of words
    input_text, target_text = prepare_data_for_t5(puzzle, answer)
    input_enc, target_enc = tokenize_input_output(input_text, target_text)
    inputs.append(input_enc)
    targets.append(target_enc)

# Split dataset into training and validation sets (80% train, 20% validation)
train_inputs, val_inputs, train_targets, val_targets = train_test_split(inputs, targets, test_size=0.2)


# Create dataset for validation
class ConnectionsDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            'input_ids': self.inputs[idx]['input_ids'].squeeze(),
            'attention_mask': self.inputs[idx]['attention_mask'].squeeze(),
            'labels': self.targets[idx]['input_ids'].squeeze()  # We use input_ids of target as labels
        }


train_dataset = ConnectionsDataset(train_inputs, train_targets)
val_dataset = ConnectionsDataset(val_inputs, val_targets)

# Set up Trainer
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
)

trainer = Trainer(
    model=t5_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # Provide validation dataset
)

# Fine-tune the model
trainer.train()


# Function to generate answers for new puzzles
def predict_answer_with_t5(puzzle):
    input_text = "puzzle: " + " ".join(puzzle.split(','))
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=16)

    # Generate the answer
    outputs = t5_model.generate(inputs['input_ids'], max_length=16, num_return_sequences=1)

    # Decode the prediction
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction


# Example usage
new_puzzle = "discount,bonus,animal,eloise,forget,equity,club,plaza,ignore,promotion,goldfish,pug,overlook,raise,ritz,turtle"
predicted_answer = predict_answer_with_t5(new_puzzle)
print(f"Predicted answer: {predicted_answer}")
