import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
import numpy as np
import ast


# CNN Class
class ConnectionsCNN(nn.Module):
    def __init__(self, embedding_dim=300, num_groups=4):
        super(ConnectionsCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=2, stride=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(128 * 6, 128)  # Adjust based on input size
        self.fc2 = nn.Linear(128, num_groups * 16)  # Output: 16 × 4

    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # Debug input shape
        x = x.permute(0, 2, 1)  # Transpose for Conv1D
        # print(f"After permute: {x.shape}")  # Debug shape after transpose
        x = F.relu(self.conv1(x))
        # print(f"After conv1: {x.shape}")  # Debug shape after conv1
        x = F.relu(self.conv2(x))
        # print(f"After conv2: {x.shape}")  # Debug shape after conv2
        x = self.pool(x)
        # print(f"After pooling: {x.shape}")  # Debug shape after pooling
        x = x.view(x.size(0), -1)
        # print(f"After flatten: {x.shape}")  # Debug flattened shape
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x.view(-1, 16, 4)


# Custom Dataset Class
class ConnectionsDataset(Dataset):
    def __init__(self, data, word_vectors):
        self.data = data
        self.word_vectors = word_vectors

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        words = row['Puzzle'].lower().split(', ')
        groups = ast.literal_eval(row['Answer'].lower())

        valid_embeddings = [self.word_vectors[word] for word in words if word in self.word_vectors]
        mean_embedding = np.mean(valid_embeddings, axis=0) if valid_embeddings else np.zeros(
            self.word_vectors.vector_size
        )
        embeddings = [
            self.word_vectors[word] if word in self.word_vectors else mean_embedding for word in words
        ]
        embeddings = torch.tensor(embeddings, dtype=torch.float32)

        labels = np.zeros((16, 4), dtype=np.float32)
        for group_idx, group in enumerate(groups):
            for word in group.split(', '):
                if word in words:
                    word_idx = words.index(word)
                    labels[word_idx][group_idx] = 1

        return embeddings, torch.tensor(labels, dtype=torch.float32)


# Training Function
def train_cnn(train_data, model_path, save_model_path, num_epochs=20, batch_size=16, learning_rate=0.001):
    word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=True)
    dataset = ConnectionsDataset(train_data, word_vectors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = ConnectionsCNN()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.state_dict(), save_model_path)
    print(f"Model saved to {save_model_path}")
    return model


# Main Function
if __name__ == "__main__":
    # Paths to resources
    data_path = "../data/connection_answers_aggregate.csv"
    model_path = '../GoogleNews-vectors-negative300.bin'
    save_model_path = "cnn_model.pth"

    # Train the model on all data
    # train_cnn(data_path, model_path, save_model_path, num_epochs=20, batch_size=16, learning_rate=0.001)

    # Train the data on first 400 rows
    data = pd.read_csv(data_path)
    train_data = data[:400]
    train_cnn(train_data, model_path, save_model_path, num_epochs=20, batch_size=16, learning_rate=0.001)