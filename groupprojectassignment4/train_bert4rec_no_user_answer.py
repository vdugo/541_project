# Install necessary packages
!pip uninstall -y tensorflow
!pip install tensorflow-cpu
!pip install torch transformers

# Imports
import pandas as pd
import numpy as np
from transformers import BertModel
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Path to save the model in Google Drive
model_save_path = '/content/drive/My Drive/bert_model'
os.makedirs(model_save_path, exist_ok=True)

# Define constants
SEQ_LEN = 1  # Only 1 feature now (elapsed_time)
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4

# Load the dataset
print("Loading dataset with progress bar...")
file_path = "/content/drive/MyDrive/Vincent_Monitha_541_Dataset_Folder/kt1_sampled_25_percent_no_user_answer.csv"
chunk_size = 10000  # Adjust based on memory capacity

# Get the total number of rows in the dataset
total_rows = sum(1 for _ in open(file_path)) - 1  # Subtract 1 for the header row

chunks = []
print("Loading dataset with progress bar...")
with tqdm(total=total_rows, desc="Loading Dataset", unit="rows") as pbar:
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunks.append(chunk)
        pbar.update(len(chunk))
        
# Combine chunks into a single DataFrame
df = pd.concat(chunks, ignore_index=True)
print("Dataset loaded successfully.")

# Preprocess features and labels
sequences = torch.tensor(df["elapsed_time"].values, dtype=torch.float32).unsqueeze(1)  # Add dimension for single feature
labels = torch.tensor(df["got_question_correct"].values, dtype=torch.float32)

# PyTorch Dataset
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

dataset = SequenceDataset(sequences, labels)

# Split into train and test datasets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Define a simple model (no tokenization, single numeric input)
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(SEQ_LEN, 128)
        self.fc2 = torch.nn.Linear(128, 1)
        self.activation = torch.nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return self.activation(x).squeeze()

# Training
print("Training the model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.BCELoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

# Evaluation
print("Evaluating the model...")
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="Evaluating"):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        all_preds.extend(outputs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Metrics
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
accuracy = accuracy_score(all_labels, all_preds > 0.5)
auc = roc_auc_score(all_labels, all_preds)
logloss = log_loss(all_labels, all_preds)

print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print(f"Log Loss: {logloss:.4f}")

# Save the model
model_save_file = os.path.join(model_save_path, 'simple_model.pth')
torch.save(model.state_dict(), model_save_file)
print(f"Model saved to {model_save_file}")
