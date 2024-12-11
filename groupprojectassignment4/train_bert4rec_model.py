# Install necessary packages
!pip uninstall -y tensorflow
!pip install tensorflow-cpu
!pip install torch transformers

# Imports
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertConfig, BertModel
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
SEQ_LEN = 50  # Sequence length for padding
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

# Filter out rows where 'user_answer' is NaN
df = df.dropna(subset=['user_answer']).reset_index(drop=True)

# Tokenizer
print("Initializing tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize 'user_answer'
def tokenize_column(df, column, tokenizer, seq_len):
    tokenized = []
    for value in df[column]:
        tokens = tokenizer.encode(
            str(value), add_special_tokens=True, max_length=seq_len, truncation=True
        )
        tokenized.append(tokens)
    return tokenized

print("Tokenizing user_answer...")
df["tokenized_answer"] = tokenize_column(df, "user_answer", tokenizer, SEQ_LEN)

# Pad sequences
def pad_sequences(sequences, seq_len):
    return [
        seq + [0] * (seq_len - len(seq)) if len(seq) < seq_len else seq[:seq_len]
        for seq in sequences
    ]

print("Padding tokenized sequences...")
padded_sequences = pad_sequences(df["tokenized_answer"].tolist(), SEQ_LEN)
padded_sequences = torch.tensor(padded_sequences)

# Convert labels to tensors
labels = torch.tensor(df["got_question_correct"].values)

# PyTorch Dataset
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

dataset = SequenceDataset(padded_sequences, labels)

# Split into train and test datasets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Define the BERT model
class BERT4RECModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = torch.nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.last_hidden_state[:, -1, :])
        return torch.sigmoid(logits).squeeze()
    
# Training
print("Training the model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERT4RECModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.BCELoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        input_ids, labels = batch
        input_ids, labels = input_ids.to(device), labels.to(device)
        attention_mask = (input_ids != 0).to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
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
        input_ids, labels = batch
        input_ids, labels = input_ids.to(device), labels.to(device)
        attention_mask = (input_ids != 0).to(device)

        outputs = model(input_ids, attention_mask)
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
model_save_file = os.path.join(model_save_path, 'model.pth')
torch.save(model.state_dict(), model_save_file)
print(f"Model saved to {model_save_file}")