# -*- coding: utf-8 -*-
"""Neural Network.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1y01DdJHOEUVt4RJbAzgcHvNwaZ1TB_3g
"""

import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from tqdm import tqdm

from google.colab import drive
drive.mount('/content/drive')

print("Loading dataset with progress bar...")
file_path = "/content/drive/MyDrive/Vincent_Monitha_541_Dataset_Folder/kt1_sampled_25_percent.csv"
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

print(df["elapsed_time"])

# Specify features and target
features = ["user_answer", "elapsed_time"]
target = "got_question_correct"

# Split into features and target
X = df[features]
y = df[target]

# One-hot encode categorical features and scale numerical features
categorical_features = [ "user_answer"]
numerical_features = [ "elapsed_time"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

print(df["user_answer"])

processed_data = preprocessor.fit_transform(df[features])

print(df["user_answer"])

import tensorflow as tf
X_processed = preprocessor.fit_transform(df[features])
y = df[target].astype("int32")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)


X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)

# Train the model on TPU
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
num_features = X_train.shape[1]

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # Detect TPUs
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("TPU is available.")
except ValueError:
    strategy = tf.distribute.get_strategy()  # Use default strategy if no TPU is found
    print("No TPU found, using default strategy.")
with strategy.scope():
  # Define the neural network model
    model = keras.Sequential([
        layers.InputLayer(input_shape=(num_features,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

# Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=5, batch_size=128)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# Calculate metrics
results = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1 Score": f1_score(y_test, y_pred),
    "Log Loss": log_loss(y_test, y_pred_proba),
    "AUROC": roc_auc_score(y_test, y_pred_proba),
}

# Print results
print("\nModel Evaluation Metrics:")
for metric, value in results.items():
    print(f"{metric}: {value:.4f}")

# Save the trained model to a file
model_filename = "logistic_regression_model.joblib"
joblib.dump(logistic_regression, model_filename)
print(f"Model saved to {model_filename}.")