import joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from tqdm import tqdm

# Load the dataset with a progress bar
print("Loading dataset with progress bar...")
file_path = "./data/kt1_combined_with_correctness.csv"
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

# Specify features and target
features = ["solving_id", "question_id", "user_answer", "elapsed_time"]
target = "got_question_correct"

# Split into features and target
X = df[features]
y = df[target]

# One-hot encode categorical features and scale numerical features
categorical_features = ["question_id", "user_answer"]
numerical_features = ["solving_id", "elapsed_time"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

# Define the Logistic Regression pipeline
logistic_regression = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train the model with tqdm progress bar
print("Training Logistic Regression model...")
with tqdm(
    total=logistic_regression.named_steps["classifier"].max_iter,
    desc="Training Progress",
    unit="iteration",
) as pbar:
    logistic_regression.fit(X_train, y_train)
    pbar.update(logistic_regression.named_steps["classifier"].max_iter)

# Evaluate the model
y_pred = logistic_regression.predict(X_test)
y_pred_proba = logistic_regression.predict_proba(X_test)[:, 1]

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