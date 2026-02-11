import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import time

print("===================================")
print("Starting Random Forest Training...")

start_time = time.time()

# Load data
X_train = pd.read_parquet("../data/split/X_train.parquet")
y_train = pd.read_parquet("../data/split/y_train.parquet").values.ravel()

print(f"Training data shape: {X_train.shape}")
print(f"Positive rate (train): {y_train.mean():.6f}")

# Initialize model
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

print("Model parameters:")
print(model)

# Train
model.fit(X_train, y_train)

print("Training completed.")

# Save model
os.makedirs("../models", exist_ok=True)
model_path = "../models/rf_model.pkl"
joblib.dump(model, model_path)

end_time = time.time()

print("===================================")
print(f"Model saved to: {model_path}")
print(f"Training time: {end_time - start_time:.2f} seconds")
print("Ready to run: train_xgb.py or evaluate.py")
print("===================================")
