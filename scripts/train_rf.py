import os
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib


DATA_DIR = "data/split"
MODEL_DIR = "models"
RANDOM_STATE = 42


def load_data():
    print("===================================")
    print("Starting Random Forest Training...")

    X_path = f"{DATA_DIR}/X_train.parquet"
    y_path = f"{DATA_DIR}/y_train.parquet"

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Train split not found. Please run split.py first.")

    X_train = pd.read_parquet(X_path)
    y_train = pd.read_parquet(y_path).values.ravel()

    print(f"Training data shape: {X_train.shape}")
    print(f"Positive rate (train): {y_train.mean():.6f}")

    return X_train, y_train


def build_model():
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("Model initialized with parameters:")
    print(model)

    return model


def train_model(model, X_train, y_train):
    print("Training model...")
    start_time = time.time()

    model.fit(X_train, y_train)

    end_time = time.time()
    print("Training completed.")
    print(f"Training time: {end_time - start_time:.2f} seconds")

    return model


def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)

    model_path = f"{MODEL_DIR}/rf_model.pkl"
    joblib.dump(model, model_path)

    print("===================================")
    print(f"Model saved to: {model_path}")
    print("Ready to run: train_xgb.py or evaluate.py")
    print("===================================")


def main():
    X_train, y_train = load_data()
    model = build_model()
    trained_model = train_model(model, X_train, y_train)
    save_model(trained_model)


if __name__ == "__main__":
    main()
