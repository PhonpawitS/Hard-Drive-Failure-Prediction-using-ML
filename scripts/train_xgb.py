import os
import time
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib


DATA_DIR = "data/split"
MODEL_DIR = "models"
RANDOM_STATE = 42


def load_data():
    print("===================================")
    print("Starting XGBoost Training with Random Search...")

    X_path = f"{DATA_DIR}/X_train.parquet"
    y_path = f"{DATA_DIR}/y_train.parquet"

    if not os.path.exists(X_path) or not os.path.exists(y_path):
        raise FileNotFoundError("Train split not found. Please run split.py first.")

    X_train = pd.read_parquet(X_path)
    y_train = pd.read_parquet(y_path).values.ravel()

    print(f"Training data shape: {X_train.shape}")
    print(f"Positive rate (train): {y_train.mean():.6f}")

    return X_train, y_train


def build_model(y_train):
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("Base XGB model initialized.")

    return model


def run_random_search(model, X_train, y_train):
    param_dist = {
        "n_estimators": [200, 300, 400],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1, 0.2],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0]
    }

    print("Starting RandomizedSearchCV...")
    print("Scoring metric: average_precision (PR-AUC)")

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=15,
        scoring="average_precision",
        cv=3,
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    start_time = time.time()
    random_search.fit(X_train, y_train)
    end_time = time.time()

    print("Random search completed.")
    print(f"Best CV PR-AUC: {random_search.best_score_:.6f}")
    print("Best parameters found:")
    print(random_search.best_params_)
    print(f"Tuning time: {end_time - start_time:.2f} seconds")

    return random_search.best_estimator_


def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = f"{MODEL_DIR}/xgb_model.pkl"

    joblib.dump(model, model_path)

    print("===================================")
    print(f"Model saved to: {model_path}")
    print("Ready to run: evaluate.py")
    print("===================================")


def main():
    X_train, y_train = load_data()
    base_model = build_model(y_train)
    best_model = run_random_search(base_model, X_train, y_train)
    save_model(best_model)


if __name__ == "__main__":
    main()
