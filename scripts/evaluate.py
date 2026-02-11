import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from xgboost import plot_importance
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


# ==========================================================
# Utility Functions
# ==========================================================

def load_test_data():
    print("\n[1] Loading test dataset...")
    X_test = pd.read_parquet("data/split/X_test.parquet")
    y_test = pd.read_parquet("data/split/y_test.parquet").values.ravel()

    print(f"X_test shape: {X_test.shape}")
    print(f"Positive rate (test): {y_test.mean():.6f}")
    return X_test, y_test


def load_models():
    print("\n[2] Loading trained models...")
    rf_model = joblib.load("models/rf_model.pkl")
    xgb_model = joblib.load("models/xgb_model.pkl")
    print("Models loaded successfully.")
    return rf_model, xgb_model


def evaluate_global_metrics(y_true, y_proba, model_name):
    roc = roc_auc_score(y_true, y_proba)
    pr = average_precision_score(y_true, y_proba)

    print(f"\n{model_name} Performance")
    print("-" * 40)
    print(f"ROC-AUC : {roc:.6f}")
    print(f"PR-AUC  : {pr:.6f}")

    return roc, pr


def evaluate_top_pct(y_true, y_proba, pct, model_name):
    print(f"\n{model_name} — Top {int(pct * 100)}% High-Risk Strategy")
    print("-" * 40)

    k = int(len(y_proba) * pct)
    idx = np.argsort(y_proba)[::-1][:k]

    y_pred = np.zeros_like(y_proba)
    y_pred[idx] = 1

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))


def plot_xgb_importance(model):
    print("\n[6] Plotting XGBoost Feature Importance...")
    plt.figure(figsize=(8, 6))
    plot_importance(model, max_num_features=10)
    plt.title("Top 10 Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.show()


# ==========================================================
# Main Execution
# ==========================================================

def main():

    print("=" * 60)
    print("MODEL EVALUATION PIPELINE")
    print("=" * 60)

    # Load
    X_test, y_test = load_test_data()
    rf_model, xgb_model = load_models()

    # Predict probabilities
    print("\n[3] Generating predictions...")
    y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
    y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
    print("Predictions generated.")

    # Global metrics
    print("\n[4] Overall Metrics")
    evaluate_global_metrics(y_test, y_proba_rf, "Random Forest")
    evaluate_global_metrics(y_test, y_proba_xgb, "XGBoost")

    # Business metric (Top 1%)
    print("\n[5] Business-Oriented Evaluation (Top 1%)")
    evaluate_top_pct(y_test, y_proba_rf, 0.01, "Random Forest")
    evaluate_top_pct(y_test, y_proba_xgb, 0.01, "XGBoost")

    # Feature importance
    plot_xgb_importance(xgb_model)

    print("\nEvaluation completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
