import os
import time
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
import joblib

print("===================================")
print("Starting XGBoost Training with Random Search...")

start_time = time.time()

# Load data
X_train = pd.read_parquet("../data/split/X_train.parquet")
y_train = pd.read_parquet("../data/split/y_train.parquet").values.ravel()

print(f"Training data shape: {X_train.shape}")
print(f"Positive rate (train): {y_train.mean():.6f}")

# Handle imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

# Base model
xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    n_jobs=-1
)

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
    xgb,
    param_distributions=param_dist,
    n_iter=15,
    scoring="average_precision",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("Random search completed.")
print(f"Best CV PR-AUC: {random_search.best_score_:.6f}")
print("Best parameters found:")
print(random_search.best_params_)

xgb_model = random_search.best_estimator_

# Save model
os.makedirs("../models", exist_ok=True)
model_path = "../models/xgb_model.pkl"
joblib.dump(xgb_model, model_path)

end_time = time.time()

print("===================================")
print(f"Model saved to: {model_path}")
print(f"Total training time: {end_time - start_time:.2f} seconds")
print("Ready to run: evaluate.py")
print("===================================")
