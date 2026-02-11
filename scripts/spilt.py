import os
import pandas as pd
from sklearn.model_selection import train_test_split

print("===================================")
print("Starting Data Splitting Pipeline...")

df = pd.read_parquet("../data/processed/feature_engineered.parquet")
print(f"Loaded dataset shape: {df.shape}")

target = "fail_in_7_days"

drop_cols = [
    target,
    "failure",
    "date",
    "serial_number",
    "model",
    "capacity_bytes",
    "fail_date"
]

X = df.drop(columns=drop_cols)
y = df[target]

print(f"Feature matrix shape: {X.shape}")
print(f"Target positive rate (overall): {y.mean():.6f}")

# Missing indicators
for col in ["smart_187_raw", "smart_188_raw"]:
    X[col + "_missing"] = X[col].isna().astype(int)

X = X.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Split completed.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Train positive rate: {y_train.mean():.6f}")
print(f"Test positive rate: {y_test.mean():.6f}")

os.makedirs("../data/split", exist_ok=True)

X_train.to_parquet("../data/split/X_train.parquet", index=False)
X_test.to_parquet("../data/split/X_test.parquet", index=False)
y_train.to_parquet("../data/split/y_train.parquet", index=False)
y_test.to_parquet("../data/split/y_test.parquet", index=False)

print("===================================")
print("Train/Test datasets saved to ../data/split/")
print("Ready to run: train_rf.py or train_xgb.py")
print("===================================")
