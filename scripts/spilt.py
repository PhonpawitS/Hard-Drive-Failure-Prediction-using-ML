import os
import pandas as pd
from sklearn.model_selection import train_test_split


INPUT_PATH = "../data/processed/feature_engineered.parquet"
OUTPUT_DIR = "../data/split"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_data():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"File not found: {INPUT_PATH}")

    print("===================================")
    print("Starting Data Splitting Pipeline...")

    df = pd.read_parquet(INPUT_PATH)
    print(f"Loaded dataset shape: {df.shape}")

    return df


def prepare_features(df):
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

    # Add missing indicators
    for col in ["smart_187_raw", "smart_188_raw"]:
        if col in X.columns:
            X[col + "_missing"] = X[col].isna().astype(int)

    X = X.fillna(0)

    return X, y


def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("Split completed.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.6f}")
    print(f"Test positive rate: {y_test.mean():.6f}")

    return X_train, X_test, y_train, y_test


def save_split(X_train, X_test, y_train, y_test):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train.to_parquet(f"{OUTPUT_DIR}/X_train.parquet", index=False)
    X_test.to_parquet(f"{OUTPUT_DIR}/X_test.parquet", index=False)
    y_train.to_frame().to_parquet(f"{OUTPUT_DIR}/y_train.parquet", index=False)
    y_test.to_frame().to_parquet(f"{OUTPUT_DIR}/y_test.parquet", index=False)

    print("===================================")
    print(f"Train/Test datasets saved to {OUTPUT_DIR}")
    print("Ready to run: train_rf.py or train_xgb.py")
    print("===================================")


def main():
    df = load_data()
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_split(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
