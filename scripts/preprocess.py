import os
import glob
import pandas as pd


USE_SAMPLE = False
DATA_PATH = "data/raw/data_Q3_2025/*.csv"
OUTPUT_PATH = "data/processed/feature_engineered.parquet"


def load_data():
    files = glob.glob(DATA_PATH)

    if USE_SAMPLE:
        files = files[:10]

    if len(files) == 0:
        raise FileNotFoundError("No CSV files found in data directory.")

    print("===================================")
    print("Starting Preprocessing Pipeline...")
    print(f"Number of files loaded: {len(files)}")

    important_smarts = [
        "smart_5_raw",
        "smart_9_raw",
        "smart_187_raw",
        "smart_188_raw",
        "smart_197_raw",
        "smart_198_raw"
    ]

    use_cols = [
        "date",
        "serial_number",
        "model",
        "capacity_bytes",
        "failure"
    ] + important_smarts

    df = pd.concat(
        (pd.read_csv(f, usecols=use_cols) for f in files),
        ignore_index=True
    )

    print(f"Raw dataset shape: {df.shape}")

    return df


def create_target(df, window=7):
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["serial_number", "date"])

    # Remove old fail_date if exists
    df = df.drop(columns=[col for col in df.columns if "fail_date" in col], errors="ignore")

    fail_dates = (
        df.loc[df["failure"] == 1]
        .groupby("serial_number")["date"]
        .min()
        .rename("fail_date")
    )

    df = df.merge(fail_dates, on="serial_number", how="left")

    df["fail_in_7_days"] = (
        (df["fail_date"].notna()) &
        (df["date"] < df["fail_date"]) &
        (df["date"] >= df["fail_date"] - pd.Timedelta(days=window))
    ).astype(int)

    df_train = df[
        (df["fail_date"].isna()) |
        (df["date"] <= df["fail_date"])
    ]

    print(f"Training dataset shape: {df_train.shape}")
    print(f"Positive rate (fail_in_7_days): {df_train['fail_in_7_days'].mean():.6f}")

    return df_train


def save_data(df):
    os.makedirs("../data/processed", exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)

    print("===================================")
    print("Preprocessing completed successfully.")
    print(f"Saved to: {OUTPUT_PATH}")
    print("Ready to run: split.py")
    print("===================================")


def main():
    df = load_data()
    df_train = create_target(df)
    save_data(df_train)


if __name__ == "__main__":
    main()
