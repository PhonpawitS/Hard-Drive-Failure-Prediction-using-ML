import os
import pandas as pd
import glob

USE_SAMPLE = False

files = glob.glob("../data/raw/data_Q3_2025/*.csv")

if USE_SAMPLE:
    files = files[:10]

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

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["serial_number", "date"])

df = df.drop(columns=[col for col in df.columns if "fail_date" in col], errors="ignore")

fail_dates = (
    df.loc[df["failure"] == 1]
      .groupby("serial_number")["date"]
      .min()
      .rename("fail_date")
)

df = df.merge(fail_dates, on="serial_number", how="left")

window = 7

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

os.makedirs("../data/processed", exist_ok=True)
df_train.to_parquet("../data/processed/feature_engineered.parquet", index=False)

print("===================================")
print("Preprocessing completed successfully.")
print("Saved to: ../data/processed/feature_engineered.parquet")
print("Ready to run: split.py")
print("===================================")
