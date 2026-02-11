# Hard-Drive-Failure-Prediction-using-ML

Predict hard drive failure using SMART attributes from the Backblaze dataset using Machine Learning.

---

## Project Overview

This project aims to predict hard drive failure using SMART (Self-Monitoring, Analysis and Reporting Technology) metrics from the Backblaze dataset.

We build a complete ML pipeline including:

- Data download
- Data preprocessing
- Train/test split
- Model training (Random Forest, XGBoost)
- Model evaluation

The objective is to classify whether a hard drive will fail (`failure = 1`) or remain healthy (`failure = 0`).

---

## Project Structure

```
Hard-Drive-Failure-Prediction-using-ML/
│
├── data/
│   ├── raw/            # Raw downloaded data
│   ├── processed/      # Cleaned & feature-selected data
│   └── split/          # Train/test datasets (parquet files)
│
├── models/             # Saved trained models
│
├── scripts/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── split.py
│   ├── train_rf.py
│   ├── train_xgb.py
│   └── evaluate.py
│
└── README.md
```

---

## Dataset

This project uses the Backblaze Hard Drive SMART dataset.

Example SMART attributes:

- `smart_5`  – Reallocated Sectors Count  
- `smart_9`  – Power-On Hours  
- `smart_197` – Current Pending Sector  
- `smart_198` – Offline Uncorrectable  

Target variable:

- `failure`
  - 1 = drive failed
  - 0 = drive healthy

---

## How to Download Data

```bash
python scripts/data_loader.py
```

Raw data will be stored in:

```
data/raw/
```

---

## Data Preprocessing

```bash
python scripts/preprocess.py
```

This step:

- Selects important SMART features
- Handles missing values
- Cleans dataset
- Saves processed dataset to:

```
data/processed/
```

---

## Train-Test Split

```bash
python scripts/split.py
```

Output:

```
data/split/
├── X_train.parquet
├── X_test.parquet
├── y_train.parquet
└── y_test.parquet
```

---

## Train Random Forest

```bash
python scripts/train_rf.py
```

Model configuration:

- `n_estimators = 200`
- `class_weight = "balanced_subsample"`
- `random_state = 42`
- `n_jobs = -1`

Saved model:

```
models/rf_model.pkl
```

---

## Train XGBoost

```bash
python scripts/train_xgb.py
```

Model details:

- Objective: `binary:logistic`
- Evaluation metric: `logloss`
- Hyperparameter tuning: `RandomizedSearchCV`
- Scoring metric: `average_precision` (PR-AUC)
- Cross-validation: 3-fold

Saved model:

```
models/xgb_model.pkl
```

---

## Model Evaluation

```bash
python scripts/evaluate.py
```

Evaluation includes:

- ROC-AUC
- PR-AUC
- Confusion Matrix
- Classification Report
- Business evaluation (Top 1% high-risk disks)
- XGBoost Feature Importance plot

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Main libraries used:

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- joblib

---

