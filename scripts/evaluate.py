import matplotlib.pyplot as plt
from xgboost import plot_importance
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import joblib
import numpy as np
from sklearn.metrics import classification_report

X_test = pd.read_parquet("../data/processed/X_test.parquet")
y_test = pd.read_parquet("../data/processed/y_test.parquet").values.ravel()

rf_model = joblib.load("../models/rf_model.pkl")
xgb_model = joblib.load("../models/xgb_model.pkl")

y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
y_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]


print("RF ROC-AUC:", roc_auc_score(y_test, y_proba_rf))
print("XGB ROC-AUC:", roc_auc_score(y_test, y_proba_xgb))

print("RF PR-AUC:", average_precision_score(y_test, y_proba_rf))
print("XGB PR-AUC:", average_precision_score(y_test, y_proba_xgb))

def evaluate_top_pct(y_true, y_proba, pct):
    k = int(len(y_proba) * pct)
    idx = np.argsort(y_proba)[::-1][:k]
    y_pred = np.zeros_like(y_proba)
    y_pred[idx] = 1
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))

print("RF Top 1%")
evaluate_top_pct(y_test, y_proba_rf, 0.01)

print("XGB Top 1%")
evaluate_top_pct(y_test, y_proba_xgb, 0.01)


plt.figure(figsize=(8,6))
plot_importance(xgb_model, max_num_features=10)
plt.title("Feature Importance (XGBoost)")
plt.show()