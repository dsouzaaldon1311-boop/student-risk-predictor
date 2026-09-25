import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
import numpy as np

df = pd.read_csv("../data/students_cleaned.csv")
X = df.drop(columns=["At_Risk"])
y = df["At_Risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = joblib.load("../models/model_rf.pkl")

# Get probability of "At Risk" (class 1) instead of a hard 0/1 prediction
y_probs = rf.predict_proba(X_test)[:, 1]

# Try a few thresholds and compare
for threshold in [0.5, 0.4, 0.35, 0.3, 0.25]:
    y_pred_thresh = (y_probs >= threshold).astype(int)
    print(f"\n--- Threshold: {threshold} ---")
    print(classification_report(y_test, y_pred_thresh, target_names=["Not At Risk", "At Risk"]))

    import joblib
joblib.dump(0.35, "../models/threshold.pkl")
print("Saved threshold: 0.35")