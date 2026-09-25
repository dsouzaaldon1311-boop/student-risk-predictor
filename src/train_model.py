import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# 1. Load cleaned data
df = pd.read_csv("../data/students_cleaned.csv")

# 2. Split features (X) from target (y)
X = df.drop(columns=["At_Risk"])
y = df["At_Risk"]   

# 3. Train/test split — 80% to train on, 20% held back to honestly test on
#    stratify=y keeps the same 32%/68% risk ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# 4. Scale features — Logistic Regression is sensitive to feature scale
#    (e.g. "Age" ranges 17-70 but "Debtor" is just 0/1 — scaling puts them on equal footing)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. MODEL 1: Logistic Regression (simple, interpretable baseline)
log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred_log = log_reg.predict(X_test_scaled)

print("\n" + "="*50)
print("LOGISTIC REGRESSION RESULTS")
print("="*50)
print(classification_report(y_test, y_pred_log, target_names=["Not At Risk", "At Risk"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_log))

# 6. MODEL 2: Random Forest (usually stronger, handles nonlinearity, doesn't need scaling)
rf = RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\n" + "="*50)
print("RANDOM FOREST RESULTS")
print("="*50)
print(classification_report(y_test, y_pred_rf, target_names=["Not At Risk", "At Risk"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf))

# 7. Save everything we'll need later (model, scaler, column names)
joblib.dump(rf, "../models/model_rf.pkl")
joblib.dump(log_reg, "../models/model_logreg.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
joblib.dump(list(X.columns), "../models/feature_columns.pkl")
print("\nModels saved: model_rf.pkl, model_logreg.pkl, scaler.pkl, feature_columns.pkl")