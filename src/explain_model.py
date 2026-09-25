import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# 1. Load everything we saved earlier
df = pd.read_csv("../data/students_cleaned.csv")
X = df.drop(columns=["At_Risk"])
rf = joblib.load("../models/model_rf.pkl")

# 2. Build a SHAP explainer for our Random Forest model
#    TreeExplainer is fast and exact for tree-based models like Random Forest
explainer = shap.TreeExplainer(rf)

# 3. Calculate SHAP values for a sample of students (all 4424 would be slow)
sample = X.sample(200, random_state=42)
shap_values = explainer.shap_values(sample)

# shap_values is a list [class_0_values, class_1_values] for binary classification
# We care about class 1 = "At Risk"
if isinstance(shap_values, list):
    # older SHAP versions: list of arrays, one per class
    shap_values_at_risk = shap_values[1]
elif shap_values.ndim == 3:
    # newer SHAP versions: single array shaped (samples, features, classes)
    shap_values_at_risk = shap_values[:, :, 1]
else:
    shap_values_at_risk = shap_values

# 4. GLOBAL explanation: which features matter most overall?
plt.figure()
shap.summary_plot(shap_values_at_risk, sample, show=False)
plt.tight_layout()
plt.savefig("../data/shap_summary.png", dpi=150)
plt.close()
print("Saved global feature importance chart to shap_summary.png")

# 5. LOCAL explanation: explain ONE specific student's prediction
#    Let's pick the first student in our sample as an example
student_index = 0
student_data = sample.iloc[[student_index]]
student_prob = rf.predict_proba(student_data)[0][1]
print(f"\nExample student risk probability: {student_prob:.2%}")

# Show the top factors pushing this student's risk up or down
single_shap = shap_values_at_risk[student_index]
feature_impact = pd.DataFrame({
    "feature": sample.columns,
    "value": student_data.values[0],
    "shap_impact": single_shap
}).sort_values("shap_impact", key=abs, ascending=False)

print("\nTop 10 factors driving this student's risk score:")
print(feature_impact.head(10).to_string(index=False))