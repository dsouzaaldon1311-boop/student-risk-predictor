import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
from mappings import decode_value

# ─────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────
st.set_page_config(page_title="Student Academic Risk Predictor", layout="wide")
st.title("🎓 Student Academic Risk & Support System")
st.write("Identify students at risk of dropping out, and understand *why*.")

# ─────────────────────────────────────────────
# Cached loading — runs once, not on every interaction
# ─────────────────────────────────────────────
@st.cache_resource
def load_model_and_explainer():
    rf = joblib.load("../models/model_rf.pkl")
    explainer = shap.TreeExplainer(rf)
    return rf, explainer

@st.cache_data
def load_data():
    df = pd.read_csv("../data/students_cleaned.csv")
    return df

rf, explainer = load_model_and_explainer()
df = load_data()
X = df.drop(columns=["At_Risk"])

# ─────────────────────────────────────────────
# Sidebar controls
# ─────────────────────────────────────────────
st.sidebar.header("Controls")

threshold = st.sidebar.slider(
    "Risk threshold (flag as 'At Risk' if probability ≥ this)",
    min_value=0.10, max_value=0.90, value=0.35, step=0.05
)
st.sidebar.caption("Lower = catches more at-risk students, but more false alarms.")

student_index = st.sidebar.selectbox(
    "Select a student (by row number)",
    options=X.index.tolist(),
    index=0
)

# ─────────────────────────────────────────────
# Main panel: prediction for selected student
# ─────────────────────────────────────────────
student_row = X.loc[[student_index]]
probability = rf.predict_proba(student_row)[0][1]
is_at_risk = probability >= threshold

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Risk Assessment")
    st.metric("Risk Probability", f"{probability:.1%}")
    if is_at_risk:
        st.error("⚠️ FLAGGED: At Risk")
    else:
        st.success("✅ Not Flagged: Likely Safe")

    st.caption(f"Current threshold: {threshold:.0%}")

with col2:
    st.subheader("Student's Raw Data")
    display_df = student_row.T.rename(columns={student_index: "Value"})
    display_df["Value"] = display_df["Value"].apply(lambda x: int(x) if float(x).is_integer() else x)
    display_df["Decoded"] = [decode_value(feat, val) for feat, val in zip(display_df.index, display_df["Value"])]
    st.dataframe(display_df)
# ─────────────────────────────────────────────
# SHAP explanation for THIS specific student
# ─────────────────────────────────────────────
st.subheader("Why this prediction? (SHAP explanation)")

shap_values = explainer.shap_values(student_row)
if isinstance(shap_values, list):
    single_shap = shap_values[1][0]
elif shap_values.ndim == 3:
    single_shap = shap_values[0, :, 1]
else:
    single_shap = shap_values[0]

impact_df = pd.DataFrame({
    "Feature": X.columns,
    "Value": student_row.values[0],
    "Impact on Risk": single_shap
})
impact_df["Decoded Value"] = [
    decode_value(feat, val) for feat, val in zip(impact_df["Feature"], impact_df["Value"])
]
impact_df = impact_df.sort_values("Impact on Risk", key=abs, ascending=False).head(10)

st.dataframe(impact_df[["Feature", "Decoded Value", "Impact on Risk"]], use_container_width=True)
st.caption("Positive impact = pushes risk UP. Negative impact = pushes risk DOWN.")

# ─────────────────────────────────────────────
# Global feature importance (reference)
# ─────────────────────────────────────────────
with st.expander("See overall model behavior (all students)"):
    st.image("../data/shap_summary.png", caption="Global SHAP summary")