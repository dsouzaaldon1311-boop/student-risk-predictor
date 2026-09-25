import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/students_cleaned.csv")

# 1. Which features correlate most with being at-risk?
correlations = df.corr(numeric_only=True)["At_Risk"].sort_values(ascending=False)
print("Top 10 positively correlated with At_Risk:")
print(correlations.head(11)[1:])  # skip At_Risk correlating with itself
print("\nTop 10 negatively correlated with At_Risk:")
print(correlations.tail(10))

# 2. Compare average grades between at-risk and not-at-risk students
comparison = df.groupby("At_Risk")[[
    "Curricular units 1st sem (grade)",
    "Curricular units 2nd sem (grade)",
    "Age at enrollment",
    "Admission grade"
]].mean()
print("\nAverage values by risk group:")
print(comparison)

# 3. Save a quick visual: 2nd semester grade distribution by risk group
fig, ax = plt.subplots(figsize=(8, 5))
df[df["At_Risk"] == 0]["Curricular units 2nd sem (grade)"].hist(alpha=0.6, label="Not At Risk", bins=20, ax=ax)
df[df["At_Risk"] == 1]["Curricular units 2nd sem (grade)"].hist(alpha=0.6, label="At Risk", bins=20, ax=ax)
ax.set_xlabel("2nd Semester Grade")
ax.set_ylabel("Number of Students")
ax.set_title("Grade Distribution: At Risk vs Not At Risk")
ax.legend()
plt.savefig("../data/grade_distribution.png")
print("\nSaved chart to grade_distribution.png")