import pandas as pd

url = "https://archive.ics.uci.edu/static/public/697/predict+students+dropout+and+academic+success.zip"
df = pd.read_csv(url, sep=";", compression="zip")

# 1. Fix the stray tab character in that one column name
df.columns = df.columns.str.strip()

# 2. Create our binary target: 1 = At Risk (Dropout), 0 = Not At Risk (Graduate or Enrolled)
df["At_Risk"] = (df["Target"] == "Dropout").astype(int)

# 3. Drop the original 3-class Target column now that we have our binary one
df = df.drop(columns=["Target"])

# 4. Sanity checks
print(df.columns.tolist())
print()
print(df["At_Risk"].value_counts())
print()
print(df["At_Risk"].value_counts(normalize=True).round(3))  # proportions

# 5. Save cleaned version so we don't re-download every time
df.to_csv("../data/students_cleaned.csv", index=False)
print("\nSaved to students_cleaned.csv")