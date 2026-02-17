# train_final_severity_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

print("🔹 Loading MIMIC severity dataset...")

df = pd.read_csv("data/mimic_severity_dataset.csv")

print("Columns:", df.columns)
print("Dataset shape:", df.shape)
print("\nSeverity distribution:\n", df["severity"].value_counts())

# ---------------------------------------------------
# STEP 1: Select real clinical features
# ---------------------------------------------------

X = df[["num_diagnoses", "icu", "los_days", "mortality"]]
y = df["severity"]

# ---------------------------------------------------
# STEP 2: Train-test split
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ---------------------------------------------------
# STEP 3: Train Random Forest (best for tabular clinical data)
# ---------------------------------------------------

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ---------------------------------------------------
# STEP 4: Evaluation
# ---------------------------------------------------

y_pred = model.predict(X_test)

print("\n=== FINAL SEVERITY MODEL RESULTS ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# ---------------------------------------------------
# STEP 5: Save model
# ---------------------------------------------------

joblib.dump(model, "models/final_severity_model.pkl")

print("\n✅ Final clinical severity model saved in /models/")
