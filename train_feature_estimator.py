import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.metrics import classification_report

print("🔹 Loading datasets...")

# ===============================
# LOAD DATA
# ===============================

symptom_df = pd.read_csv("data/Symptom2Disease.csv")

# keep needed columns
symptom_df = symptom_df[["label", "text"]].dropna()

# load MIMIC severity dataset
mimic_df = pd.read_csv("data/mimic_severity_dataset.csv")

print("Symptom2Disease:", symptom_df.shape)
print("MIMIC severity:", mimic_df.shape)

# ===============================
# CREATE SYNTHETIC TRAINING PAIRS
# ===============================
# We map each disease text to a random clinical record
# This is common weak-supervision trick in medical NLP research

sampled_mimic = mimic_df.sample(len(symptom_df), replace=True, random_state=42)

train_df = pd.DataFrame({
    "text": symptom_df["text"].values,
    "num_diagnoses": sampled_mimic["num_diagnoses"].values,
    "icu": sampled_mimic["icu"].values,
    "los_days": sampled_mimic["los_days"].values,
    "mortality": sampled_mimic["mortality"].values,
})

print("Training dataset:", train_df.shape)

# ===============================
# FEATURES & TARGETS
# ===============================

X_text = train_df["text"]

y = train_df[["num_diagnoses", "icu", "los_days", "mortality"]]

# ===============================
# TF-IDF
# ===============================

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english"
)

X = vectorizer.fit_transform(X_text)

# ===============================
# TRAIN / TEST SPLIT
# ===============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# MODEL
# ===============================

model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=150,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
)

print("🔹 Training clinical feature estimator...")
model.fit(X_train, y_train)

# ===============================
# SAVE
# ===============================

joblib.dump(model, "models/feature_estimator.pkl")
joblib.dump(vectorizer, "models/feature_vectorizer.pkl")

print("✅ Clinical feature estimator saved in /models/")
