from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware



# =========================
# Load models
# =========================
severity_model = joblib.load("models/final_severity_model.pkl")

feature_model = joblib.load("models/feature_estimator.pkl")
feature_vectorizer = joblib.load("models/feature_vectorizer.pkl")

app = FastAPI(title="MedAI Clinical Severity API")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow frontend access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Request schema
# =========================
class SymptomRequest(BaseModel):
    text: str
    age: int
    gender: str
    duration_days: int


# =========================
# Advice helper
# =========================
def get_advice(severity: str):
    if severity == "High":
        return "⚠ Seek immediate medical attention or visit the nearest hospital."
    elif severity == "Medium":
        return "👨‍⚕ Consult a doctor soon for proper diagnosis."
    else:
        return "🟢 Monitor symptoms at home and rest. Seek care if worsening."


# =========================
# Predict endpoint
# =========================
@app.post("/predict")
def predict(data: SymptomRequest):

    # ---------------------------------
    # 1️⃣ Estimate clinical features
    # ---------------------------------
    feature_input = f"{data.text} age {data.age} gender {data.gender} duration {data.duration_days}"
    feature_vec = feature_vectorizer.transform([feature_input])

    clinical_features = feature_model.predict(feature_vec)[0]

    # Expected order from training:
    # [icu_risk, mortality_risk, los_days, num_diagnoses]
    icu_risk = float(clinical_features[0])
    mortality_risk = float(clinical_features[1])
    los_days = float(clinical_features[2])
    num_diagnoses = float(clinical_features[3])

    # ---------------------------------
    # 2️⃣ Predict severity USING 4 features
    # ---------------------------------
    severity_input = np.array([[icu_risk, mortality_risk, los_days, num_diagnoses]])

    predicted_severity = severity_model.predict(severity_input)[0]
    confidence = float(np.max(severity_model.predict_proba(severity_input)))

    # ---------------------------------
    # 3️⃣ Explanation
    # ---------------------------------
    explanation = (
        f"ICU risk={icu_risk:.2f}, mortality risk={mortality_risk:.2f}, "
        f"LOS={los_days:.1f} days, diagnoses={num_diagnoses:.0f} "
        f"→ predicted severity={predicted_severity}."
    )

    # ---------------------------------
    # 4️⃣ Response
    # ---------------------------------
    return {
        "severity": predicted_severity,
        "confidence": round(confidence, 3),
        "advice": get_advice(predicted_severity),
        "clinical_features": {
            "icu_risk": round(icu_risk, 3),
            "mortality_risk": round(mortality_risk, 3),
            "length_of_stay_days": round(los_days, 2),
            "num_diagnoses": int(num_diagnoses),
        },
        "explanation": explanation,
    }


@app.get("/")
def home():
    return {"message": "MedAI Clinical Severity API running"}
