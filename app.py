from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, Field
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from scipy.sparse import hstack
import re



# =========================
# Load models - TWO-STAGE PIPELINE
# =========================
# Stage 1: Symptoms → Severity
severity_model = joblib.load("models/severity_classifier.pkl")
text_vectorizer = joblib.load("models/text_vectorizer.pkl")
scaler = joblib.load("models/numeric_scaler.pkl")
gender_encoder = joblib.load("models/gender_encoder.pkl")

# Stage 2: Severity → Clinical Outcomes (trained on MIMIC-IV)
clinical_model = joblib.load("models/clinical_predictor.pkl")
clinical_scaler = joblib.load("models/clinical_scaler.pkl")

app = FastAPI(title="MedAI Clinical Severity API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow frontend access
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# Request schema with validation
# =========================
class SymptomRequest(BaseModel):
    text: str = Field(..., min_length=5, max_length=2000, description="Symptom description")
    age: int = Field(..., ge=0, le=120, description="Patient age (0-120)")
    gender: str = Field(..., description="Patient gender")
    duration_days: int = Field(..., ge=0, le=3650, description="Symptom duration in days (0-3650)")
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        """Validate symptom text input"""
        # Remove extra whitespace
        v = ' '.join(v.split())
        
        # Check if empty after cleaning
        if len(v.strip()) < 5:
            raise ValueError('Symptom description must be at least 5 characters')
        
        # Check if text contains at least one letter (not just numbers/symbols)
        if not re.search(r'[a-zA-Z]', v):
            raise ValueError('Symptom description must contain actual text, not just numbers or symbols')
        
        text_lower = v.lower()
        
        # POSITIVE VALIDATION: Must contain at least one actual medical symptom/term
        # This is much more robust than trying to catch every bad input
        valid_medical_terms = [
            # Common symptoms
            'fever', 'pain', 'ache', 'cough', 'cold', 'flu', 'headache', 'migraine',
            'nausea', 'vomit', 'diarrhea', 'constipation', 'fatigue', 'tired', 'weakness',
            'dizzy', 'dizziness', 'vertigo', 'confusion', 'disoriented',
            
            # Respiratory
            'breath', 'breathing', 'wheez', 'asthma', 'shortness', 'suffocating',
            'congestion', 'stuffy', 'runny nose', 'sneez', 'sinus',
            
            # Cardiovascular
            'chest', 'heart', 'cardiac', 'palpitation', 'racing', 'irregular',
            'pressure', 'tight', 'angina',
            
            # Gastrointestinal
            'stomach', 'abdominal', 'belly', 'cramp', 'bloat', 'indigestion',
            'heartburn', 'reflux', 'gas', 'nauseous',
            
            # Neurological
            'seizure', 'convulsion', 'tremor', 'numbness', 'tingling', 'paralysis',
            'stroke', 'blurred vision', 'double vision',
            
            # Musculoskeletal
            'joint', 'muscle', 'back', 'neck', 'shoulder', 'leg', 'arm',
            'sprain', 'strain', 'fracture', 'broken', 'swell', 'swollen',
            
            # Skin
            'rash', 'itch', 'hives', 'burn', 'wound', 'cut', 'bruise', 'bleed',
            'sore', 'lesion', 'blisters',
            
            # ENT
            'ear', 'throat', 'tonsil', 'nose', 'hearing', 'tinnitus',
            
            # Urinary
            'urin', 'bladder', 'kidney', 'frequent urination', 'painful urination',
            
            # General
            'inflammation', 'infection', 'discharge', 'pus', 'blood',
            'chills', 'sweats', 'appetite', 'weight loss', 'weight gain',
            'insomnia', 'sleep', 'anxiety', 'depression', 'stress',
            'dehydrat', 'thirst', 'dry mouth', 'pale', 'jaundice',
            
            # Severity descriptors with symptoms
            'severe', 'acute', 'chronic', 'persistent', 'intermittent',
            'sharp', 'dull', 'throbbing', 'burning', 'stabbing'
        ]
        
        # Check if at least one medical term is present
        has_medical_term = any(term in text_lower for term in valid_medical_terms)
        
        if not has_medical_term:
            raise ValueError(
                'Please describe actual medical symptoms (e.g., fever, pain, cough, nausea). '
                'Vague descriptions like "problem", "not feeling well", or "something wrong" are not sufficient.'
            )
        
        # Quick check for obviously bad inputs (much simpler now)
        bad_phrases = [
            'just testing', 'just test', 'test test', 'nothing', 'no symptoms',
            'i died', 'i am dead', 'passed away', 'lol lol', 'haha haha', 'joke'
        ]
        
        for phrase in bad_phrases:
            if phrase in text_lower:
                raise ValueError('Please enter serious medical symptoms only.')
        
        return v.strip()
    
    @field_validator('gender')
    @classmethod
    def validate_gender(cls, v):
        """Validate gender input"""
        v = v.lower().strip()
        if v not in ['male', 'female', 'm', 'f']:
            raise ValueError('Gender must be male or female')
        # Normalize to full name
        return 'male' if v in ['male', 'm'] else 'female'
    
    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        """Validate age is reasonable"""
        if v < 0:
            raise ValueError('Age cannot be negative')
        if v > 120:
            raise ValueError('Age must be 120 or less')
        return v
    
    @field_validator('duration_days')
    @classmethod
    def validate_duration(cls, v):
        """Validate symptom duration is reasonable"""
        if v < 0:
            raise ValueError('Duration cannot be negative')
        if v > 3650:  # ~10 years
            raise ValueError('Duration seems unrealistic. Please verify (max 3650 days / ~10 years)')
        return v


# =========================
# Input validation helper
# =========================
def validate_input_consistency(data: SymptomRequest):
    """
    Check for logical inconsistencies in input data
    Returns (is_valid, warning_message)
    """
    warnings = []
    
    # Check age-duration consistency
    if data.age < 1 and data.duration_days > 365:
        warnings.append("Infant with symptoms >1 year seems inconsistent")
    
    # Check for extremely long duration relative to age
    if data.duration_days > (data.age * 365 * 0.5) and data.age > 0:
        warnings.append(f"Duration ({data.duration_days} days) seems very long relative to age ({data.age} years)")
    
    # Warn about chronic symptoms
    if data.duration_days > 365:
        warnings.append(f"Symptoms lasting {data.duration_days} days ({data.duration_days//365}+ years) require comprehensive medical evaluation")
    
    return warnings


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

def get_detailed_recommendations(severity: str):
    """Provide detailed recommendations based on severity"""
    if severity == "High":
        return {
            "urgency": "IMMEDIATE",
            "action": "Emergency Care Required",
            "timeframe": "Seek help NOW or call emergency services",
            "care_level": "Emergency Department / Hospital",
            "details": [
                "Do not wait - these symptoms may indicate a serious condition",
                "Call emergency services or go to nearest emergency room",
                "If alone, ask someone to accompany you",
                "Bring list of current medications and medical history",
                "Do not drive yourself if symptoms are severe"
            ]
        }
    elif severity == "Medium":
        return {
            "urgency": "PROMPT",
            "action": "Medical Consultation Needed",
            "timeframe": "Schedule appointment within 24-48 hours",
            "care_level": "Urgent Care / Primary Care Physician",
            "details": [
                "These symptoms require professional medical evaluation",
                "Contact your doctor or visit urgent care clinic",
                "Monitor symptoms - seek emergency care if they worsen",
                "Keep track of temperature and other vital signs",
                "Stay hydrated and get adequate rest"
            ]
        }
    else:  # Low
        return {
            "urgency": "ROUTINE",
            "action": "Home Care with Monitoring",
            "timeframe": "Monitor for 3-5 days, seek care if symptoms persist",
            "care_level": "Self-Care / Telemedicine",
            "details": [
                "These symptoms are likely manageable with home care",
                "Get plenty of rest and stay well-hydrated",
                "Use over-the-counter medications as appropriate",
                "Monitor for any worsening or new symptoms",
                "Contact healthcare provider if symptoms persist beyond 5-7 days"
            ]
        }

def check_critical_symptoms(text: str, duration_days: int = None):
    """
    Safety override: Check for critical symptom combinations that should be HIGH severity.
    Uses negation detection and anatomical specificity to avoid false alarms.
    Returns True if critical symptoms detected
    """
    text_lower = text.lower()
    
    # ===== NEGATION DETECTION =====
    # Check if cardiac/chest symptoms are negated
    negation_patterns = [
        r'\bno\s+(heart|chest|cardiac)',           # "no heart pain"
        r'\b(heart|chest|cardiac)\s+is\s+not',     # "heart is not"
        r'\bnot?\s+(heart|chest|cardiac)',         # "not heart"
        r'\bwithout\s+(heart|chest|cardiac)',      # "without chest pain"
        r'\b(heart|chest|cardiac)\s+is\s+absent',  # "chest pain is absent"
        r'\babsence\s+of\s+(heart|chest|cardiac)', # "absence of heart"
        r'\bno\s+\w*\s*(heart|chest|cardiac)',     # "no severe heart"
    ]
    
    import re
    has_negation = any(re.search(pattern, text_lower) for pattern in negation_patterns)
    
    if has_negation:
        # If cardiac symptoms are explicitly negated, don't override to HIGH
        return False
    
    # ===== ANATOMICAL SPECIFICITY =====
    # Exclude non-cardiac pain areas (these should NOT trigger emergency override)
    non_cardiac_pain = [
        'ankle', 'foot', 'toe', 'leg', 'knee', 'thigh', 'hip',
        'wrist', 'hand', 'finger', 'elbow', 'shoulder', 'neck',
        'ear', 'eye', 'tooth', 'jaw', 'throat', 'nose',
        'stomach', 'belly', 'abdomen', 'groin', 'genital',
        'penis', 'dick', 'testicle', 'testicular', 'scrotum',
        'vagina', 'vaginal', 'breast', 'boob', 'nipple',
        'back', 'spine', 'lower back', 'upper back',
        'head', 'scalp', 'skin'
    ]
    
    # If non-cardiac pain is mentioned, don't treat as cardiac emergency
    if any(area in text_lower for area in non_cardiac_pain):
        # Exception: if explicitly says "chest pain" or "heart pain", still critical
        if not ('chest pain' in text_lower or 'heart pain' in text_lower or 
                'chest ache' in text_lower or 'heart ache' in text_lower or
                'cardiac pain' in text_lower):
            return False
    
    # ===== SPECIFIC CRITICAL PATTERNS =====
    # Only trigger on explicit cardiac emergencies
    critical_patterns = [
        'chest pain', 'heart pain', 'heart attack', 'cardiac arrest',
        'chest pressure', 'crushing chest', 'chest tightness',
        'chest discomfort', 'angina', 'heart racing',
        'stroke', 'seizure', 'unconscious', 'loss of consciousness',
        'severe bleeding', 'hemorrhage', 'coughing blood', 'vomiting blood',
        'can\'t breathe', 'cannot breathe', 'difficulty breathing',
        'shortness of breath', 'suffocating', 'choking'
    ]
    
    # Check if any critical pattern is present (exact phrase matching)
    has_critical_pattern = any(pattern in text_lower for pattern in critical_patterns)
    
    if has_critical_pattern:
        return True
    
    # Duration-based escalation (chronic symptoms need investigation)
    if duration_days is not None and duration_days > 30:
        return True
    
    return False


# =========================
# Predict endpoint
# =========================
@app.post("/predict")
def predict(data: SymptomRequest):
    
    # Check for input consistency warnings
    warnings = validate_input_consistency(data)
    
    try:
        # ---------------------------------
        # 1️⃣ Prepare text features
        # ---------------------------------
        text_vec = text_vectorizer.transform([data.text])
        
        # ---------------------------------
        # 2️⃣ Prepare numeric features
        # ---------------------------------
        numeric_features = np.array([[data.age, data.duration_days]])
        numeric_scaled = scaler.transform(numeric_features)
        
        # ---------------------------------
        # 3️⃣ Prepare gender
        # ---------------------------------
        gender_enc = gender_encoder.transform([[data.gender]])
        
        # ---------------------------------
        # 4️⃣ Combine features
        # ---------------------------------
        X_combined = hstack([text_vec, numeric_scaled, gender_enc])
        
        # ---------------------------------
        # 5️⃣ Predict severity directly
        # ---------------------------------
        predicted_severity = severity_model.predict(X_combined)[0]
        confidence = float(np.max(severity_model.predict_proba(X_combined)))
        
        # ---------------------------------
        # 5.5️⃣ Safety Override: Check for critical symptoms or chronic duration
        # ---------------------------------
        if check_critical_symptoms(data.text, data.duration_days):
            # Override if model predicted Medium or Low but symptoms are critical
            if predicted_severity in ["Medium", "Low"]:
                predicted_severity = "High"
                confidence = max(confidence, 0.75)  # Boost confidence for safety
        
        # ---------------------------------
        # 6️⃣ Predict clinical outcomes using MIMIC-trained model (Stage 2)
        # ---------------------------------
        # Map severity to numeric encoding (same as training)
        severity_map = {'Low': 0, 'Medium': 1, 'High': 2}
        severity_encoded = severity_map[predicted_severity]
        
        # Map gender to numeric
        gender_encoded = 1 if data.gender.lower() == 'male' else 0
        
        # Prepare features for clinical predictor
        # [severity_encoded, age, gender_encoded, duration_days]
        clinical_features = np.array([[
            severity_encoded,
            data.age,
            gender_encoded,
            data.duration_days
        ]])
        
        # Scale features using the same scaler from training
        clinical_features_scaled = clinical_scaler.transform(clinical_features)
        
        # Predict clinical outcomes
        # Outputs: [icu_risk, mortality_risk, los_days, num_diagnoses]
        clinical_predictions = clinical_model.predict(clinical_features_scaled)[0]
        
        icu_risk = clinical_predictions[0]  # Binary (0 or 1), convert to probability
        mortality_risk = clinical_predictions[1]
        los_days = clinical_predictions[2]
        num_diagnoses = clinical_predictions[3]
        
        # ---------------------------------
        # 7️⃣ Explanation
        # ---------------------------------
        explanation = (
            f"Stage 1: Symptom analysis (text pattern, age {data.age}, "
            f"gender {data.gender}, duration {data.duration_days} days) "
            f"→ Severity: {predicted_severity}. "
            f"\nStage 2: Clinical outcome prediction based on MIMIC-IV ICU data "
            f"for {predicted_severity} severity patients with similar demographics."
        )
        
        # Add warnings to explanation if any
        if warnings:
            explanation += "\n\n⚠️ Notes:\n" + "\n".join(f"• {w}" for w in warnings)

        # ---------------------------------
        # 8️⃣ Response
        # ---------------------------------
        detailed_rec = get_detailed_recommendations(predicted_severity)
        
        return {
            "severity": predicted_severity,
            "confidence": round(confidence, 3),
            "advice": get_advice(predicted_severity),
            "recommendations": detailed_rec,
            "clinical_features": {
                "icu_risk": round(max(0, min(1, icu_risk)), 3),
                "mortality_risk": round(max(0, min(1, mortality_risk)), 3),
                "length_of_stay_days": round(max(0, los_days), 2),
                "num_diagnoses": int(max(1, num_diagnoses)),
            },
            "explanation": explanation,
            "warnings": warnings,  # Include warnings in response
        }
        
    except Exception as e:
        # Catch any unexpected errors during prediction
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing prediction: {str(e)}"
        )


@app.get("/")
def home():
    return {"message": "MedAI Clinical Severity API running"}
