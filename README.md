# MedAI – AI-Driven Symptom-Based Clinical Severity Prediction

MedAI is an AI-powered clinical decision support prototype that predicts **patient severity level** (Low, Medium, High) from:

- Natural language symptom description  
- Age  
- Gender  
- Duration of symptoms  

The system combines **clinical ML**, **NLP**, and **risk estimation** to perform **early triage support**.

---

## 🚀 Features

- Symptom → Clinical feature estimation  
- ICU risk prediction  
- Mortality risk estimation  
- Hospital length-of-stay estimation  
- Final severity classification (Low / Medium / High)  
- Explainable AI reasoning output  
- FastAPI backend + simple web UI  

---

## 🧠 Models Used

### 1️⃣ NLP Symptom Understanding
- **TF-IDF Vectorization**
- Converts free-text symptoms → numerical vectors

### 2️⃣ Clinical Feature Estimator
- **Random Forest Regressor**
- Predicts:
  - ICU risk
  - Mortality risk
  - Length of stay
  - Number of diagnoses

### 3️⃣ Final Severity Classifier
- **Random Forest Classifier**
- Uses predicted clinical features → severity label

---

## 📊 Model Performance

### Final Severity Model
- **Accuracy:** 98.18%
- **Precision:**  
  - High: 1.00  
  - Medium: 0.91  
  - Low: 1.00  

- **Recall:**  
  - High: 1.00  
  - Medium: 1.00  
  - Low: 0.95  

- **F1-Score:**  
  - High: 1.00  
  - Medium: 0.95  
  - Low: 0.97  

---

## 🗂️ Datasets Used

### 1️⃣ Symptoms2Disease Dataset
- Natural language symptom descriptions
- Used for **NLP training**

### 2️⃣ MIMIC-IV Clinical Dataset (PhysioNet)
- ICU admission data
- Mortality outcomes
- Length of stay
- Diagnoses count  
➡ Used for **clinical severity modeling**

---

## 🏗️ Project Structure

MedAI/
│
├── data/
│ ├── Symptoms2Disease.csv
│ ├── mimic_severity_dataset.csv
│
├── models/
│ ├── severity_model.pkl
│ ├── severity_vectorizer.pkl
│ ├── feature_estimator.pkl
│
├── app.py
├── train_final_severity_model.py
├── train_feature_estimator.py
├── requirements.txt
└── README.md


---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/MedAI.git
cd MedAI
pip install -r requirements.txt

▶️ Run the Backend
uvicorn app:app --reload

🔬 Research Contribution

This work proposes an:

AI-Driven Symptom-Based Triage System for Early Clinical Risk Assessment

Key novelty:

Combines NLP + Clinical ML

Uses real ICU dataset (MIMIC-IV)

Provides explainable severity reasoning

Designed for early triage in low-resource settings

⚠️ Disclaimer

MedAI is a research prototype and NOT a medical device.
Predictions must not replace professional medical judgment.
