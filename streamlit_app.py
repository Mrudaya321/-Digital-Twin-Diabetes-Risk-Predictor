# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import numpy as np
import json
from cryptography.fernet import Fernet
import os

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

# =============== CONFIG ===============
KEY_FILE = "fernet_key.key"
MODEL_PATH = "final_model_clinical_ready.joblib"
SCALER_PATH = "scaler.joblib"
ORIGINAL_DATA_PATH = "diabetes_prediction_dataset.csv"  # adjust if needed
ENCRYPTED_FEEDBACK_FILE = "encrypted_feedback_records.bin"
THRESHOLD = 0.42  # your optimized threshold


# =============== ENCRYPTION SETUP ===============
if not os.path.exists(KEY_FILE):
    with open(KEY_FILE, "wb") as f:
        f.write(Fernet.generate_key())

with open(KEY_FILE, "rb") as f:
    FERNET_KEY = f.read()

fernet = Fernet(FERNET_KEY)


# =============== MODEL / SCALER LOAD ===============
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler, THRESHOLD


model, scaler, threshold = load_artifacts()


# =============== CONTINUOUS LEARNING HELPERS ===============
def save_encrypted_feedback_record(
    patient_features: dict,
    pred_proba: float,
    pred_label: int,
    feedback: str,          # "Correct" or "Incorrect"
    prescription_text: str, # free-text
    doctor_id: str = None
):
    """
    Save one feedback + prescription record in encrypted form.
    One Fernet-encrypted JSON line per record.
    """

    # Determine true label from feedback
    if feedback == "Correct":
        true_label = pred_label
    elif feedback == "Incorrect":
        true_label = 1 - pred_label  # flip
    else:
        true_label = None  # not used for retraining

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "patient_features": patient_features,  # raw unscaled input
        "pred_proba": float(pred_proba),
        "pred_label": int(pred_label),
        "doctor_feedback": feedback,
        "true_label": true_label,
        "prescription": prescription_text,
        "doctor_id": doctor_id,
    }

    json_str = json.dumps(record)
    token = fernet.encrypt(json_str.encode("utf-8"))

    with open(ENCRYPTED_FEEDBACK_FILE, "ab") as f:
        f.write(token + b"\n")


def load_decrypted_feedback_records() -> pd.DataFrame:
    """
    Decrypt all feedback records and return a DataFrame.
    Expands patient_features and keeps only rows with a valid true_label.
    """
    if not os.path.exists(ENCRYPTED_FEEDBACK_FILE):
        return pd.DataFrame()

    records = []
    with open(ENCRYPTED_FEEDBACK_FILE, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json_str = fernet.decrypt(line).decode("utf-8")
                rec = json.loads(json_str)
                records.append(rec)
            except Exception:
                continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df = df[df["true_label"].notnull()].copy()
    df["true_label"] = df["true_label"].astype(int)

    # expand patient_features dict to columns
    features_df = pd.json_normalize(df["patient_features"])
    features_df["label"] = df["true_label"]
    return features_df


def load_original_training_data() -> pd.DataFrame:
    """
    Load original diabetes dataset used for training.
    """
    df = pd.read_csv(ORIGINAL_DATA_PATH)
    return df


def build_retraining_dataset(target_col: str = "diabetes") -> pd.DataFrame:
    """
    Combine original dataset with feedback-labelled cases.
    Uses only columns that exist in BOTH datasets (plus target).
    """
    df_orig = load_original_training_data()
    df_feedback = load_decrypted_feedback_records()

    if df_feedback.empty:
        return df_orig

    if target_col not in df_orig.columns:
        raise ValueError(
            f"Target column '{target_col}' not found in original dataset. "
            f"Please update target_col in build_retraining_dataset()."
        )

    # feedback label column is called "label"
    df_feedback = df_feedback.rename(columns={"label": target_col})

    # intersection of feature columns
    orig_features = [c for c in df_orig.columns if c != target_col]
    fb_features = [c for c in df_feedback.columns if c != target_col]
    common_features = list(sorted(set(orig_features).intersection(fb_features)))

    if not common_features:
        raise ValueError("No common feature columns between original and feedback data.")

    df_orig_sub = df_orig[common_features + [target_col]].copy()
    df_fb_sub = df_feedback[common_features + [target_col]].copy()

    df_combined = pd.concat([df_orig_sub, df_fb_sub], axis=0).reset_index(drop=True)
    return df_combined


def retrain_model_and_save(
    target_col: str = "diabetes",
    version_prefix: str = "model"
):
    """
    Retrain an MLPClassifier on combined dataset (original + feedback),
    save model and scaler with a new version tag, and also update the
    active model/scaler used by the app.
    """
    df = build_retraining_dataset(target_col=target_col)

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # New scaler for retrained model
    scaler_new = StandardScaler()
    X_train_scaled = scaler_new.fit_transform(X_train)
    X_val_scaled = scaler_new.transform(X_val)

    # Simple MLP — tune as needed
    model_new = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        max_iter=200,
        random_state=42
    )

    model_new.fit(X_train_scaled, y_train)

    # Evaluate with same threshold
    val_proba = model_new.predict_proba(X_val_scaled)[:, 1]
    val_pred = (val_proba >= THRESHOLD).astype(int)

    acc = accuracy_score(y_val, val_pred)
    prec = precision_score(y_val, val_pred, zero_division=0)
    rec = recall_score(y_val, val_pred, zero_division=0)

    # Versioning
    version_idx = 1
    while os.path.exists(f"{version_prefix}_v{version_idx}.joblib"):
        version_idx += 1

    model_filename = f"{version_prefix}_v{version_idx}.joblib"
    scaler_filename = f"scaler_v{version_idx}.joblib"

    joblib.dump(model_new, model_filename)
    joblib.dump(scaler_new, scaler_filename)

    # Also update main model/scaler used in app
    joblib.dump(model_new, MODEL_PATH)
    joblib.dump(scaler_new, SCALER_PATH)

    # clear cache so new model is picked up
    load_artifacts.clear()

    return {
        "version": f"v{version_idx}",
        "model_path": model_filename,
        "scaler_path": scaler_filename,
        "accuracy": acc,
        "precision": prec,
        "recall": rec
    }


# -------- FUTURISTIC DIGITAL TWIN THEME CSS --------
st.markdown("""
<style>

body {
    background: #0a0f24;
}

.css-ffhzg2, .css-18e3th9 {
    background: #0a0f24 !important;
}

/* Neon glowing header */
.title-glow {
    font-size: 45px;
    font-weight: 800;
    text-align: center;
    color: #00eaff;
    text-shadow: 0px 0px 15px #00eaff, 0px 0px 25px #009dff;
}

/* Sub-header */
.sub-title {
    text-align: center;
    font-size: 20px;
    color: #9bd6ff;
}

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.08);
    border-radius: 15px;
    padding: 25px;
    border: 1px solid rgba(0, 174, 255, 0.3);
    box-shadow: 0px 0px 15px rgba(0, 174, 255, 0.4);
    backdrop-filter: blur(12px);
}

/* Prediction card animation */
@keyframes pulseGlow {
    0%   { box-shadow: 0px 0px 8px #00eaff; }
    50%  { box-shadow: 0px 0px 20px #00eaff; }
    100% { box-shadow: 0px 0px 8px #00eaff; }
}

.prediction-box {
    background: rgba(0, 255, 255, 0.1);
    border: 1px solid #00eaff;
    padding: 25px;
    border-radius: 20px;
    animation: pulseGlow 2s infinite;
    text-align: center;
}

/* Text color */
.text-glow {
    color: #00eaff;
    font-size: 24px;
    font-weight: bold;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(10px);
    border-right: 1px solid rgba(0, 174, 255, 0.2);
}

/* Sidebar title */
.sidebar-title {
    color: #00eaff;
    font-size: 22px;
}

</style>
""", unsafe_allow_html=True)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Diabetes Digital Twin",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- HEADER ----------
st.markdown("""
<h1 class='title-glow'>🧬 Digital Twin — Diabetes Risk Predictor</h1>
<p class='sub-title'>AI-powered holographic risk analysis | Futuristic Healthcare Interface</p>
""", unsafe_allow_html=True)

st.markdown("---")


# ---------- SIDEBAR INPUT ----------
with st.sidebar:
    st.markdown("<h2 class='sidebar-title'>⚙ Input Parameters</h2>", unsafe_allow_html=True)
    st.markdown("---")

    gender = st.selectbox("🧍 Gender", ["Male", "Female"])
    age = st.slider("📅 Age", 1, 100, 45)

    hypertension = st.selectbox("❤️ Hypertension", ["No", "Yes"])
    heart_disease = st.selectbox("💔 Heart Disease", ["No", "Yes"])

    bmi = st.slider("⚖ BMI", 10.0, 60.0, 27.0)
    hba1c = st.slider("🧪 HbA1c Level", 3.0, 15.0, 6.0)
    glucose = st.slider("🩸 Blood Glucose", 50.0, 400.0, 120.0)
    smoking = st.selectbox("🚬 Smoking History", ["never", "former", "current", "unknown"])

    predict = st.button("🔍 Predict Risk")

    # ===== CONTINUOUS LEARNING SIDEBAR =====
    st.markdown("---")
    st.markdown("<h3 class='sidebar-title'>🧠 Continuous Learning</h3>", unsafe_allow_html=True)
    st.caption("Retrain using doctor-validated cases (manual).")

    if st.button("🔁 Retrain model now"):
        with st.spinner("Retraining model on original + feedback data..."):
            try:
                result = retrain_model_and_save(target_col="diabetes")
                st.success(
                    f"Retraining complete. Saved as {result['version']} "
                    f"(Acc: {result['accuracy']:.3f}, "
                    f"Prec: {result['precision']:.3f}, "
                    f"Rec: {result['recall']:.3f})"
                )
            except Exception as e:
                st.error(f"Retraining failed: {e}")


# ---------- PREPARE INPUT ----------
def prepare_input():
    g = 1 if gender == "Male" else 0
    hyper = 1 if hypertension == "Yes" else 0
    heart = 1 if heart_disease == "Yes" else 0

    row = pd.DataFrame([{
        "gender": g,
        "age": age,
        "hypertension": hyper,
        "heart_disease": heart,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glucose,
        "smoking_history_current": 1 if smoking == "current" else 0,
        "smoking_history_former": 1 if smoking == "former" else 0,
        "smoking_history_unknown": 1 if smoking == "unknown" else 0
    }])
    return row


def get_zone(prob):
    if prob >= 0.60:
        return "High Risk — Diabetic", "💀", "#FF4B4B"
    elif prob >= 0.35:
        return "Borderline / Pre-Diabetic", "⚠", "#FFA500"
    else:
        return "Low Risk — Non-Diabetic", "💚", "#00B050"


# ---------- MAIN PREDICTION ----------
if predict:
    # raw (unscaled) for storage
    df_raw = prepare_input()
    df_scaled = df_raw.copy()

    df_scaled[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']] = scaler.transform(
        df_scaled[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
    )

    # probability of positive class
    try:
        prob = float(model.predict(df_scaled)[0][0])
    except Exception:
        prob = float(model.predict_proba(df_scaled)[:, 1][0])

    risk = prob * 100
    zone, emoji, color = get_zone(prob)
    pred_label = 1 if prob >= THRESHOLD else 0

    st.markdown(
        f"""
        <div class='prediction-box'>
            <h2>{emoji} <span class='text-glow'>Risk Score: {risk:.2f}%</span></h2>
            <h3 style='color:{color};'>{zone}</h3>
        </div>
        """, unsafe_allow_html=True
    )

    # ---------- Recommendations ----------
    st.markdown("<h2 class='text-glow'>🔮 AI Recommendations</h2>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    if "High" in zone:
        st.write("""
        - 🚨 Immediate medical follow-up required  
        - 🩸 Confirmatory HbA1c / FPG tests recommended  
        - 🥗 Begin structured diabetic diet  
        - 🏃 Daily activity: 30–45 min  
        - 🧘 Stress reduction essential  
        """)
    elif "Borderline" in zone:
        st.write("""
        - ⚠ Lifestyle modification recommended  
        - 🍎 Reduce sugar + refined carbs  
        - 🏃 Exercise 150 min/week  
        - 📅 Re-test HbA1c in 3 months  
        """)
    else:
        st.write("""
        - 💚 Maintain healthy habits  
        - 🥦 Balanced diet  
        - 🏃 Regular exercise  
        - 📅 Annual screening recommended  
        """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Clinician Action Panel ----------
    st.markdown("<h2 class='text-glow'>👨‍⚕ Clinician Action Panel</h2>", unsafe_allow_html=True)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.write("""
    - Document comorbidities  
    - Order confirmatory labs  
    - Evaluate metabolic risk factors  
    - Provide lifestyle education  
    - Schedule structured follow-ups  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---------- NEW: DOCTOR FEEDBACK + PRESCRIPTION ----------
    st.markdown("<h2 class='text-glow'>🧪 Doctor Feedback & Prescription</h2>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        feedback = st.radio(
            "Clinical evaluation of this prediction:",
            ["Not reviewed", "Correct", "Incorrect"],
            index=0
        )

    with col2:
        doctor_id = st.text_input(
            "Doctor ID (optional)",
            value="",
            help="Clinic/hospital identifier (not required)."
        )

    prescription_text = st.text_area(
        "Prescription / Clinical Notes (optional)",
        placeholder="e.g., Start metformin 500 mg OD; advise diet + exercise...",
        height=120
    )

    if st.button("💾 Save feedback & prescription (encrypted)"):
        if feedback == "Not reviewed":
            st.warning("Please mark this prediction as Correct or Incorrect before saving.")
        else:
            save_encrypted_feedback_record(
                patient_features=df_raw.iloc[0].to_dict(),
                pred_proba=prob,
                pred_label=pred_label,
                feedback=feedback,
                prescription_text=prescription_text,
                doctor_id=doctor_id or None
            )
            st.success("Feedback and prescription saved securely (encrypted).")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("🧠 Developed as part of the Federated Learning–Enabled Digital Twin for Predictive Diabetes Care project.")
    st.caption("⚠ This tool is for educational and research purposes only — not for clinical diagnosis.")
