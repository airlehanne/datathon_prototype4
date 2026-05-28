import pickle
from pathlib import Path

import pandas as pd
import streamlit as st


# ---------------------------------------------------
# File paths
# ---------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_CANDIDATES = [
    BASE_DIR / "brfss_model.pkl",
    BASE_DIR / "brfss_model(1).pkl",
]

SCALER_CANDIDATES = [
    BASE_DIR / "brfss_scaler.pkl",
    BASE_DIR / "brfss_scaler(1).pkl",
]

FEATURES_CANDIDATES = [
    BASE_DIR / "brfss_features.pkl",
    BASE_DIR / "brfss_features(1).pkl",
]


def find_existing_file(possible_paths, file_description):
    for path in possible_paths:
        if path.exists():
            return path

    st.error(
        f"Could not find the {file_description}. Please make sure the required .pkl "
        f"file is in the same folder as this Streamlit app."
    )
    st.stop()


MODEL_PATH = find_existing_file(MODEL_CANDIDATES, "saved model file")
SCALER_PATH = find_existing_file(SCALER_CANDIDATES, "saved scaler file")
FEATURES_PATH = find_existing_file(FEATURES_CANDIDATES, "saved feature list file")


@st.cache_resource
def load_saved_files():
    with open(MODEL_PATH, "rb") as file:
        model = pickle.load(file)

    with open(SCALER_PATH, "rb") as file:
        scaler = pickle.load(file)

    with open(FEATURES_PATH, "rb") as file:
        features = pickle.load(file)

    return model, scaler, features


model, scaler, FEATURES = load_saved_files()


# ---------------------------------------------------
# Page setup
# ---------------------------------------------------

st.set_page_config(
    page_title="Diabetes Health Insights",
    page_icon="🩺",
    layout="centered",
)


# ---------------------------------------------------
# Custom styling
# ---------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #B8C0CC;
        margin-bottom: 25px;
    }

    .result-card {
        padding: 24px;
        border-radius: 18px;
        background: #121826;
        border: 1px solid #2D3748;
        margin-bottom: 20px;
    }

    .success-card {
        padding: 24px;
        border-radius: 18px;
        background: #123D2A;
        border: 1px solid #2ECC71;
        margin-bottom: 20px;
    }

    .risk-card {
        padding: 24px;
        border-radius: 18px;
        background: #4A1F1F;
        border: 1px solid #FF6B6B;
        margin-bottom: 20px;
    }

    .metric-number {
        font-size: 46px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .section-label {
        color: #AAB2C0;
        font-size: 15px;
        margin-bottom: 5px;
    }

    .recommendation-box {
        background: #111827;
        padding: 18px;
        border-radius: 14px;
        border: 1px solid #374151;
        margin-bottom: 12px;
    }

    .timeline-box {
        background: #101820;
        padding: 16px;
        border-radius: 14px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 12px;
    }

    .small-muted {
        color: #AAB2C0;
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------

with st.sidebar:
    st.title("🩺 App Menu")
    st.write("Diabetes Health Insights")
    st.markdown("---")
    st.write("**Sections**")
    st.write("🏠 Home")
    st.write("📋 Assessment")
    st.write("📊 Results")
    st.write("💡 Guidance")
    st.write("ℹ️ About")
    st.markdown("---")
    st.caption(
        "Educational prototype only. Not intended to diagnose or replace clinical advice."
    )


# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------

def yes_no_input(label, help_text=None):
    choice = st.selectbox(label, ["No", "Yes"], help=help_text)
    return 1 if choice == "Yes" else 0


def coded_selectbox(label, options, help_text=None):
    selected_label = st.selectbox(label, list(options.keys()), help=help_text)
    return options[selected_label]


def convert_age_to_brfss_group(age):
    if age <= 24:
        return 1
    elif age <= 29:
        return 2
    elif age <= 34:
        return 3
    elif age <= 39:
        return 4
    elif age <= 44:
        return 5
    elif age <= 49:
        return 6
    elif age <= 54:
        return 7
    elif age <= 59:
        return 8
    elif age <= 64:
        return 9
    elif age <= 69:
        return 10
    elif age <= 74:
        return 11
    elif age <= 79:
        return 12
    else:
        return 13


def build_recommendations(inputs, probability_diabetes):
    recommendations = []

    if inputs["HighBP"] == 1:
        recommendations.append(
            "🩺 You reported high blood pressure. Regular monitoring and clinical follow-up may help reduce longer-term health risk."
        )

    if inputs["HighChol"] == 1:
        recommendations.append(
            "❤️ You reported high cholesterol. Managing cholesterol can support both cardiovascular and metabolic health."
        )

    if inputs["CholCheck"] == 0:
        recommendations.append(
            "🧪 You have not had a cholesterol check in the past 5 years. A check-up may help identify risk factors earlier."
        )

    if inputs["BMI"] >= 30:
        recommendations.append(
            "⚖️ Your BMI is in the obese range. Gradual lifestyle changes and professional support may help reduce diabetes risk."
        )
    elif inputs["BMI"] >= 25:
        recommendations.append(
            "⚖️ Your BMI is in the overweight range. Small, sustainable diet and physical activity changes may help reduce future risk."
        )

    if inputs["HeartDiseaseorAttack"] == 1:
        recommendations.append(
            "❤️ You reported a history of heart disease or heart attack. Personalised medical advice may be important."
        )

    if inputs["PhysActivity"] == 0:
        recommendations.append(
            "🏃 You reported no physical activity outside regular work in the past month. Adding manageable activity, such as walking, may help lower risk."
        )

    if inputs["GenHlth"] >= 4:
        recommendations.append(
            "🩺 You rated your general health as fair or poor. A general health review may help identify areas for support."
        )

    if probability_diabetes >= 0.60:
        recommendations.append(
            "📌 Because the model classified you into the diabetes / pre-diabetes group, consider using this result as a prompt to seek medical advice."
        )

    if not recommendations:
        recommendations.append(
            "✅ Your responses do not strongly trigger any specific recommendation. Continue regular check-ups, physical activity, and healthy lifestyle habits."
        )

    return recommendations


def build_contributing_factors(inputs):
    factors = []

    if inputs["BMI"] >= 30:
        factors.append(("BMI", "Obese range", 90))
    elif inputs["BMI"] >= 25:
        factors.append(("BMI", "Overweight range", 70))

    if inputs["PhysActivity"] == 0:
        factors.append(("Physical activity", "No activity reported in past month", 75))

    if inputs["HighBP"] == 1:
        factors.append(("Blood pressure", "High blood pressure reported", 70))

    if inputs["HighChol"] == 1:
        factors.append(("Cholesterol", "High cholesterol reported", 65))

    if inputs["CholCheck"] == 0:
        factors.append(("Preventative screening", "No cholesterol check in past 5 years", 55))

    if inputs["HeartDiseaseorAttack"] == 1:
        factors.append(("Heart health", "Heart disease or heart attack history reported", 80))

    if inputs["GenHlth"] >= 4:
        factors.append(("General health", "Fair or poor health reported", 60))

    if not factors:
        factors.append(("Overall profile", "No major lifestyle or clinical flags entered", 30))

    return factors


def create_summary_text(predicted_class, confidence, recommendations, factors):
    factor_text = "\n".join(
        [f"- {name}: {description}" for name, description, score in factors]
    )

    recommendation_text = "\n".join([f"- {rec}" for rec in recommendations])

    return f"""
Diabetes Health Insights - Result Summary

Predicted classification:
{predicted_class}

Model confidence:
{confidence:.1f}%

Key contributing factors:
{factor_text}

Personalised guidance:
{recommendation_text}

Disclaimer:
This prototype is intended for educational and screening purposes only. It does not replace professional medical advice, diagnosis, or treatment.
"""


# ---------------------------------------------------
# Header
# ---------------------------------------------------

st.markdown(
    '<div class="main-title">🩺 Diabetes Health Insights</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">A patient-facing AI prototype for diabetes screening, lifestyle insight, and prevention support.</div>',
    unsafe_allow_html=True,
)

st.info(
    "This tool is for educational/demo purposes only. It does not diagnose diabetes "
    "and does not replace advice from a doctor or qualified health professional."
)


# ---------------------------------------------------
# How it works
# ---------------------------------------------------

st.subheader("How It Works")

step1, step2, step3 = st.columns(3)

with step1:
    st.markdown("### 1️⃣ Enter")
    st.caption("Provide lifestyle and health information.")

with step2:
    st.markdown("### 2️⃣ Analyse")
    st.caption("The trained model compares your inputs with learned diabetes patterns.")

with step3:
    st.markdown("### 3️⃣ Guide")
    st.caption("Receive a predicted classification and personalised guidance.")


# ---------------------------------------------------
# Inputs
# ---------------------------------------------------

st.header("Patient Information")

st.caption(
    "Answer the questions below to receive a predicted classification, confidence score, and personalised guidance."
)

col1, col2 = st.columns(2)

with col1:
    HighBP = yes_no_input(
        "High blood pressure",
        "Have you ever been told by a doctor, nurse, or other health professional that you have high blood pressure?",
    )

    HighChol = yes_no_input(
        "High cholesterol",
        "Have you ever been told by a doctor, nurse, or other health professional that your cholesterol is high?",
    )

    CholCheck = yes_no_input(
        "Cholesterol checked in the past 5 years",
        "Have you had your cholesterol checked in the past 5 years?",
    )

    BMI = st.slider(
        "BMI",
        min_value=10.0,
        max_value=70.0,
        value=25.0,
        step=0.1,
        help="Body Mass Index calculated from height and weight in kg/m².",
    )

    HeartDiseaseorAttack = yes_no_input(
        "Heart disease or heart attack",
        "Have you ever been told you had coronary heart disease or a heart attack?",
    )

with col2:
    PhysActivity = yes_no_input(
        "Physical activity in the past month",
        "During the past month, other than your regular job, did you participate in physical activities or exercises?",
    )

    GenHlth = coded_selectbox(
        "General health",
        {
            "Excellent": 1,
            "Very good": 2,
            "Good": 3,
            "Fair": 4,
            "Poor": 5,
        },
    )

    Sex = coded_selectbox(
        "Sex",
        {
            "Female": 0,
            "Male": 1,
        },
    )

    actual_age = st.slider(
        "Age",
        min_value=0,
        max_value=100,
        value=30,
        step=1,
    )

    Age = convert_age_to_brfss_group(actual_age)


required_inputs = {
    "HighBP": HighBP,
    "HighChol": HighChol,
    "CholCheck": CholCheck,
    "BMI": BMI,
    "HeartDiseaseorAttack": HeartDiseaseorAttack,
    "PhysActivity": PhysActivity,
    "GenHlth": GenHlth,
    "Sex": Sex,
    "Age": Age,
}


default_model_values = {
    "Smoker": 0,
    "Stroke": 0,
    "NoDocbcCost": 0,
    "MentHlth": 0,
    "PhysHlth": 0,
    "DiffWalk": 0,
    "Education": 4,
    "Income": 4,
}


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

if st.button("Assess Diabetes Risk", use_container_width=True):

    input_values = {**default_model_values, **required_inputs}

    input_data = pd.DataFrame(
        [{feature: input_values.get(feature, 0) for feature in FEATURES}]
    )

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    prediction_probabilities = model.predict_proba(input_scaled)[0]

    probability_non_diabetes = prediction_probabilities[0]
    probability_diabetes = prediction_probabilities[1]

    if prediction == 1:
        predicted_class = "Diabetes / Pre-diabetes Group"
        display_prediction = "Higher Diabetes Likelihood Detected"
        model_confidence = probability_diabetes * 100
        risk_score = probability_diabetes * 100
    else:
        predicted_class = "Non-Diabetes Group"
        display_prediction = "Current Indicators Suggest Lower Diabetes Likelihood"
        model_confidence = probability_non_diabetes * 100
        risk_score = probability_diabetes * 100

    recommendations = build_recommendations(required_inputs, probability_diabetes)
    contributing_factors = build_contributing_factors(required_inputs)

    st.markdown("---")
    st.header("📊 Assessment Results")

    # Result card
    if prediction == 1:
        st.markdown(
            f"""
            <div class="risk-card">
                <div class="section-label">Predicted Classification</div>
                <div class="metric-number">⚠️ {display_prediction}</div>
                <p>{predicted_class}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="success-card">
                <div class="section-label">Predicted Classification</div>
                <div class="metric-number">✅ {display_prediction}</div>
                <p>{predicted_class}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Risk meter
    st.subheader("Visual Risk Meter")
    st.write(f"Estimated diabetes likelihood: **{risk_score:.1f}%**")
    st.progress(risk_score / 100)

    # Confidence
    st.subheader("Model Confidence")
    st.write(f"Model confidence in this classification: **{model_confidence:.1f}%**")
    st.progress(model_confidence / 100)

    # What this means
    st.subheader("What This Means")

    if prediction == 1:
        st.write(
            "Your current health and lifestyle indicators are more similar to patterns seen in the diabetes / pre-diabetes group. "
            "This does not mean you have diabetes, but it may be useful as a prompt for professional follow-up."
        )
    else:
        st.write(
            "Your current health and lifestyle indicators are more similar to patterns seen in the non-diabetes group. "
            "This does not rule out diabetes, but it suggests lower likelihood based on the information entered."
        )

    # Key contributing factors
    st.subheader("Why This Result Was Predicted")

    st.caption(
        "These are simple explainability indicators based on the user inputs and known risk-related features."
    )

    for factor_name, factor_description, factor_strength in contributing_factors:
        st.write(f"**{factor_name}:** {factor_description}")
        st.progress(factor_strength / 100)

    # Disclaimer
    st.warning(
        "This prototype is intended for educational and screening purposes only and does not replace professional medical advice, diagnosis, or treatment. "
        "Please consult a qualified healthcare professional for personal medical guidance."
    )

    # Personalised guidance
    st.subheader("💡 Personalised Guidance")

    for recommendation in recommendations:
        st.markdown(
            f"""
            <div class="recommendation-box">
                {recommendation}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Timeline
    st.subheader("Recommended Next Steps")

    st.markdown(
        """
        <div class="timeline-box">
            <b>This week:</b><br>
            Review your result and identify one manageable lifestyle change, such as increasing daily walking.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="timeline-box">
            <b>Next 1–3 months:</b><br>
            Monitor changes in physical activity, weight, blood pressure, or other relevant health indicators.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="timeline-box">
            <b>Within 6 months:</b><br>
            Consider a routine health check, especially if multiple risk factors were identified.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Download summary
    summary_text = create_summary_text(
        predicted_class,
        model_confidence,
        recommendations,
        contributing_factors,
    )

    st.download_button(
        label="Download Result Summary",
        data=summary_text,
        file_name="diabetes_result_summary.txt",
        mime="text/plain",
        use_container_width=True,
    )

    # Debug section
    with st.expander("Show model input values"):
        st.write(
            "The app asks for actual age from 0 to 100, but the model uses grouped BRFSS age codes."
        )
        st.write(f"Actual age entered: {actual_age}")
        st.write(f"Converted BRFSS age group code: {Age}")
        st.dataframe(input_data)