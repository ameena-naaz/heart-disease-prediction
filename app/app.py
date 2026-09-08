import streamlit as st
import pandas as pd
import joblib
import base64
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent

MODEL_PATH = PROJECT_DIR / "models" / "heart_disease_model.pkl"

HEART_PATH_1 = APP_DIR / "heart_real.png"
HEART_PATH_2 = APP_DIR / "assets" / "heart_real.png"


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as error:
    st.error(f"Model could not be loaded: {error}")
    st.stop()


# ============================================================
# LOAD HEART SVG
# ============================================================

def load_heart():

    if HEART_PATH_1.exists():
        heart_path = HEART_PATH_1

    elif HEART_PATH_2.exists():
        heart_path = HEART_PATH_2

    else:
        return None

    with open(heart_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")

    return f"data:image/png;base64,{encoded}"


heart_src = load_heart()


# ============================================================
# SESSION STATE
# ============================================================

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "probability" not in st.session_state:
    st.session_state.probability = None

if "patient_data" not in st.session_state:
    st.session_state.patient_data = None


# ============================================================
# GLOBAL CSS
# ============================================================

st.html(
    """
<style>

/* =========================================================
   BASE
========================================================= */

.stApp {
    background:
        radial-gradient(
            circle at 78% 18%,
            rgba(255, 73, 73, 0.10),
            transparent 30%
        ),
        radial-gradient(
            circle at 45% 88%,
            rgba(54, 108, 255, 0.06),
            transparent 35%
        ),
        #060A11;

    color: #F5F7FA;
}

.block-container {
    max-width: 1480px;
    padding-top: 2.4rem;
    padding-bottom: 4rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

#MainMenu,
footer {
    visibility: hidden;
}


/* =========================================================
   SIDEBAR
========================================================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #080C13 0%,
            #070A10 100%
        );

    border-right: 1px solid rgba(255,255,255,0.07);
}

.brand {
    color: #F7F8FA;
    font-size: 25px;
    font-weight: 850;
    margin-bottom: 2px;
}

.brand-sub {
    color: #FF6257;
    font-size: 10px;
    font-weight: 850;
    letter-spacing: 2.3px;
    margin-bottom: 30px;
}

.sidebar-label {
    color: #586476;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2px;
    margin-bottom: 10px;
}


/* =========================================================
   TYPOGRAPHY
========================================================= */

.eyebrow-red {
    color: #FF6B61;
    font-size: 12px;
    letter-spacing: 3px;
    font-weight: 850;
    margin-bottom: 17px;
}

.eyebrow-cyan {
    color: #67E8F9;
    font-size: 12px;
    letter-spacing: 3px;
    font-weight: 850;
    margin-bottom: 17px;
}

.hero-title {
    color: #F7F8FA;
    font-size: clamp(48px, 5vw, 67px);
    line-height: 1.04;
    font-weight: 850;
    letter-spacing: -2px;
    margin-bottom: 22px;
}

.page-title {
    color: #F5F7FA;
    font-size: 43px;
    line-height: 1.08;
    font-weight: 850;
    margin-bottom: 14px;
}

.body-copy {
    color: #98A2B3;
    font-size: 16px;
    line-height: 1.75;
    max-width: 760px;
}


/* =========================================================
   OVERVIEW METRICS
========================================================= */

.metric-box {
    background:
        linear-gradient(
            145deg,
            rgba(18, 27, 42, 0.96),
            rgba(9, 15, 25, 0.97)
        );

    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;

    padding: 24px;

    min-height: 145px;

    box-shadow:
        0 18px 45px rgba(0,0,0,0.18);
}

.metric-label {
    color: #98A2B3;
    font-size: 10px;
    font-weight: 850;
    letter-spacing: 1.8px;
}

.metric-value {
    color: #F5F7FA;
    font-size: 34px;
    font-weight: 850;
    margin-top: 12px;
}

.metric-note {
    color: #667085;
    font-size: 12px;
    margin-top: 6px;
}


/* =========================================================
   HEART STAGE
========================================================= */

.heart-stage {
    min-height: 485px;

    display: flex;
    align-items: center;
    justify-content: center;

    position: relative;

    perspective: 900px;

    overflow: visible;
}

.heart-orbit {
    position: absolute;

    width: 390px;
    height: 390px;

    border-radius: 50%;

    border: 1px solid rgba(255,98,87,0.12);

    box-shadow:
        0 0 60px rgba(255,98,87,0.08),
        inset 0 0 60px rgba(255,98,87,0.04);

    animation:
        orbitPulse 3s ease-in-out infinite;
}

.heart-glow {
    position: absolute;

    width: 320px;
    height: 320px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(255,98,87,0.24),
            rgba(255,98,87,0.09) 35%,
            rgba(255,98,87,0.025) 58%,
            transparent 73%
        );

    filter: blur(5px);

    animation:
        glowPulse 1.35s ease-in-out infinite;
}

.heart-3d-wrapper {
    position: relative;

    z-index: 3;

    transform-style: preserve-3d;

    animation:
        heart3D 7s ease-in-out infinite,
        heartbeat 1.35s ease-in-out infinite;
}

.heart-3d {
    width: 300px;

    max-height: 380px;

    object-fit: contain;

    transform-style: preserve-3d;

    filter:
        drop-shadow(18px 18px 16px rgba(0,0,0,0.55))
        drop-shadow(-8px -5px 15px rgba(255,130,120,0.10))
        drop-shadow(0 0 35px rgba(255,98,87,0.32))
        drop-shadow(0 0 85px rgba(255,98,87,0.14));
}

.heart-shadow {
    position: absolute;

    width: 240px;
    height: 55px;

    bottom: 48px;

    background:
        radial-gradient(
            ellipse,
            rgba(255,98,87,0.18),
            rgba(0,0,0,0) 70%
        );

    filter: blur(8px);

    animation:
        shadowPulse 1.35s ease-in-out infinite;
}


@keyframes heartbeat {

    0% {
        scale: 1;
    }

    10% {
        scale: 1.055;
    }

    19% {
        scale: 1;
    }

    28% {
        scale: 1.035;
    }

    39% {
        scale: 1;
    }

    100% {
        scale: 1;
    }
}


@keyframes heart3D {

    0% {
        transform:
            rotateX(-4deg)
            rotateY(-13deg)
            rotateZ(-2deg)
            translateY(0px);
    }

    25% {
        transform:
            rotateX(2deg)
            rotateY(4deg)
            rotateZ(1deg)
            translateY(-7px);
    }

    50% {
        transform:
            rotateX(-2deg)
            rotateY(15deg)
            rotateZ(2deg)
            translateY(0px);
    }

    75% {
        transform:
            rotateX(3deg)
            rotateY(3deg)
            rotateZ(-1deg)
            translateY(-6px);
    }

    100% {
        transform:
            rotateX(-4deg)
            rotateY(-13deg)
            rotateZ(-2deg)
            translateY(0px);
    }
}


@keyframes glowPulse {

    0%,
    100% {
        transform: scale(0.93);
        opacity: 0.62;
    }

    14% {
        transform: scale(1.08);
        opacity: 1;
    }

    30% {
        transform: scale(0.98);
        opacity: 0.78;
    }

    100% {
        transform: scale(0.93);
        opacity: 0.62;
    }
}


@keyframes orbitPulse {

    0%,
    100% {
        transform: scale(0.96);
        opacity: 0.5;
    }

    50% {
        transform: scale(1.04);
        opacity: 0.9;
    }
}


@keyframes shadowPulse {

    0%,
    100% {
        transform: scaleX(0.90);
        opacity: 0.55;
    }

    15% {
        transform: scaleX(1.05);
        opacity: 0.8;
    }

    100% {
        transform: scaleX(0.90);
        opacity: 0.55;
    }
}


/* =========================================================
   RESULT CARD
========================================================= */

.result-card {
    padding: 30px;

    border-radius: 20px;

    margin-top: 26px;

    box-shadow:
        0 18px 60px rgba(0,0,0,0.24);
}

.result-small {
    font-size: 11px;
    font-weight: 850;
    letter-spacing: 2px;
    margin-bottom: 12px;
}

.result-title {
    color: #F5F7FA;
    font-size: 28px;
    font-weight: 850;
}

.result-percent {
    font-size: 49px;
    font-weight: 900;
    margin-top: 12px;
}

.result-caption {
    color: #98A2B3;
    font-size: 12px;
}

.progress-track {
    width: 100%;
    height: 9px;

    background: rgba(255,255,255,0.08);

    border-radius: 20px;

    overflow: hidden;

    margin: 18px 0;
}

.result-text {
    color: #D0D5DD;
    font-size: 14px;
    line-height: 1.75;
}


/* =========================================================
   MODEL PERFORMANCE
========================================================= */

.performance-card {
    background:
        linear-gradient(
            145deg,
            rgba(18,27,42,0.96),
            rgba(9,15,25,0.98)
        );

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 18px;

    padding: 23px;

    min-height: 130px;
}

.performance-number {
    color: #F5F7FA;
    font-size: 32px;
    font-weight: 850;
    margin-top: 10px;
}


/* =========================================================
   MODEL COMPARISON
========================================================= */

.model-card {
    background:
        linear-gradient(
            145deg,
            rgba(17,25,39,0.95),
            rgba(9,15,24,0.97)
        );

    border: 1px solid rgba(255,255,255,0.08);

    border-radius: 17px;

    padding: 20px;

    margin-bottom: 13px;
}

.model-card-selected {
    border:
        1px solid rgba(103,232,249,0.38);

    box-shadow:
        0 0 28px rgba(103,232,249,0.06);
}

.model-name {
    color: #F5F7FA;
    font-size: 17px;
    font-weight: 800;
}

.selected-pill {
    display: inline-block;

    background: rgba(103,232,249,0.12);

    color: #67E8F9;

    border:
        1px solid rgba(103,232,249,0.24);

    border-radius: 30px;

    padding: 4px 9px;

    font-size: 9px;

    letter-spacing: 1.3px;

    font-weight: 800;

    margin-left: 8px;
}

.model-data {
    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 12px;

    margin-top: 18px;
}

.model-data-label {
    color: #667085;
    font-size: 10px;
    letter-spacing: 1px;
}

.model-data-value {
    color: #D0D5DD;
    font-size: 16px;
    font-weight: 750;
    margin-top: 3px;
}


/* =========================================================
   CONFUSION MATRIX
========================================================= */

.matrix-wrapper {
    background:
        linear-gradient(
            145deg,
            rgba(17,25,39,0.95),
            rgba(9,15,24,0.97)
        );

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius: 18px;

    padding: 24px;
}

.matrix-heading {
    display: grid;

    grid-template-columns:
        150px 1fr 1fr;

    gap: 8px;

    margin-bottom: 8px;
}

.matrix-grid {
    display: grid;

    grid-template-columns:
        150px 1fr 1fr;

    gap: 8px;
}

.matrix-label {
    display: flex;

    align-items: center;

    color: #98A2B3;

    font-size: 11px;

    padding: 15px 8px;
}

.matrix-top-label {
    text-align: center;

    color: #667085;

    font-size: 10px;

    letter-spacing: 0.8px;
}

.matrix-cell {
    border-radius: 13px;

    padding: 24px 10px;

    text-align: center;
}

.matrix-correct {
    background:
        rgba(50,213,131,0.10);

    border:
        1px solid rgba(50,213,131,0.22);
}

.matrix-error {
    background:
        rgba(255,98,87,0.08);

    border:
        1px solid rgba(255,98,87,0.18);
}

.matrix-number {
    color: #F5F7FA;
    font-size: 29px;
    font-weight: 850;
}

.matrix-sub {
    color: #667085;
    font-size: 10px;
    margin-top: 4px;
}


/* =========================================================
   ABOUT
========================================================= */

.workflow-card {
    background:
        linear-gradient(
            145deg,
            rgba(17,25,39,0.94),
            rgba(9,15,24,0.97)
        );

    border:
        1px solid rgba(255,255,255,0.08);

    border-radius: 18px;

    padding: 25px;
}

.workflow-line {
    color: #D0D5DD;
    font-size: 14px;
    line-height: 2;
}


/* =========================================================
   FORM
========================================================= */

.stButton > button,
[data-testid="stFormSubmitButton"] > button {

    width: 100%;

    min-height: 48px;

    border-radius: 12px;

    color: white;

    font-weight: 800;

    border:
        1px solid rgba(255,98,87,0.55);

    background:
        linear-gradient(
            90deg,
            #FF6257,
            #FF746B
        );
}


/* =========================================================
   RESPONSIVE
========================================================= */

@media (max-width: 900px) {

    .hero-title {
        font-size: 42px;
    }

    .page-title {
        font-size: 34px;
    }

    .heart-stage {
        min-height: 340px;
    }

    .heart-3d {
        width: 220px;
    }

    .heart-orbit {
        width: 290px;
        height: 290px;
    }

    .heart-glow {
        width: 250px;
        height: 250px;
    }

}

</style>
"""
)


# ============================================================
# HELPERS
# ============================================================

def page_header(
    label,
    title,
    description,
    cyan=False
):

    css_class = (
        "eyebrow-cyan"
        if cyan
        else "eyebrow-red"
    )

    st.html(
        f"""
<div style="padding-top:25px;">

    <div class="{css_class}">
        {label}
    </div>

    <div class="page-title">
        {title}
    </div>

    <div class="body-copy">
        {description}
    </div>

</div>
"""
    )


def show_result(
    prediction,
    probability
):

    risk_percent = probability * 100


    if prediction == 1:

        title = "Higher Predicted Risk"

        icon = "⚠"

        color = "#FF6257"

        background = "rgba(255,98,87,0.10)"

        text = (
            "The submitted cardiovascular indicators produced "
            "a pattern associated with a higher predicted "
            "likelihood of heart disease."
        )


    else:

        title = "Lower Predicted Risk"

        icon = "✓"

        color = "#32D583"

        background = "rgba(50,213,131,0.10)"

        text = (
            "The submitted cardiovascular indicators produced "
            "a pattern associated with a lower predicted "
            "likelihood of heart disease."
        )


    st.html(
        f"""
<div
    class="result-card"
    style="
        background:{background};
        border:1px solid {color};
    "
>

    <div
        class="result-small"
        style="color:{color};"
    >
        PREDICTION RESULT
    </div>

    <div class="result-title">
        {icon} {title}
    </div>

    <div
        class="result-percent"
        style="color:{color};"
    >
        {risk_percent:.1f}%
    </div>

    <div class="result-caption">
        Predicted probability of heart disease
    </div>

    <div class="progress-track">

        <div style="
            width:{risk_percent:.1f}%;
            height:100%;
            background:{color};
            border-radius:20px;
        ">
        </div>

    </div>

    <div class="result-text">
        {text}
    </div>

</div>
"""
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
<div class="brand">
    ❤️ CardioSense
</div>

<div class="brand-sub">
    HEART RISK PREDICTION
</div>

<div class="sidebar-label">
    WORKSPACE
</div>
"""
    )


    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Patient Assessment",
            "Prediction",
            "Model Insights",
            "About"
        ],
        label_visibility="collapsed"
    )


    st.divider()


    st.caption(
        "UCI Heart Disease Dataset"
    )

    st.caption(
        "Cleveland subset · 303 records"
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    left, right = st.columns(
        [1.15, 0.85],
        gap="large"
    )


    with left:

        st.html(
            """
<div style="padding-top:78px;">

    <div class="eyebrow-red">
        HEART DISEASE RISK PREDICTION
    </div>

    <div class="hero-title">
        Data-driven insights for<br>
        heart risk prediction.
    </div>

    <div class="body-copy">

        Analyze key cardiovascular indicators using
        machine learning and explore how patient data
        contributes to predicted heart disease risk.

    </div>

</div>
"""
        )


        st.write("")
        st.write("")


        c1, c2, c3 = st.columns(3)


        with c1:

            st.html(
                """
<div class="metric-box">

    <div class="metric-label">
        DATASET
    </div>

    <div class="metric-value">
        303
    </div>

    <div class="metric-note">
        Patient records
    </div>

</div>
"""
            )


        with c2:

            st.html(
                """
<div class="metric-box">

    <div class="metric-label">
        MODEL INPUTS
    </div>

    <div class="metric-value">
        13
    </div>

    <div class="metric-note">
        Cardiovascular features
    </div>

</div>
"""
            )


        with c3:

            st.html(
                """
<div class="metric-box">

    <div class="metric-label">
        ROC-AUC
    </div>

    <div class="metric-value">
        0.951
    </div>

    <div class="metric-note">
        Held-out test set
    </div>

</div>
"""
            )


    with right:


        if heart_src is not None:

            st.html(
                f"""
<div class="heart-stage">

    <div class="heart-orbit"></div>

    <div class="heart-glow"></div>

    <div class="heart-shadow"></div>

    <div class="heart-3d-wrapper">

        <img
            src="{heart_src}"
            class="heart-3d"
        >

    </div>

</div>

<div style="
    text-align:center;
    color:#667085;
    letter-spacing:2px;
    font-size:10px;
    font-weight:800;
">
    CARDIOVASCULAR ANALYSIS
</div>
"""
            )


        else:

            st.error(
                "heart.svg was not found in the app folder."
            )


# ============================================================
# PATIENT ASSESSMENT
# ============================================================

elif page == "Patient Assessment":

    page_header(
        "PATIENT ASSESSMENT",
        "Cardiovascular Assessment",
        (
            "Enter the patient's cardiovascular indicators "
            "below to generate a machine-learning prediction."
        )
    )


    st.write("")


    with st.form("assessment_form"):


        st.subheader("Patient Profile")


        c1, c2, c3 = st.columns(3)


        with c1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=54,
                step=1
            )


        with c2:

            sex_label = st.selectbox(
                "Sex",
                [
                    "Female",
                    "Male"
                ]
            )

            sex = (
                0
                if sex_label == "Female"
                else 1
            )


        with c3:

            cp_label = st.selectbox(
                "Chest Pain Type",
                [
                    "Typical angina",
                    "Atypical angina",
                    "Non-anginal pain",
                    "Asymptomatic"
                ]
            )


            cp = {
                "Typical angina": 1,
                "Atypical angina": 2,
                "Non-anginal pain": 3,
                "Asymptomatic": 4
            }[cp_label]


        st.subheader(
            "Clinical Measurements"
        )


        c1, c2, c3 = st.columns(3)


        with c1:

            trestbps = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=70,
                max_value=250,
                value=130,
                step=1
            )


        with c2:

            chol = st.number_input(
                "Serum Cholesterol (mg/dl)",
                min_value=80,
                max_value=700,
                value=240,
                step=1
            )


        with c3:

            fbs_label = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dl",
                [
                    "No",
                    "Yes"
                ]
            )

            fbs = (
                0
                if fbs_label == "No"
                else 1
            )


        st.subheader(
            "Cardiac & Exercise Indicators"
        )


        c1, c2, c3 = st.columns(3)


        with c1:

            restecg_label = st.selectbox(
                "Resting ECG",
                [
                    "Normal",
                    "ST-T wave abnormality",
                    "Left ventricular hypertrophy"
                ]
            )


            restecg = {
                "Normal": 0,
                "ST-T wave abnormality": 1,
                "Left ventricular hypertrophy": 2
            }[restecg_label]


        with c2:

            thalach = st.number_input(
                "Maximum Heart Rate Achieved",
                min_value=60,
                max_value=230,
                value=150,
                step=1
            )


        with c3:

            exang_label = st.selectbox(
                "Exercise-Induced Angina",
                [
                    "No",
                    "Yes"
                ]
            )

            exang = (
                0
                if exang_label == "No"
                else 1
            )


        st.subheader(
            "Advanced Indicators"
        )


        c1, c2, c3, c4 = st.columns(4)


        with c1:

            oldpeak = st.number_input(
                "ST Depression (Oldpeak)",
                min_value=0.0,
                max_value=7.0,
                value=1.0,
                step=0.1
            )


        with c2:

            slope_label = st.selectbox(
                "Slope of Peak Exercise ST",
                [
                    "Upsloping",
                    "Flat",
                    "Downsloping"
                ]
            )


            slope = {
                "Upsloping": 1,
                "Flat": 2,
                "Downsloping": 3
            }[slope_label]


        with c3:

            ca = st.selectbox(
                "Number of Major Vessels",
                [
                    0,
                    1,
                    2,
                    3
                ]
            )


        with c4:

            thal_label = st.selectbox(
                "Thalassemia",
                [
                    "Normal",
                    "Fixed defect",
                    "Reversible defect"
                ]
            )


            thal = {
                "Normal": 3,
                "Fixed defect": 6,
                "Reversible defect": 7
            }[thal_label]


        analyze = st.form_submit_button(
            "Analyze Heart Risk →"
        )


    if analyze:


        patient_df = pd.DataFrame(
            [
                {
                    "age": age,
                    "sex": sex,
                    "cp": cp,
                    "trestbps": trestbps,
                    "chol": chol,
                    "fbs": fbs,
                    "restecg": restecg,
                    "thalach": thalach,
                    "exang": exang,
                    "oldpeak": oldpeak,
                    "slope": slope,
                    "ca": ca,
                    "thal": thal
                }
            ]
        )


        try:


            prediction = int(
                model.predict(
                    patient_df
                )[0]
            )


            probability = float(
                model.predict_proba(
                    patient_df
                )[0][1]
            )


            st.session_state.prediction = (
                prediction
            )

            st.session_state.probability = (
                probability
            )

            st.session_state.patient_data = (
                patient_df
                .iloc[0]
                .to_dict()
            )


        except Exception as error:


            st.error(
                f"Prediction failed: {error}"
            )

            st.stop()


    if (
        st.session_state.prediction
        is not None
    ):


        show_result(
            st.session_state.prediction,
            st.session_state.probability
        )


# ============================================================
# PREDICTION PAGE
# ============================================================

elif page == "Prediction":


    page_header(
        "PREDICTION ANALYSIS",
        "Your Heart Risk Result",
        (
            "Review the prediction generated from "
            "the submitted cardiovascular indicators."
        ),
        cyan=True
    )


    if (
        st.session_state.prediction
        is None
    ):


        st.info(
            "Complete the Patient Assessment "
            "to generate a prediction."
        )


    else:


        show_result(
            st.session_state.prediction,
            st.session_state.probability
        )


        patient = (
            st.session_state.patient_data
        )


        st.write("")
        st.write("")


        st.subheader(
            "Submitted Cardiovascular Profile"
        )


        c1, c2, c3, c4 = st.columns(4)


        c1.metric(
            "Age",
            int(patient["age"])
        )


        c2.metric(
            "Resting BP",
            f'{int(patient["trestbps"])} mm Hg'
        )


        c3.metric(
            "Cholesterol",
            f'{int(patient["chol"])} mg/dl'
        )


        c4.metric(
            "Max Heart Rate",
            int(patient["thalach"])
        )


# ============================================================
# MODEL INSIGHTS
# ============================================================

elif page == "Model Insights":


    page_header(
        "MODEL PERFORMANCE",
        "Model Performance & Evaluation",
        (
            "Compare the evaluated machine-learning models "
            "and inspect the classification results of the "
            "selected Logistic Regression model."
        ),
        cyan=True
    )


    st.write("")
    st.write("")


    c1, c2, c3, c4 = st.columns(4)


    with c1:

        st.html(
            """
<div class="performance-card">

    <div class="metric-label">
        TEST ACCURACY
    </div>

    <div class="performance-number">
        86.89%
    </div>

    <div class="metric-note">
        Logistic Regression
    </div>

</div>
"""
        )


    with c2:

        st.html(
            """
<div class="performance-card">

    <div class="metric-label">
        ROC-AUC
    </div>

    <div class="performance-number">
        0.951
    </div>

    <div class="metric-note">
        Held-out test set
    </div>

</div>
"""
        )


    with c3:

        st.html(
            """
<div class="performance-card">

    <div class="metric-label">
        POSITIVE RECALL
    </div>

    <div class="performance-number">
        93%
    </div>

    <div class="metric-note">
        Heart-disease class
    </div>

</div>
"""
        )


    with c4:

        st.html(
            """
<div class="performance-card">

    <div class="metric-label">
        5-FOLD CV
    </div>

    <div class="performance-number">
        82.83%
    </div>

    <div class="metric-note">
        ± 3.60%
    </div>

</div>
"""
        )


    st.write("")
    st.write("")


    left, right = st.columns(
        [1, 1],
        gap="large"
    )


    # --------------------------------------------------------
    # MODEL COMPARISON
    # --------------------------------------------------------

    with left:


        st.subheader(
            "Model Comparison"
        )


        st.html(
            """
<div class="
    model-card
    model-card-selected
">

    <div class="model-name">

        Logistic Regression

        <span class="selected-pill">
            SELECTED MODEL
        </span>

    </div>


    <div class="model-data">

        <div>

            <div class="model-data-label">
                TEST ACCURACY
            </div>

            <div class="model-data-value">
                86.89%
            </div>

        </div>


        <div>

            <div class="model-data-label">
                5-FOLD CV
            </div>

            <div class="model-data-value">
                82.83%
            </div>

        </div>


        <div>

            <div class="model-data-label">
                CV STD
            </div>

            <div class="model-data-value">
                3.60%
            </div>

        </div>

    </div>

</div>


<div class="model-card">

    <div class="model-name">
        Random Forest
    </div>

    <div class="model-data">

        <div>

            <div class="model-data-label">
                TEST ACCURACY
            </div>

            <div class="model-data-value">
                90.16%
            </div>

        </div>


        <div>

            <div class="model-data-label">
                5-FOLD CV
            </div>

            <div class="model-data-value">
                81.49%
            </div>

        </div>


        <div>

            <div class="model-data-label">
                CV STD
            </div>

            <div class="model-data-value">
                5.26%
            </div>

        </div>

    </div>

</div>


<div class="model-card">

    <div class="model-name">
        Decision Tree
    </div>

    <div class="model-data">

        <div>

            <div class="model-data-label">
                TEST ACCURACY
            </div>

            <div class="model-data-value">
                78.69%
            </div>

        </div>


        <div>

            <div class="model-data-label">
                5-FOLD CV
            </div>

            <div class="model-data-value">
                76.89%
            </div>

        </div>


        <div>

            <div class="model-data-label">
                CV STD
            </div>

            <div class="model-data-value">
                7.13%
            </div>

        </div>

    </div>

</div>
"""
        )


        st.caption(
            "Logistic Regression was selected "
            "for its stronger cross-validation "
            "stability and interpretability."
        )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    with right:


        st.subheader(
            "Confusion Matrix"
        )


        st.html(
            """
<div class="matrix-wrapper">

    <div class="matrix-heading">

        <div></div>

        <div class="matrix-top-label">
            PREDICTED<br>
            NO DISEASE
        </div>

        <div class="matrix-top-label">
            PREDICTED<br>
            DISEASE
        </div>

    </div>


    <div class="matrix-grid">


        <div class="matrix-label">
            ACTUAL<br>
            NO DISEASE
        </div>


        <div class="
            matrix-cell
            matrix-correct
        ">

            <div class="matrix-number">
                27
            </div>

            <div class="matrix-sub">
                TRUE NEGATIVE
            </div>

        </div>


        <div class="
            matrix-cell
            matrix-error
        ">

            <div class="matrix-number">
                6
            </div>

            <div class="matrix-sub">
                FALSE POSITIVE
            </div>

        </div>


        <div class="matrix-label">
            ACTUAL<br>
            DISEASE
        </div>


        <div class="
            matrix-cell
            matrix-error
        ">

            <div class="matrix-number">
                2
            </div>

            <div class="matrix-sub">
                FALSE NEGATIVE
            </div>

        </div>


        <div class="
            matrix-cell
            matrix-correct
        ">

            <div class="matrix-number">
                26
            </div>

            <div class="matrix-sub">
                TRUE POSITIVE
            </div>

        </div>

    </div>

</div>
"""
        )


        st.success(
            "53 of 61 test records "
            "were classified correctly."
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":


    page_header(
        "ABOUT THE PROJECT",
        "Heart Disease Prediction Using Machine Learning",
        (
            "CardioSense demonstrates an end-to-end "
            "data-science workflow covering preprocessing, "
            "exploratory analysis, model evaluation "
            "and interactive prediction."
        )
    )


    st.write("")
    st.write("")


    left, right = st.columns(
        [1.15, 0.85],
        gap="large"
    )


    with left:


        st.subheader(
            "Project Workflow"
        )


        st.html(
            """
<div class="workflow-card">

    <div class="workflow-line">

        01 — UCI Cleveland heart-disease data collection

        <br>

        02 — Missing-value identification and preprocessing

        <br>

        03 — Binary target transformation

        <br>

        04 — Exploratory data analysis

        <br>

        05 — Train/test split and preprocessing

        <br>

        06 — Logistic Regression, Random Forest and Decision Tree

        <br>

        07 — Five-fold cross-validation and model comparison

        <br>

        08 — Final Logistic Regression pipeline training

        <br>

        09 — Streamlit prediction application

    </div>

</div>
"""
        )


    with right:


        st.subheader(
            "Project Facts"
        )


        st.html(
            """
<div class="workflow-card">

    <div class="metric-label">
        DATA SOURCE
    </div>

    <div style="
        color:#F5F7FA;
        font-size:19px;
        font-weight:800;
        margin:8px 0 20px 0;
    ">
        UCI Heart Disease Dataset
    </div>


    <div class="metric-label">
        DATASET SUBSET
    </div>

    <div style="
        color:#F5F7FA;
        font-size:19px;
        font-weight:800;
        margin:8px 0 20px 0;
    ">
        Processed Cleveland
    </div>


    <div class="metric-label">
        PATIENT RECORDS
    </div>

    <div style="
        color:#F5F7FA;
        font-size:19px;
        font-weight:800;
        margin:8px 0 20px 0;
    ">
        303
    </div>


    <div class="metric-label">
        MODEL FEATURES
    </div>

    <div style="
        color:#F5F7FA;
        font-size:19px;
        font-weight:800;
        margin-top:8px;
    ">
        13
    </div>

</div>
"""
        )