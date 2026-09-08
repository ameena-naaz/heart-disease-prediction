import streamlit as st
import pandas as pd
import base64
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "heart_disease_model.pkl"
model = joblib.load(MODEL_PATH)

heart_path = Path(__file__).resolve().parent / "heart.svg"

with open(heart_path, "rb") as heart_file:
    heart_base64 = base64.b64encode(heart_file.read()).decode()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM STYLING
# ============================================================

st.markdown("""
<style>


/* ------------------------------------------------------------
   MAIN APPLICATION
------------------------------------------------------------ */

.stApp {
    background:
        radial-gradient(
            circle at 78% 35%,
            rgba(180, 25, 35, 0.13),
            transparent 32%
        ),
        #050506;

    color: #F5F5F5;
}


/* Main Streamlit content */

.block-container {
    padding-top: 2rem;
    padding-left: 3rem;
    padding-right: 3rem;
    max-width: 1500px;
}


/* ------------------------------------------------------------
   STREAMLIT DEFAULT UI
------------------------------------------------------------ */

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stDecoration"] {
    display: none;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ------------------------------------------------------------
   SIDEBAR
------------------------------------------------------------ */

section[data-testid="stSidebar"] {
    background: #080809;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}


/* ------------------------------------------------------------
   HERO SECTION
------------------------------------------------------------ */

.hero-section {
    width: 100%;
    min-height: 650px;

    display: flex;
    align-items: center;
}


.hero-content {
    width: 100%;

    display: grid;

    grid-template-columns:
        1.15fr
        0.85fr;

    align-items: center;

    gap: 40px;
}


/* ------------------------------------------------------------
   HERO TEXT
------------------------------------------------------------ */

.hero-label {
    color: #FF5C5C;

    letter-spacing: 3px;

    font-size: 12px;

    font-weight: 700;

    margin-bottom: 28px;
}


.hero-text h1 {
    font-size: 58px;

    line-height: 1.08;

    margin: 0;

    color: #F7F7F8;

    font-weight: 750;
}


.hero-text h1 span {
    background:
        linear-gradient(
            90deg,
            #FF4545,
            #FF8066
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}


.hero-description {
    color: #9CA3AF;

    font-size: 17px;

    line-height: 1.8;

    max-width: 650px;

    margin-top: 32px;
}


/* ------------------------------------------------------------
   HEART VISUAL AREA
------------------------------------------------------------ */

.heart-area {
    position: relative;

    min-height: 470px;

    display: flex;

    justify-content: center;

    align-items: center;

    overflow: hidden;
}


/* Background glow */

.heart-glow {
    position: absolute;

    width: 310px;

    height: 310px;

    border-radius: 50%;

    background:
        radial-gradient(
            circle,
            rgba(255, 40, 55, 0.30),
            rgba(255, 40, 55, 0.08) 45%,
            transparent 70%
        );

    filter: blur(18px);

    animation:
        glowPulse 2.5s
        ease-in-out
        infinite;
}


/* Heart */

.heart-icon {
    position: relative;

    z-index: 3;

    font-size: 210px;

    line-height: 1;

    color: #E32636;

    text-shadow:
        0 0 25px rgba(255, 45, 55, 0.45),
        0 0 60px rgba(255, 25, 40, 0.20);

    animation:
        heartbeat 1.6s ease-in-out infinite,
        heartFloat 5s ease-in-out infinite;
}


/* ------------------------------------------------------------
   HEART RINGS
------------------------------------------------------------ */

.heart-ring {
    position: absolute;

    border-radius: 50%;

    border:
        1px solid
        rgba(255, 75, 85, 0.18);
}


.ring-one {
    width: 330px;

    height: 330px;

    animation:
        ringPulse
        3s
        infinite;
}


.ring-two {
    width: 410px;

    height: 410px;

    border-color:
        rgba(255, 75, 85, 0.08);

    animation:
        ringPulse
        3s
        infinite
        1s;
}


/* ------------------------------------------------------------
   HEART STATUS
------------------------------------------------------------ */

.heart-status {
    position: absolute;

    bottom: 35px;

    font-size: 10px;

    letter-spacing: 2px;

    color: #8D949E;
}


.status-dot {
    width: 7px;

    height: 7px;

    background: #FF4D4D;

    border-radius: 50%;

    display: inline-block;

    margin-right: 9px;

    box-shadow:
        0 0 10px
        rgba(255, 70, 70, 0.8);

    animation:
        dotPulse
        1.4s
        infinite;
}


/* ============================================================
   ANIMATIONS
============================================================ */


/* Heartbeat */

@keyframes heartbeat {

    0%, 100% {
        transform: scale(1);
    }

    12% {
        transform: scale(1.07);
    }

    24% {
        transform: scale(1);
    }

    36% {
        transform: scale(1.04);
    }

    48% {
        transform: scale(1);
    }
}


/* Floating movement */

@keyframes heartFloat {

    0%, 100% {
        translate: 0 0;
    }

    50% {
        translate: 0 -10px;
    }
}


/* Glow */

@keyframes glowPulse {

    0%, 100% {
        opacity: 0.55;
        transform: scale(0.95);
    }

    50% {
        opacity: 1;
        transform: scale(1.08);
    }
}


/* Expanding rings */

@keyframes ringPulse {

    0% {
        transform: scale(0.9);
        opacity: 0.30;
    }

    70% {
        transform: scale(1.08);
        opacity: 0.08;
    }

    100% {
        transform: scale(1.12);
        opacity: 0;
    }
}


/* Status light */

@keyframes dotPulse {

    0%, 100% {
        opacity: 0.5;
    }

    50% {
        opacity: 1;
    }
}


/* ============================================================
   MOBILE RESPONSIVENESS
============================================================ */

@media (max-width: 900px) {

    .block-container {
        padding-left: 1.5rem;
        padding-right: 1.5rem;
    }

    .hero-content {
        grid-template-columns: 1fr;
    }

    .hero-text h1 {
        font-size: 42px;
    }

    .hero-section {
        min-height: auto;
    }

    .heart-area {
        min-height: 360px;
    }

    .heart-icon {
        font-size: 160px;
    }
    .anatomical-heart {
    width: 250px;
    height: auto;
    display: block;

    filter:
        drop-shadow(0 0 18px rgba(255, 45, 55, 0.40))
        drop-shadow(0 0 45px rgba(180, 20, 30, 0.18));
}

}


</style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR NAVIGATION
# =============================================================
with st.sidebar:

    st.markdown(
        """
<div style="padding:18px 4px 24px 4px;">
<div style="font-size:22px;font-weight:800;color:#F5F7FA;">
❤️ CardioSense
</div>
<div style="color:#FF6257;font-size:10px;letter-spacing:2px;margin-top:4px;">
HEART RISK PREDICTION
</div>
</div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Assessment",
            "Prediction",
            "Model Insights",
            "About"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.caption("Educational machine learning project")


# ============================================================
# PAGE ROUTING
# ============================================================

if page == "Assessment":

    st.markdown(
        """
<div style="padding-top:55px;">

<p style="color:#FF6257;letter-spacing:3px;font-size:12px;font-weight:700;">
PATIENT ASSESSMENT
</p>

<h1 style="color:#F5F7FA;font-size:48px;margin-bottom:16px;">
Cardiovascular assessment
</h1>

<p style="color:#98A2B3;font-size:16px;line-height:1.7;max-width:700px;">
Enter the patient's cardiovascular indicators to generate a
machine-learning-based heart disease risk estimate.
</p>

</div>
        """,
        unsafe_allow_html=True
    )
if page == "Assessment":
    st.markdown(
        """
<div style="padding-top:55px;">

<p style="color:#FF6257;letter-spacing:3px;font-size:12px;font-weight:700;">
PATIENT ASSESSMENT
</p>

<h1 style="color:#F5F7FA;font-size:48px;margin-bottom:16px;">
Cardiovascular assessment
</h1>

<p style="color:#98A2B3;font-size:16px;line-height:1.7;max-width:700px;">
Enter the patient's cardiovascular indicators below and run the model
to generate a heart disease risk prediction.
</p>

</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Patient Profile")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", min_value=18, max_value=100, value=50)

    with c2:
        sex_label = st.selectbox("Sex", ["Female", "Male"])
        sex = 1 if sex_label == "Male" else 0

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

        cp_map = {
            "Typical angina": 1,
            "Atypical angina": 2,
            "Non-anginal pain": 3,
            "Asymptomatic": 4
        }

        cp = cp_map[cp_label]

    st.markdown("### Clinical Measurements")
    c1, c2, c3 = st.columns(3)

    with c1:
        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80,
            max_value=220,
            value=130
        )

    with c2:
        chol = st.number_input(
            "Cholesterol (mg/dl)",
            min_value=100,
            max_value=600,
            value=240
        )

    with c3:
        fbs_label = st.selectbox(
            "Fasting Blood Sugar > 120 mg/dl",
            ["No", "Yes"]
        )
        fbs = 1 if fbs_label == "Yes" else 0

    st.markdown("### Cardiac & Exercise Indicators")
    c1, c2, c3 = st.columns(3)

    with c1:
        restecg_label = st.selectbox(
            "Resting ECG",
            [
                "Normal",
                "ST-T abnormality",
                "Left ventricular hypertrophy"
            ]
        )

        restecg_map = {
            "Normal": 0,
            "ST-T abnormality": 1,
            "Left ventricular hypertrophy": 2
        }

        restecg = restecg_map[restecg_label]

    with c2:
        thalach = st.number_input(
            "Maximum Heart Rate Achieved",
            min_value=60,
            max_value=220,
            value=150
        )

    with c3:
        exang_label = st.selectbox(
            "Exercise-Induced Angina",
            ["No", "Yes"]
        )
        exang = 1 if exang_label == "Yes" else 0

    st.markdown("### Advanced Indicators")
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        oldpeak = st.number_input(
            "ST Depression (Oldpeak)",
            min_value=0.0,
            max_value=10.0,
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

        slope_map = {
            "Upsloping": 1,
            "Flat": 2,
            "Downsloping": 3
        }

        slope = slope_map[slope_label]

    with c3:
        ca = st.selectbox(
            "Number of Major Vessels (CA)",
            [0, 1, 2, 3]
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

        thal_map = {
            "Normal": 3,
            "Fixed defect": 6,
            "Reversible defect": 7
        }

        thal = thal_map[thal_label]

    analyze = st.button(
        "Analyze Heart Risk →",
        use_container_width=True
    )

    if analyze:
        patient_data = pd.DataFrame([{
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
        }])

        prediction = model.predict(patient_data)[0]
        probability = model.predict_proba(patient_data)[0][1]

        st.session_state["prediction"] = int(prediction)
        st.session_state["probability"] = float(probability)

        risk_percent = probability * 100

    if "prediction" in st.session_state:
        prediction = st.session_state["prediction"]
        probability = st.session_state["probability"]
        risk_percent = probability * 100

        if prediction == 1:
            risk_title = "Higher Predicted Risk"
            risk_icon = "⚠"
            risk_color = "#FF6257"
            risk_bg = "rgba(255, 98, 87, 0.10)"
            interpretation = (
                "The model identified a pattern associated with a higher "
                "likelihood of heart disease in the training data."
            )
        else:
            risk_title = "Lower Predicted Risk"
            risk_icon = "✓"
            risk_color = "#32D583"
            risk_bg = "rgba(50, 213, 131, 0.10)"
            interpretation = (
                "The model identified a pattern associated with a lower "
                "likelihood of heart disease in the training data."
            )
        st.markdown(
    f"""
<div style="
    margin-top:24px;
    padding:26px 30px;
    border-radius:16px;
    background:{risk_bg};
    border:1px solid {risk_color}55;
">

<div style="
    color:{risk_color};
    font-size:13px;
    font-weight:700;
    letter-spacing:2px;
    margin-bottom:10px;
">
PREDICTION RESULT
</div>

<div style="
    font-size:27px;
    font-weight:700;
    color:#F5F7FA;
    margin-bottom:8px;
">
{risk_icon} {risk_title}
</div>

<div style="
    font-size:42px;
    font-weight:800;
    color:{risk_color};
    margin-bottom:4px;
">
{risk_percent:.1f}%
</div>

<div style="
    color:#98A2B3;
    font-size:13px;
    margin-bottom:18px;
">
Predicted probability of heart disease
</div>

<div style="
    width:100%;
    height:8px;
    background:rgba(255,255,255,0.08);
    border-radius:10px;
    overflow:hidden;
    margin-bottom:18px;
">
    <div style="
        width:{risk_percent:.1f}%;
        height:100%;
        background:{risk_color};
        border-radius:10px;
    "></div>
</div>

<div style="
    color:#D0D5DD;
    font-size:14px;
    line-height:1.7;
">
{interpretation}
</div>

</div>
""",
    unsafe_allow_html=True
)
       
if page == "Prediction":

    st.markdown(
        """
        <div style="padding-top:35px;">
            <p style="color:#67E8F9; letter-spacing:3px; font-size:12px; font-weight:700;">
                PREDICTION ANALYSIS
            </p>

            <h1 style="color:#F5F7FA; font-size:42px; margin-bottom:10px;">
                Your Heart Risk Result
            </h1>

            <p style="color:#98A2B3; font-size:16px; max-width:700px;">
                Review the machine-learning prediction generated from your cardiovascular assessment.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "prediction" not in st.session_state:
        st.info("Complete the Patient Assessment first to generate your prediction.")
        st.stop()

    prediction = st.session_state["prediction"]
    probability = st.session_state["probability"]
    risk_percent = probability * 100

    if prediction == 1:
        risk_title = "Higher Predicted Risk"
        risk_icon = "⚠"
        risk_color = "#FF6257"
        risk_bg = "rgba(255, 98, 87, 0.10)"
        result_text = (
            "The submitted cardiovascular indicators produced a pattern "
            "associated with a higher predicted likelihood of heart disease."
        )
    else:
        risk_title = "Lower Predicted Risk"
        risk_icon = "✓"
        risk_color = "#32D583"
        risk_bg = "rgba(50, 213, 131, 0.10)"
        result_text = (
            "The submitted cardiovascular indicators produced a pattern "
            "associated with a lower predicted likelihood of heart disease."
        )

    st.markdown(
        f"""
        <div style="
            margin-top:30px;
            padding:30px;
            border-radius:18px;
            background:{risk_bg};
            border:1px solid {risk_color};
        ">
            <p style="
                color:{risk_color};
                letter-spacing:2px;
                font-size:12px;
                font-weight:700;
            ">
                CURRENT PREDICTION
            </p>

            <h2 style="color:#F5F7FA;">
                {risk_icon} {risk_title}
            </h2>

            <div style="
                color:{risk_color};
                font-size:46px;
                font-weight:800;
                margin:18px 0 4px 0;
            ">
                {risk_percent:.1f}%
            </div>

            <p style="color:#98A2B3; font-size:13px;">
                Predicted probability of heart disease
            </p>

            <div style="
                width:100%;
                height:10px;
                background:rgba(255,255,255,0.08);
                border-radius:10px;
                overflow:hidden;
                margin:18px 0;
            ">
                <div style="
                    width:{risk_percent:.1f}%;
                    height:100%;
                    background:{risk_color};
                    border-radius:10px;
                "></div>
            </div>

            <p style="
                color:#D0D5DD;
                font-size:14px;
                line-height:1.7;
            ">
                {result_text}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


if page == "Model Insights":

    st.markdown("## Model Insights")
    st.caption("Model performance, ROC-AUC and evaluation results will appear here.")
    st.stop()


if page == "About":

    st.markdown("## About This Project")
    st.caption(
        "Heart Disease Prediction using the UCI Heart Disease dataset "
        "and Logistic Regression."
    )
    st.stop()
# ============================================================
# HERO
# ============================================================

st.markdown(f"""
<div class="hero-section">
<div class="hero-content">

<div class="hero-text">
<p class="hero-label">HEART DISEASE RISK PREDICTION</p>

<h1>
Data-driven insights for<br>
<span>heart risk prediction.</span>
</h1>

<p class="hero-description">
Analyze key cardiovascular indicators using machine learning
and explore how patient data contributes to predicted heart disease risk.
</p>
</div>

<div class="heart-area">
<div class="heart-glow"></div>
<div class="heart-ring ring-one"></div>
<div class="heart-ring ring-two"></div>

<div class="heart-icon">
<img src="data:image/svg+xml;base64,{heart_base64}" class="anatomical-heart">
</div>


<div class="heart-status">
<span class="status-dot"></span>
CARDIOVASCULAR ANALYSIS
</div>
</div>

</div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# PATIENT ASSESSMENT
# ============================================================
st.markdown("#### PATIENT ASSESSMENT")

st.markdown(
    "<h2 style='font-size:38px; margin:0; color:#F5F7FA;'>"
    "Enter cardiovascular indicators"
    "</h2>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='color:#98A2B3; font-size:15px; margin-top:12px;'>"
    "Provide the patient information below to generate a "
    "machine-learning-based heart disease risk estimate."
    "</p>",
    unsafe_allow_html=True
)