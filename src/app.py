# app.py
import os
import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Exoplanet Classifier", page_icon="assets/favicon.png", layout="wide")

# -----------------------------
# Global styling (Dark Space Theme)
# -----------------------------
st.markdown("""
<style>
/* Space blue gradient across the whole app */
body, .stApp {
    background: linear-gradient(180deg, #001d3d 0%, #003566 35%, #000814 100%);
    color: #e0f7fa;
    font-family: 'Inter', system-ui, -apple-system, Segoe UI, Roboto, 'Helvetica Neue', Arial, 'Noto Sans', 'Liberation Sans', sans-serif;
}

/* Remove white backgrounds anywhere */
.block-container, .main, .stTabs, .stPlotlyChart, .stFileUploader, .stDataFrame {
    background: transparent !important;
}

/* Headings */
h1, h2, h3, h4 {
    color: #90e0ef;
    text-align: center;
    text-shadow: 0 0 8px rgba(144, 224, 239, 0.6);
}

/* Header */
header[data-testid="stHeader"] {
    background: linear-gradient(90deg, #001d3d, #003566) !important;
    color: #e0f7fa !important;
    border-bottom: 1px solid #0077b6 !important;
}

header[data-testid="stHeader"] * {
    color: #e0f7fa !important;
}

/* Labels and helper text with good contrast */
label, .stFileUploader label div, .stMarkdown, .stCaption, .stTextInput, .stNumberInput {
    color: #caf0f8 !important;
}

/* Inputs */
.stNumberInput>div>div>input {
    background-color: #001d3d !important;
    color: #f1faee !important;
    border: 1px solid #0077b6 !important;
}

/* Dataframe table container */
.stDataFrame {
    background-color: rgba(255, 255, 255, 0.05) !important;
    border-radius: 10px !important;
    color: #f1faee !important;
}

/* Buttons */
button[kind="primary"], button[kind="secondary"], div[data-testid="stFileUploaderDropzone"] {
    background: linear-gradient(90deg, #004080, #00b4d8) !important;
    color: #ffffff !important;
    border: 1px solid #00b4d8 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    box-shadow: 0 0 10px rgba(0, 180, 216, 0.4) !important;
    transition: all 0.3s ease-in-out !important;
}
button[kind="primary"]:hover, button[kind="secondary"]:hover, div[data-testid="stFileUploaderDropzone"]:hover {
    background: linear-gradient(90deg, #00b4d8, #48cae4) !important;
    color: #001d3d !important;
    border-color: #48cae4 !important;
    box-shadow: 0 0 18px rgba(72, 202, 228, 0.6) !important;
}

/* Predict*/
button[data-testid="stBaseButton-secondaryFormSubmit"] {
    background: linear-gradient(90deg, #004080, #0077b6) !important;
    color: #ffffff !important;
    border: 1px solid #00b4d8 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    padding: 0.6rem 1.4rem !important;
    box-shadow: 0 0 10px rgba(0, 180, 216, 0.35) !important;
    transition: all 0.3s ease-in-out !important;
}

/* Hover */
button[data-testid="stBaseButton-secondaryFormSubmit"]:hover {
    background: linear-gradient(90deg, #00b4d8, #48cae4) !important;
    color: #001d3d !important;
    border-color: #48cae4 !important;
    box-shadow: 0 0 18px rgba(72, 202, 228, 0.6) !important;
    transform: translateY(-1px);
}

.stFileUploader label div[data-testid="stFileUploaderDropzone"] p {
    color: #ffffff !important;
    font-weight: 500 !important;
}

/* Result box card */
.result-box {
    background: rgba(10, 25, 47, 0.78) !important;
    padding: 20px !important;
    margin-top: 16px !important;
}

/* Footer */
footer {
    text-align: center;
    color: #90e0ef;
    font-size: 13px;
    margin-top: 40px;
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Constants & labels
# -----------------------------
FRIENDLY_NAMES = {
    "koi_period": "Orbital period (days)",
    "koi_duration": "Transit duration (hours)",
    "koi_depth": "Transit depth (ppm)",
    "koi_prad": "Planet radius (Earth radii)",
    "koi_srad": "Stellar radius (Solar radii)",
    "koi_steff": "Stellar effective temperature (K)",
    "koi_slogg": "Stellar surface gravity (log g)"
}
FEATURES = list(FRIENDLY_NAMES.keys())

# Colors per class (semantic)
CLASS_COLORS = {
    "CONFIRMED": "#2ecc71",      # green
    "CANDIDATE": "#f1c40f",      # amber
    "FALSE POSITIVE": "#e74c3c"  # red
}

# -----------------------------
# Load artifacts
# -----------------------------
MODEL_DIR = "models"
model_path = os.path.join(MODEL_DIR, "exoplanet_model.pkl")
encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
means_path = os.path.join(MODEL_DIR, "class_means.json")

if not (os.path.exists(model_path) and os.path.exists(encoder_path)):
    st.error("No trained model found. Please run `train_model.py` first to train and save a model in /models.")
    st.stop()

# Load model & encoder
clf = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# Try to load class means (for educational explanation)
class_means = None
if os.path.exists(means_path):
    with open(means_path, "r") as f:
        class_means = json.load(f)
    # Patch: if keys are numeric strings (0/1/2), map back to labels
    # Expected structure: feature -> class_name -> mean
    # If we see digits as keys, remap them:
    for feat, by_class in class_means.items():
        if by_class and all(k.isdigit() for k in by_class.keys()):
            mapping = {str(i): lab for i, lab in enumerate(label_encoder.classes_)}
            class_means[feat] = {mapping[k]: v for k, v in by_class.items()}
else:
    st.warning("class_means.json not found. Educational comparison will be skipped.")

# -----------------------------
# Header
# -----------------------------
col1, col2, col3 = st.columns([2.2, 2, 1])
# Contenedor para la imagen
with col2:
    with st.container():
        st.image("assets/exo_ai_logo.png", width=200)

# Contenedor para el texto, con margen superior para separarlo verticalmente
with st.container():
    st.markdown(
        """
        <div style="margin-top: 80px; text-align: center;">
            <h3>Explore the universe of Kepler data through Artificial Intelligence.</h3>
        </div>
        """,
        unsafe_allow_html=True
    )

st.divider()

# -----------------------------
# CSV Upload
# -----------------------------
st.markdown("## 📂 Upload CSV Data")
st.markdown("<p>Upload your Kepler-like table (columns in dataset naming) to batch predict.</p>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload a CSV with the required columns", type=["csv"])
if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file, comment="#")
        st.write("**Preview of uploaded data:**")
        st.dataframe(df.head())

        missing = [f for f in FEATURES if f not in df.columns]
        if missing:
            st.error(f"CSV must contain columns: {FEATURES}\nMissing: {missing}")
        else:
            preds = clf.predict(df[FEATURES])
            probs = clf.predict_proba(df[FEATURES])
            df["Prediction"] = label_encoder.inverse_transform(preds)
            for i, label in enumerate(label_encoder.classes_):
                df[f"Prob_{label}"] = probs[:, i]

            st.success("Predictions complete!")
            st.dataframe(df[["Prediction"] + [f"Prob_{label}" for label in label_encoder.classes_]])
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

st.divider()

# -----------------------------
# Manual Input Prediction
# -----------------------------
st.markdown("## 🧠 Manual Input Prediction")
st.markdown("<p>Enter values for a single candidate and get an instant prediction.</p>", unsafe_allow_html=True)

with st.form("manual_input_form"):
    st.markdown("#### Candidate parameters")
    cols = st.columns(3)
    koi_period = cols[0].number_input(FRIENDLY_NAMES["koi_period"], min_value=0.0, value=10.0)
    koi_duration = cols[1].number_input(FRIENDLY_NAMES["koi_duration"], min_value=0.0, value=5.0)
    koi_depth = cols[2].number_input(FRIENDLY_NAMES["koi_depth"], min_value=0.0, value=500.0)

    cols2 = st.columns(3)
    koi_prad = cols2[0].number_input(FRIENDLY_NAMES["koi_prad"], min_value=0.0, value=1.0)
    koi_srad = cols2[1].number_input(FRIENDLY_NAMES["koi_srad"], min_value=0.0, value=1.0)
    koi_steff = cols2[2].number_input(FRIENDLY_NAMES["koi_steff"], min_value=0.0, value=5700.0)

    koi_slogg = st.number_input(FRIENDLY_NAMES["koi_slogg"], min_value=0.0, value=4.4)

    submitted = st.form_submit_button("🚀 Predict")
    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        # Prepare input
        sample = {
            "koi_period": koi_period,
            "koi_duration": koi_duration,
            "koi_depth": koi_depth,
            "koi_prad": koi_prad,
            "koi_srad": koi_srad,
            "koi_steff": koi_steff,
            "koi_slogg": koi_slogg
        }
        df_input = pd.DataFrame([sample], columns=FEATURES)

        # Predict
        pred = clf.predict(df_input)[0]
        proba = clf.predict_proba(df_input)[0]
        predicted_class = label_encoder.inverse_transform([pred])[0]

        # Result card
        st.markdown(f"<h3 style='color:#90e0ef;'>Prediction: {predicted_class}</h3>", unsafe_allow_html=True)

        # Build proba df in class order as encoder exposes
        proba_df = pd.DataFrame({"Class": label_encoder.classes_, "Probability": proba})

        # Create a color map that respects our semantic mapping, with fallback
        color_map = {c: CLASS_COLORS.get(c, "#00b4d8") for c in proba_df["Class"].tolist()}

        # Pie chart (no white background, good contrast)
        fig = px.pie(
            proba_df,
            values="Probability",
            names="Class",
            title="Prediction Probability Distribution",
            hole=0.3,
            color="Class",
            color_discrete_map=color_map
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0f7fa", size=16),
            legend=dict(
                bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e0f7fa", size=14)
            ),
            title_font=dict(color="#90e0ef", size=20),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Educational explanation (if means available)
        st.markdown("### Educational Interpretation")
        if class_means is None:
            st.caption("Class means not available. Run training step that saves models/class_means.json to enable this section.")
        else:
            explanations = []
            for feat in FEATURES:
                user_val = sample[feat]
                # Means stored as: feature -> class_name -> mean
                feat_means = class_means.get(feat, {})
                if not feat_means:
                    continue
                # Find class whose mean is closest to user value
                closest_class = min(feat_means, key=lambda k: abs(user_val - feat_means[k]))
                friendly = FRIENDLY_NAMES.get(feat, feat)
                explanations.append(
                    f"For **{friendly}**, your value is {user_val:.2f}, closest to **{closest_class}** (avg: {feat_means[closest_class]:.2f})."
                )
            if explanations:
                for line in explanations:
                    st.markdown(f"- {line}")
            else:
                st.caption("No feature statistics available to compare.")

        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Model Insights
# -----------------------------
st.divider()
st.markdown("## 🔍 Model Insights")
st.markdown(
    "Explore which physical features influence the model's predictions the most, "
    "based on SHAP (SHapley Additive exPlanations) analysis."
)

EXPLAIN_DIR = "models/explainability"
summary_path = os.path.join(EXPLAIN_DIR, "shap_summary.png")
bar_path = os.path.join(EXPLAIN_DIR, "shap_bar.png")
force_path = os.path.join(EXPLAIN_DIR, "shap_force_sample.png")

if os.path.exists(summary_path) and os.path.exists(bar_path):
    col1, col2 = st.columns(2)
    with col1:
        st.image(bar_path, caption="Mean Absolute SHAP Values", use_container_width=True)
    with col2:
        st.image(summary_path, caption="Global SHAP Summary Plot", use_container_width=True)

    if os.path.exists(force_path):
        st.markdown("### Example Prediction Explanation")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(force_path, caption="SHAP Force Plot — Example Candidate", use_container_width=False, width=700)  

    st.markdown(
        """
        **Interpretation:**  
        - Each bar represents how strongly a feature contributes to the model's predictions.  
        - Larger absolute SHAP values indicate greater influence.  
        - For example, if *Stellar Effective Temperature* (koi_steff) has a large SHAP value,  
          it means the model relies heavily on it to distinguish confirmed planets from false positives.
        """
    )
else:
    st.warning("SHAP explainability results not found. Run `python src/explain_model.py` to generate them.")

# -----------------------------
# Footer
# -----------------------------
st.markdown("<footer>Made by Team Leaf | NASA Space Apps 2025</footer>", unsafe_allow_html=True)
