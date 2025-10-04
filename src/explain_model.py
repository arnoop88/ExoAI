"""
explain_model.py
Generate SHAP explanations for the trained exoplanet model.
Works with RandomForest, XGBoost, and CatBoost models.
"""
import numpy as np
import os
import joblib
import shap
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
MODEL_PATH = "models/best_model.pkl"
ENCODER_PATH = "models/label_encoder.pkl"
DATA_PATH = "data/kepler_cumulative.csv"
OUTPUT_DIR = "models/explainability"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load model and data
# -----------------------------
print("Loading model and dataset...")
model = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

df = pd.read_csv(DATA_PATH, comment="#")

features = [
    "koi_period", "koi_duration", "koi_depth",
    "koi_prad", "koi_srad", "koi_steff", "koi_slogg"
]
target = "koi_disposition"

df = df[features + [target]].dropna()
X = df[features]
y = encoder.transform(df[target])

FRIENDLY_NAMES = {
    "koi_period": "Orbital period",
    "koi_duration": "Transit duration ",
    "koi_depth": "Transit depth ",
    "koi_prad": "Planet radius ",
    "koi_srad": "Stellar radius ",
    "koi_steff": "Stellar effective temperature ",
    "koi_slogg": "Stellar surface gravity "
}

X_renamed = X.rename(columns=FRIENDLY_NAMES)


# -----------------------------
# SHAP Explainer
# -----------------------------
print("Initializing SHAP explainer...")

# handle model type
model_name = type(model).__name__.lower()
if "xgb" in model_name:
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
elif "catboost" in model_name:
    explainer = shap.TreeExplainer(model)
else:
    explainer = shap.TreeExplainer(model)

print("Explainer initialized")

# Compute SHAP values
print("Computing SHAP values (this may take a few minutes)...")
shap_values = explainer.shap_values(X)

# -----------------------------
# Save summary plots
# -----------------------------
print("Generating SHAP summary plots...")

# SHAP summary plot (global importance)
plt.figure()
shap.summary_plot(shap_values, X, show=False)
plt.title("Feature Importance — SHAP Summary")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), dpi=150)

# SHAP bar plot
plt.figure()
class_names = ["Candidate", "Confirmed", "False Positive"]
shap.summary_plot(shap_values, X_renamed, class_names=class_names, plot_type="bar", show=False)
plt.xlabel("")
plt.title("Mean Absolute SHAP Values")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_bar.png"), dpi=150)

print(f"SHAP plots saved in {OUTPUT_DIR}/")

# -----------------------------
# Single-sample explanation (final stable)
# -----------------------------
print("Generating explanation for sample index 0")
example_idx = 0
sample = X.iloc[[example_idx]]

# Handle multi-class or single-class SHAP arrays
if isinstance(shap_values, list):
    pred_class = int(model.predict(sample)[0])
    shap_vec = shap_values[pred_class][example_idx]
    expected_val = explainer.expected_value[pred_class]
else:
    shap_array = shap_values[example_idx]
    if shap_array.ndim == 2:  # multi-class 2D array
        pred_class = int(model.predict(sample)[0])
        shap_vec = shap_array[:, pred_class]
        expected_val = explainer.expected_value[pred_class]
    else:
        shap_vec = shap_array
        expected_val = explainer.expected_value

# ensure base value is scalar (if it's a list, take mean)
if isinstance(expected_val, (list, tuple)):
    expected_val = float(np.mean(expected_val))

# Build a single Explanation object
expl = shap.Explanation(
    values=shap_vec,
    base_values=expected_val,
    data=sample.iloc[0].values,
    feature_names=X_renamed.columns
)

# Plot and save
import matplotlib.pyplot as plt
plt.figure()
shap.plots.waterfall(expl, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_force_sample.png"), dpi=150)
print("Waterfall plot saved successfully.")
