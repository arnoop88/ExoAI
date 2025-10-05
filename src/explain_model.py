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

# -----------------------------
# Interaction summary (heavy)
# -----------------------------
try:
    print("Computing SHAP interaction values (this can be heavy)...")
    shap_inter = explainer.shap_interaction_values(X)

    def _friendly_cols(df):
        return df.rename(columns={
            "koi_period":  "days",
            "koi_duration":"hours",
            "koi_depth":   "ppm",
            "koi_prad":    "Earth radii",
            "koi_srad":    "Solar radii",
            "koi_steff":   "K",
            "koi_slogg":   "log g"
        })

    class_names = list(encoder.classes_) if encoder is not None else [f"Class {i}" for i in range(3)]
    class_idx = 1  # confirmed
    class_label = class_names[class_idx] if class_idx < len(class_names) else f"Class{class_idx}"

    inter_to_plot = None
    shape_info = None

    import numpy as np
    import matplotlib.pyplot as plt

    # --- Cases ---
    if isinstance(shap_inter, list):
        if len(shap_inter) > class_idx and hasattr(shap_inter[class_idx], "ndim"):
            shape_info = f"list[{len(shap_inter)}] -> {np.array(shap_inter[class_idx]).shape}"
            inter_to_plot = shap_inter[class_idx]  # (N, F, F)
    elif hasattr(shap_inter, "ndim"):
        shape_info = str(shap_inter.shape)
        if shap_inter.ndim == 3:
            # (N, F, F)
            inter_to_plot = shap_inter
        elif shap_inter.ndim == 5 and shap_inter.shape[2] == len(class_names) and shap_inter.shape[4] == len(class_names):
            # (N, F, C, F, C)
            inter_to_plot = shap_inter[:, :, class_idx, :, class_idx]
        elif shap_inter.ndim == 4:
            # (N, F, F, C)
            if shap_inter.shape[1] == shap_inter.shape[2] and shap_inter.shape[3] == len(class_names):
                inter_to_plot = shap_inter[:, :, :, class_idx]
            # (N, F, C, F)
            elif shap_inter.shape[2] == len(class_names) and shap_inter.shape[1] == shap_inter.shape[3]:
                inter_to_plot = shap_inter[:, :, class_idx, :]

    if inter_to_plot is not None and hasattr(inter_to_plot, "ndim") and inter_to_plot.ndim == 3:
        X_ren = _friendly_cols(X.copy())
        plt.close("all")
        fig = plt.gcf()
        fig.set_size_inches(12, 6)
        shap.summary_plot(inter_to_plot, X_ren, show=False, max_display=len(X.columns))
        out_inter = os.path.join(OUTPUT_DIR, f"shap_interactions_{class_label}.png")
        plt.tight_layout()
        plt.savefig(out_inter, dpi=160, bbox_inches="tight")
        plt.close()
        print(f"Saved interaction summary: {out_inter} (from shape {shape_info})")
    else:
        print(f"Interaction values format not recognized; got type={type(shap_inter)}"
              + (f", shape={shape_info}" if shape_info else "") + ". Skipping interaction plot.")

except Exception as e:
    print(f"Skipping interaction summary (reason: {e})")



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
# Force Plot
# -----------------------------
print("Generating explanation for sample CONFIRMED")
example_idx = 1 # Confirmed
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

# Force Plot
plt.figure()
shap.plots.waterfall(expl, show=False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "shap_force_sample.png"), dpi=150)
print("Waterfall plot saved successfully.")
