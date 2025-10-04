"""
benchmark_models.py
Compare RandomForest, XGBoost and CatBoost for exoplanet classification.
Outputs metrics and saves best model + results table.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------
DATA_PATH = "data/kepler_cumulative.csv"
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# 1. Load and preprocess
# -----------------------------
print("Loading dataset...")
df = pd.read_csv(DATA_PATH, comment="#")

features = [
    "koi_period", "koi_duration", "koi_depth",
    "koi_prad", "koi_srad", "koi_steff", "koi_slogg"
]
target = "koi_disposition"

# drop missing & select relevant
df = df[features + [target]].dropna()
print(f"Dataset shape: {df.shape}")

# encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df[target])
X = df[features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -----------------------------
# 2. Define models
# -----------------------------
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=None, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="mlogloss"
    ),
    "CatBoost": CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_seed=42
    )
}

# -----------------------------
# 3. Train and evaluate
# -----------------------------
results = []
print("\nTraining models...")

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, preds, average="weighted")
    f1 = f1_score(y_test, preds, average="weighted")
    try:
        auc = roc_auc_score(y_test, proba, multi_class="ovr")
    except Exception:
        auc = np.nan

    results.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    })

# -----------------------------
# 4. Save metrics & best model
# -----------------------------
df_results = pd.DataFrame(results)
best_row = df_results.sort_values(by="f1", ascending=False).iloc[0]
best_model_name = best_row["model"]
best_model = models[best_model_name]

print("\nBest model:", best_model_name)
print(df_results)

df_results.to_json(os.path.join(OUTPUT_DIR, "model_results.json"), orient="records", indent=2)
joblib.dump(best_model, os.path.join(OUTPUT_DIR, "best_model.pkl"))
joblib.dump(label_encoder, os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

# -----------------------------
# 5. Plot comparison
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 4))
df_results.plot(x="model", y=["accuracy", "f1", "roc_auc"], kind="bar", ax=ax)
ax.set_title("Model Comparison — Kepler Dataset")
ax.set_ylabel("Score")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "benchmark_results.png"))
print(f"\nResults saved in {OUTPUT_DIR}/")

