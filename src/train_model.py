# train_model.py
import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load dataset (NASA cumulative Kepler table, skipping metadata lines)
df = pd.read_csv("data/kepler_cumulative.csv", comment="#")

# 2. Select relevant features and target column
features = [
    "koi_period",    # Orbital period
    "koi_duration",  # Transit duration
    "koi_depth",     # Transit depth
    "koi_prad",      # Planet radius
    "koi_srad",      # Stellar radius
    "koi_steff",     # Stellar effective temperature
    "koi_slogg"      # Stellar surface gravity
]
target = "koi_disposition"

# Drop rows with missing values
df = df[features + [target]].dropna()

# 3. Encode target labels
label_encoder = LabelEncoder()
df[target] = label_encoder.fit_transform(df[target])

# 4. Train/test split
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Train RandomForest model
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# 6. Validate model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# 7. Ensure output folder exists
output_dir = "models"
os.makedirs(output_dir, exist_ok=True)

# Save model + encoder
joblib.dump(clf, os.path.join(output_dir, "exoplanet_model.pkl"))
joblib.dump(label_encoder, os.path.join(output_dir, "label_encoder.pkl"))

print(f"Model saved in {output_dir}/exoplanet_model.pkl")

# Compute class averages (group by encoded target)
grouped = df.groupby(target)[features].mean()

# Replace numeric indices with original class names
class_labels = label_encoder.inverse_transform(grouped.index)
grouped.index = class_labels

# Convert to dict and save
class_means = grouped.to_dict()

with open(os.path.join(output_dir, "class_means.json"), "w") as f:
    json.dump(class_means, f)

print("Class means saved with readable class labels")
