# predict.py
import os
import sys
import joblib
import pandas as pd

# Paths
MODEL_DIR = "models"
model_path = os.path.join(MODEL_DIR, "exoplanet_model.pkl")
encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")

# 1. Check if trained model exists
if not os.path.exists(model_path) or not os.path.exists(encoder_path):
    print("ERROR: No trained model found.")
    print("Please run `train_model.py` first to train and save a model in the 'models/' folder.")
    sys.exit(1)

# 2. Load model + encoder
clf = joblib.load(model_path)
label_encoder = joblib.load(encoder_path)

# 3. Define features used during training
FEATURES = [
    "koi_period",
    "koi_duration",
    "koi_depth",
    "koi_prad",
    "koi_srad",
    "koi_steff",
    "koi_slogg"
]

def predict_exoplanet(input_data: dict):
    """
    input_data: dictionary with keys matching FEATURES
    Example:
    {
      "koi_period": 5.0,
      "koi_duration": 3.5,
      "koi_depth": 200.0,
      "koi_prad": 1.2,
      "koi_srad": 0.9,
      "koi_steff": 5700,
      "koi_slogg": 4.3
    }
    """
    df = pd.DataFrame([input_data], columns=FEATURES)
    pred = clf.predict(df)[0]
    proba = clf.predict_proba(df)[0]

    predicted_class = label_encoder.inverse_transform([pred])[0]

    result = {
        "prediction": predicted_class,
        "probabilities": {
            label_encoder.classes_[i]: float(proba[i])
            for i in range(len(proba))
        }
    }
    return result

# Example usage
if __name__ == "__main__":
    sample = {
        "koi_period": 10.0,
        "koi_duration": 5.0,
        "koi_depth": 500.0,
        "koi_prad": 1.5,
        "koi_srad": 1.0,
        "koi_steff": 5800,
        "koi_slogg": 4.4
    }
    output = predict_exoplanet(sample)
    print("Prediction:", output["prediction"])
    print("Probabilities:", output["probabilities"])
