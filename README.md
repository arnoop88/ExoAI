# 🌌 A World Away — Exoplanet Classifier
### NASA Space Apps Challenge 2025

## Overview
*A World Away* is an artificial intelligence tool for **detecting and classifying exoplanets** using open data from NASA’s **Kepler**, **K2**, and **TESS** missions.

The app enables users to:
- Train an AI model using real NASA data.
- Predict whether an object is a **confirmed exoplanet**, a **candidate**, or a **false positive**.
- Explore how physical parameters (radius, transit duration, stellar temperature, etc.) affect the classification.
- Interact with predictions through an **educational, web-based interface**.

## Objective
To accelerate and democratize exoplanet detection through **interpretable machine learning**, making it accessible both to researchers and enthusiasts.

## Data Sources
- [Kepler Cumulative Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=cumulative)
- [TESS Object of Interest Table](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=TOI)
- [K2 Candidate Catalog](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=k2pandc)

Research references:
- [MNRAS 2021: Machine Learning for Exoplanet Classification](https://academic.oup.com/mnras/article/513/4/5505/6472249)
- [Electronics 2024: ML Techniques for Exoplanet Discovery](https://www.mdpi.com/2079-9292/13/19/3950)

### Model pipeline
1. **Preprocessing**
- Loads NASA’s Kepler dataset and ignores commented header lines.
- Cleans missing or invalid values and normalizes key numeric features.
- Filters to the subset of physically meaningful variables.
2. **Feature selection**
Each chosen feature corresponds directly to a measurable property in the transit method:
   - `koi_period` — orbital period (days)
   - `koi_duration` — transit duration (hours)
   - `koi_depth` — transit depth (ppm)
   - `koi_prad` — planet radius (Earth radii)
   - `koi_srad` — stellar radius (Solar radii)
   - `koi_steff` — stellar effective temperature (K)
   - `koi_slogg` — stellar surface gravity (log g)
3. **Model Benchmarking & Training**
- Trains and compares `RandomForest`, `XGBoost`, and `CatBoost` classifiers on the labeled classes: `CONFIRMED`, `CANDIDATE`, and `FALSE POSITIVE`
- Selects the model with the best accuracy–interpretability trade-off based on validation metrics (Accuracy, F1, ROC-AUC).
4. **Export**
- Saves the best-performing model and the fitted label encoder in `/models/`.
- Generates `model_results.json` and benchmark plots for documentation.
5. **Class Means**
- Stores per-class mean values for each feature, used to produce interpretative comparisons (e.g., how a candidate differs from typical confirmed planets).

## Web Application
Built with **Streamlit**, the app allows:
- **CSV upload** for batch predictions.  
- **Manual input** for single-candidate prediction.  
- **Pie chart** of predicted class probabilities.  
- **Educational interpretation** comparing user inputs with per-class averages.  
- **Global and local SHAP visualizations** for explainability.

## Technical Stack
| Component | Technology |
|------------|-------------|
| Backend / Model | Python 3.10, scikit-learn, pandas, joblib |
| Frontend | Streamlit + Plotly |
| Explainability | SHAP (planned) |
| Styling | Custom CSS |

## Installation & Run
```bash
git clone https://github.com/arnoop88/ExoAI.git
cd ExoAI
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 src/train_model.py
python3 src/benchmark_models.py
python3 src/explain_model.py
streamlit run src/app.py
```
