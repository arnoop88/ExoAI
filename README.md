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
1. **Preprocessing:** dataset loading, cleaning, and feature filtering.  
2. **Feature selection:** physical features such as:
   - `koi_period` — orbital period  
   - `koi_duration` — transit duration  
   - `koi_depth` — transit depth  
   - `koi_prad` — planet radius  
   - `koi_srad` — stellar radius  
   - `koi_steff` — stellar effective temperature  
   - `koi_slogg` — stellar surface gravity  
3. **Model:** `RandomForestClassifier` trained on labeled data (CONFIRMED, CANDIDATE, FALSE POSITIVE).  
4. **Export:** saves model and encoder to `/models`.  
5. **Class Means:** stores average feature values per class for interpretability.

## Web Application
Built with **Streamlit**, the app allows:
- **CSV upload** for batch predictions.  
- **Manual input** for single candidates.  
- **Pie chart visualization** of class probabilities.  
- **Educational explanations** comparing inputs with class averages.  
- **Space-themed UI** with gradient blue night mode.

## Technical Stack
| Component | Technology |
|------------|-------------|
| Backend / Model | Python 3.10, scikit-learn, pandas, joblib |
| Frontend | Streamlit + Plotly |
| Explainability | SHAP (planned) |
| Styling | Custom CSS |

## Installation & Run
```bash
git clone https://github.com/arnoop88/exoplanet-ai.git
cd exoplanet-ai
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python src/train_model.py
streamlit run src/app.py
```
