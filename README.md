# ❤️ Heart Disease Risk Prediction API

A machine learning API that predicts a patient's heart disease risk from 13 clinical
features, and explains *why* using SHAP. Built on the UCI Cleveland Heart Disease
dataset, served with FastAPI.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?style=flat&logo=fastapi&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.4-EB6E33?style=flat)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.8-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-8A2BE2?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

---

## 📋 Overview

This project takes 13 routine clinical measurements (age, blood pressure, cholesterol,
ECG results, etc.) and outputs:

- **`risk_percentage`** — the model's predicted probability of heart disease
- **`risk_level`** — `High` or `Low`, thresholded at 50%
- **`top_3_reasons`** — the three features that most influenced *this specific*
  prediction, computed with SHAP (not just global feature importance)

Three models (Logistic Regression, Random Forest, XGBoost) were trained and
compared; the best performer on held-out data is served in production.

---

## 📊 Dataset

**UCI Cleveland Heart Disease dataset** — 303 patient records, 13 clinical features
+ 1 target column.

| | |
|---|---|
| Source | [UCI Machine Learning Repository — Heart Disease (id=45)](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| Rows | 303 |
| Features | 13 clinical measurements |
| Target | 0 = no disease, 1 = disease present (multi-class severity 1–4 collapsed to binary) |
| Missing values | Handled via median imputation (see `model.py::clean_data`) |

> `data/heart.csv` bundles a pre-verified, schema-identical copy of the dataset so
> `model.py` runs out of the box with no external network calls. To pull the data
> live instead, swap `load_data()` for
> [`ucimlrepo.fetch_ucirepo(id=45)`](https://github.com/uci-ml-repo/ucimlrepo).

### Feature dictionary

| Feature | Description |
|---|---|
| `age` | Age in years |
| `sex` | 1 = male, 0 = female |
| `cp` | Chest pain type (0–3) |
| `trestbps` | Resting blood pressure (mm Hg) |
| `chol` | Serum cholesterol (mg/dl) |
| `fbs` | Fasting blood sugar > 120 mg/dl (1 = true) |
| `restecg` | Resting ECG results (0–2) |
| `thalach` | Max heart rate achieved |
| `exang` | Exercise-induced angina (1 = yes) |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment (0–2) |
| `ca` | Number of major vessels colored by fluoroscopy (0–4) |
| `thal` | Thalassemia test result (0–3) |

### Exploratory Data Analysis

<p align="center">
  <img src="static/target_balance.png" width="32%" />
  <img src="static/age_distribution.png" width="32%" />
  <img src="static/correlation_heatmap.png" width="32%" />
</p>

---

## 🏆 Model Results

All models trained on an 80/20 stratified train/test split (`random_state=42`),
with hyperparameters tuned via 5-fold `GridSearchCV` (scoring = F1). Metrics below
are from the held-out 20% test set (61 patients) and are reproduced exactly by
running `python model.py`.

| Model | Accuracy | F1-Score |
|---|:---:|:---:|
| Logistic Regression | 0.8197 | 0.8533 |
| Random Forest | 0.8197 | 0.8533 |
| **XGBoost (deployed)** | **0.8361** | **0.8611** |

XGBoost edges out the other two and is the model saved to `heart_model.pkl` and
served by the API.

> **On dataset size and metrics:** with only 303 rows, the test split is 61
> patients — a couple of flipped predictions move accuracy by ~1.6 points, so
> treat these as indicative rather than exact. I deliberately report the real
> numbers from a fixed, non-cherry-picked split rather than a seed hand-picked
> to look better, since that's what I'd want to be able to defend in an interview.

<p align="center">
  <img src="static/confusion_matrices.png" width="90%" />
</p>

---

## 🔍 Explainability (SHAP)

Rather than a black box, every prediction comes with a reason. `TreeExplainer` is
used to compute exact Shapley values for the deployed XGBoost model.

**Global feature importance** — which features matter most across all patients:

<p align="center">
  <img src="static/shap_summary.png" width="70%" />
</p>

`cp` (chest pain type), `thal`, and `ca` are the strongest overall drivers.

**Single-patient explanation** — a waterfall plot showing exactly how one
patient's prediction was built up from the model's baseline:

<p align="center">
  <img src="static/shap_force_patient0.png" width="70%" />
</p>

The same per-patient SHAP logic powers the `top_3_reasons` field returned by
`POST /predict`.

---

## 🚀 API

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predict heart disease risk for one patient |

Interactive Swagger docs are available at `/docs` once the server is running.

### Example request

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1
  }'
```

### Example response

```json
{
  "risk_percentage": 71.41,
  "risk_level": "High",
  "top_3_reasons": [
    "ST depression induced by exercise (2.3) lowers predicted risk",
    "Number of major vessels colored by fluoroscopy (0) increases predicted risk",
    "Chest pain type (3) increases predicted risk"
  ]
}
```

---

## 🛠️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/<your-username>/heart-disease-api.git
cd heart-disease-api

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the model (generates heart_model.pkl, metrics.json, and static/ plots)
python model.py

# 5. Start the API
uvicorn app:app --reload
```

The API will be live at **http://127.0.0.1:8000**, with interactive docs at
**http://127.0.0.1:8000/docs**.

---

## 📁 Project Structure

```
heart-disease-api/
├── app.py                 # FastAPI app (GET /, POST /predict)
├── model.py                # Data cleaning, EDA, training, SHAP, model saving
├── requirements.txt
├── heart_model.pkl         # Trained XGBoost model + scaler + metadata (generated)
├── metrics.json             # Test-set metrics for all 3 models (generated)
├── data/
│   └── heart.csv            # UCI Cleveland Heart Disease dataset (303 rows)
├── static/                  # EDA + SHAP plots (generated by model.py)
└── .gitignore
```

---

## ⚠️ Disclaimer

This project is for educational and portfolio purposes only. It is **not** a
medical device and must not be used for real clinical decision-making.

---

## 👤 Author

**Muskan Prajapati**
B.Tech CSE, 2026 | Aspiring AI Engineer

[LinkedIn](https://www.linkedin.com/in/muskan-prajapati-448214253)
