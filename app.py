"""
app.py
------
FastAPI service that serves the trained XGBoost heart-disease-risk model.

Endpoints:
    GET  /          -> health check
    POST /predict    -> risk_percentage, risk_level, and the top 3 SHAP-based
                        reasons behind that particular prediction

Run locally:
    uvicorn app:app --reload
"""

from contextlib import asynccontextmanager

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "heart_model.pkl"
HIGH_RISK_THRESHOLD = 50.0  # risk_percentage >= this -> "High"

# Populated at startup by load_artifacts()
artifact = {}
explainer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model, scaler and SHAP explainer once, at startup."""
    global artifact, explainer
    try:
        artifact = joblib.load(MODEL_PATH)
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"{MODEL_PATH} not found. Run `python model.py` first to train "
            "and save the model."
        ) from exc

    explainer = shap.TreeExplainer(artifact["model"])
    print(f"Loaded model: {artifact['model_name']}  |  SHAP explainer ready")
    yield
    artifact.clear()


app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts heart disease risk from 13 clinical features "
    "(UCI Cleveland Heart Disease dataset) and explains each prediction with SHAP.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Request / response schemas
# --------------------------------------------------------------------------- #
class PatientData(BaseModel):
    age: int = Field(..., ge=1, le=120, description="Age in years", examples=[63])
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female", examples=[1])
    cp: int = Field(..., ge=0, le=3, description="Chest pain type (0-3)", examples=[3])
    trestbps: int = Field(..., ge=50, le=250, description="Resting blood pressure (mm Hg)", examples=[145])
    chol: int = Field(..., ge=100, le=700, description="Serum cholesterol (mg/dl)", examples=[233])
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl (1 = true)", examples=[1])
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)", examples=[0])
    thalach: int = Field(..., ge=50, le=250, description="Max heart rate achieved", examples=[150])
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina (1 = yes)", examples=[0])
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise", examples=[2.3])
    slope: int = Field(..., ge=0, le=2, description="Slope of peak exercise ST segment (0-2)", examples=[0])
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored by fluoroscopy (0-4)", examples=[0])
    thal: int = Field(..., ge=0, le=3, description="Thalassemia result (0-3)", examples=[1])

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
                    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
                    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    risk_percentage: float
    risk_level: str
    top_3_reasons: list[str]


# --------------------------------------------------------------------------- #
# Explanation helpers
# --------------------------------------------------------------------------- #
def _describe_feature(feature: str, value, shap_value: float) -> str:
    """
    Turn one (feature, value, SHAP contribution) triple into a short,
    human-readable reason for the prediction.
    """
    label = artifact["feature_descriptions"].get(feature, feature)
    direction = "increases" if shap_value > 0 else "lowers"

    # A handful of features read better with a short qualitative note.
    if feature == "sex":
        value_str = "male" if value == 1 else "female"
    elif feature in ("fbs", "exang"):
        value_str = "yes" if value == 1 else "no"
    else:
        value_str = str(value)

    return f"{label} ({value_str}) {direction} predicted risk"


def explain_prediction(input_df: pd.DataFrame, top_n: int = 3) -> list[str]:
    """Return the top_n features driving this specific prediction, by |SHAP value|."""
    shap_values = explainer(input_df)
    values = shap_values.values[0]
    features = input_df.columns.tolist()

    order = np.argsort(np.abs(values))[::-1][:top_n]
    reasons = [
        _describe_feature(features[i], input_df.iloc[0, i], values[i]) for i in order
    ]
    return reasons


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/")
def health_check():
    """Simple health check / service info."""
    return {
        "status": "ok",
        "message": "Heart Disease Risk Prediction API is running",
        "model": artifact.get("model_name", "not loaded"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientData):
    """
    Predict heart disease risk for one patient and explain the prediction
    with the top 3 SHAP feature contributions.
    """
    if not artifact:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    input_df = pd.DataFrame([patient.model_dump()])[artifact["feature_names"]]

    proba = artifact["model"].predict_proba(input_df)[0][1]
    risk_percentage = round(float(proba) * 100, 2)
    risk_level = "High" if risk_percentage >= HIGH_RISK_THRESHOLD else "Low"
    reasons = explain_prediction(input_df)

    return PredictionResponse(
        risk_percentage=risk_percentage,
        risk_level=risk_level,
        top_3_reasons=reasons,
    )
