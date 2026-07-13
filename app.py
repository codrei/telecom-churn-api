"""Telecom Churn Prediction API.

Serves the tuned XGBoost model trained by train.py (methodology in
notebooks/telco_customer_churn.ipynb). Interactive docs at /docs.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

MODELS_DIR = Path(__file__).parent / "models"

app = FastAPI(
    title="Telecom Churn Prediction API",
    description=(
        "Predicts whether a telecom customer is at risk of churning. "
        "Tuned XGBoost classifier optimized for recall on the churn class "
        "(0.80 recall / 0.63 F1 / 0.85 ROC-AUC)."
    ),
    version="2.0.0",
)

model = joblib.load(MODELS_DIR / "churn_xgboost_model.joblib")
scaler = joblib.load(MODELS_DIR / "churn_scaler.joblib")
meta = json.loads((MODELS_DIR / "model_meta.json").read_text())

FEATURE_NAMES: list[str] = meta["feature_names"]
NUMERIC_COLS: list[str] = meta["numeric_cols"]


class CustomerData(BaseModel):
    """The model's input features (one-hot encoded) for a single customer."""

    features: dict


def risk_tier(probability: float) -> str:
    if probability >= 0.65:
        return "High"
    if probability >= 0.35:
        return "Medium"
    return "Low"


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_features": len(FEATURE_NAMES)}


@app.post("/predict")
def predict_churn(customer: CustomerData) -> dict:
    missing = [f for f in FEATURE_NAMES if f not in customer.features]
    if missing:
        raise HTTPException(
            status_code=422,
            detail={"error": "Missing required features", "missing": missing},
        )

    # Reorder into the exact column order the model was trained on —
    # dict insertion order from the client must never decide feature order.
    df = pd.DataFrame([customer.features])[FEATURE_NAMES]
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])

    probability = float(model.predict_proba(df)[0, 1])
    churn = probability >= 0.5
    return {
        "status": "success",
        "churn": churn,
        "churn_probability": round(probability, 4),
        "risk_tier": risk_tier(probability),
        "prediction": "Churn (High Risk)" if churn else "Stay (Low Risk)",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
