"""Telecom Churn Prediction API.

Serves the tuned XGBoost model from notebooks/telco_customer_churn.ipynb
behind a FastAPI endpoint. Interactive docs available at /docs when running.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MODELS_DIR = Path(__file__).parent / "models"

# Numeric features that must be scaled with the same StandardScaler
# fitted during training — everything else is one-hot encoded 0/1.
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

app = FastAPI(
    title="Telecom Churn Prediction API",
    description=(
        "Predicts whether a telecom customer is at risk of churning. "
        "Tuned XGBoost classifier optimized for recall on the churn class "
        "(0.80 recall / 0.63 F1)."
    ),
    version="1.0.0",
)

model = joblib.load(MODELS_DIR / "churn_xgboost_model.joblib")
scaler = joblib.load(MODELS_DIR / "churn_scaler.joblib")


class CustomerData(BaseModel):
    """The 31 model features (one-hot encoded) for a single customer."""

    features: dict


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict")
def predict_churn(customer: CustomerData) -> dict:
    df = pd.DataFrame([customer.features])
    df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    churn = bool(model.predict(df)[0])
    return {
        "status": "success",
        "churn": churn,
        "prediction": "Churn (High Risk)" if churn else "Stay (Low Risk)",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
