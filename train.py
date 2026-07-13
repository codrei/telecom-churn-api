"""Reproducible training pipeline for the Telco churn model.

Rebuilds the exact model served by app.py from the raw CSV:
data cleaning -> encoding -> stratified split -> scaling -> XGBoost with the
hyperparameters found via RandomizedSearchCV (n_iter=20, scoring='f1', cv=3;
see notebooks/telco_customer_churn.ipynb for the search itself).

Usage:
    python train.py [--csv Telco.csv]

Dataset: IBM Telco Customer Churn (not committed; see README).
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Winning configuration from RandomizedSearchCV (see notebook).
BEST_PARAMS = {
    "learning_rate": 0.01,
    "max_depth": 6,
    "n_estimators": 200,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
}


def load_and_prepare(csv_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    # Blank TotalCharges are brand-new customers (tenure 0) -> impute 0.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df = pd.get_dummies(df, drop_first=True, dtype=int)
    return df.drop("Churn", axis=1), df["Churn"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(ROOT / "Telco.csv"))
    args = parser.parse_args()

    X, y = load_and_prepare(Path(args.csv))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train[NUMERIC_COLS])
    X_train[NUMERIC_COLS] = scaler.transform(X_train[NUMERIC_COLS])
    X_test[NUMERIC_COLS] = scaler.transform(X_test[NUMERIC_COLS])

    # Weight the minority (churn) class by the observed imbalance.
    ratio = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        random_state=42,
        scale_pos_weight=ratio,
        eval_metric="logloss",
        n_jobs=-1,
        **BEST_PARAMS,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, model.predict(X_test), digits=3))
    print(f"ROC-AUC: {roc_auc_score(y_test, proba):.3f}")
    print(f"PR-AUC:  {average_precision_score(y_test, proba):.3f}")

    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(model, models_dir / "churn_xgboost_model.joblib")
    joblib.dump(scaler, models_dir / "churn_scaler.joblib")
    (models_dir / "model_meta.json").write_text(
        json.dumps(
            {
                "feature_names": list(X.columns),
                "numeric_cols": NUMERIC_COLS,
                "imbalance_ratio": round(float(ratio), 4),
                "params": BEST_PARAMS,
            },
            indent=2,
        )
    )
    print(f"\nSaved model, scaler, and metadata to {models_dir}/")


if __name__ == "__main__":
    main()
