# 📡 Telecom Customer Churn Prediction & API

An end-to-end machine learning project: from raw telecom data to a tuned
**XGBoost** classifier served behind a **FastAPI** endpoint. Built to answer a
real business question — *which customers are about to leave, while there's
still time to retain them?*

## 🎯 The problem & the trade-off

Telecoms lose revenue every time a subscriber leaves, but by the time churn
shows up in reports it's too late to act. This model flags at-risk customers
ahead of time — and it's deliberately tuned to **prioritize recall over
precision**: a missed at-risk customer (lost revenue) costs far more than a
wasted retention offer (a discount email). That business framing drove every
modeling decision below.

## 📊 Results

Test set: 1,409 customers (churn class = 374). Class imbalance 2.77 : 1,
handled with `scale_pos_weight`.

| Model                          | Churn Precision | Churn Recall | Churn F1 |
| ------------------------------ | :-------------: | :----------: | :------: |
| Logistic Regression (baseline) |      0.66       |     0.56     |   0.60   |
| XGBoost (default)              |      0.61       |     0.52     |   0.56   |
| XGBoost + class weighting      |      0.54       |     0.68     |   0.60   |
| **XGBoost tuned (final)**      |      0.53       |   **0.80**   | **0.63** |

The final model **catches 4 out of 5 customers who will actually churn**,
with **0.847 ROC-AUC** and **0.662 PR-AUC** on the held-out test set.
Hyperparameters were tuned with `RandomizedSearchCV` (60 fits, scored on F1).
The strongest churn signal in the data: **contract type** — two-year contracts
are the best retention predictor.

### Is this the ceiling? (verified: yes)

To check whether the model left performance on the table, I ran a wider
follow-up experiment: a 40-candidate regularized search (`min_child_weight`,
`gamma`, `reg_lambda`) scored on average precision, plus decision-threshold
tuning selected on out-of-fold training predictions (never on the test set).
The result **matched but did not beat** this model on every metric (±0.01).
On this dataset, ~0.63 churn-class F1 is the practical ceiling — further gains
would have to come from richer features or more data, not from hyperparameters.

## 🏗️ Project structure

```
├── app.py                          # FastAPI service: POST /predict, GET /health
├── train.py                        # one-command reproducible training pipeline
├── requirements.txt
├── models/
│   ├── churn_xgboost_model.joblib  # trained, tuned classifier
│   ├── churn_scaler.joblib         # StandardScaler fitted on training data
│   └── model_meta.json             # feature order + hyperparameters (used by the API)
└── notebooks/
    └── telco_customer_churn.ipynb  # full pipeline: EDA → cleaning → training → tuning
```

## 🚀 Run the API locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

Then open **http://127.0.0.1:8000/docs** for interactive Swagger docs, or POST
directly:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {"tenure": 12, "MonthlyCharges": 70.35, "TotalCharges": 845.5, "Contract_Two year": 0, ...}}'
```

The payload takes all 30 one-hot encoded features (the API validates them and
returns a helpful 422 listing anything missing). Response:

```json
{
  "status": "success",
  "churn": true,
  "churn_probability": 0.6629,
  "risk_tier": "High",
  "prediction": "Churn (High Risk)"
}
```

The probability lets a retention team **rank** customers instead of just
flagging them; `risk_tier` buckets it for dashboards.

## 🔁 Reproduce the model

```bash
python train.py --csv Telco.csv
```

Retrains from the raw CSV with the winning hyperparameters and rewrites
`models/` (classification report + ROC/PR-AUC printed on completion).

## 📚 Dataset

[IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
(7,043 customers, 21 raw features). The CSV is intentionally not committed —
download it and save as `Telco.csv` next to the notebook to reproduce
training.

## 🔬 Methodology (see the notebook)

1. **Cleaning** — coerce `TotalCharges` to numeric; impute blank values for
   new customers
2. **Encoding** — one-hot encoding for categoricals; `StandardScaler` for the
   3 numeric features
3. **Imbalance** — `scale_pos_weight = 2.77` (computed from the training
   split, not guessed)
4. **Tuning** — `RandomizedSearchCV` over learning rate, depth, estimators,
   and subsampling
5. **Serving** — model + scaler exported with joblib and loaded by the FastAPI
   app

## 👤 Author

**Marco Andrei R. Belen** — Computer Science (Machine Learning) student, NU Lipa

[Portfolio](https://marcobelen.vercel.app) · [GitHub](https://github.com/codrei) · [LinkedIn](https://www.linkedin.com/in/marco-andrei-belen/)
