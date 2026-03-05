# 📡 Telecom Customer Churn Prediction & API

An end-to-end Machine Learning pipeline developed to predict customer churn with **XGBoost**, featuring a live **FastAPI** deployment.

## 🚀 Project Overview
This project identifies high-risk customers likely to cancel their telecom subscriptions. By analyzing features like contract type, tenure, and monthly charges, the model provides actionable business insights to improve retention.

* **Final Model:** Tuned XGBoost Classifier
* **Performance:** **0.63 F1-Score** (optimized for the minority churn class)
* **Key Driver:** Two-Year Contracts (highest feature importance)

## 🛠️ Tech Stack
* **Data Science:** Python, Pandas, Scikit-learn, XGBoost
* **Deployment:** FastAPI, Uvicorn, Joblib
* **Environment:** Conda (environment_ko)

## 📦 How to Use the API
1. Activate the environment: `conda activate environment_ko`
2. Start the server: `uvicorn my_first_app:app --reload`
3. Send a POST request to `http://127.0.0.1:8000/predict` with customer features.

## 📊 Methodology
Included in this repository is the `My Customer Churn Prediction.ipynb` notebook, which details:
* Data cleaning and One-Hot Encoding
* Handling class imbalance via `scale_pos_weight`
* RandomizedSearchCV for hyperparameter tuning