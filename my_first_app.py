from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize the FastAPI Web Server
app = FastAPI(title="Telecom Churn Prediction API")

# 2. Load the saved AI Brain and Scaler into the server's memory
model = joblib.load('churn_xgboost_model.joblib')
scaler = joblib.load('churn_scaler.joblib')

# 3. Define the expected incoming data structure from the website
class CustomerData(BaseModel):
    # Expecting a dictionary containing all 31 feature names and their values
    features: dict 

# 4. Create the API Endpoint that the website will call
@app.post("/predict")
def predict_churn(customer: CustomerData):
    # Convert the incoming JSON data into a single-row Pandas DataFrame
    df = pd.DataFrame([customer.features])
    
    # Scale the 3 numerical columns using the exact scaler from Day 1
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    df[num_cols] = scaler.transform(df[num_cols])
    
    # Feed the processed data to XGBoost for a prediction
    prediction = model.predict(df)
    
    # Translate the math (0 or 1) back into English for the front-end
    result = "Churn (High Risk)" if prediction[0] == 1 else "Stay (Low Risk)"
    
    return {"status": "success", "prediction": result}