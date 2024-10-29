from fastapi import FastAPI
from pydantic import BaseModel
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from sqlalchemy import create_engine

app = FastAPI()

# Define request model
class UserRequest(BaseModel):
    user_id: int
    db_url: str = None

@app.post("/predict")
async def predict_cash_flow(request: UserRequest):
    user_id = request.user_id
    db_url = request.db_url or os.getenv('DB_URL')

    # Call the main function from your script with the given user_id and db_url
    predictions = main(user_id, db_url)
    return {"predictions": predictions.tolist()}
