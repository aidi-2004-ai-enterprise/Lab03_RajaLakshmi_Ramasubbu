"""
main.py
FastAPI app for predicting penguin species using XGBoost model.
"""
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from enum import Enum
import logging
from typing import Dict, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Enums & Model
class Island(str, Enum):
    Torgersen = "Torgersen"
    Biscoe = "Biscoe"
    Dream = "Dream"

class Sex(str, Enum):
    male = "male"
    female = "female"

class PenguinFeatures(BaseModel):
    bill_length_mm: float
    bill_depth_mm: float
    flipper_length_mm: float
    body_mass_g: float
    year: int
    sex: Sex
    island: Island

# Load model and encoders at startup
MODEL_PATH = "app/data/model.pkl"
model_bundle = joblib.load(MODEL_PATH)
model = model_bundle["model"]
model_columns = model_bundle["columns"]
label_encoder = model_bundle["label_encoder"]

logger.info(f"Model and encoders loaded from {MODEL_PATH}")

app = FastAPI(
    title="Penguin Species Classifier",
    description="Predict penguin species using XGBoost",
    version="1.0"
)

@app.post("/predict", response_model=Dict[str, str])
def predict(features: PenguinFeatures)-> Dict[str, str]:
    """
        Predict the penguin species from the input features.

    Args:
        features (PenguinFeatures): Input features as a Pydantic model.
    Returns:
        dict: Predicted species.
    """
    try:
        # Convert input to DataFrame
        input_df = pd.DataFrame([features.dict()])

        # One-hot encode 'sex' and 'island'
        input_df = pd.get_dummies(input_df, columns=["sex", "island"])

        # Add any missing columns (set to 0)
        for col in model_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[model_columns]

        # Predict
        pred = model.predict(input_df)[0]
        species = label_encoder.inverse_transform([pred])[0]

        logger.info(f"Prediction success: {species}")
        return {"species": species}

    except Exception as e:
        logger.debug(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid input or internal error.")

# Custom error handler for input validation (enum restrictions)
@app.exception_handler(HTTPException)
async def custom_http_exception_handler(request, exc: HTTPException)-> Any:
    """
    Custom HTTPException handler that logs details.
    """
    logger.debug(f"HTTPException: {exc.detail}")
    return await FastAPI.default_exception_handler(request, exc)

# Root endpoint
@app.get("/", response_model=Dict[str, str])
def read_root()-> Dict[str,str]:
    """
    Root endpoint returns a welcome message.
    """
    return {"message": "Welcome to the Penguin Species Predictor!"}
