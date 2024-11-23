from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import logging

# Initialize the FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Environment variables for dynamic configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")  # Default to allow all origins
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")  # Default model path

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in CORS_ORIGINS.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    model_retrained = joblib.load(MODEL_PATH)
    important_features = model_retrained.feature_names_in_.tolist()
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise RuntimeError("Model loading failed. Please check the model file and path.")

# Define the Pydantic model for input validation
class PredictionRequest(BaseModel):
    Roll_No: int
    IA1: int
    IA2: int

    class Config:
        schema_extra = {
            "example": {
                "Roll_No": 1,
                "IA1": 75,
                "IA2": 85,
            }
        }


@app.post("/predict")
async def predict(request: PredictionRequest):
    """
    Predict based on the input data.
    """
    try:
        # Log the input request
        logging.info(f"Received request data: {request}")

        # Prepare the input DataFrame
        request_data = request.dict()
        df_input = pd.DataFrame([request_data])

        # Map the input fields to model features
        input_mapping = {
            "Roll_No": "Roll NO",
            "IA1": "IA1",
            "IA2": "IA2",
        }
        df_input.rename(columns=input_mapping, inplace=True)

        # Ensure the DataFrame matches the model's feature order
        df_input = df_input[important_features]

        # Perform the prediction
        prediction = model_retrained.predict(df_input)

        # Return the prediction
        return {"prediction": prediction[0]}

    except KeyError as ke:
        logging.error(f"Missing or mismatched input feature: {ke}")
        raise HTTPException(status_code=400, detail=f"Input feature error: {ke}")

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {e}")


@app.get("/")
async def root():
    """
    Health check endpoint.
    """
    return {"message": "Welcome to the FastAPI Prediction API!"}
