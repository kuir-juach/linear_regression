import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Logging Configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize FastAPI app
app = FastAPI()

# Load Model
MODEL_PATH = "model.pkl"  # Ensure this matches your model file location
try:
    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded successfully from '%s'.", MODEL_PATH)
except Exception as e:
    logging.error("Error loading model: %s", str(e))
    raise RuntimeError("Model could not be loaded. Check the file path and format.")

# Request Body Schema
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float

# Root Endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Prediction API!"}

# Prediction Endpoint
@app.post("/predict")
async def predict(data: InputData):
    try:
        # Prepare input data
        input_df = pd.DataFrame([data.dict()])
        logging.info("Received data for prediction: %s", input_df.to_dict(orient="records"))

        # Make prediction
        prediction = model.predict(input_df)
        logging.info("Prediction result: %s", prediction)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        logging.error("Prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail="Prediction failed. Check input data or model.")

# Start server
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Use PORT environment variable or default to 8000
    logging.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
