import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Create FastAPI app
app = FastAPI()

# Load your model (make sure model.pkl is in the same directory as this script or provide correct path)
MODEL_PATH = "model.pkl"
try:
    model = joblib.load(MODEL_PATH)
    logging.info("Model loaded successfully from '%s'.", MODEL_PATH)
except Exception as e:
    logging.error("Error loading model: %s", str(e))
    raise RuntimeError("Model could not be loaded. Check the file path and format.")

# Define InputData schema
class InputData(BaseModel):
    feature1: float
    feature2: float
    feature3: float

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to the FastAPI Prediction API!"}

# Prediction endpoint
@app.post("/predict")
async def predict(data: InputData):
    try:
        # Convert input data to pandas DataFrame for prediction
        input_df = pd.DataFrame([data.dict()])
        logging.info("Received data for prediction: %s", input_df.to_dict(orient="records"))

        # Make prediction
        prediction = model.predict(input_df)
        logging.info("Prediction result: %s", prediction)

        return {"prediction": prediction.tolist()}
    except Exception as e:
        logging.error("Prediction error: %s", str(e))
        raise HTTPException(status_code=500, detail="Prediction failed. Check input data or model.")

# Entry point to start the FastAPI app
if __name__ == "__main__":
    import uvicorn

    # Get the port from the environment (Render provides this)
    port = int(os.environ.get("PORT", 8000))
    logging.info(f"Starting server on port {port}")

    # Run FastAPI on 0.0.0.0 so that it's accessible externally
    uvicorn.run(app, host="0.0.0.0", port=port)
