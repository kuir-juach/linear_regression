from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import logging

# Initialize the app
app = FastAPI()

# Configure CORS middleware
origins = [
    "http://localhost",         
    "http://localhost:3000",    
    "http://127.0.0.1:8000",    
    "*"                         
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Enable logging for debugging
logging.basicConfig(level=logging.INFO)

# Load the retrained model
model_retrained = joblib.load('model.pkl')

# Extract the feature names used during training
important_features = model_retrained.feature_names_in_.tolist()

# Define the Pydantic model for request validation
class PredictionRequest(BaseModel):
    # Dynamically generate fields based on the feature names
    Roll_No: int
    IA1: int
    IA2: int
  

    class Config:
        json_schema_extra = {  # Updated to use json_schema_extra
            "example": {
                "Roll_No": 1.0,
                "IA1": 0.0,
                "IA2": 0.0,
                           }
        }

@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Log the incoming request
        logging.info(f"Received request: {request}")

        # Convert input data to a DataFrame
        request_data = request.dict()
        df_input = pd.DataFrame([request_data])

        # Map input fields to model feature names
        input_mapping = {
            "Roll_No": "Roll NO",
            "IA1": "IA1",
            "IA2": "IA2"
        }
        df_input.rename(columns=input_mapping, inplace=True)

        # Ensure input DataFrame matches the model's feature order
        df_input = df_input[important_features]

        # Make a prediction
        prediction = model_retrained.predict(df_input)

        # Return the prediction result
        return {"prediction": prediction[0]}

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
