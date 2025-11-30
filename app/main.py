"""REST API for ML Model Serving"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import generate_latest
import joblib
import time
import numpy as np
from pathlib import Path
from app.metrics import predictions_total, prediction_duration_seconds


app = FastAPI(title="ML Model API", version="1.0.0")

# Load model at startup
MODEL_PATH = Path("models/baseline.joblib")
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None
    print(f"Warning: Model not found at {MODEL_PATH}. /predict endpoint will not work.")
except Exception as e:
    model = None
    print(f"Error loading model: {e}")


# Pydantic models
class PredictRequest(BaseModel):
    x1: float
    x2: float


class PredictResponse(BaseModel):
    score: float
    model_version: str


class HealthResponse(BaseModel):
    status: str


# Endpoints
@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint"""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Make prediction on input features"""
    # Check if model is loaded
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model not loaded. Please ensure models/baseline.joblib exists."
        )
    
    # Start timing
    start_time = time.time()
    
    try:
        # Prepare input features
        features = np.array([[request.x1, request.x2]])
        
        # Make prediction
        prediction = model.predict_proba(features)[0][1]  # Get probability of positive class
        
        # Track metrics
        predictions_total.inc()
        prediction_duration_seconds.observe(time.time() - start_time)
        
        return {
            "score": float(prediction),
            "model_version": "v1.0"
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics"""
    return Response(content=generate_latest(), media_type="text/plain")


