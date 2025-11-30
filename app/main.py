"""FastAPI application for serving model metrics."""

import os

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from app.metrics import calculate_metrics, load_predictions

app = FastAPI(
    title="ML Metrics API",
    description="API for serving machine learning model metrics",
    version="1.0.0"
)


@app.get("/")
def root():
    """Root endpoint returning a welcome message."""
    return {"message": "Welcome to the ML Metrics API"}


@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/metrics")
def get_metrics():
    """Get model performance metrics from predictions.
    
    Returns:
        JSON response containing accuracy, precision, recall, and f1 score
    """
    predictions_path = os.getenv("PREDICTIONS_PATH", "data/predictions.csv")
    
    if not os.path.exists(predictions_path):
        raise HTTPException(
            status_code=404,
            detail=f"Predictions file not found at {predictions_path}"
        )
    
    try:
        predictions_df = load_predictions(predictions_path)
        
        if "actual" not in predictions_df.columns or "predicted" not in predictions_df.columns:
            raise HTTPException(
                status_code=400,
                detail="Predictions file must contain 'actual' and 'predicted' columns"
            )
        
        metrics = calculate_metrics(
            predictions_df["actual"],
            predictions_df["predicted"]
        )
        
        return JSONResponse(content={"metrics": metrics})
        
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Predictions file is empty")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)