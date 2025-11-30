"""Metrics calculation module for model predictions."""

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def calculate_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Calculate classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary containing accuracy, precision, recall, and f1 score
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def load_predictions(predictions_path: str) -> pd.DataFrame:
    """Load predictions from CSV file.
    
    Args:
        predictions_path: Path to the predictions CSV file
        
    Returns:
        DataFrame containing predictions with 'actual' and 'predicted' columns
    """
    return pd.read_csv(predictions_path)