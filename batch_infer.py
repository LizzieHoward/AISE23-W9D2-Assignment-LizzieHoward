"""Batch inference script for model predictions."""

import argparse
import os

import joblib
import pandas as pd


def load_model(model_path: str):
    """Load a trained model from disk.
    
    Args:
        model_path: Path to the saved model file
        
    Returns:
        Loaded model object
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    return joblib.load(model_path)


def load_input_data(input_path: str) -> pd.DataFrame:
    """Load input data from CSV file.
    
    Args:
        input_path: Path to the input CSV file
        
    Returns:
        DataFrame containing input features
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")
    return pd.read_csv(input_path)


def run_inference(model, input_data: pd.DataFrame, label_column: str = "label") -> pd.DataFrame:
    """Run batch inference on input data.
    
    Args:
        model: Trained model object
        input_data: DataFrame containing input features
        label_column: Name of the label column (if present in input)
        
    Returns:
        DataFrame with actual and predicted values
    """
    # Separate features and labels if label column exists
    if label_column in input_data.columns:
        features = input_data.drop(columns=[label_column])
        actual = input_data[label_column]
    else:
        features = input_data
        actual = None
    
    # Run predictions
    predictions = model.predict(features)
    
    # Create output DataFrame
    if actual is not None:
        result_df = pd.DataFrame({
            "actual": actual,
            "predicted": predictions
        })
    else:
        result_df = pd.DataFrame({
            "predicted": predictions
        })
    
    return result_df


def save_predictions(predictions: pd.DataFrame, output_path: str):
    """Save predictions to CSV file.
    
    Args:
        predictions: DataFrame containing predictions
        output_path: Path to save the output CSV file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def main():
    """Main function for batch inference."""
    parser = argparse.ArgumentParser(description="Run batch inference on input data")
    parser.add_argument(
        "--model",
        type=str,
        default="models/baseline.joblib",
        help="Path to the trained model file"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input.csv",
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/predictions.csv",
        help="Path to save the output predictions"
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default="label",
        help="Name of the label column in input data"
    )
    
    args = parser.parse_args()
    
    print(f"Loading model from {args.model}...")
    model = load_model(args.model)
    
    print(f"Loading input data from {args.input}...")
    input_data = load_input_data(args.input)
    print(f"Input data shape: {input_data.shape}")
    
    print("Running inference...")
    predictions = run_inference(model, input_data, args.label_column)
    
    save_predictions(predictions, args.output)
    print("Batch inference completed successfully!")


if __name__ == "__main__":
    main()
