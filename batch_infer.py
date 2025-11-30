"""Batch inference script for processing CSV files"""
import csv
import sys
import time
import joblib
import numpy as np
from pathlib import Path


def main():
    # Check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python batch_infer.py <input_csv> <output_csv>")
        print("Example: python batch_infer.py data/input.csv data/predictions.csv")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Start timing
    start_time = time.time()
    
    # Load model
    model_path = Path("models/baseline.joblib")
    try:
        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Read input CSV
    try:
        print(f"Reading input data from {input_path}...")
        with open(input_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
            if not rows:
                print("Error: Input CSV is empty")
                sys.exit(1)
            
            # Check for required columns
            fieldnames = reader.fieldnames
            if 'x1' not in fieldnames or 'x2' not in fieldnames:
                print(f"Error: Input CSV must have 'x1' and 'x2' columns")
                print(f"Found columns: {fieldnames}")
                sys.exit(1)
    
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading input file: {e}")
        sys.exit(1)
    
    # Generate predictions
    print(f"Processing {len(rows)} rows...")
    predictions = []
    
    try:
        for row in rows:
            # Extract features
            x1 = float(row['x1'])
            x2 = float(row['x2'])
            features = np.array([[x1, x2]])
            
            # Get prediction probability (probability of positive class)
            prob = model.predict_proba(features)[0][1]
            predictions.append(prob)
    
    except ValueError as e:
        print(f"Error: Invalid data format in CSV. x1 and x2 must be numbers.")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)
    
    # Write output CSV
    try:
        print(f"Writing predictions to {output_path}...")
        output_fieldnames = fieldnames + ['prediction']
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=output_fieldnames)
            writer.writeheader()
            
            for row, pred in zip(rows, predictions):
                row['prediction'] = pred
                writer.writerow(row)
    
    except Exception as e:
        print(f"Error writing output file: {e}")
        sys.exit(1)
    
    # Calculate statistics
    elapsed_time = time.time() - start_time
    rows_per_second = len(rows) / elapsed_time if elapsed_time > 0 else 0
    
    # Print statistics
    print("\n" + "="*50)
    print("Batch Inference Complete!")
    print("="*50)
    print(f"Rows processed: {len(rows)}")
    print(f"Time taken: {elapsed_time:.3f} seconds")
    print(f"Processing speed: {rows_per_second:.2f} rows/second")
    print(f"Output saved to: {output_path}")
    print("="*50)


if __name__ == "__main__":
    main()