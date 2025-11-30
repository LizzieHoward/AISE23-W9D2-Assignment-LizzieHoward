# AISE23-W9D2-Assignment-LizzieHoward

## ML Model API with FastAPI and Prometheus

A machine learning model API built with FastAPI, featuring Prometheus metrics monitoring and batch inference capabilities.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
cd models
python train_model.py
cd ..
```

This creates a simple LogisticRegression model and saves it to `models/baseline.joblib`.

### 3. Run the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## Usage

### API Endpoints

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Make Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"x1\": 3.5, \"x2\": 4.5}"
```

#### View Prometheus Metrics
```bash
curl http://localhost:8000/metrics
```

### Batch Inference

Process multiple predictions from a CSV file:

```bash
python batch_infer.py data/input.csv data/predictions.csv
```

**Input CSV Format:**
- Must have `x1` and `x2` columns
- Can have additional columns (will be preserved)

**Output:**
- Creates CSV with original data plus `prediction` column (probability 0.0-1.0)
- Prints processing statistics (rows processed, time taken)

## API Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI application
│   └── metrics.py       # Prometheus instrumentation
├── data/
│   ├── input.csv        # Sample input data (100 rows)
│   └── predictions.csv  # Sample output from batch script
├── models/
│   ├── baseline.joblib  # Trained model
│   └── train_model.py   # Model training script
├── batch_infer.py       # Batch inference script
├── requirements.txt     # Dependencies
└── README.md
```

## Troubleshooting

### Port already in use
Change to a different port:
```bash
uvicorn app.main:app --port 8001
```

### Module not found
Check that `app/__init__.py` exists

### Metrics not showing
Import prometheus_client and ensure counter.inc() is called before returning