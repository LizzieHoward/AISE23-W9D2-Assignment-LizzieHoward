FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app/ ./app/
COPY data/ ./data/
COPY models/ ./models/
COPY batch_infer.py .

# Expose the port for the API
EXPOSE 8000

# Set environment variables
ENV PREDICTIONS_PATH=/app/data/predictions.csv

# Run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
