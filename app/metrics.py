"""Prometheus metrics for monitoring predictions"""
from prometheus_client import Counter, Histogram

# Track total number of predictions
predictions_total = Counter(
    'predictions_total',
    'Total number of predictions made'
)

# Track prediction latency
prediction_duration_seconds = Histogram(
    'prediction_duration_seconds',
    'Time spent processing predictions',
    buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)