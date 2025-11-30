"""Simple script to create a LogisticRegression model"""
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np

# Create simple dummy data (2 features: x1, x2)
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])
y = np.array([0, 0, 0, 1, 1, 1])

# Train simple logistic regression
model = LogisticRegression()
model.fit(X, y)

# Save model (in current directory since we're in models/)
joblib.dump(model, 'baseline.joblib')
print("Model trained and saved to baseline.joblib")
print(f"Model score: {model.score(X, y)}")
