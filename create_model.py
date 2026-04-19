import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

# Create dummy training data (must match your API inputs)
data = pd.DataFrame({
    "education": [1, 2, 3, 1, 2, 3],
    "experience": [1, 5, 10, 3, 7, 12],
    "location": [1, 2, 3, 2, 1, 3],
    "job_title": [1, 2, 3, 2, 1, 3],
    "age": [22, 30, 40, 28, 35, 45],
    "gender": [0, 1, 0, 1, 0, 1],
    "salary": [40000, 60000, 90000, 50000, 70000, 100000]
})

# Split features and target
X = data.drop("salary", axis=1)
y = data["salary"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "salary_model.pkl")

print("Model saved as salary_model.pkl")