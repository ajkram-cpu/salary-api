from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

# Load your model
model = joblib.load("salary_model.pkl")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        prediction = model.predict(input_df)[0]

        return jsonify({
            "prediction": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)