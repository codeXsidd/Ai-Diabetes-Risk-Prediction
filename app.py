
from flask import Flask, render_template, request
import numpy as np
import joblib
import os
app = Flask(__name__)

model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    values = [float(x) for x in request.form.values()]
    data = np.array([values])
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)[0]

    if prediction == 1:
        result = "High Diabetes Risk ⚠️"
    else:
        result = "Low Diabetes Risk ✅"

    return render_template("index.html", prediction_text=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
