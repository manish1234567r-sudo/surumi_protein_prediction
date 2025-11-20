from flask import Flask, request, jsonify, render_template_string
import numpy as np
import joblib
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Input

app = Flask(__name__)

MODEL_FILE = "protein_model.keras"
SCALER_FILE = "scaler.pkl"


# ----------------------------
# TRAIN MODEL (Runs ONLY if model is missing)
# ----------------------------
def train_and_save_model():
    print("Training new model...")

    # Dummy training data (replace with real)
    X = np.array([
        [6.8, 40, 20, 200],
        [7.2, 50, 30, 250],
        [6.5, 35, 25, 220],
        [7.0, 45, 20, 300]
    ])

    y = np.array([40, 45, 35, 50])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = Sequential([
        Input(shape=(4,)),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    model.fit(X_scaled, y, epochs=50, verbose=0)

    # Save in new Keras format (NO .h5)
    model.save(MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    print("Model + Scaler saved.")


# ----------------------------
# SAFE MODEL LOAD (Fixes keras.metrics.mse error)
# ----------------------------
def load_safe_model():
    try:
        return load_model(MODEL_FILE, compile=False)
    except:
        print("Model corrupted. Retraining...")
        train_and_save_model()
        return load_model(MODEL_FILE, compile=False)


# ----------------------------
# Load or Train model on startup
# ----------------------------
if not os.path.exists(MODEL_FILE) or not os.path.exists(SCALER_FILE):
    train_and_save_model()

model = load_safe_model()
scaler = joblib.load(SCALER_FILE)


# ----------------------------
# PREDICT FUNCTION
# ----------------------------
def predict_protein(pH, temp, time, volume):
    X = np.array([[pH, temp, time, volume]])
    X_scaled = scaler.transform(X)
    return float(model.predict(X_scaled)[0][0])


# ----------------------------
# FLASK ROUTES
# ----------------------------
@app.route("/")
def home():
    return """
    <h2>Surimi Protein Prediction</h2>
    <form method="post" action="/predict">
        pH: <input name="ph" /><br>
        Temp: <input name="temp" /><br>
        Time: <input name="time" /><br>
        Volume: <input name="vol" /><br>
        <button>Predict</button>
    </form>
    """


@app.route("/predict", methods=["POST"])
def predict():
    pH = float(request.form["ph"])
    temp = float(request.form["temp"])
    time = float(request.form["time"])
    vol = float(request.form["vol"])

    result = predict_protein(pH, temp, time, vol)

    return f"<h2>Predicted Protein Yield: {result:.2f}</h2>"


# ----------------------------
# PORT BINDING (Fix for Render)
# ----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
