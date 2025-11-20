# app.py
import os
from pathlib import Path
from flask import Flask, render_template_string, request
import numpy as np
import joblib

# Keras / TensorFlow imports
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

app = Flask(__name__)

# Filenames
MODEL_FILE = "protein_model.h5"
SCALER_FILE = "scaler.pkl"

# Small dataset (your provided data)
DATA = np.array([
    [4, 55, 20, 0.320, 0.43],
    [6.5, 55, 20, 0.320, 0.36],
    [4.0, 65, 20, 0.320, 1.00],
    [6.5, 65, 20, 0.320, 1.15],
    [4.0, 60, 15, 0.320, 1.17],
    [6.5, 60, 15, 0.320, 0.86],
    [4.0, 60, 25, 0.320, 1.06],
    [6.5, 60, 25, 0.320, 0.99],
    [5.5, 55, 15, 0.320, 0.95],
    [5.5, 65, 15, 0.320, 0.98],
    [5.5, 55, 25, 0.320, 1.30],
    [5.5, 65, 25, 0.320, 1.04],
])

# Training helper
def train_and_save_model():
    from sklearn.preprocessing import MinMaxScaler
    X = DATA[:, :-1]
    y = DATA[:, -1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_FILE)
    # Build model
    model = Sequential([
        Dense(16, input_dim=4, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.01), loss='mse')
    model.fit(X_scaled, y, epochs=400, verbose=0)
    model.save(MODEL_FILE)
    print("Trained and saved model + scaler.")

# Ensure model + scaler exist (train on first run if missing)
if not (Path(MODEL_FILE).exists() and Path(SCALER_FILE).exists()):
    print("Model or scaler missing. Training model now...")
    train_and_save_model()
else:
    print("Model and scaler found. Skipping training.")

# Load model + scaler once for all requests
MODEL = load_model(MODEL_FILE)
SCALER = joblib.load(SCALER_FILE)

# Prediction function
def predict_protein(pH, temp, time_min, volume):
    base_volume = 0.320  # same base used in training data
    X_input = np.array([[pH, temp, time_min, base_volume]])
    X_scaled = SCALER.transform(X_input)
    base_yield = float(MODEL.predict(X_scaled, verbose=0)[0][0])
    # scale linearly with volume
    adjusted_yield = base_yield * (volume / base_volume)
    return adjusted_yield

# Template (same as your UI but uses the server variables)
TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <title>Protein Yield Predictor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Poppins', sans-serif; background: radial-gradient(circle at top, #0a1a2f, #000814); color: white; min-height: 100vh; }
        .container { display:flex; justify-content:center; padding:40px 10px; }
        .card { width:100%; max-width:480px; background: rgba(255,255,255,0.06); padding:30px; border-radius:18px; box-shadow:0 8px 40px rgba(0,255,255,0.08); }
        h1 { text-align:center; margin-bottom:16px; text-shadow:0 0 8px cyan; }
        label { font-size:14px; margin-top:12px; display:block; }
        select, input { width:100%; padding:12px; margin-top:6px; border-radius:10px; background: rgba(255,255,255,0.12); border:1px solid rgba(255,255,255,0.12); color:#000; }
        select:focus, input:focus { outline:none; box-shadow:0 0 10px cyan; }
        button { width:100%; margin-top:18px; padding:12px; border:none; background:linear-gradient(90deg,#00eaff,#0077ff); color:white; border-radius:10px; font-weight:600; cursor:pointer; }
        .output-box { margin-top:18px; padding:14px; background:rgba(0,0,0,0.6); border-radius:10px; text-align:center; box-shadow:0 0 10px cyan; }
        footer { margin-top:20px; text-align:center; font-size:13px; opacity:0.8; }
    </style>
</head>
<body>
<header style="padding:14px 20px;">
    <h2 style="text-shadow:0 0 10px cyan;">🐟 Surimi Protein Yield Predictor</h2>
</header>
<div class="container">
    <div class="card">
        <h1>Enter Parameters</h1>
        <form method="POST">
            <label>Fish Type</label>
            <select name="fish">
                {% for f in ['Threadfin Bream','Lizardfish','Croaker','Ribbonfish','Silver Carp','Tilapia','Catfish'] %}
                    <option {{ 'selected' if fish==f else '' }}>{{ f }}</option>
                {% endfor %}
            </select>

            <label>pH Value</label>
            <input type="number" step="0.1" name="ph" value="{{ ph or '' }}" placeholder="Enter pH" required>

            <label>Temperature (°C)</label>
            <input type="number" name="temp" value="{{ temp or '' }}" placeholder="Enter Temperature" required>

            <label>Time (min)</label>
            <input type="number" name="time" value="{{ time or '' }}" placeholder="Enter Time" required>

            <label>Volume (L)</label>
            <input type="number" step="0.001" name="volume" value="{{ volume or '' }}" placeholder="Enter Volume" required>

            <button type="submit">Predict Yield</button>
        </form>

        {% if result is not none %}
        <div class="output-box">
            Predicted Protein Yield: <br><b>{{ result }} g</b>
        </div>
        {% endif %}
    </div>
</div>

<footer>© 2025 Surimi Research | Designed by Victo Hosting</footer>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    ph = temp = time_val = volume = fish = None

    if request.method == "POST":
        try:
            fish = request.form.get("fish")
            ph = float(request.form.get("ph", "0"))
            temp = float(request.form.get("temp", "0"))
            time_val = float(request.form.get("time", "0"))
            volume = float(request.form.get("volume", "0.320"))
            pred = predict_protein(ph, temp, time_val, volume)
            result = round(float(pred), 4)
        except Exception as e:
            result = f"Error: {e}"

    return render_template_string(
        TEMPLATE,
        result=result,
        ph=ph,
        temp=temp,
        time=time_val,
        volume=volume,
        fish=fish
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
