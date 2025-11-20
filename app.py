# app.py
import os
from pathlib import Path
from flask import Flask, request, render_template_string
import numpy as np
import joblib
import traceback
import logging

# TensorFlow / Keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("surimi_app")

app = Flask(__name__)

# Filenames
MODEL_FILE = "protein_model.keras"
SCALER_FILE = "scaler.pkl"

# Training data you provided (columns: pH, temp, time, volume_base, yield_g)
DATA = np.array([
    [4.0, 55, 20, 0.320, 0.43],
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

BASE_VOLUME = 0.320  # liters used in training features

# -------------------------
# Model build / train / load
# -------------------------
def build_model():
    model = Sequential([
        Input(shape=(4,)),
        Dense(16, activation="relu"),
        Dense(8, activation="relu"),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.01), loss="mse")  # no metrics to avoid deserialization issues
    return model

def train_and_save_model(epochs=250):
    logger.info("Training model (runs only if model or scaler are missing)...")
    X = DATA[:, :-1]  # pH, temp, time, base_volume
    y = DATA[:, -1]   # yield
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_FILE)
    model = build_model()
    model.fit(X_scaled, y, epochs=epochs, verbose=0)
    # Save in new Keras format
    model.save(MODEL_FILE)
    logger.info("Training finished — model and scaler saved.")

def load_safe_model():
    # If missing, train (this happens only on first run if you didn't upload model files)
    if not Path(MODEL_FILE).exists() or not Path(SCALER_FILE).exists():
        train_and_save_model()
    # load with compile=False to avoid legacy metric deserialization issues
    model = load_model(MODEL_FILE, compile=False)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler

try:
    MODEL, SCALER = load_safe_model()
    logger.info("Model & scaler loaded successfully.")
except Exception:
    logger.error("Model load failed; printing traceback and attempting retrain.")
    logger.error(traceback.format_exc())
    train_and_save_model()
    MODEL, SCALER = load_safe_model()

# -------------------------
# Prediction
# -------------------------
def predict_protein(pH, temp, time_min, requested_volume):
    # Build the input with base volume (match training)
    X_input = np.array([[pH, temp, time_min, BASE_VOLUME]])
    X_scaled = SCALER.transform(X_input)
    base_yield = float(MODEL.predict(X_scaled, verbose=0)[0][0])
    # scale by volume linearly (same method used earlier)
    adjusted_yield = base_yield * (requested_volume / BASE_VOLUME)
    return adjusted_yield

# -------------------------
# HTML template (glassmorphism UI)
# -------------------------
TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>Surimi Protein Yield Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
<style>
:root{--bg1:#071123;--bg2:#02111a;--glass: rgba(255,255,255,0.06);--glass-strong: rgba(255,255,255,0.12);--accent: rgba(0,238,255,0.9);--muted: rgba(255,255,255,0.75);}
*{box-sizing:border-box;font-family:'Poppins',sans-serif}
body{margin:0;min-height:100vh;background: radial-gradient(circle at 10% 10%, #08304a 0%, var(--bg1) 20%, var(--bg2) 60%);color:var(--muted);display:flex;align-items:center;justify-content:center;padding:36px;}
.card{width:100%;max-width:720px;border-radius:20px;padding:28px;background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));box-shadow: 0 8px 40px rgba(4,12,20,0.6), inset 0 1px 0 rgba(255,255,255,0.02);backdrop-filter: blur(12px) saturate(120%);display:grid;grid-template-columns: 1fr 380px;gap:22px;align-items:center;}
.left{padding:18px}
.brand{display:flex;gap:12px;align-items:center;margin-bottom:6px}
.logo{width:52px;height:52px;border-radius:12px;background:linear-gradient(135deg,var(--accent),#0066ff);display:flex;align-items:center;justify-content:center;font-weight:700;color:#002;box-shadow:0 6px 18px rgba(0,230,255,0.08)}
h1{font-size:20px;margin:0 0 6px 0;color:white}
p.lead{margin:0 0 14px 0;opacity:0.9;font-size:14px}
form{display:flex;flex-direction:column;gap:10px}
label{font-size:13px;opacity:0.9;margin-bottom:6px}
input, select{padding:12px;border-radius:12px;border:1px solid rgba(255,255,255,0.08);background:var(--glass-strong);color:#001;font-size:14px;outline:none}
select { color: #001; background: rgba(255,255,255,0.95); }
input:focus, select:focus { box-shadow:0 6px 20px rgba(0,238,255,0.06); border-color: rgba(0,238,255,0.6); }
.btn{margin-top:10px;padding:12px;border-radius:12px;border:none;font-weight:600;background:linear-gradient(90deg,var(--accent),#0077ff);color:#012;cursor:pointer;box-shadow:0 10px 30px rgba(0,120,255,0.12)}
.right{padding:18px;border-radius:16px;background: rgba(255,255,255,0.03);backdrop-filter: blur(8px);box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);display:flex;flex-direction:column;gap:12px;align-items:center;justify-content:center;}
.result-title{font-size:14px;color:var(--muted);opacity:0.9}
.result-box{margin-top:6px;padding:18px;border-radius:12px;background:linear-gradient(180deg, rgba(0,0,0,0.35), rgba(0,0,0,0.45));width:100%;text-align:center;color:white;font-weight:700;font-size:20px;box-shadow:0 8px 30px rgba(0,0,0,0.6)}
.meta{font-size:12px;color:rgba(255,255,255,0.7);opacity:0.9}
footer{grid-column:1/-1;text-align:center;margin-top:12px;color:rgba(255,255,255,0.6);font-size:13px}
@media (max-width:880px){.card{grid-template-columns:1fr;padding:18px}.right{order:2}}
</style>
</head>
<body>
  <div class="card">
    <div class="left">
      <div class="brand">
        <div class="logo">SP</div>
        <div>
          <h1>Surimi Protein Yield Predictor</h1>
          <div class="meta">Glassmorphism UI • Model trained with your dataset</div>
        </div>
      </div>
      <p class="lead">Enter processing parameters below. Volume scaling uses base volume <strong>{{ base_volume }} L</strong> (matching training).</p>
      <form method="POST" action="/">
        <label>Fish Type</label>
        <select name="fish">
          {% for f in ['Threadfin Bream','Lizardfish','Armor Croaker','Ribbonfish','Silver Carp','Tilapia','Catfish'] %}
            <option {{ 'selected' if fish==f else '' }}>{{ f }}</option>
          {% endfor %}
        </select>
        <label>pH Value</label>
        <input name="ph" type="number" step="0.1" value="{{ ph or '' }}" required placeholder="e.g. 5.5" />
        <label>Temperature (°C)</label>
        <input name="temp" type="number" step="1" value="{{ temp or '' }}" required placeholder="e.g. 60" />
        <label>Time (min)</label>
        <input name="time" type="number" step="1" value="{{ time_val or '' }}" required placeholder="e.g. 20" />
        <label>Volume (L)</label>
        <input name="volume" type="number" step="0.001" value="{{ volume or '' }}" required placeholder="e.g. 0.320" />
        <button class="btn" type="submit">Predict Yield</button>
      </form>
    </div>
    <div class="right">
      <div class="result-title">Predicted Protein Yield</div>
      <div class="result-box">
        {% if result is not none %}
          {{ result }} g
        {% else %}
          — enter values and click Predict —
        {% endif %}
      </div>
      <div class="meta">Base model saved as <code>{{ model_file }}</code></div>
    </div>
    <footer>© 2025 Surimi Research • Designed by Victo Hosting</footer>
  </div>
</body>
</html>
"""

# -------------------------
# Flask route
# -------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    ph = temp = time_val = volume = fish = None

    if request.method == "POST":
        try:
            fish = request.form.get("fish")
            ph = float(request.form.get("ph", 5.5))
            temp = float(request.form.get("temp", 60))
            time_val = float(request.form.get("time", 20))
            volume = float(request.form.get("volume", BASE_VOLUME))
            pred = predict_protein(ph, temp, time_val, volume)
            result = round(float(pred), 4)
        except Exception as e:
            # Log full traceback so you can see it in Render logs
            logger.error("Prediction error:\n" + traceback.format_exc())
            result = f"Error: {str(e)}"

    return render_template_string(
        TEMPLATE,
        result=result,
        ph=ph,
        temp=temp,
        time_val=time_val,
        volume=volume,
        fish=fish,
        base_volume=BASE_VOLUME,
        model_file=MODEL_FILE
    )

# -------------------------
# Render-friendly port binding
# -------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
