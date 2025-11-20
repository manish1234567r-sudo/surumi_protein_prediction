# app.py
import os
from flask import Flask, request, render_template_string
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from pathlib import Path

app = Flask(__name__)

MODEL_FILE = "protein_model.pkl"
SCALER_FILE = "scaler.pkl"

# training data
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

BASE_VOLUME = 0.320

def train_model():
    X = DATA[:, :-1]
    y = DATA[:, -1]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_FILE)

    model = MLPRegressor(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=3000,
        random_state=42
    )
    model.fit(X_scaled, y)
    joblib.dump(model, MODEL_FILE)
    print("Model trained and saved!")

def load_model_safe():
    if not Path(MODEL_FILE).exists() or not Path(SCALER_FILE).exists():
        train_model()
    model = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    return model, scaler

MODEL, SCALER = load_model_safe()

def predict_protein(pH, temp, time_min, requested_volume):
    X_input = np.array([[pH, temp, time_min, BASE_VOLUME]])
    X_scaled = SCALER.transform(X_input)
    base_yield = float(MODEL.predict(X_scaled)[0])
    final = base_yield * (requested_volume / BASE_VOLUME)
    return final

# ---------------------- HTML UI ----------------------
TEMPLATE = """
<!doctype html>
<html>
<head>
<title>Surimi Protein Yield Predictor</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap" rel="stylesheet">
<style>
  body{
    margin:0;
    background:#071123;
    font-family:'Poppins';
    display:flex;
    justify-content:center;
    align-items:center;
    height:100vh;
    color:white;
  }
  .card{
    width:90%;max-width:750px;
    background:rgba(255,255,255,0.08);
    backdrop-filter:blur(12px);
    padding:25px;border-radius:18px;
  }
  input,select{
    width:100%;padding:12px;margin-bottom:14px;
    border-radius:10px;border:none;
  }
  button{
    width:100%;padding:12px;
    background:#00e1ff;border:none;border-radius:10px;
    font-weight:600;
  }
  .result{
    margin-top:15px;font-size:20px;font-weight:600;
  }
</style>
</head>

<body>
<div class="card">
  <h2>Surimi Protein Yield Predictor</h2>

  <form method="POST">
    <label>Fish Type</label>
    <select name="fish">
      {% for f in ['Threadfin Bream','Lizardfish','Armor Croaker','Ribbonfish','Silver Carp','Tilapia','Catfish'] %}
        <option {{ 'selected' if fish==f else '' }}>{{ f }}</option>
      {% endfor %}
    </select>

    <label>pH</label>
    <input type="number" step="0.1" name="ph" required value="{{ ph or '' }}">

    <label>Temperature (°C)</label>
    <input type="number" name="temp" required value="{{ temp or '' }}">

    <label>Time (min)</label>
    <input type="number" name="time" required value="{{ time or '' }}">

    <label>Volume (L)</label>
    <input type="number" step="0.001" name="volume" required value="{{ volume or '' }}">

    <button type="submit">Predict</button>
  </form>

  <div class="result">
    {% if result is not none %}
      Predicted Yield: {{ result }} g
    {% endif %}
  </div>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET","POST"])
def home():
    result = None
    ph = temp = time_val = volume = fish = None

    if request.method == "POST":
        try:
            fish = request.form.get("fish")
            ph = float(request.form.get("ph"))
            temp = float(request.form.get("temp"))
            time_val = float(request.form.get("time"))
            volume = float(request.form.get("volume"))

            pred = predict_protein(ph, temp, time_val, volume)
            result = round(pred, 4)

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
    port = int(os.environ.get("PORT",5000))
    app.run(host="0.0.0.0", port=port)
