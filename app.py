import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

# --- Model Training (runs once on startup) ---
def build_model():
    csv_path = os.path.join(os.path.dirname(__file__), "emails.csv")
    df = pd.read_csv(csv_path)
    X = df["text"]
    y = df["label"]
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42)
    model = make_pipeline(
        TfidfVectorizer(stop_words="english"),
        RandomForestClassifier(n_estimators=100, random_state=42),
    )
    model.fit(X_train, y_train)
    return model

model = build_model()

# --- Simple HTML UI ---
HOME_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Phishing Email Detector</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', sans-serif;
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      color: #e0e0e0;
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .card {
      background: rgba(255,255,255,0.06);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255,255,255,0.12);
      border-radius: 16px;
      padding: 2.5rem;
      max-width: 540px;
      width: 90%;
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    h1 { font-size: 1.6rem; margin-bottom: 0.3rem; }
    p.sub { color: #aaa; margin-bottom: 1.5rem; font-size: 0.9rem; }
    textarea {
      width: 100%; height: 120px; border: 1px solid rgba(255,255,255,0.15);
      border-radius: 10px; background: rgba(0,0,0,0.3); color: #fff;
      padding: 12px; font-size: 0.95rem; resize: vertical;
    }
    textarea:focus { outline: none; border-color: #7c5cfc; }
    button {
      margin-top: 1rem; width: 100%; padding: 12px;
      border: none; border-radius: 10px; cursor: pointer;
      font-size: 1rem; font-weight: 600;
      background: linear-gradient(135deg, #7c5cfc, #a855f7);
      color: #fff; transition: opacity .2s;
    }
    button:hover { opacity: .85; }
    #result {
      margin-top: 1.2rem; padding: 14px; border-radius: 10px;
      text-align: center; font-weight: 600; font-size: 1.05rem;
      display: none;
    }
    .safe { background: rgba(16,185,129,0.2); color: #34d399; }
    .phishing { background: rgba(239,68,68,0.2); color: #f87171; }
  </style>
</head>
<body>
  <div class="card">
    <h1>&#128274; Phishing Email Detector</h1>
    <p class="sub">Paste an email below to check if it's phishing or safe.</p>
    <textarea id="email" placeholder="Paste email text here..."></textarea>
    <button onclick="predict()">Analyze Email</button>
    <div id="result"></div>
  </div>
  <script>
    async function predict() {
      const text = document.getElementById('email').value.trim();
      if (!text) return;
      const res = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text})
      });
      const data = await res.json();
      const el = document.getElementById('result');
      el.style.display = 'block';
      if (data.prediction === 'Phishing') {
        el.className = 'phishing';
        el.textContent = '⚠️ Phishing Detected';
      } else {
        el.className = 'safe';
        el.textContent = '✅ Email Looks Safe';
      }
    }
  </script>
</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HOME_HTML)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    prediction = model.predict([text])[0]
    return jsonify({"prediction": prediction})

# Vercel uses `app` as the WSGI entrypoint automatically.
# For local development you can run: python app.py
if __name__ == "__main__":
    app.run(debug=True)
