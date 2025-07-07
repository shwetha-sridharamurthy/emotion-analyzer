from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)  # ✅ enable CORS once and keep this app instance

HF_TOKEN = os.getenv("HF_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/bhadresh-savani/distilbert-base-uncased-emotion"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_mood():
    text = request.json.get("text", "")
    response = requests.post(API_URL, headers=headers, json={"inputs": text})
    try:
        result = response.json()
        top_emotion = max(result[0], key=lambda x: x["score"])
        return jsonify({"emotion": top_emotion["label"], "score": round(top_emotion["score"], 2)})
    except Exception as e:
        return jsonify({"error": str(e), "raw_response": response.text}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=81)
