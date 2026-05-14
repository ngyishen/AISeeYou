from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import traceback

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

app = Flask(__name__)
CORS(app)

SAVE_PATH = "ngyishen/AISeeYou"

try:
    print("Loading tokenizer...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(SAVE_PATH)

    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(SAVE_PATH)

    model.eval()
    print("Model loaded successfully")

except Exception:
    print("FAILED DURING LOAD")
    traceback.print_exc()
    raise


def predict_single(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    return {
        "human_prob": float(probs[0][0]),
        "ai_prob": float(probs[0][1])
    }


@app.route("/")
def home():
    return "AI Detector API is running"


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    return jsonify(predict_single(data["text"]))


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json(silent=True)

    if not data or "texts" not in data:
        return jsonify({"error": "Missing 'texts' field"}), 400

    results = [predict_single(t) for t in data["texts"]]
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)