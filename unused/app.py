from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
import json
import traceback
import sys


from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

    print("Model loaded successfully")
except Exception as e:
    print("FAILED DURING LOAD")
    traceback.print_exc()
    sys.exit(1)


app = Flask(__name__)
CORS(app)


# SAVE_PATH = "./ai_detector_model" # local
SAVE_PATH = "ngyishen/AISeeYou"


tokenizer = DistilBertTokenizerFast.from_pretrained(SAVE_PATH)
model = DistilBertForSequenceClassification.from_pretrained(SAVE_PATH)
model.eval()

def predict_single(text):
    inputs = tokenizer(
        text.lower(),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)

    return {
        "human_prob": probs[0][0].item(),
        "ai_prob": probs[0][1].item()
    }

@app.route("/")
def home():
    return "AI Detector API is running"

# Single text endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    result = predict_single(data["text"])
    return jsonify(result)

# Batch endpoint
@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.json
    texts = data["texts"]

    results = []
    for t in texts:
        results.append(predict_single(t))

    return jsonify(results)