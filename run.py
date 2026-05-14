import torch
from flask import Flask, request, jsonify
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import json
import os

SAVE_PATH = "./model"

# Load tokenizer & model
tokenizer = DistilBertTokenizerFast.from_pretrained(SAVE_PATH)
model = DistilBertForSequenceClassification.from_pretrained(SAVE_PATH)

model.eval()

# Load label map
with open(os.path.join(SAVE_PATH, "label_map.json"), "r") as f:
    id2label = json.load(f)


def debug_tokens(text):
    encoded = tokenizer(
        text.lower(),
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"][0].tolist()
    attention_mask = encoded["attention_mask"][0].tolist()

    tokens = tokenizer.convert_ids_to_tokens(input_ids)

    print("normalized:", text.lower())
    print("tokens:", tokens)
    print("input_ids:", input_ids)
    print("attention_mask:", attention_mask)



def predict_text(text):
    debug_tokens(text)

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

    human_prob = probs[0][0].item()
    ai_prob = probs[0][1].item()

    if human_prob > 0.6:
        final = "Human"
    elif ai_prob > 0.6:
        final = "AI"
    else:
        final = "Uncertain"

    return {
        "final_label": final,
        "human_prob": human_prob,
        "ai_prob": ai_prob
    }

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <html>
    <body>
        <h2>AI Text Detector</h2>
        <textarea id="text" rows="10" cols="60"></textarea><br><br>
        <button onclick="predict()">Analyze</button>
        <h3 id="result"></h3>

        <script>
        async function predict() {
            const text = document.getElementById("text").value;

            const res = await fetch("/predict", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify({text})
            });

            const data = await res.json();

            document.getElementById("result").innerText =
                "Final: " + data.final_label +
                " | Human: " + data.human_prob.toFixed(3) +
                " | AI: " + data.ai_prob.toFixed(3);
        }
        </script>
    </body>
    </html>
    """

@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.json
    print("RECEIVED TEXT:", repr(data["text"][:200]))
    return jsonify(predict_text(data["text"]))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

