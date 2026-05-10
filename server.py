from fastapi import FastAPI
from pydantic import BaseModel

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

import torch

app = FastAPI()

MODEL_PATH = "./ai_detector_model"

tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)

model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

class TextRequest(BaseModel):
    text: str

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict(req: TextRequest):

    inputs = tokenizer(
        req.text.lower(),
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

    final = "Human" if human_prob > ai_prob else "AI"

    return {
        "final_label": final,
        "human_prob": human_prob,
        "ai_prob": ai_prob
    }