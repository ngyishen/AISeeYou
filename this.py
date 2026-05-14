import torch, numpy as np, onnxruntime as ort
from transformers import DistilBertForSequenceClassification

input_ids = [101, 1037, 2691, 2110, 1997, 4722, 6980, 18653, 2043, 1037, 2711, 15970, 1996, 3484, 1997, 2037, 2166, 3236, 2090, 1996, 1000, 3638, 1000, 1997, 1996, 2627, 1998, 1996, 1000, 11162, 1000, 1997, 1996, 2925, 1012, 1529, 2156, 2062, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
attn      = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# PyTorch
model = DistilBertForSequenceClassification.from_pretrained("./model")
model.eval()
ids_t  = torch.tensor([input_ids], dtype=torch.long)
attn_t = torch.tensor([attn],      dtype=torch.long)

with torch.no_grad():
    out = model(input_ids=ids_t)
pt = torch.softmax(out.logits, dim=1)[0].tolist()
print("PyTorch  logits:", out.logits[0].tolist())
print("PyTorch  probs:", pt)

# ONNX
sess = ort.InferenceSession("./extension/onnx_model/model.onnx")
res = sess.run(None, {
    "input_ids":      np.array([input_ids], dtype=np.int64),
    "attention_mask": np.array([attn],      dtype=np.int64),
})[0][0]
e = np.exp(res - res.max()); onnx_probs = e / e.sum()
print("ONNX     logits:", res.tolist())
print("ONNX     probs:", onnx_probs.tolist())