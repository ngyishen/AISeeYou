import torch
from transformers import DistilBertForSequenceClassification

model = DistilBertForSequenceClassification.from_pretrained("./model")
model.eval()

dummy_ids  = torch.zeros(1, 128, dtype=torch.long)
dummy_attn = torch.ones(1, 128, dtype=torch.long)

torch.onnx.export(
    model,
    ({"input_ids": dummy_ids, "attention_mask": dummy_attn},),
    "./onnx_model/model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch"}, "attention_mask": {0: "batch"}},
    opset_version=14
)
print("exported")