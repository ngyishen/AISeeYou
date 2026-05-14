from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "C:/FYP/model"

model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.save_pretrained("C:/FYP/model")
tokenizer.save_pretrained("C:/FYP/model")