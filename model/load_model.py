import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "Shresthhan/NepaliSentimentBERT"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

id2label = {
    0: "negative",
    1: "positive",
    2: "neutral",
}

def predict_sentiment(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=228,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    label_id = torch.argmax(probs).item()
    confidence = probs[label_id].item()
    label = id2label.get(label_id, str(label_id))
    return label, confidence
