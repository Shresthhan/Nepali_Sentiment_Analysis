# model/load_model.py
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification

# This path is relative to the project root when you run your app
MODEL_DIR = "model/saved_model"

# Choose CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model from the fine-tuned checkpoint folder
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
model.eval()

# Map label IDs to human-readable strings
id2label = {
    0: "negative",
    1: "positive",
    2: "neutral",
}

def predict_sentiment(text: str):
    """
    Run a forward pass on a single text and return (label, confidence).
    """
    # Tokenize input text exactly like in training
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=228,   # same MAX_LENGTH you used in Kaggle
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    label_id = torch.argmax(probs).item()
    confidence = probs[label_id].item()
    label = id2label.get(label_id, str(label_id))
    return label, confidence
