from fastapi import FastAPI
from pydantic import BaseModel
from model.load_model import predict_sentiment  # uses your Kaggle model

app = FastAPI(title="Nepali Sentiment Analysis API")

class TextIn(BaseModel):
    text: str

class PredictionOut(BaseModel):
    label: str
    confidence: float

@app.get("/")
def root():
    return {"message": "Nepali Sentiment API is running"}

@app.post("/predict", response_model=PredictionOut)
def predict(payload: TextIn):
    label, conf = predict_sentiment(payload.text)
    return PredictionOut(label=label, confidence=conf)
