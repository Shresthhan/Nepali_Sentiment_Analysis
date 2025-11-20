from model.load_model import predict_sentiment

if __name__ == "__main__":
    text = "यो फिल्म निकै राम्रो छ।"
    label, conf = predict_sentiment(text)
    print("Text:", text)
    print("Predicted sentiment:", label)
    print("Confidence:", conf)
