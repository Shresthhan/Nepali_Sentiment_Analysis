from huggingface_hub import HfApi

api = HfApi()

api.upload_folder(
    folder_path="model/saved_model",                 # local folder with all model files
    repo_id="Shresthhan/NepaliSentimentBERT",       # your HF repo
    token="PASTE_YOUR_WRITE_TOKEN_HERE"             
)
print("Model uploaded successfully!")