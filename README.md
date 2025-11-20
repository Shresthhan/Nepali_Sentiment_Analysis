# Nepali Sentiment Analysis using BERT

This project fine‑tunes a Nepali BERT model for sentiment analysis, serves the trained model via a FastAPI REST API, provides a Streamlit web UI, and packages everything into Docker containers using Docker Compose.

---

## 1. Project overview

- **Task:** Classify Nepali sentences into sentiment classes: negative, positive, neutral.  
- **Model:** Fine‑tuned BERT model trained on a Nepali sentiment dataset (Hugging Face / Kaggle).  
- **Backend:** FastAPI + Uvicorn, PyTorch, Hugging Face Transformers.  
- **Frontend:** Streamlit app that calls the FastAPI `/predict` endpoint.  
- **Deployment:** Two Docker containers (API + UI) orchestrated with Docker Compose.

This README explains how to get the model, set up the project, and run everything locally and with Docker.

---

## 2. Repository structure

Nepali_Sentiment_App/  
model/  
saved_model/ # (optional) local cache; model is loaded from Hugging Face  
load_model.py # shared model loading + predict_sentiment()  

api/  
main.py # FastAPI app exposing /predict  
requirements.txt # backend dependencies (fastapi, uvicorn, transformers, ...)  

ui/  
app.py # Streamlit UI  
requirements.txt # frontend dependencies (streamlit, requests)  

docker/  
Dockerfile.api # builds API image  
Dockerfile.ui # builds UI image  
docker-compose.yml # runs both containers together  

README.md # this file  

---

## 3. Getting the trained model (Hugging Face)

You do **not** need to download any model files into this repository.

The fine‑tuned BERT model is published on Hugging Face as:

Shresthhan/NepaliSentimentBERT


The code in `model/load_model.py` uses:

MODEL_NAME = "Shresthhan/NepaliSentimentBERT"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


When you run the project for the first time:

- Transformers will automatically download the model and tokenizer from Hugging Face.  
- The files are cached under your local Hugging Face cache directory (typically in your user home folder).  
- Subsequent runs reuse the cached copy; no extra setup is required.

So, to run this project on any machine, you only need:
- Internet access the first time you run the code, and  
- A Hugging Face account **is not required** just to download the public model.

---

## 4. Prerequisites

To run this project you need:

- Python 3.10+ installed on your machine  
- pip (Python package manager)  
- Docker Desktop (Windows/macOS) or Docker Engine + Docker Compose (Linux) if you want to run with Docker  
- Internet access on first run so the model `Shresthhan/NepaliSentimentBERT` can be downloaded from Hugging Face

---

## 5. Running locally (without Docker)

These steps run FastAPI and Streamlit directly with Python.

### 5.1. Create and activate a virtual environment

cd Nepali_Sentiment_App

python -m venv .venv

Windows
.venv\Scripts\activate

Linux/macOS
source .venv/bin/activate


### 5.2. Install backend (API) dependencies + CPU‑only PyTorch

`api/requirements.txt`:

fastapi

uvicorn[standard]

transformers


Install:

pip install -r api/requirements.txt
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.3.1


This installs FastAPI, Transformers, and a CPU‑only version of PyTorch (no CUDA required).

### 5.3. Install frontend (UI) dependencies

`ui/requirements.txt`:

streamlit

requests


Install:

pip install -r ui/requirements.txt


### 5.4. Start the FastAPI backend

From the project root:

uvicorn api.main:app --reload --port 8000


- Open `http://localhost:8000/docs` in your browser.  
- Under `POST /predict`, click **Try it out**, change the JSON to:

{
"text": "यो फिल्म निकै राम्रो छ।"
}

- Click **Execute** and check that you get a JSON response with `"label"` and `"confidence"`.

### 5.5. Start the Streamlit UI

Open a **second** terminal, activate the same virtual env, and run:

cd Nepali_Sentiment_App
streamlit run ui/app.py --server.port 8501


Then:

- Open `http://localhost:8501` in your browser.  
- Type a Nepali sentence and click **Analyze sentiment**.  
- You should see the predicted label and confidence, coming from the FastAPI backend.

---

## 6. Running with Docker

The project uses two containers:

- `api`: FastAPI backend + BERT model  
- `ui`: Streamlit frontend that calls the backend via HTTP

### 6.1. Build the images

Make sure Docker is running, then:

cd Nepali_Sentiment_App/docker
docker compose build


If you added `image:` fields in `docker-compose.yml`, this will also tag the images (for example, `nepali-sentiment-api:latest` and `nepali-sentiment-ui:latest`).

### 6.2. Start the containers

docker compose up


Keep this terminal open to see logs from both containers.

### 6.3. Access the app

- **FastAPI docs:** `http://localhost:8000/docs`  
- **Streamlit UI:** `http://localhost:8501`

To stop the containers, press `Ctrl + C` in the terminal, or run:

docker compose down


If you retrain the model later and update the Hugging Face repository, just rebuild the API image so Docker pulls the new version on first run:

cd Nepali_Sentiment_App/docker
docker compose build api
docker compose up


---

## 7. Example sentences to test

Use these Nepali sentences to check different sentiment outputs:

- Positive: `यो फिल्म निकै राम्रो छ।`  
- Negative: `यो उत्पादनले मेरो पैसा मात्रै बर्बाद गर्यो।`  
- Neutral: `मीटिङ भोलि बिहान दस बजे सुरु हुन्छ।`  

You can paste them into:

- the `text` field of the `/predict` endpoint at `http://localhost:8000/docs`, or  
- the textarea in the Streamlit UI at `http://localhost:8501`.

---

## 8. How it works (short explanation)

1. **Model loading (`model/load_model.py`)**  
   - Loads tokenizer and `AutoModelForSequenceClassification` from the Hugging Face model `Shresthhan/NepaliSentimentBERT` (downloaded and cached automatically).  
   - Moves the model to CPU (or GPU if available) and defines a `predict_sentiment(text)` function that returns `(label, confidence)`.

2. **FastAPI backend (`api/main.py`)**  
   - On startup, imports `predict_sentiment`.  
   - Exposes `POST /predict` that accepts JSON `{"text": "..."}` and returns `{"label": "...", "confidence": 0.xx}`.

3. **Streamlit UI (`ui/app.py`)**  
   - Simple web page with a textarea for Nepali text.  
   - When you click **Analyze sentiment**, it sends a POST request to the FastAPI `/predict` endpoint and shows the response.

4. **Docker**  
   - `Dockerfile.api` installs Python dependencies, copies `model/` and `api/`, and starts Uvicorn.  
   - `Dockerfile.ui` installs Streamlit, copies `ui/`, and starts Streamlit.  
   - `docker-compose.yml` builds both images, connects them in one network, and exposes ports 8000 and 8501 to your host.

---

## 9. Notes and future improvements

- Replace the training dataset with a larger or cleaned Nepali corpus for better accuracy.  
- Add evaluation metrics plots (accuracy, F1, confusion matrix) and show them in the README or Streamlit.  
- Deploy the Docker images to a cloud service (Render, Railway, AWS ECS, etc.) for public access.  
- Add authentication, logging, or monitoring (e.g., Prometheus + Grafana, Evidently) around the FastAPI service.

---

## 10. Author and license

- **Author:** Ankit Shrestha  
- **License:** MIT  
- This repository is licensed under the MIT License. See the `LICENSE` file for details.




















