import os
import requests
import streamlit as st

# In Docker we'll override this with an environment variable
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.title("Nepali Sentiment Analysis")
st.write("Enter a Nepali sentence and get its sentiment prediction.")

text = st.text_area("Nepali text:", height=150)

if st.button("Analyze sentiment"):
    if not text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Contacting API..."):
            try:
                resp = requests.post(API_URL, json={"text": text})
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Sentiment: {data['label']}")
                    st.write(f"Confidence: {data['confidence']:.2f}")
                else:
                    st.error(f"API error: {resp.status_code} - {resp.text}")
            except Exception as e:
                st.error(f"Error calling API: {e}")
