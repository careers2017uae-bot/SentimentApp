import streamlit as st
import os
from io import StringIO
from PyPDF2 import PdfReader
import requests

# Page configuration
st.set_page_config(page_title="Sentiment Analysis App", page_icon="📝", layout="wide")

st.title("📝 Sentiment Analysis App with Groq")
st.markdown("""
Analyze the sentiment of any text or uploaded document.
Supports **.txt** and **.pdf** files.
""")

# Load API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("⚠️ GROQ_API_KEY not found. Please set it as an environment variable.")
    st.stop()

# Text input or file upload
tab1, tab2 = st.tabs(["Enter Text", "Upload File"])

text_input = ""
with tab1:
    text_input = st.text_area("Enter your text here:", height=200)

uploaded_file = None
with tab2:
    uploaded_file = st.file_uploader("Upload your .txt or .pdf file", type=["txt", "pdf"])

# Function to read uploaded file
def read_file(file):
    if file.type == "text/plain":
        return str(file.read(), "utf-8")
    elif file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    return ""

# Function to call Groq API
def analyze_sentiment_groq(text):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = f"""
Analyze the sentiment of the following text and classify it as Positive, Negative, or Neutral.
Provide a short explanation for your choice.

Text:
\"\"\"{text}\"\"\"
"""
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        try:
            result = response.json()
            return result['choices'][0]['message']['content']
        except:
            return "⚠️ Error parsing Groq response."
    else:
        return f"⚠️ Groq API Error: {response.status_code} - {response.text}"

# Analyze sentiment
if st.button("Analyze Sentiment"):
    final_text = text_input
    if uploaded_file:
        final_text = read_file(uploaded_file)

    if final_text.strip() == "":
        st.warning("Please enter text or upload a file for analysis.")
    else:
        with st.spinner("Analyzing sentiment with Groq..."):
            analysis_result = analyze_sentiment_groq(final_text)
        st.subheader("Analysis Result")
        st.markdown(analysis_result)
