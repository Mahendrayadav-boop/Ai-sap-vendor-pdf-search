# Ai-sap-vendor-pdf-search
Lightweight AI-powered tool to search and extract information from SAP Vendor Service Request PDFs using keyword and semantic search.
🧠 AI PDF Search for SAP Vendor Service Requests

This project implements a lightweight AI-based search utility to extract and find information from SAP Vendor Service Request PDFs.
It combines keyword search and semantic (contextual) search to quickly locate key details like Vendor Codes, Request IDs, Dates, and Approval Notes — without complex infrastructure.

🚀 Features

🔍 Extracts structured and unstructured text from SAP PDF forms

🧩 Supports keyword and semantic similarity search

🗂️ Handles multiple PDF uploads

💡 Lightweight architecture (no external DB required)

💬 Optional LLM integration (e.g., GPT or Llama2) for smart summaries

🧱 Architecture
PDF Input → Text Extraction → Preprocessing → Embedding Generation → Search Engine (FAISS) → Query Interface (Streamlit)

⚙️ Tools & Libraries
Purpose	Library
PDF Extraction	pdfplumber, PyMuPDF, pytesseract (for OCR)
Text Cleaning	re, nltk
Semantic Embeddings	sentence-transformers (all-MiniLM-L6-v2)
Vector Search	faiss
UI / App	Streamlit
Optional	OpenAI API for smart contextual answers
🧩 Directory Structure
ai-sap-vendor-pdf-search/
│
├── data/                      # Sample PDFs (non-sensitive)
├── notebooks/                 # Jupyter notebooks for experiments
├── src/
│   ├── extract_text.py        # PDF text extraction
│   ├── preprocess.py          # Cleaning and chunking
│   ├── embed_search.py        # Embedding + FAISS retrieval
│   ├── app.py                 # Streamlit app
│   └── config.py              # Configuration parameters
│
├── requirements.txt
├── README.md
└── LICENSE

🧰 Installation
git clone https://github.com/<your-username>/ai-sap-vendor-pdf-search.git
cd ai-sap-vendor-pdf-search
pip install -r requirements.txt


requirements.txt

pdfplumber
sentence-transformers
faiss-cpu
nltk
streamlit
PyMuPDF
pytesseract
pdf2image

🧠 Example Code (Simplified)
src/extract_text.py
import pdfplumber

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

src/embed_search.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

def build_vector_index(chunks):
    embeddings = model.encode(chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def search_query(query, chunks, index, top_k=3):
    q_emb = model.encode([query])
    _, indices = index.search(np.array(q_emb), top_k)
    return [chunks[i] for i in indices[0]]

src/app.py (Streamlit)
import streamlit as st
from src.extract_text import extract_text_from_pdf
from src.embed_search import build_vector_index, search_query

st.title("🔍 AI Search – SAP Vendor Service Request PDF")

uploaded_file = st.file_uploader("Upload SAP PDF", type=["pdf"])
if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    chunks = [t.strip() for t in text.split("\n") if t.strip()]
    index, _ = build_vector_index(chunks)

    query = st.text_input("Enter your search term or question:")
    if st.button("Search") and query:
        results = search_query(query, chunks, index)
        st.subheader("Top Matches:")
        for res in results:
            st.write(f"- {res}")

🧪 Example Use Case

Upload a Vendor Service Request PDF from SAP.

Enter queries like:

“Vendor Code”

“Approval date”

“Status of request”

The tool returns exact or semantically similar sentences from the document.

🗓️ Roadmap
Phase	Description
v1.0	Local prototype (keyword + semantic search)
v1.1	Add OCR for scanned PDFs
v1.2	Integrate HANA Vector DB for enterprise use
v2.0	Add LLM (OpenAI or Llama2) for smart contextual Q&A
👨‍💻 Author

Mahendra Yadav Vavitikalva
AI/ML Engineer


