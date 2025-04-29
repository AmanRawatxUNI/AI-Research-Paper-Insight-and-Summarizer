# AI Research Paper Insight and Summarizer

## Overview
This project is a Streamlit-based web application that summarizes and compares AI research papers using natural language processing. It allows users to upload PDFs, get concise summaries, and view comparative insights between papers.

##  Features
- 📄 Upload multiple research papers (PDF format)
- ✂️ Summarize key insights using NLP
- 📊 Compare research papers on various aspects
- ⚡ Real-time analysis and interactive UI via Streamlit

##  Tech Stack
- Python
- Streamlit
- PyMuPDF / PDFplumber
- Transformers / Hugging Face
- OpenAI API / T5 / BART (choose your summarizer)
- scikit-learn / NLTK (for NLP utilities)

##  How to Run
```bash
pip install -r requirements.txt
streamlit run my_streamlit_app.py
