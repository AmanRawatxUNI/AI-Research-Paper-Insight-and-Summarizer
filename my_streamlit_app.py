import streamlit as st
from transformers import pipeline
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rake_nltk import Rake  # For key phrase extraction
import nltk
import plotly.graph_objects as go
import pandas as pd

# Download punkt and stopwords manually
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Streamlit App Config
st.set_page_config(page_title="AI Research Paper Insight & Summarizer", layout="wide")
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            color: #e6e6e6;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #59c1ff;
        }
        .stButton>button {
            background-color: #0059b3;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 0.5rem 1rem;
        }
        .stFileUploader {
            background-color: #1e2229;
            padding: 10px;
            border-radius: 8px;
            color: #e6e6e6;
        }
        .stExpander {
            background-color: #1e2229;
            color: #e6e6e6;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .stSuccess {
            background-color: #2e8b57 !important;
            color: white !important;
            border-radius: 8px;
            padding: 0.5rem;
        }
        .stCaption {
            color: #a9a9a9;
        }
        .stTextArea textarea {
            background-color: #1e2229;
            color: #e6e6e6;
            border-radius: 8px;
            padding: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📄 AI Research Paper Insight & Summarizer")
st.markdown("""
Upload one or more research paper PDFs to generate summaries, gain deeper insights, and compare their content.
""")

uploaded_files = st.file_uploader("Upload one or more Research Paper PDFs", type="pdf", accept_multiple_files=True)

summarizer = pipeline("summarization", model="t5-small", device="cpu")
question_answerer = pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad", device="cpu")
sentiment_analyzer = pipeline("sentiment-analysis", device="cpu")
rake = Rake()

papers_data = []

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content
    return text

def extract_summary(text):
    try:
        return summarizer(text[:1000], max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    except Exception as e:
        return f"Error during summarization: {e}"

def compute_similarity(summary1, summary2):
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform([summary1, summary2])
    similarity_array = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])
    return similarity_array[0][0]

def extract_keywords(text, top_n=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform([text])
    dense_array = X.toarray()
    indices = dense_array[0].argsort()[-top_n:][::-1]
    keywords = [vectorizer.get_feature_names_out()[i] for i in indices]
    return keywords

def extract_key_phrases(text, top_n=5):
    rake.extract_keywords_from_text(text)
    return rake.get_ranked_phrases()[:top_n]

if uploaded_files:
    for uploaded_file in uploaded_files:
        title = uploaded_file.name
        text = extract_text_from_pdf(uploaded_file)
        summary = extract_summary(text)
        keywords = extract_keywords(text)
        key_phrases = extract_key_phrases(text)
        sentiment = sentiment_analyzer(summary)[0]
        papers_data.append({
            "title": title,
            "text": text,
            "summary": summary,
            "keywords": keywords,
            "key_phrases": key_phrases,
            "sentiment": sentiment
        })

    for i, paper in enumerate(papers_data):
        with st.expander(f"📑 {paper['title']}"):
            st.subheader("📝 Summary")
            st.success(paper['summary'])

            st.subheader("😊 Sentiment of Summary")
            st.caption(f"Label: {paper['sentiment']['label']}, Score: {paper['sentiment']['score']:.2f}")

            st.subheader("🔑 Top Keywords")
            st.caption(", ".join(paper['keywords']))

            st.subheader("🔑 Key Phrases")
            st.caption(", ".join(paper['key_phrases']))

            st.subheader("❓ Ask Questions about the Paper")
            question = st.text_area(f"Enter your question for {paper['title']}:", key=f"question_{i}")
            ask_button = st.button("Ask", key=f"ask_button_{i}")

            if ask_button and question:
                try:
                    answer = question_answerer(question=question, context=paper['text'][:4000])
                    if answer and answer.get('answer') and answer.get('score', 0) > 0.5:  # Adjust the confidence threshold as needed
                        st.info(f"**Question:** {question}\n\n**Answer:** {answer['answer']}")
                    else:
                        st.info(f"**Question:** {question}\n\n**Answer:** No relevant answer found in the text.")
                except Exception as e:
                    st.error(f"Error during question answering: {e}")

    if len(papers_data) > 1:
        st.markdown("---")
        st.subheader("📊 Comparative Analysis")

        similarity_scores = []
        paper_pairs = []
        common_keyword_counts = []

        for i in range(len(papers_data)):
            for j in range(i + 1, len(papers_data)):
                paper1 = papers_data[i]
                paper2 = papers_data[j]
                similarity_score = compute_similarity(paper1['summary'], paper2['summary'])
                similarity_scores.append(similarity_score)
                paper_pairs.append(f"{paper1['title']} vs {paper2['title']}")
                common_keywords = list(set(paper1['keywords']) & set(paper2['keywords']))
                common_keyword_counts.append(len(common_keywords))

                st.markdown(f"#### 🤝 {paper1['title']} vs {paper2['title']}")
                st.write(f"**Summary Similarity Score:** {similarity_score:.2f} ({'Highly similar' if similarity_score > 0.8 else 'Moderately similar' if 0.5 < similarity_score <= 0.8 else 'Low similarity'})")

                if common_keywords:
                    st.caption(f"🔑 Common Keywords: {', '.join(common_keywords[:10])}. This suggests some overlap in the topics discussed.")
                else:
                    st.caption("No significant common keywords found in the top keywords, indicating potentially different areas of focus.")

        if paper_pairs:
            # Bar chart for similarity scores
            fig_similarity = go.Figure(data=[go.Bar(x=paper_pairs, y=similarity_scores, text=[f"{s:.2f}" for s in similarity_scores], textposition='outside')])
            fig_similarity.update_layout(title='Summary Similarity Comparison', xaxis_title='Paper Pair', yaxis_title='Similarity Score (0-1)')
            st.plotly_chart(fig_similarity)

            # Bar chart for common keyword counts
            fig_keywords = go.Figure(data=[go.Bar(x=paper_pairs, y=common_keyword_counts, text=common_keyword_counts, textposition='outside')])
            fig_keywords.update_layout(title='Number of Common Keywords', xaxis_title='Paper Pair', yaxis_title='Common Keyword Count')
            st.plotly_chart(fig_keywords)

if papers_data:
    all_summaries_text = ""
    for paper in papers_data:
        all_summaries_text += f"## {paper['title']}\n\n{paper['summary']}\n\n---\n\n"

    st.download_button(
        label="📥 Download All Summaries",
        data=all_summaries_text,
        file_name="all_summaries.md",
        mime="text/markdown",
    )