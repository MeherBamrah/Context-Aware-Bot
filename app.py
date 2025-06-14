import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
from dotenv import load_dotenv
import os
import textwrap
import fitz 

# ---- Gemini API Config ----
api_key = st.secrets["GEMINI"]["API_KEY"]
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-1.5-flash")

# ---- Streamlit Config ----
st.set_page_config(page_title="Gemini Chatbot", layout="centered")
st.title("Retrieval-Based Chatbot with Local Semantic Search")

# ---- Session State Setup ----
if "history" not in st.session_state:
    st.session_state.history = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []
    st.session_state.vectorizer = None
    st.session_state.chunk_vectors = None

# ---- Helper: Chunk Text ----
def chunk_text(text, chunk_size=300):
    return textwrap.wrap(text, chunk_size)

# ---- Helper: Extract PDF Text ----
def extract_text_from_pdf(uploaded_pdf):
    doc = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# ---- Helper: Process File ----
def process_file(uploaded_file, file_type):
    if file_type == "pdf":
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")

    chunks = chunk_text(text)
    vectorizer = TfidfVectorizer().fit(chunks)
    chunk_vectors = vectorizer.transform(chunks)

    st.session_state.chunks = chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.chunk_vectors = chunk_vectors

# ---- Helper: Get Relevant Chunks ----
def get_relevant_chunks(query):
    vec = st.session_state.vectorizer.transform([query])
    similarities = cosine_similarity(vec, st.session_state.chunk_vectors).flatten()
    top_indices = similarities.argsort()[-3:][::-1]
    return "\n\n".join([st.session_state.chunks[i] for i in top_indices])

# ---- Helper: Build Prompt ----
def build_prompt(query):
    chat_history = "\n".join(
        [f"User: {u}\nBot: {b}" for u, b in st.session_state.history[-3:]]
    )
    context = get_relevant_chunks(query)
    prompt = f"""You are a helpful assistant, that answers based on the context and document provided. Use the provided context and chat history to answer the user's question.

Context:
{context}

Chat History:
{chat_history}

User: {query}
Bot:"""
    return prompt

# ---- File Upload UI ----
st.sidebar.header("📄 Upload File")
uploaded_file = st.sidebar.file_uploader("Upload a .txt or .pdf file", type=["txt", "pdf"])
if uploaded_file:
    file_type = uploaded_file.type.split("/")[-1]
    process_file(uploaded_file, file_type)
    st.sidebar.success("✅ File processed!")

# ---- Chat Interface ----
st.subheader("💬 Ask Queries")

user_query = st.text_input("You:", placeholder="Ask about courses, hostel, fees, etc...")

if st.button("Ask") and user_query and st.session_state.vectorizer:
    prompt = build_prompt(user_query)
    try:
        response = model.generate_content(prompt)
        bot_reply = response.text.strip()
        st.session_state.history.append((user_query, bot_reply))
        st.success("✅ Response generated")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ---- Display Chat ----
if st.session_state.history:
    st.subheader("🗨️ Conversation")
    for user, bot in reversed(st.session_state.history):
        st.markdown(f"**You:** {user}")
        st.markdown(f"**Bot:** {bot}")
        st.markdown("---")