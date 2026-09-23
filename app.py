"""
Streamlit chat interface for NeuroLit-RAG, with multiple chat sessions.

Usage:
    streamlit run app.py
"""

import os
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai

DB_PATH = Path("vector_db")
HISTORY_PATH = Path("chat_sessions.json")
TOP_K = 5
MAX_DISTANCE = 1.0
MODELS_TO_TRY = ["gemini-3.6-flash", "gemini-3.5-flash-lite"]

TITLE_FIXES = {
    "10278_2024_Article_1136.txt": "Gradient-Based Saliency Maps Are Not Trustworthy Visual Explanations of Automated AI Musculoskeletal Diagnoses",
    "2009.14260v1.txt": "Trustworthy Convolutional Neural Networks: A Gradient Penalized-based Approach",
    "2106.10649v1.txt": "CAMERAS: Enhanced Resolution And Sanity Preserving Class Activation Mapping for Image Saliency",
    "2410.07613v1.txt": "Explainability of Deep Neural Networks for Brain Tumor Detection",
    "2506.07228v1.txt": "Transfer Learning and Explainable AI for Brain Tumor Classification: A Study Using MRI Data from Bangladesh",
    "2506.07327v3.txt": "CASE: Contrastive Activation for Saliency Estimation",
    "2508.11880v1.txt": "PCA- and SVM-Grad-CAM for Convolutional Neural Networks: Closed-form Jacobian Expression",
    "2605.01999v1.txt": "TumorXAI: Self-Supervised Deep Learning Framework for Explainable Brain MRI Tumor Classification",
    "applsci-15-05412-v2.txt": "A Novel Hybrid Deep Learning Model Enhanced with Explainable AI for Brain Tumor Multi-Classification from MRI Images",
    "fonc-15-1535478.txt": "Explainable AI in Medical Imaging: An Interpretable and Collaborative Federated Learning Model for Brain Tumor Classification",
    "frai-8-1700214.txt": "Explainable AI-driven MRI-based Brain Tumor Classification: A Novel Deep Learning Approach",
    "make-06-00111.txt": "Empowering Brain Tumor Diagnosis through Explainable Deep Learning",
    "s10462-025-11410-8.txt": "Exploring the Potential of Explainable AI in Brain Tumor Detection and Classification: A Systematic Review",
    "s11548-022-02619-x.txt": "Explainability of Deep Neural Networks for MRI Analysis of Brain Tumors",
    "s12880-024-01292-7.txt": "Enhancing Brain Tumor Detection in MRI Images through Explainable AI using Grad-CAM with ResNet50",
    "s40708-025-00257-y.txt": "Explainable CNN for Brain Tumor Detection and Classification through XAI Based Key Features Identification",
}

PAPER_URLS = {
    "10278_2024_Article_1136.txt": "https://doi.org/10.1007/s10278-024-01136-4",
    "2009.14260v1.txt": "https://arxiv.org/abs/2009.14260",
    "2106.10649v1.txt": "https://arxiv.org/abs/2106.10649",
    "2410.07613v1.txt": "https://arxiv.org/abs/2410.07613",
    "2506.07228v1.txt": "https://arxiv.org/abs/2506.07228",
    "2506.07327v3.txt": "https://arxiv.org/abs/2506.07327",
    "2508.11880v1.txt": "https://arxiv.org/abs/2508.11880",
    "2605.01999v1.txt": "https://arxiv.org/abs/2605.01999",
    "applsci-15-05412-v2.txt": "https://doi.org/10.3390/app15105412",
    "fonc-15-1535478.txt": "https://doi.org/10.3389/fonc.2025.1535478",
    "frai-8-1700214.txt": "https://doi.org/10.3389/frai.2025.1700214",
    "make-06-00111.txt": "https://doi.org/10.3390/make6040111",
    "s10462-025-11410-8.txt": "https://doi.org/10.1007/s10462-025-11410-8",
    "s11548-022-02619-x.txt": "https://doi.org/10.1007/s11548-022-02619-x",
    "s12880-024-01292-7.txt": "https://doi.org/10.1186/s12880-024-01292-7",
    "s40708-025-00257-y.txt": "https://doi.org/10.1186/s40708-025-00257-y",
}


def clean_title(source_file: str, fallback_title: str) -> str:
    return TITLE_FIXES.get(source_file, fallback_title)


def get_url(source_file: str):
    return PAPER_URLS.get(source_file)


# --- Multi-session history persistence ---

def load_all_sessions() -> dict:
    if HISTORY_PATH.exists():
        return json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    return {}


def save_all_sessions(sessions: dict):
    HISTORY_PATH.write_text(json.dumps(sessions, indent=2), encoding="utf-8")


def create_new_session() -> str:
    session_id = str(uuid.uuid4())
    st.session_state.sessions[session_id] = {
        "title": "New chat",
        "created": datetime.now().isoformat(),
        "messages": [],
    }
    return session_id


def delete_session(session_id: str):
    """Remove a session. If it was the current one, switch to another (or make a new one)."""
    del st.session_state.sessions[session_id]

    if st.session_state.current_session_id == session_id:
        if st.session_state.sessions:
            # Switch to the most recently created remaining session
            st.session_state.current_session_id = max(
                st.session_state.sessions,
                key=lambda sid: st.session_state.sessions[sid]["created"],
            )
        else:
            # No sessions left at all — start a fresh one
            st.session_state.current_session_id = create_new_session()


# --- Cached setup: these run ONCE, not on every message ---

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(name="neurolit_papers")


@st.cache_resource
def load_gemini_client():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Check your .env file.")
    return genai.Client(api_key=api_key)


# --- Core RAG logic ---

def retrieve_chunks(collection, model, question: str, top_k: int = TOP_K, max_distance: float = MAX_DISTANCE):
    question_embedding = model.encode([question])
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=top_k,
    )

    closest_distance = results["distances"][0][0]
    if closest_distance > max_distance:
        return []

    chunks = []
    for i in range(len(results["ids"][0])):
        source_file = results["metadatas"][0][i]["source_file"]
        raw_title = results["metadatas"][0][i]["paper_title"]
        chunks.append({
            "text": results["documents"][0][i],
            "paper_title": clean_title(source_file, raw_title),
            "source_file": source_file,
            "url": get_url(source_file),
        })
    return chunks


def build_prompt(question: str, chunks: list) -> str:
    context_sections = []
    for i, chunk in enumerate(chunks, 1):
        context_sections.append(
            f"[Source {i}: {chunk['paper_title']}]\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_sections)

    prompt = f"""You are a research assistant answering questions about brain tumor XAI (explainable AI) literature.

Answer the question below using ONLY the excerpts provided. Follow these rules strictly:
1. Every claim you make must be supported by one of the excerpts below.
2. Cite the source paper by name after each claim, like this: (Source: <paper title>).
3. If the excerpts do not contain enough information to answer the question, say so clearly instead of guessing or using outside knowledge.
4. Do not make up information that isn't in the excerpts.

Excerpts:
{context_text}

Question: {question}

Answer:"""
    return prompt


def answer_question(question: str, embed_model, collection, gemini_client):
    chunks = retrieve_chunks(collection, embed_model, question)

    if not chunks:
        return (
            "This question doesn't appear to be covered by the NeuroLit-RAG paper collection. "
            "Try asking about XAI methods, brain tumor classification, or Grad-CAM/saliency techniques."
        ), []

    prompt = build_prompt(question, chunks)
    source_list = [{"paper_title": c["paper_title"], "url": c["url"]} for c in chunks]

    for model_name in MODELS_TO_TRY:
        for attempt in range(2):
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                return response.text, source_list
            except Exception:
                if attempt == 0:
                    time.sleep(3)
                continue

    return "🕐 The system is busy right now. Please try again in a moment.", []


def render_sources(sources: list):
    if not sources:
        return
    seen = {}
    for s in sources:
        seen[s["paper_title"]] = s["url"]
    links = []
    for title, url in seen.items():
        links.append(f"[{title}]({url})" if url else title)
    st.caption("📄 Sources: " + " · ".join(links))


# --- Streamlit UI ---

st.set_page_config(page_title="NeuroLit-RAG", page_icon="🧠", layout="wide")

embed_model = load_embedding_model()
collection = load_collection()
gemini_client = load_gemini_client()

if "sessions" not in st.session_state:
    st.session_state.sessions = load_all_sessions()

if "current_session_id" not in st.session_state:
    if st.session_state.sessions:
        st.session_state.current_session_id = max(
            st.session_state.sessions,
            key=lambda sid: st.session_state.sessions[sid]["created"],
        )
    else:
        st.session_state.current_session_id = create_new_session()

# --- Sidebar: New Chat button + list of past conversations (with delete) ---
with st.sidebar:
    st.title("🧠 NeuroLit-RAG")

    if st.button("➕ New chat", use_container_width=True):
        st.session_state.current_session_id = create_new_session()
        save_all_sessions(st.session_state.sessions)
        st.rerun()

    st.divider()
    st.caption("Chat history")

    sorted_sessions = sorted(
        st.session_state.sessions.items(),
        key=lambda item: item[1]["created"],
        reverse=True,
    )

    for session_id, session_data in sorted_sessions:
        is_current = session_id == st.session_state.current_session_id
        label = ("🟢 " if is_current else "") + session_data["title"]

        col_select, col_delete = st.columns([5, 1])

        with col_select:
            if st.button(label, key=f"session_{session_id}", use_container_width=True):
                st.session_state.current_session_id = session_id
                st.rerun()

        with col_delete:
            if st.button("🗑️", key=f"delete_{session_id}"):
                delete_session(session_id)
                save_all_sessions(st.session_state.sessions)
                st.rerun()

# --- Main chat area ---
current_session = st.session_state.sessions[st.session_state.current_session_id]

st.caption("Ask questions about XAI methods for brain tumor MRI classification, grounded in 16 research papers.")

for message in current_session["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []))

if question := st.chat_input("Ask a question about the papers..."):
    current_session["messages"].append({"role": "user", "content": question})

    if current_session["title"] == "New chat":
        current_session["title"] = question[:40] + ("..." if len(question) > 40 else "")

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            answer, sources = answer_question(question, embed_model, collection, gemini_client)
        st.markdown(answer)
        render_sources(sources)

    current_session["messages"].append({"role": "assistant", "content": answer, "sources": sources})

    save_all_sessions(st.session_state.sessions)