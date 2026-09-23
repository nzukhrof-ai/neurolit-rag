"""
Full RAG pipeline: retrieve relevant chunks, then generate a cited answer.

Usage:
    python generate_answer.py vector_db "your question here"
"""

import sys
import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from google import genai


def retrieve_chunks(collection, model, question: str, top_k: int = 5, max_distance: float = 1.0):
    """Retrieve chunks, but reject the whole batch if nothing is actually relevant."""
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
        chunks.append({
            "text": results["documents"][0][i],
            "paper_title": results["metadatas"][0][i]["paper_title"],
            "source_file": results["metadatas"][0][i]["source_file"],
        })
    return chunks


def build_prompt(question: str, chunks: list) -> str:
    """Build the instruction prompt that tells the LLM how to answer safely."""
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


def generate_answer(question: str, db_path: Path, top_k: int = 5):
    # Load API key from .env
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Check your .env file.")

    # NEW SDK: create a client instead of calling configure()
    client = genai.Client(api_key=api_key)

    # Load embedding model and ChromaDB collection
    print("Loading embedding model and vector store...")
    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    chroma_client = chromadb.PersistentClient(path=str(db_path))
    collection = chroma_client.get_or_create_collection(name="neurolit_papers")

    # Step 1: Retrieve
    print("Retrieving relevant chunks...")
    chunks = retrieve_chunks(collection, embed_model, question, top_k)

    if not chunks:
        print("=" * 60)
        print("QUESTION:", question)
        print("=" * 60)
        print("This question doesn't appear to be covered by the NeuroLit-RAG paper collection. "
              "Try asking about XAI methods, brain tumor classification, or Grad-CAM/saliency techniques.")
        print("=" * 60)
        return
    # Step 2: Build prompt
    prompt = build_prompt(question, chunks)

    # Step 3: Generate with Gemini (new SDK call pattern)
    print("Generating answer...\n")
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    print("=" * 60)
    print("QUESTION:", question)
    print("=" * 60)
    print(response.text)
    print("=" * 60)
    print(f"\n(Based on {len(chunks)} retrieved excerpts)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print('Usage: python generate_answer.py <db_folder> "<question>"')
        sys.exit(1)

    db_path = Path(sys.argv[1])
    question = sys.argv[2]

    generate_answer(question, db_path)