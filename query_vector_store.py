"""
Test retrieval: ask a question, get back the most relevant chunks.

Usage:
    python query_vector_store.py vector_db
"""

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


def query(db_path: Path, question: str, top_k: int = 5):
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    client = chromadb.PersistentClient(path=str(db_path))
    collection = client.get_or_create_collection(name="neurolit_papers")

    # Embed the question the SAME way we embedded the chunks
    question_embedding = model.encode([question])

    # Ask ChromaDB for the top_k closest chunks
    results = collection.query(
        query_embeddings=question_embedding.tolist(),
        n_results=top_k,
    )

    print(f"\nQuestion: {question}\n")
    print(f"Top {top_k} matching chunks:\n")

    for i in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][i]
        distance = results["distances"][0][i]
        metadata = results["metadatas"][0][i]
        text_preview = results["documents"][0][i][:200]

        print(f"--- Match {i+1} (distance: {distance:.4f}) ---")
        print(f"Paper: {metadata['paper_title']}")
        print(f"Text preview: {text_preview}...")
        print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python query_vector_store.py <db_folder>")
        sys.exit(1)

    db_path = Path(sys.argv[1])

    # Change this question to test different things
    test_question = "What does Grad-CAM highlight in brain tumor MRI images?"
    query(db_path, test_question)