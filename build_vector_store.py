"""
Embed all chunks and store them in ChromaDB for retrieval.

Usage:
    python build_vector_store.py chunks.json vector_db
"""

import sys
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


def load_chunks(chunks_path: Path) -> list:
    """Load the chunks.json file we created earlier."""
    return json.loads(chunks_path.read_text(encoding="utf-8"))


def build_vector_store(chunks: list, db_path: Path):
    # Load the same embedding model we tested earlier
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create a persistent ChromaDB client — this saves to disk at db_path,
    # so next time we run this, the data is still there (not rebuilt from scratch)
    client = chromadb.PersistentClient(path=str(db_path))

    # A "collection" is basically ChromaDB's version of a table.
    # get_or_create so re-running this script doesn't error out if it already exists.
    collection = client.get_or_create_collection(name="neurolit_papers")

    print(f"Embedding {len(chunks)} chunks...")

    # ChromaDB wants these as separate lists, all in matching order
    ids = []
    texts = []
    metadatas = []

    for chunk in chunks:
        ids.append(str(chunk["chunk_id"]))
        texts.append(chunk["text"])
        metadatas.append({
            "paper_title": chunk["paper_title"],
            "source_file": chunk["source_file"],
            "chunk_index_in_paper": chunk["chunk_index_in_paper"],
        })

    # Embed everything in one batch (much faster than one-by-one in a loop)
    embeddings = model.encode(texts, show_progress_bar=True)

    # Add everything into ChromaDB at once
    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    print(f"\nDone. {collection.count()} chunks stored in ChromaDB at {db_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python build_vector_store.py <chunks_json_path> <db_folder>")
        sys.exit(1)

    chunks_path = Path(sys.argv[1])
    db_path = Path(sys.argv[2])

    chunks = load_chunks(chunks_path)
    build_vector_store(chunks, db_path)