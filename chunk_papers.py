"""
Chunk extracted paper text into overlapping pieces for embedding.

Usage:
    python chunk_papers.py extracted chunks.json
"""

import sys
import json
import re
from pathlib import Path


def split_into_sentences(text: str) -> list:
    """Split text into sentences using punctuation as the boundary."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Drop any empty strings that might result from splitting
    return [s.strip() for s in sentences if s.strip()]


def force_split_long_sentence(sentence: str, max_words: int = 300) -> list:
    """
    Safety net for pseudocode/equation blocks that have no periods,
    so split_into_sentences() treats them as one giant 'sentence'.
    If a single sentence is itself longer than max_words, force-split
    it into fixed-size word chunks. This ignores the no-mid-sentence
    rule, but only for this rare edge case.
    """
    words = sentence.split()
    if len(words) <= max_words:
        return [sentence]

    pieces = []
    for i in range(0, len(words), max_words):
        piece = " ".join(words[i:i + max_words])
        pieces.append(piece)
    return pieces


def chunk_text(text: str, target_words: int = 300, overlap_sentences: int = 2) -> list:
    """Split text into ~target_words chunks, never cutting mid-sentence
    (except for the rare oversized-sentence safety case)."""
    raw_sentences = split_into_sentences(text)

    # NEW: pass every sentence through the safety net before chunking.
    # Normal sentences (under 300 words) pass through unchanged.
    # Oversized ones (e.g. pseudocode blocks with no periods) get
    # force-split here, so the main loop below never has to swallow
    # a single giant "sentence" whole.
    sentences = []
    for s in raw_sentences:
        sentences.extend(force_split_long_sentence(s, max_words=target_words))

    chunks = []
    current_chunk = []
    current_word_count = 0

    for sentence in sentences:
        current_chunk.append(sentence)
        current_word_count += len(sentence.split())

        if current_word_count >= target_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap_sentences:]
            current_word_count = sum(len(s.split()) for s in current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def process_extracted_folder(input_dir: Path, output_path: Path):
    txt_files = sorted(input_dir.glob("*.txt"))

    if not txt_files:
        print(f"No .txt files found in {input_dir}")
        return

    print(f"Found {len(txt_files)} extracted papers. Chunking...\n")

    all_chunks = []
    chunk_counter = 0

    for txt_path in txt_files:
        # Read the paper's cleaned text
        paper_text = txt_path.read_text(encoding="utf-8")

        # Read the matching metadata file to get the title
        json_path = txt_path.with_suffix(".json")
        title = txt_path.stem  # fallback if metadata is missing
        if json_path.exists():
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            title = metadata.get("title_guess", title)

        # Chunk this paper's text
        paper_chunks = chunk_text(paper_text)

        for i, chunk in enumerate(paper_chunks):
            chunk_counter += 1
            all_chunks.append({
                "chunk_id": chunk_counter,
                "paper_title": title,
                "source_file": txt_path.name,
                "chunk_index_in_paper": i,
                "text": chunk,
                "word_count": len(chunk.split()),
            })

        print(f"  {txt_path.name}  ->  {len(paper_chunks)} chunks")

    output_path.write_text(json.dumps(all_chunks, indent=2), encoding="utf-8")
    print(f"\nDone. {chunk_counter} total chunks written to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python chunk_papers.py <extracted_folder> <output_json_path>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    process_extracted_folder(input_dir, output_path)