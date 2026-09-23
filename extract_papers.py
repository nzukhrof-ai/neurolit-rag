"""
Extract clean text from a folder of academic PDFs for a RAG pipeline.

Usage:
    python extract_papers.py papers extracted
"""

import sys
import json
import re
from pathlib import Path
from collections import Counter

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from a PDF, page by page."""
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc:
        pages_text.append(page.get_text("text"))
    doc.close()
    return "\n".join(pages_text)


def remove_repeated_headers(lines: list, threshold: int = 3) -> list:
    """
    Removes lines that repeat too often (likely running headers/footers),
    while protecting short lines (like real section titles) from being
    wrongly caught by this rule.
    """
    # Only consider lines longer than 15 characters as "header candidates."
    # Short lines like "Introduction" or "Results" might legitimately
    # repeat as section names and shouldn't be treated as noise.
    candidates = [line for line in lines if len(line) > 15]

    # Count how many times each candidate line appears
    counts = Counter(candidates)

    # Any line appearing MORE than `threshold` times is probably a
    # repeated header/footer (e.g. once per page)
    noisy_lines = {line for line, count in counts.items() if count > threshold}

    # Keep every line that is NOT in the noisy set
    return [line for line in lines if line not in noisy_lines]


def clean_text(raw_text: str) -> str:
    """Basic cleanup: collapse whitespace, drop common noise lines,
    and remove repeated headers/footers."""
    text = re.sub(r"\n{3,}", "\n\n", raw_text)
    text = re.sub(r" {2,}", " ", text)

    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        if len(stripped) < 3:
            continue
        cleaned_lines.append(stripped)

    cleaned_lines = remove_repeated_headers(cleaned_lines)

    return "\n".join(cleaned_lines)


def strip_references_section(text: str) -> str:
    """
    Cut the text off at the start of the references/bibliography section.
    Looks for a line that is just 'References' or similar, appearing after
    at least 40% of the document (to avoid false positives near the top).
    """
    lines = text.split("\n")
    cutoff_index = None
    search_start = int(len(lines) * 0.4)

    for i in range(search_start, len(lines)):
        line_clean = lines[i].strip().lower()
        if line_clean in ("references", "bibliography", "works cited"):
            cutoff_index = i
            break

    if cutoff_index is not None:
        return "\n".join(lines[:cutoff_index])
    return text


def guess_title(pdf_path: Path, text: str) -> str:
    """Rough title guess: first non-trivial line of extracted text."""
    for line in text.split("\n"):
        stripped = line.strip()
        if len(stripped) > 15 and not stripped.isupper():
            return stripped
    return pdf_path.stem


def process_folder(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(input_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDFs found in {input_dir}")
        return

    print(f"Found {len(pdf_files)} PDFs. Extracting...\n")

    for pdf_path in pdf_files:
        try:
            raw_text = extract_text_from_pdf(pdf_path)
            cleaned = clean_text(raw_text)
            no_refs = strip_references_section(cleaned)
            title = guess_title(pdf_path, cleaned)

            out_txt = output_dir / f"{pdf_path.stem}.txt"
            out_txt.write_text(no_refs, encoding="utf-8")

            out_meta = output_dir / f"{pdf_path.stem}.json"
            out_meta.write_text(
                json.dumps(
                    {
                        "source_file": pdf_path.name,
                        "title_guess": title,
                        "char_count_raw": len(raw_text),
                        "char_count_cleaned": len(no_refs),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            print(f"  OK  {pdf_path.name}  ({len(no_refs)} chars after cleanup)")
        except Exception as e:
            print(f"  FAIL {pdf_path.name}: {e}")

    print(f"\nDone. Output written to {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_papers.py <pdf_folder> <output_folder>")
        sys.exit(1)

    input_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    process_folder(input_dir, output_dir)