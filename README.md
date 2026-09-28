# NeuroLit-RAG

A retrieval-augmented generation (RAG) system for answering questions about
explainable AI (XAI) methods in brain tumor MRI classification, grounded in
and cited to a curated collection of 16 peer-reviewed research papers.

Built as an independent extension of a final-year brain tumor diagnosis
project, letting research questions about the literature be answered
conversationally instead of manually searching through PDFs — with every
claim traceable back to its source paper.

## What it does

- Ask a natural-language question about Grad-CAM, LIME, SHAP, or other XAI
  techniques applied to brain tumor MRI classification
- Get back an answer generated strictly from the retrieved paper excerpts,
  with clickable citations linking to the original source
- If a question falls outside the scope of the 16-paper corpus, the system
  explicitly says so instead of guessing or hallucinating an answer
- Multiple saved chat sessions, browsable and deletable from a sidebar,
  persisted across restarts

## Architecture

```
16 PDFs
  → PDF extraction & cleaning         (PyMuPDF)
  → Sentence-aware chunking           (~300 words/chunk, 2-sentence overlap)
  → Semantic embedding                (all-MiniLM-L6-v2)
  → Vector storage & similarity search (ChromaDB)
  → Cited answer generation           (Gemini API, with retry/fallback)
  → Chat interface                    (Streamlit)
```

The retrieval step embeds the user's question with the same model used for
the corpus, finds the closest matching chunks by cosine similarity, and
passes both the question and the retrieved excerpts to the LLM with explicit
instructions to cite every claim and to decline rather than fabricate an
answer when the excerpts are insufficient. A distance threshold on the
closest retrieved match is used to detect and reject out-of-scope questions
before the LLM is even called.

## Tech stack

- **PyMuPDF** — PDF text extraction
- **sentence-transformers** (`all-MiniLM-L6-v2`) — text embedding
- **ChromaDB** — vector storage and nearest-neighbor search
- **Google Gemini API** — answer generation, with automatic retry and
  model fallback for reliability
- **Streamlit** — chat interface
- **Python 3**

## Corpus

16 open-access papers spanning two themes:

1. **Applied XAI for brain tumor classification** — CNN/transfer-learning
   models paired with Grad-CAM, Grad-CAM++, LIME, and SHAP for tumor
   detection and classification
2. **Mathematical and reproducibility limitations of saliency methods** —
   critical analyses of Grad-CAM's trustworthiness, class-insensitivity,
   gradient noise, and repeatability (CAMERAS, CASE, and related work)

## Setup

```bash
git clone https://github.com/<your-username>/neurolit-rag.git
cd neurolit-rag
pip install -r requirements.txt
```

Create a `.env` file in the project root with a free Gemini API key
(obtained from [Google AI Studio](https://aistudio.google.com)):

```
GEMINI_API_KEY=your_key_here
```

Build the vector store from the included `chunks.json`:

```bash
python build_vector_store.py chunks.json vector_db
```

Run the app:

```bash
streamlit run app.py
```

## Project pipeline scripts

| Script | Purpose |
|---|---|
| `extract_papers.py` | Extracts and cleans text from source PDFs |
| `chunk_papers.py` | Splits cleaned text into sentence-aware chunks |
| `build_vector_store.py` | Embeds chunks and stores them in ChromaDB |
| `query_vector_store.py` | Standalone retrieval test (no generation) |
| `generate_answer.py` | Standalone full RAG pipeline test (CLI) |
| `app.py` | Full Streamlit chat interface |

## Design notes

- **Refusal on out-of-scope questions**: retrieval results are checked
  against a distance threshold before being passed to the LLM, so
  unrelated questions are rejected deterministically rather than left to
  the LLM's judgment on irrelevant injected context.
- **Retry and model fallback**: generation calls retry briefly on
  transient failures and fall back to a lighter model if the primary one
  is persistently unavailable, rather than surfacing raw API errors to
  the user.
- **Citation integrity**: every generated answer is required to cite the
  specific source paper for each claim; source titles and links are
  resolved from a verified mapping rather than trusting auto-extracted
  metadata, which was unreliable for a handful of papers.

## Limitations

- Corpus is fixed at 16 papers; not a general-purpose literature search
  tool
- Mathematical notation and pseudocode in source PDFs can degrade during
  text extraction, a known tradeoff of PDF-to-text conversion
- Answer quality depends on the underlying LLM API's availability

## Relation to broader research

This project was built alongside a final-year project on dual-faithfulness
in explainable AI for brain tumor MRI classification. Roughly half of the
corpus consists of literature critiquing the mathematical reliability of
the same saliency methods (Grad-CAM and its variants) used elsewhere in
that work, directly supporting its literature review and research
motivation.

## License

This project is intended for academic and educational use.