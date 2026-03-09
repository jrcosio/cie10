
# MutuaCIE10

Automated ICD-10 medical coding system using RAG (Retrieval-Augmented Generation) with Google Gemini and Qdrant vector database.

## Prerequisites

- **Docker** running Qdrant: `docker run -p 6333:6333 qdrant/qdrant`
- **Environment variable** `GEMINI_API_KEY` set, or `.env` file in project root with `GEMINI_API_KEY=your_key`
- **Python 3.12+** with `uv` package manager

## Quick Start

```bash
# Install dependencies
uv sync

# Run the application
uv run main.py

# Add a dependency
uv add <package>
```

## Architecture

Single-file RAG pipeline (`main.py`) for automatic ICD-10 encoding:

```
Medical history (free text)
    → [Gemini 2.5 Flash] extracts primary diagnosis (5 words)
    → [gemini-embedding-001] vectorizes (768 dims)
    → [Qdrant] cosine search → top 5 ICD-10 catalog candidates
    → [Gemini 2.5 Flash, temperature=0] selects final code
    → Returns: {codigo, justificacion, enfermedad_detectada, candidatos}
```

## Data Source

`CIE10ES_2026_Finales.xlsx` — Column 0: ICD-10 code, Column 1: description.

## Vector Database

docker run -d --name qdrant -p 6333:6333 -p 6334:6334 -v ${PWD}/qdrant_storage:/qdrant/storage qdrant/qdrant

Local Qdrant at `http://localhost:6333`, collection `cie10_catalogo`. Idempotent loading with batch size 100 and 0.5s pause between batches.

## Models

- **Generation:** `gemini-2.5-flash` (v1beta API)
- **Embeddings:** `gemini-embedding-001` with `output_dimensionality=768`
