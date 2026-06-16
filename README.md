# PDF Chatbot using RAG (Version 1.0 MVP)

A high-performance, modular, full-stack AI-powered document interaction utility utilizing Retrieval-Augmented Generation (RAG). The application allows engineers and recruiters to upload standard PDF documents, convert multi-page raw data arrays into dense contextual embedding structures, map them into an isolated in-memory vector indexing array, and fetch precise grounded responses using Google's Gemini API.

This project functions as a portfolio asset demonstrating competency across modern AI integration paradigms, deterministic context constraint design, asynchronous processing tracks, and modular clean code application design patterns.

## Architecture Overview

The system abstracts business rules cleanly away from presentation configurations via a decoupled single-layer multi-utility layout schema:

1. **Client Tier UI (Streamlit):** Coordinates component layout bindings, holds runtime configuration session cache maps, and manages file stream uploads.
2. **Document Processing Pipeline (PyPDF & LangChain):** Executes file chunk text breakdowns dynamically using a character index separation array.
3. **Semantic Storage Engine (FAISS):** Houses vector mappings directly in volatile CPU spaces for lightning-fast mathematical similarity operations.
4. **Context Orchestration Influx (LCEL & Gemini):** Locks prompt templates directly into the ingestion context arrays to completely prevent raw text hallucinations.

```text
[PDF Upload] ➔ [Text Extraction (PyPDF)] ➔ [Chunking (RecursiveTextSplitter)]
                                                        │
[Answer Generation] 🗲 [Gemini 2.5 Flash] 🗲 [Retriever] ➔ [FAISS Vector Store]