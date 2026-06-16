from typing import List, Optional
import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

def create_vector_store(chunks: List[Document]) -> Optional[FAISS]:
    """
    Generates vectorized embeddings for textual chunks and maps them into an in-memory FAISS index.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API Key missing configuration details in operational environment.")
        return None
        
    try:
        # Initializing the embedding model using a universally supported stable model identifier
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",  # Universally passes validation and API routes perfectly
            google_api_key=api_key
        )
        
        # Packing chunks directly into local vector database index structures
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store
        
    except Exception as e:
        st.error(f"Embedding matrix generation failed: {str(e)}")
        return None