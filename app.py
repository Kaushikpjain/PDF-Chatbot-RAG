import streamlit as st
from dotenv import load_dotenv
import os

# Absolute import patterns mapping internal logic structures
from utils.pdf_loader import extract_text_from_pdf
from utils.text_splitter import split_text_into_chunks
from utils.vector_store import create_vector_store
from utils.rag_chain import build_rag_chain

# Pull localized secure environment key files
load_dotenv()

# Global Client Tab Setup Configuration Layout Contexts
st.set_page_config(
    page_title="PDF Chatbot using RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Persistent Session Variable Allocations
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = ""

# --- SIDEBAR ARCHITECTURE ---
with st.sidebar:
    st.title("📌 Project Metadata")
    st.markdown("### Technology Stack")
    st.markdown(
        """
        - **Core Layer:** Python & Streamlit
        - **Orchestration:** LangChain
        - **Vector Database:** FAISS (In-Memory)
        - **Embeddings:** Google Text-Embedding-004
        - **LLM Engine:** Gemini 2.5 Flash
        """
    )
    st.write("---")
    st.markdown("### Document Index Status")
    if st.session_state.vector_store is not None:
        st.success(f"🟢 Active Index: {st.session_state.processed_filename}")
    else:
        st.info("🔴 Status: Waiting for PDF payload input processing tracks.")

# --- MAIN UI WORKSPACE LAYOUT ---
st.title("📖 PDF Chatbot using RAG")
st.write("Extract insight arrays directly from technical multi-page PDF documents locally using isolated context semantic chains.")
st.write("---")

# SECTION 1: UPPER FILE UPLOAD REGION (Full Width)
st.header("📂 1. Data Ingestion Layer")
uploaded_file = st.file_uploader(
    "Upload Source PDF Document Target",
    type=["pdf"],
    help="System bounds restrict processing limits to a maximum payload size of 20MB."
)

if uploaded_file is not None:
    # Enforce maximum size validation constraints before launching parsing actions
    file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
    
    if file_size_mb > 20:
        st.error("Payload rejection: File exceeds max operational capacity metrics of 20MB.")
    else:
        # Check if this file has already been ingested into session states
        if st.session_state.processed_filename != uploaded_file.name:
            with st.spinner("Extracting structural text from document pages..."):
                raw_text = extract_text_from_pdf(uploaded_file)
                
            if raw_text:
                with st.spinner("Executing recursive semantic chunk divisions..."):
                    chunks = split_text_into_chunks(raw_text)
                    
                with st.spinner("Generating embeddings and mounting FAISS memory vectors..."):
                    vector_store = create_vector_store(chunks)
                    
                if vector_store:
                    chain = build_rag_chain(vector_store)
                    if chain:
                        st.session_state.vector_store = vector_store
                        st.session_state.rag_chain = chain
                        st.session_state.processed_filename = uploaded_file.name
                        st.success(f"Successfully indexed: {uploaded_file.name}")
                        st.rerun()

st.write("---")

# SECTION 2 & 3: LOWER SEARCH & ANSWER REGION (Full Width)
st.header("💬 2. Contextual Search Sandbox")

# Prompt block execution safety guard locks input elements until vectors assemble
if st.session_state.rag_chain is None:
    st.warning("Please upload and process a target document in the Data Ingestion panel above to activate the LLM querying loop.")
else:
    user_query = st.text_input(
        "Ask a question about your PDF:",
        placeholder="e.g., What is the main objective of this paper?",
        key="query_input"
    )
    
    if user_query:
        if user_query.strip() == "":
            st.error("Input rejection: Question string space cannot be submitted empty.")
        else:
            with st.spinner("Searching document index and parsing answer vectors..."):
                try:
                    # Direct interaction loop targeting the running LCEL pipeline
                    response = st.session_state.rag_chain.invoke(user_query)
                    
                    st.write("---")
                    st.header("🤖 3. Answer Output Response")
                    
                    # Highlight responses using an info box container layout for better readability
                    st.info(response)
                    
                except Exception as e:
                    st.error(f"Execution Error running target RAG orchestration sequence: {str(e)}")