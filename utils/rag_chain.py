from typing import Optional
import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

def build_rag_chain(vector_store: FAISS):
    """
    Constructs a deterministic LCEL RAG execution pipeline using the vector store retriever and Gemini.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API Key missing configuration details in operational execution tracks.")
        return None

    try:
        # Initializing the target foundation model pipeline
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.0,  # Minimize creativity to enforce grounding
            google_api_key=api_key
        )
        
        # Establishing standard top-k similarity search parameters
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        
        # Designing strict operational context confinement templates
        template = """You are a helpful AI assistant.
Answer the user's question only using the provided PDF context.
If the answer is not present in the context, respond:
"I could not find this information in the uploaded PDF."

Context:
{context}

Question:
{question}

Answer:"""
        
        prompt = PromptTemplate.from_template(template)
        
        # Helper framework utility to join document string fragments together
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
            
        # Generating functional LangChain Expression Language (LCEL) runtime chains
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain
        
    except Exception as e:
        st.error(f"Initialization parameters failed for context layer assembly: {str(e)}")
        return None