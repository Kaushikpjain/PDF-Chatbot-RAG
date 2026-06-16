from typing import Optional
import pypdf
import streamlit as st

def extract_text_from_pdf(pdf_file) -> Optional[str]:
    """
    Extracts all raw text content from an uploaded PDF file wrapper.
    """
    try:
        pdf_reader = pypdf.PdfReader(pdf_file)
        extracted_text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text:
                extracted_text += page_text + "\n"
                
        if not extracted_text.strip():
            st.error("The uploaded PDF appears to contain no readable text layers.")
            return None
            
        return extracted_text
        
    except Exception as e:
        st.error(f"Failed to parse PDF document error: {str(e)}")
        return None