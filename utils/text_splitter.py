from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits continuous text string into decoupled Document structures using semantic separators.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    # Convert raw text into standard LangChain Document object structure
    documents = [Document(page_content=text)]
    return text_splitter.split_documents(documents)