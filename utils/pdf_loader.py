from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.preprocessing import clean_text
from config.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(path):

    loader = PyPDFLoader(path)
    docs = loader.load()

    # Clean the raw page text
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    # Better chunking settings for PDFs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],  # better semantic splits
    )

    chunks = splitter.split_documents(docs)

    # Add chunk_id metadata (very useful later)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks