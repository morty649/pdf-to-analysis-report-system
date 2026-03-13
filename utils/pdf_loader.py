from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.preprocessing import clean_text
from config.config import CHUNK_SIZE, CHUNK_OVERLAP


def load_pdf(path):

    loader = PyPDFLoader(path)
    docs = loader.load()

    for doc in docs:
        doc.page_content = clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(docs)

    return chunks