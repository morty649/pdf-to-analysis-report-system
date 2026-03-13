from pathlib import Path
from typing import List, Tuple, Any, Union
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from models.embeddings import load_embeddings
from config.config import VECTOR_DB_PATH

def create_vector_db(chunks: List[Union[dict, Document, str]]) -> Tuple[Any, List[Document]]:

    # ensuring the directory exists
    path = VECTOR_DB_PATH or "./vector_db"
    Path(path).mkdir(parents=True, exist_ok=True)

    embeddings = load_embeddings()

    docs: List[Document] = []
    for chunk in chunks:
        if isinstance(chunk, Document):
            docs.append(chunk)
            continue

        # dictionary type chunking
        if isinstance(chunk, dict):
            text = chunk.get("page_content", "") or ""
            metadata = chunk.get("metadata", {}) or {}
            docs.append(Document(page_content=text, metadata=metadata))
            continue

        # if its a string
        docs.append(Document(page_content=str(chunk), metadata={}))

   
    db = Chroma.from_documents(
        docs,
        embeddings,
        persist_directory=path
    )

    try:    
        db.persist()
    except Exception:
        pass

    return db, docs