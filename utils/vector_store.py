from langchain_community.vectorstores import Chroma
from models.embeddings import load_embeddings
from config.config import VECTOR_DB_PATH


def create_vector_db(chunks):

    embeddings = load_embeddings()

    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=VECTOR_DB_PATH
    )

    return db,chunks