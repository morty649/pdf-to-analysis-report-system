import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

LLM_MODEL = "openai/gpt-oss-120b"

EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

VECTOR_DB_PATH = "./vector_db"