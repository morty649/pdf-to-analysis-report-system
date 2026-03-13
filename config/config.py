import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") or st.secrets.get("TAVILY_API_KEY", "")
from pathlib import Path

LLM_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"


CHUNK_SIZE =  600
CHUNK_OVERLAP = 100

# vector db path 
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH") or "./vector_db"


Path(VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)