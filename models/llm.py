import os
from langchain.chat_models import init_chat_model
from config.config import GROQ_API_KEY


def get_chatgroq_model():
    """Initialize and return the Groq chat model"""

    try:

        
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY

       
        llm = init_chat_model(
            "groq:openai/gpt-oss-120b",
            temperature=0.2
        )

        return llm

    except Exception as e:

        raise RuntimeError(
            f"Failed to initialize Groq model: {str(e)}"
        )