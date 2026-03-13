import os
from langchain.chat_models import init_chat_model
from config.config import GROQ_API_KEY,LLM_MODEL



def get_chatgroq_model():
    """Initializes and return the Groq chat model"""

    try:

        
        os.environ["GROQ_API_KEY"] = GROQ_API_KEY

       
        llm = init_chat_model(
            f"groq:{LLM_MODEL}",
            temperature=0.2
        )

        return llm

    except Exception as e:

        raise RuntimeError(
            f"Failed to initialize Groq model: {str(e)}"
        )