import streamlit as st
import tempfile
import os
from pathlib import Path
from typing import Tuple, List, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from models.llm import get_chatgroq_model
from models.embeddings import load_embeddings
from utils.pdf_loader import load_pdf
from utils.hybrid_retriever import HybridRetriever
from utils.vector_store import create_vector_db
from utils.tavily_search import search_web
from utils.preprocessing import clean_text
from langchain.memory import ConversationBufferMemory


import os
from config.config import VECTOR_DB_PATH

os.makedirs(VECTOR_DB_PATH, exist_ok=True)
from config.config import GROQ_API_KEY, TAVILY_API_KEY, VECTOR_DB_PATH, CHUNK_SIZE, CHUNK_OVERLAP


# to get response from model

def get_chat_response(chat_model, messages: List[dict], system_prompt: str) -> str:
    
    try:
        formatted_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))

        response = chat_model.invoke(formatted_messages)
        return response.content
    except Exception as e:
        return f"Error getting response: {str(e)}"


def build_system_prompt(context: str, response_mode: str, memory_summary: str = "") -> str:
    if response_mode == "Concise":
        mode_instruction = (
            "Response Style: CONCISE "
            "- Reply in 2-4 sentences maximum "
            "- Be direct and to the point. "
            "- No long explanations unless absolutely necessary. "
        )
    else:
        mode_instruction = (
            "Response Style: DETAILED "
            "- Provide a thorough, in-depth response. "
            "- Use bullet points, numbered lists, or sections where helpful. "
            "- Cover edge cases or caveats if relevant. "
        )

    mem_section = f"\nConversation summary (brief):\n{memory_summary}\n" if memory_summary else ""

    return f"""
You are an AI assistant that helps users understand documents related to finance and regulations.
Don't hallucinate if you understand the query reason and respond , else say "I was said to not hallucinate so I don't know about it " to  the same query.
Use the provided context to answer the question accurately.


{mode_instruction}

{mem_section}

Context:
{context}
"""



# Page CSS Inspired from my previous project

PAGE_STYLE = """ 
<style> body { background: #f7f4ee; color: #1a1a18; } 
.cover { background: #1a1a18; color: #f0ece3; padding: 35px; border-radius: 8px; margin-bottom: 20px; } 
.cover-title { font-size: 32px; font-weight: 700; } 
.cover-sub { color: #a0998e; margin-top: 6px; } 
.user-bubble { background: #eef3ff; padding: 10px; border-radius: 6px; color: #111; border: 1px solid #d0d7ff; margin-bottom: 8px; } 
.bot-bubble { background: #f8f8f8; padding: 10px; border-radius: 6px; color: #111; border: 1px solid #e0e0e0; margin-bottom: 8px; } 
.mode-badge-concise { display: inline-block; background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7; border-radius: 12px; padding: 2px 10px; font-size: 12px; font-weight: 600; margin-bottom: 6px; } 
.mode-badge-detailed { display: inline-block; background: #e3f2fd; color: #1565c0; border: 1px solid #90caf9; border-radius: 12px; padding: 2px 10px; font-size: 12px; font-weight: 600; margin-bottom: 6px; } 
.section-label { font-size: 13px; font-weight: 600; color: #555; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; } </style> """



# Cached resource wrappers for deployment

@st.cache_resource
def init_chat_model():
    # Wrap the model-init so Streamlit caches the model object across runs
    try:
        model = get_chatgroq_model()
        return model
    except Exception as e:
        st.error(f"Failed initializing LLM: {e}")
        return None

@st.cache_resource
def init_embeddings():
    try:
        emb = load_embeddings()
        return emb
    except Exception:
        return None


# Main Chat Page

def chat_page():

    # prepare VECTOR_DB_PATH
    Path(VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)

    # session state defaults
    st.session_state.setdefault("vector_db", None)
    st.session_state.setdefault("docs", None)
    st.session_state.setdefault("embeddings", None)

    st.markdown(PAGE_STYLE, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="cover">
            <div class="cover-title">Regulatory AI Assistant</div>
            <div class="cover-sub">
                Upload a regulatory PDF and ask questions about it. The assistant will decide whether to use the PDF or call the web tool automatically.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # initialising LLM once per deployment (cached)
    if "chat_model" not in st.session_state:
        st.session_state.chat_model = init_chat_model()

    chat_model = st.session_state.chat_model

    # Conversation summary memory (ConversationBufferMemory)
    if "memory" not in st.session_state:
        try:
            st.session_state.memory = ConversationBufferMemory(
                return_messages=False,
                memory_key="chat_history"
            )
        except Exception:
            st.session_state.memory = None


    left, right = st.columns([1, 2])

    
    # LEFT: PDF upload + controls
    
    with left:
        with st.container():
            st.markdown('<div class="section-label"> Document </div>', unsafe_allow_html=True)

            uploaded_file = st.file_uploader("Upload PDF", type="pdf", label_visibility="visible")

            if uploaded_file and st.session_state.vector_db is None:
                # using a safe temp file that works in Streamlit Cloud / Docker
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                try:
                    tmp.write(uploaded_file.read())
                    tmp.flush()
                    tmp.close()

                    st.info("Processing document (extracting & indexing) — please wait...")

                    
                    chunks = load_pdf(tmp.name)

                    # Cleaning before embedding
                    cleaned_chunks = []
                    for c in chunks:
                        # If dict okay else assume to be a string
                        if isinstance(c, dict) and "page_content" in c:
                            cleaned = clean_text(c["page_content"])
                            # keep existing metadata if present
                            cleaned_chunks.append({**c, "page_content": cleaned})
                        else:
                            cleaned_chunks.append({"page_content": clean_text(str(c)), "metadata": {}})

                    vector_db, docs = create_vector_db(cleaned_chunks)
                    st.session_state.vector_db = vector_db
                    st.session_state.docs = docs
                    st.success("Document embedded successfully.")
                except Exception as e:
                    st.error(f"Failed to index document: {e}")
                finally:
                    # try to remove tmp file
                    try:
                        os.remove(tmp.name)
                    except Exception:
                        pass

            if st.session_state.vector_db:
                st.caption(" Document indexed and ready for retrieval.")

        st.divider()

        # Response mode selector
        with st.container():
            st.markdown('<div class="section-label"> Response Mode</div>', unsafe_allow_html=True)
            response_mode = st.radio(
                "Select response style",
                options=["Concise", "Detailed"],
                index=0 if st.session_state.get("response_mode", "Concise") == "Concise" else 1,
                label_visibility="collapsed",
            )
            st.session_state["response_mode"] = response_mode

            if response_mode == "Concise":
                st.markdown('<span class="mode-badge-concise">Concise mode active</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="mode-badge-detailed">Detailed mode active</span>', unsafe_allow_html=True)

        st.divider()

        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            if "memory" in st.session_state:
                try:
                    st.session_state.memory = ConversationBufferMemory(
                        return_messages=False,
                        memory_key="chat_history"
                    )
                except Exception:
                    st.session_state.memory = None


    
    # RIGHT: Chat UI (unchanged logic, but kept robust)
    
    with right:
        st.subheader("Chat")

        st.session_state.setdefault("messages", [])

        # render history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f'<div class="user-bubble">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-bubble">{message["content"]}</div>', unsafe_allow_html=True)

        prompt = st.chat_input("Ask something about the document...")

        if prompt:
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.markdown(f'<div class="user-bubble">{prompt}</div>', unsafe_allow_html=True)

            with st.spinner("Thinking..."):

                # 1) RAG retrieval from uploaded PDF (if present)
                retrieved_context = ""
                retrieved_docs: List[Any] = []
                if st.session_state.get("vector_db"):
                    try:
                        hybrid = HybridRetriever(
                            st.session_state.vector_db,
                            st.session_state.docs
                        )
                        retrieved_context, retrieved_docs = hybrid.retrieve(prompt)
                    except Exception as e:
                        st.warning(f"Retrieval error: {e}")
                        retrieved_context = ""
                        retrieved_docs = []

                # 2) memory summary
                memory_summary = ""
                if st.session_state.get("memory"):
                    try:
                        mem_vars = st.session_state.memory.load_memory_variables({})
                        memory_summary = mem_vars.get("chat_history", "") or mem_vars.get("summary", "") or ""
                    except Exception:
                        memory_summary = ""

                # 3) Router decision
                router_system = (
                    "You are a routing assistant. Decide whether the provided document context is sufficient to answer the user's question."
                    " Answer exactly with one token: YES or NO. Only output YES or NO."
                )
                router_user = f"Question: {prompt} \n Document context (may be empty or partial): {retrieved_context} Is the context sufficient to answer the question?"

                try:
                    router_response = ""
                    if chat_model:
                        formatted_router_msgs = [
                            SystemMessage(content=router_system),
                            HumanMessage(content=router_user),
                        ]
                        router_out = chat_model.invoke(formatted_router_msgs)
                        router_response = (router_out.content or "").strip().upper()
                    else:
                        router_response = "NO"
                except Exception as e:
                    st.warning(f"Router error: {e}")
                    router_response = "NO"

                use_web = False
                web_context = ""

                if router_response.startswith("NO"):
                    try:
                        st.info("PDF context insufficient — searching the web for additional info...")
                        web_context = search_web(prompt) or ""
                        if web_context:
                            use_web = True
                    except Exception as e:
                        st.warning(f"Web search failed: {e}")
                        web_context = ""
                        use_web = False

                final_context = ""
                if use_web and web_context:
                    final_context = web_context
                    if retrieved_context:
                        final_context = f"{retrieved_context}[PDF context was partial]{web_context}"
                else:
                    final_context = retrieved_context

                response_mode = st.session_state.get("response_mode", "Concise")
                system_prompt = build_system_prompt(final_context, response_mode, memory_summary)

                recent_messages = st.session_state.messages[-10:]
                try:
                    assistant_response = get_chat_response(
                        chat_model,
                        recent_messages,
                        system_prompt
                    )
                except Exception as e:
                    assistant_response = f"Error generating response: {e}"

                st.markdown(f'<div class="bot-bubble">{assistant_response}</div>', unsafe_allow_html=True)

                # show sources
                if retrieved_docs:
                    with st.expander("Retrieved from / Sources"):
                        for i, doc in enumerate(retrieved_docs, 1):
                            page = doc.metadata.get("page", None)
                            page_str = f"Page {page+1}" if page is not None else "Page unknown"
                            st.text_area(
                                f"Source {i} ({page_str})",
                                doc.page_content[:1200] + ("..." if len(doc.page_content) > 1200 else ""),
                                height=180,
                                disabled=True
                            )

                if use_web and web_context:
                    with st.expander("Web Search Powered by Tavily"):
                        st.text_area("Web context (top results)", web_context[:3000], height=240, disabled=True)

                st.session_state.messages.append({"role": "assistant", "content": assistant_response})

                if st.session_state.get("memory"):
                    try:
                        st.session_state.memory.save_context({"input": prompt}, {"output": assistant_response})
                    except Exception:
                        pass


# Main

def main():
    st.set_page_config(page_title="Regulatory AI Assistant",layout="wide")
    chat_page()


if __name__ == "__main__":
    main()