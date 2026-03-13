import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from models.llm import get_chatgroq_model
from models.embeddings import load_embeddings
from utils.retrieval import retrieve_context
from utils.pdf_loader import load_pdf
from utils.vector_store import create_vector_db
from utils.tavily_search import search_web


# Chat Model Helper

def get_chat_response(chat_model, messages, system_prompt):

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



# CSS Styling
# Inspiration taken from my previous project
PAGE_STYLE = """
<style>

body {
    background: #f7f4ee;
    color: #1a1a18;
}

.cover {
    background: #1a1a18;
    color: #f0ece3;
    padding: 35px;
    border-radius: 8px;
    margin-bottom: 20px;
}

.cover-title {
    font-size: 32px;
    font-weight: 700;
}

.cover-sub {
    color: #a0998e;
    margin-top: 6px;
}

.user-bubble {
    background: #eef3ff;
    padding: 10px;
    border-radius: 6px;
    color: #111;
    border: 1px solid #d0d7ff;
    margin-bottom: 8px;
}

.bot-bubble {
    background: #f8f8f8;
    padding: 10px;
    border-radius: 6px;
    color: #111;
    border: 1px solid #e0e0e0;
    margin-bottom: 8px;
}

.mode-badge-concise {
    display: inline-block;
    background: #e8f5e9;
    color: #2e7d32;
    border: 1px solid #a5d6a7;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 6px;
}

.mode-badge-detailed {
    display: inline-block;
    background: #e3f2fd;
    color: #1565c0;
    border: 1px solid #90caf9;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 6px;
}

.section-label {
    font-size: 13px;
    font-weight: 600;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

</style>
"""



# Which type of response do you expect

def build_system_prompt(context: str, response_mode: str) -> str:

    if response_mode == "Concise":
        mode_instruction = """
        Response Style: CONCISE
        - Reply in 2-4 sentences maximum.
        - Be direct and to the point.
        - No long explanations unless absolutely necessary
        - Summarize only the key answer.
        """
    else:
        mode_instruction = """
        Response Style: DETAILED
        - Provide a thorough, in-depth response.
        - Use bullet points, numbered lists, or sections where helpful.
        - Cover edge cases or caveats if relevant.
        """

    return f"""
        You are an AI assistant that helps users understand financial and regulatory documents.
        Use the provided context to answer the question accurately.
        If the answer is not present in the context, say clearly that you do not know.
        {mode_instruction}
        Context:
        {context}
        """



# Chat Page

def chat_page():

    st.markdown(PAGE_STYLE, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="cover">
            <div class="cover-title">Regulatory AI Assistant</div>
            <div class="cover-sub">
                Upload a regulatory PDF and ask questions about it. Supports live web search and adjustable response depth.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    chat_model = get_chatgroq_model()

    left, right = st.columns([1, 2])

    # ---------------------------
    # LEFT SIDE
    # ---------------------------
    with left:

        # --- PDF Upload / RAG ---
        with st.container(border=True):

            st.markdown('<div class="section-label"> PDF </div>', unsafe_allow_html=True)

            uploaded_file = st.file_uploader(
                "Upload PDF for the ",
                type="pdf",
                label_visibility="collapsed"
            )

            if uploaded_file and "vector_db" not in st.session_state:

                with open("temp.pdf", "wb") as f:
                    f.write(uploaded_file.read())

                st.info("Understand the document Please Wait...")

                chunks = load_pdf("temp.pdf")
                embedding_model = load_embeddings()
                vector_db, docs = create_vector_db(chunks)
                st.session_state.vector_db = vector_db
                st.success("Document indexed successfully!")

            if "vector_db" in st.session_state:
                st.caption("Document loaded and ready")

        st.divider()

        # --- Web Search Toggle ---
        with st.container(border=True):

            st.markdown('<div class="section-label"> Live Web Search</div>', unsafe_allow_html=True)

            use_web_search = st.toggle(
                "Enable web search",
                value=st.session_state.get("use_web_search", False),
                help="When enabled, the assistant will search the web to supplement document context."
            )
            st.session_state["use_web_search"] = use_web_search

            if use_web_search:
                st.caption("Web search is active. Results will supplement document context.")
            else:
                st.caption("Web search is off. Answers from document only.")

        st.divider()

        # --- Response Mode ---
        with st.container(border=True):

            st.markdown('<div class="section-label">Response Mode</div>', unsafe_allow_html=True)

            response_mode = st.radio(
                "Select mode",
                options=["Concise", "Detailed"],
                index=0 if st.session_state.get("response_mode", "Concise") == "Concise" else 1,
                label_visibility="collapsed",
                help="Concise: short 2-4 sentence answers. Detailed: full explanations with structure."
            )
            st.session_state["response_mode"] = response_mode

            if response_mode == "Concise":
                st.markdown('<span class="mode-badge-concise">Concise mode active</span>', unsafe_allow_html=True)
                st.caption("Short, direct answers — key points only.")
            else:
                st.markdown('<span class="mode-badge-detailed">Detailed mode active</span>', unsafe_allow_html=True)
                st.caption("In-depth answers with context, lists, and explanations.")

        st.divider()

        # --- Clear Button ---
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            '''if "vector_db" in st.session_state:
                del st.session_state.vector_db'''
            st.rerun()

    # RIGHT SIDE (CHAT)
  
    with right:

        st.subheader("Chat")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="user-bubble">{message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="bot-bubble">{message["content"]}</div>',
                    unsafe_allow_html=True
                )

        prompt = st.chat_input("Ask something about the document...")

        if prompt:

            st.session_state.messages.append({"role": "user", "content": prompt})

            st.markdown(
                f'<div class="user-bubble">{prompt}</div>',
                unsafe_allow_html=True
            )

            with st.spinner("Thinking..."):

                context = ""
                retrieved_docs = []
                web_results = []

                # --- RAG Retrieval ---
                if "vector_db" in st.session_state:
                    context, retrieved_docs = retrieve_context(
                        st.session_state.vector_db,
                        prompt
                    )

                # --- Web Search (if enabled) ---
                if st.session_state.get("use_web_search", False):
                    web_context, web_results = search_web(prompt)
                    if web_context:
                        context = (
                            context + "\n\n[Web Search Results]\n" + web_context
                            if context else web_context
                        )

                # --- Build system prompt with mode ---
                response_mode = st.session_state.get("response_mode", "Concise")
                system_prompt = build_system_prompt(context, response_mode)

                response = get_chat_response(
                    chat_model,
                    st.session_state.messages,
                    system_prompt
                )

                st.markdown(
                    f'<div class="bot-bubble">{response}</div>',
                    unsafe_allow_html=True
                )

                # --- Document source citations ---
                if retrieved_docs:
                    with st.expander("Sources: Trust Me Bro 😔"):
                        for i, doc in enumerate(retrieved_docs, 1):
                            page = doc.metadata.get("page", 0) + 1
                            st.text_area(
                                f"Source {i} (Page {page})",
                                doc.page_content[:400] + "...",
                                height=120,
                                disabled=True
                            )

                # --- Web search source citations ---
                if web_results:
                    with st.expander("Web Search Sources"):
                        for i, result in enumerate(web_results, 1):
                            st.markdown(
                                f"**{i}. [{result.get('title', 'Result')}]({result.get('url', '#')})**"
                            )
                            st.caption(result.get("snippet", "No preview available."))

            st.session_state.messages.append({"role": "assistant", "content": response})



# MAIN

def main():

    st.set_page_config(
        page_title="Regulatory AI Assistant",
        page_icon="",
        layout="wide"
    )

    chat_page()


if __name__ == "__main__":
    main()