from pathlib import Path

import streamlit as st
from langchain_core.runnables import ConfigurableField
from langchain_fireworks import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from libs.components.display_history import init_history, show_history
from libs.components.llm_settings import set_llm_settings_sidebar
from libs.components.setup_embeddings import set_embeddings
from libs.components.upload_files import (
    get_existing_db,
    process_files,
    upload_documents,
)
from libs.constant import CHECKMARK_EMOJI
from libs.create_llm_chain import create_chain
from libs.settings import llm_settings as settings

st.set_page_config(
    page_title="Rag chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="auto",
)

provider = set_llm_settings_sidebar()

embeddings = set_embeddings()

# initiate chat history
history = init_history(key="chat_messages")

# display chat
show_history(history)

# -- get retriever --
data_dir = Path("./data")
temp_dir = data_dir / "temp"
chroma_dir = data_dir / "chroma"

with st.sidebar:
    uploaded_docs = upload_documents()

    if uploaded_docs:
        retriever = process_files(
            temp_dir=temp_dir,
            chroma_dir=chroma_dir,
            _embeddings=embeddings,
            uploaded_files=uploaded_docs,
        )
    else:
        retriever = get_existing_db(chroma_dir=chroma_dir, embeddings=embeddings)

# -- set llm --
chat_fireworks = ChatFireworks(
    model=settings.fireworks_model,
    temperature=settings.temperature,
    streaming=settings.streaming,
    max_tokens=settings.max_tokens,
)
chat_gemini_pro = ChatGoogleGenerativeAI(
    model=settings.gemini_model,
    temperature=settings.temperature,
    max_output_tokens=settings.max_tokens,
    max_retries=settings.max_retries,
    convert_system_message_to_human=True,
    safety_settings=st.session_state["safety_settings"],
)
chat_groq = ChatGroq(
    model=settings.groq_model,
    temperature=settings.temperature,
    streaming=settings.streaming,
    max_tokens=settings.max_tokens,
    max_retries=settings.max_retries,
)
chat_openai = ChatOpenAI(
    model=settings.openai_model,
    temperature=settings.temperature,
    streaming=settings.streaming,
    max_tokens=settings.max_tokens,
    max_retries=settings.max_retries,
)

llm = chat_fireworks.configurable_alternatives(
    # This gives this field an id
    # When configuring the end runnable, we can then use this id to configure this field
    ConfigurableField(id="llm"),
    default_key="chat_fireworks",
    gemini_pro=chat_gemini_pro,
    groq=chat_groq,
    openai=chat_openai,
).with_fallbacks([chat_fireworks, chat_gemini_pro, chat_groq, chat_openai])

if llm and retriever:
    answer_chain = create_chain(llm, retriever)

    # Handle user input
    output_container = st.empty()
    if user_input := st.chat_input(key="input"):
        output_container = output_container.container()
        output_container.chat_message("human", avatar="👤").write(user_input)
        history.add_user_message(user_input)

        # Generate and display AI response
        answer_container = output_container.chat_message("assistant", avatar="🤖")
        full_response = ""
        msg_placeholder = answer_container.container().markdown(full_response + "▌")
        for chunk in answer_chain.stream(
            {"question": user_input, "chat_history": history.messages},
            {"configurable": {"llm": provider}},
        ):
            full_response += chunk
            msg_placeholder.markdown(full_response + "▌")
        msg_placeholder.markdown(full_response)
        history.add_ai_message(full_response)

# Sidebar options
with st.sidebar:
    with st.expander("Session state:", expanded=False):
        st.write(st.session_state)

    # Clear and export history buttons
    clear_button, export_button = st.columns([1, 1], gap="small")
    if clear_button.button("Clear History"):
        st.session_state.clear()
        st.rerun()
    if export_button.button("Export history"):
        st.download_button(
            label="Download history",
            data=str(history),
            file_name="history.txt",
            mime="text/plain",
        )
        st.toast(f"{CHECKMARK_EMOJI} History exported.")
