import logging
from pathlib import Path

import streamlit as st

from libs.components.configure_llm import set_llm
from libs.components.disable_components import disable_deploy
from libs.components.display_history import init_history, show_history
from libs.components.extra_sidbar_options import additional_options
from libs.components.header import set_page_header
from libs.components.llm_settings import set_llm_settings_sidebar
from libs.components.setup_embeddings import set_embeddings
from libs.components.upload_files import (
    DocumentProcessor
    )
from libs.create_llm_chain import create_chain

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Rag chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="auto",
    )

disable_deploy()

set_page_header()

# -- set llm --
llm = set_llm()

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
    doc_processor = DocumentProcessor(chroma_dir=chroma_dir, embeddings=embeddings)
    uploaded_docs = doc_processor.upload_documents()

    if uploaded_docs:
        retriever = doc_processor.process_files(temp_dir=temp_dir, )
    else:
        retriever = doc_processor.get_existing_db()

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

# Display additional options
additional_options(history)
