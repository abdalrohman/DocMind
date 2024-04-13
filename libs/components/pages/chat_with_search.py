import logging

import streamlit as st

from libs.components.configure_llm import set_llm
from libs.components.disable_components import disable_deploy
from libs.components.display_history import init_history, show_history
from libs.components.extra_sidbar_options import additional_options
from libs.components.header import set_page_header
from libs.components.llm_settings import set_llm_settings_sidebar
from libs.components.setup_search_engine import set_search_engine_tool
from libs.create_llm_chain import create_search_chain

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

# initiate chat history
history = init_history(key="chat_with_search_messages")

# display chat
show_history(history)

# Define tools
search = set_search_engine_tool()

if llm and search:
    search_chain = create_search_chain(llm, search)

    output_container = st.empty()
    if user_input := st.chat_input(key="input"):
        output_container = output_container.container()
        output_container.chat_message("human", avatar="👤").write(user_input)

        # Generate and display AI response
        answer_container = output_container.chat_message("assistant", avatar="🤖")
        full_response = ""
        msg_placeholder = answer_container.container().markdown(full_response + "▌")
        for chunk in search_chain.stream(
            {"question": user_input, "chat_history": history.messages},
            {"configurable": {"llm": provider}},
        ):
            full_response += chunk
            msg_placeholder.markdown(full_response + "▌")
        msg_placeholder.markdown(full_response)
        history.add_user_message(user_input)
        history.add_ai_message(full_response)

# Display additional options
additional_options(history)
