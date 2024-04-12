import logging

import streamlit as st
from langchain_core.runnables import ConfigurableField
from langchain_fireworks import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from libs.components.display_history import init_history, show_history
from libs.components.llm_settings import set_llm_settings_sidebar
from libs.components.setup_search_engine import set_search_engine_tool
from libs.constant import CHECKMARK_EMOJI
from libs.create_llm_chain import create_search_chain
from libs.settings import llm_settings as settings

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Rag chat",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="auto",
)

provider = set_llm_settings_sidebar()

# initiate chat history
history = init_history(key="chat_with_search_messages")

# display chat
show_history(history)

# Define tools
search = set_search_engine_tool()

# define LLMs
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
    ConfigurableField(id="llm"),
    default_key="chat_fireworks",
    gemini_pro=chat_gemini_pro,
    groq=chat_groq,
    openai=chat_openai,
).with_fallbacks([chat_fireworks, chat_gemini_pro, chat_groq, chat_openai])
# ).with_config(configurable={"llm": provider})

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
