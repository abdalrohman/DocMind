import streamlit as st
from langchain_community.chat_message_histories.streamlit import (
    StreamlitChatMessageHistory,
)
from langchain_core.messages import AIMessage, HumanMessage


def init_history(key: str = "chat_history"):
    return StreamlitChatMessageHistory(key=key)


def show_history(history: StreamlitChatMessageHistory):
    for msg in history.messages:
        # st.chat_message(msg.type).write(msg.content)
        if isinstance(msg, HumanMessage):
            st.chat_message("human", avatar="👤").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("ai", avatar="🤖").write(msg.content)
