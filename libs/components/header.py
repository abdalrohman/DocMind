from datetime import datetime

import streamlit as st


def set_page_header():
    st.header("Welcome to DocMind 📚!", anchor=False)
    st.caption(
        "DocMind, your personal assistant, is designed to utilize multiple providers such as "
        "Fireworks, OpenAI, Gemini, and Groq. Its purpose is to interact with your documents and "
        "extract information from the internet, among other things."
        )
    st.caption(f"Developed by M.Abdulrahman Alnaseer &copy; {datetime.now().year}")
