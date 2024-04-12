import os
from pathlib import Path

import streamlit as st
from st_pages import Page, show_pages

from libs.config import config
from libs.utilities.env import EnvironmentLoader
from libs.utilities.log import Log

# Initialize a logger to record program execution details
log = Log(
    log_file_path=Path(config.log_path) / "docmind.log",
    level=config.log_level,
)
log.get_logger()

st.set_page_config(
    page_title="DocMind",
    page_icon="📚🧠",
)

# Set the title
st.markdown("# Welcome to DocMind!")
st.markdown("## **Your Personal Document Assistant 📚**")

# Load environment variables from a .env file if it exists
env_file_path = Path(".env")
if env_file_path.exists():
    EnvironmentLoader(env_file_path).load_envs()
else:
    st.error(
        "An `.env` file was not detected. Please rename the .env.example file to .env "
        "and configure your environmental variables, "
        "or provide the necessary information in the sidebar."
    )
    langchain_api_keys = [
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_PROJECT",
    ]
    search_engines_api_keys = [
        "GOOGLE_CSE_ID",
        "GOOGLE_SEARCH_API_KEY",
        "EXA_API_KEY",
        "TAVILY_API_KEY",
    ]
    llm_providers_api_keys = [
        "GROQ_API_KEY",
        "GOOGLE_API_KEY",
        "FIREWORKS_API_KEY",
        "OPENAI_API_KEY",
    ]
    embeddings_providers_api_keys = [
        "VOYAGE_API_KEY",
        "TOGETHER_API_KEY",
        "CLOUDFLARE_API_KEY",
        "CLOUDFLARE_ACCOUNT_ID",
    ]

    api_keys_dict = {
        "Langchain API Keys": langchain_api_keys,
        "Search Engines API Keys": search_engines_api_keys,
        "LLM Providers API Keys": llm_providers_api_keys,
        "Embeddings Providers API Keys": embeddings_providers_api_keys,
    }

    st.sidebar.markdown(
        "**Note:** Set only the keys for the providers you intend to use."
    )
    for expander_title, api_keys in api_keys_dict.items():
        st.sidebar.markdown(f"**Set your {expander_title}**:")
        for key in api_keys:
            user_key = st.sidebar.text_input(
                f"{key}",
                type="password",
                help=f"Set this if not set the the value in .env file for {key}.",
                key=key.lower(),  # Store the value in st.session_state
            )
            if user_key:
                os.environ[key] = user_key
    st.stop()

# Specify what pages should be shown in the sidebar, and what their titles and icons
# should be
show_pages(
    [
        Page("app.py", "DocMind", "🧠"),
        Page("libs/ui/pages/chat_with_search.py", "Chat With Enabled Search", "🔍"),
        Page("libs/ui/pages/rag_chat.py", "Rag chat", "📚"),
        # Page("", "Rag", "❓"),
    ]
)
