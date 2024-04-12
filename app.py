import logging
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

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="DocMind",
    page_icon="📚🧠",
)

# Set the title
st.markdown("# Welcome to DocMind!")
st.markdown("## **Your Personal Document Assistant 📚**")


# Load environment variables from a .env file if it exists
@st.cache_data(experimental_allow_widgets=True)
def load_env():
    env_file_path = Path(".env")
    st_secretes = Path(".streamlit/secrets.toml")
    st_secretes_sections = [
        "LangSmith_Tracing",
        "LLMS",
        "Search_Engine",
        "Embiddings",
        "Telemetry",
    ]
    if env_file_path.exists():
        EnvironmentLoader(env_file_path).load_envs()
    elif st_secretes.exists():
        logger.info("Loading secrets from .streamlit/secrets.toml")
        for section in st_secretes_sections:
            for key, value in st.secrets[section].items():
                os.environ[key] = value
    else:
        logger.info("Loading keys from sidebar")
        api_keys_dict = {
            "Langchain API Keys": ["LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"],
            "Search Engines API Keys": [
                "GOOGLE_CSE_ID",
                "GOOGLE_SEARCH_API_KEY",
                "EXA_API_KEY",
                "TAVILY_API_KEY",
            ],
            "LLM Providers API Keys": [
                "GROQ_API_KEY",
                "GOOGLE_API_KEY",
                "FIREWORKS_API_KEY",
                "OPENAI_API_KEY",
            ],
            "Embeddings Providers API Keys": [
                "VOYAGE_API_KEY",
                "TOGETHER_API_KEY",
                "CLOUDFLARE_API_KEY",
                "CLOUDFLARE_ACCOUNT_ID",
            ],
        }
        st.session_state.setdefault("submit_env", False)

        if not st.session_state["submit_env"]:
            st.sidebar.error(
                "An `.env` file was not detected. Please rename the .env.example file to .env "
                "and configure your environmental variables, "
                "or provide the necessary information in the sidebar."
            )
            st.sidebar.markdown(
                "**Note:** Set only the keys for the providers you intend to use."
            )
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
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
            st.sidebar.button("Submit", key="submit_env", type="primary")
            st.stop()


load_env()

clear_cache = st.sidebar.button("Clear cache")
if clear_cache:
    st.cache_data.clear()
    st.rerun()

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
