from libs.components.configure_api_keys import set_api_keys
from libs.components.disable_components import disable_deploy
from libs.components.header import set_page_header

__import__("pysqlite3")
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

import logging
import os
import sqlite3
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
logger.info(f"Sqlite version: {sqlite3.sqlite_version}")

st.set_page_config(
    page_title="DocMind",
    page_icon="📚🧠",
    layout="wide",
    menu_items={
        "Get Help"    : "https://github.com/abdalrohman/DocMind/discussions",
        "Report a bug": "https://github.com/abdalrohman/DocMind/issues",
        },
    )

disable_deploy()

set_page_header()


# Load environment variables from a .env file if it exists
# @st.cache_data(experimental_allow_widgets=True)
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
        set_api_keys()


load_env()

clear_cache = st.sidebar.button("Clear cache", use_container_width=True, type="primary")
if clear_cache:
    st.cache_data.clear()
    st.rerun()

# Specify what pages should be shown in the sidebar, and what their titles and icons
# should be
show_pages(
    [
        Page("app.py", "DocMind", "🧠"),
        Page(
            "libs/components/pages/chat_with_search.py",
            "Chat With Enabled Search",
            "🔍",
            ),
        Page("libs/components/pages/rag_chat.py", "Rag chat", "📚"),
        # Page("", "Rag", "❓"),
        ]
    )
