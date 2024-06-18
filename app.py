import logging
import os
from datetime import datetime
from pathlib import Path

import streamlit as st
import yaml
from st_pages import Page, show_pages

from docmind.auth.authenticator import AuthenticatorConfig, Authenticator
from docmind.utils.common import setup_user_directory, setup_page, export_and_download_user_data
from docmind.utils.config import ProjectConfiguration
from docmind.utils.env import EnvironmentLoader
from docmind.utils.log import LogConfig

# Setup Streamlit page
setup_page("DocMind", "🧠")

# write authenticator_config file
authenticator_config = Path(".").resolve() / "authenticator_config.yaml"
if not authenticator_config.exists():
    default_config = {
        "cookie": {"expiry_days": 30, "key": "docmind_dm", "name": "docmind"},
        "credentials": {
            "usernames": {
                "docmind": {
                    "email": "docmind@gmail.com",
                    "failed_login_attempts": 0,
                    "logged_in": True,
                    "name": "docmind",
                    "password": "docmind",  # This should ideally be a hashed password in production
                }
            }
        },
        "pre-authorized": {"emails": ["docmind@gmail.com"]},
    }

    try:
        with open(authenticator_config, "w") as f:
            yaml.safe_dump(default_config, f, default_flow_style=False)
    except (IOError, yaml.YAMLError) as e:
        st.error(f"Error writing authenticator config: {e}")

# User authentication
config_manager = AuthenticatorConfig(config_path=authenticator_config)
authenticator = Authenticator(config_manager)
name, authentication_status, username = authenticator.login()
if not authentication_status:
    email, username, name = authenticator.register()
    config_manager.save_config()
    if email:
        st.success(f'User {username} registered successfully')

if not authentication_status:
    st.stop()

# Set up user directory
user_data_dir = setup_user_directory()

# Setup config
config_file_path = Path(".") / "config.yaml"
ProjectConfiguration.create_default_config_file(config_file_path, Path(".").resolve())
config = ProjectConfiguration.load_config(config_file_path)

# Setup logger
log = LogConfig(log_file_path=config.log_path / "docmind.log", level=config.log_level)
log.get_logger()
logger = logging.getLogger(__name__)

logger.info("DocMind is running...")

# check if COHERE_API_KEY inside the env
env_file_path = user_data_dir / ".env"
if not env_file_path.is_file():
    with st.container(border=True):
        st.info("Please enter your COHERE_API_KEY below. You can get it from https://dashboard.cohere.com/api-keys")
        api_key = st.text_input("Cohere Api Key:", type="password", key="COHERE_API_KEY")
        if api_key:
            with open(env_file_path, "w") as f:
                f.write(f"COHERE_API_KEY='{api_key}'\n")
            EnvironmentLoader(env_file_path=env_file_path).load_envs()
            st.rerun()
    st.stop()

if os.environ.get("COHERE_API_KEY") is None:
    EnvironmentLoader(env_file_path=env_file_path).load_envs()
    st.stop()

# Navigate between pages
show_pages(
    [
        Page("app.py", "DocMind", "🧠"),
        Page("docmind/pages/chat.py", "ChatBot", "💬"),
        Page("docmind/pages/docs_chat.py", "DocumentsChat", "📚"),
    ]
)

if authentication_status:
    with st.sidebar.container(border=True):
        st.write(f'Welcome **`{name}`**')
        authenticator.logout(button_label='Logout', location='sidebar')

    # Display information about the app
    st.header("Welcome to DocMind 🧠", anchor=False)

    # DocMind Description
    st.subheader("What is DocMind?")
    st.write(
        """DocMind is your personal AI assistant powered by Cohere, designed to interact with your documents and the internet. 
        It leverages Cohere's powerful language models and web search capabilities to provide you with insightful answers and information.
        """
    )

    # DocMind Features
    st.subheader("DocMind Features:")
    st.write(
        """
        - **Document Chat:**  Ask questions about your documents and get answers directly from the content.
        - **ChatBot:** Engage in natural conversations with a powerful language model. 
        - **Cohere Web Search Integration:** Access the vast knowledge of the internet to enrich your answers. 
        """
    )

    # Getting Started Guide
    st.subheader("Getting Started:")
    st.write(
        """
        1. **Upload Documents:** Go to the "DocumentsChat" page and upload your documents (PDF files currently supported).
        2. **Start Chatting:** Once your documents are processed, you can ask questions and get answers from your documents and the internet.
        3. **ChatBot:** Go to the "ChatBot" page to interact with a powerful language model without needing any documents.
        """
    )

    st.divider()
    st.caption(f"Developed by M.Abdulrahman Alnaseer &copy; {datetime.now().year}")
    st.info(
        "DocMind is still in development, so some features may not work as expected. "
        "Please report any issues or bugs to the GitHub repository."
    )

    clear_cache = st.sidebar.button("Clear cache", use_container_width=True, type="primary")
    if clear_cache:
        st.cache_data.clear()
        st.rerun()

    if st.sidebar.button(f"Export `{username}` Data", type="primary", use_container_width=True):
        zip_buffer, zip_filename = export_and_download_user_data(st.session_state['user_dir'], username)
        st.sidebar.download_button(
            label=f"Download `{username}` Data",
            data=zip_buffer,
            file_name=zip_filename,
            mime="application/zip",
            type="primary", use_container_width=True
        )

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

# Save config
config_manager.save_config()
