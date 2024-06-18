import json
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Union
from zipfile import ZipFile

import cohere
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from docmind.auth.authenticator import AuthenticatorConfig, Authenticator


def authenticate_user():
    config_manager = AuthenticatorConfig()
    authenticator = Authenticator(config_manager)
    name, authentication_status, username = authenticator.login()
    if not authentication_status:
        email, username, name = authenticator.register()
        config_manager.save_config()
        if email:
            st.success(f'User {username} registered successfully')
    return name, authentication_status, username


@st.cache_resource
def get_cohere_models():
    """Retrieve and cache the list of Cohere models"""
    # {'summarize', 'chat', 'generate'}
    co = cohere.Client()
    response = co.models.list()
    models = {}
    for model in response.models:
        models[model.name] = {
            'context_length': model.context_length,
            'endpoints': model.endpoints
        }
    return models


def setup_user_directory():
    """Creates the user directory if it doesn't exist."""
    if st.session_state['username']:
        user_data_dir = Path('user_data')
        user_data_dir.mkdir(exist_ok=True)
        user_dir = user_data_dir.resolve() / st.session_state['username']
        user_dir.mkdir(exist_ok=True)
        if user_dir.is_dir():
            st.session_state['user_dir'] = str(user_dir)
        return user_dir


# This must be the first Streamlit command used on an app page, and must only be set once per page.
# https://docs.streamlit.io/develop/api-reference/configuration/st.set_page_config
def setup_page(page_title: str, page_icon: str):
    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="auto"
    )

    # Remove the Streamlit `Deploy` button from the Header
    st.markdown(
        r"""
    <style>
    .stDeployButton {
            visibility: hidden;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )


def setup_chat_history(chat_folder: Union[str, Path],
                       current_chat_history_file_key: str,
                       chat_history_session_state: StreamlitChatMessageHistory = None,
                       selected_file_to_load_history: str = None,
                       action: str = "setup"):
    if isinstance(chat_folder, str):
        chat_folder = Path(chat_folder)
    chat_folder.mkdir(exist_ok=True, parents=True)

    # initialize chat file name
    chat_history_file = chat_folder / f'chat_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    if selected_file_to_load_history:
        chat_history_file = chat_folder / selected_file_to_load_history
        st.session_state[current_chat_history_file_key] = chat_history_file

    if current_chat_history_file_key not in st.session_state:
        st.session_state[current_chat_history_file_key] = chat_history_file
    chat_history_file = Path(st.session_state[current_chat_history_file_key])

    # validate actions
    valid_actions = ['new_chat', 'old_chat', 'save_chat', 'setup']
    if action not in valid_actions:
        raise ValueError(f"Invalid action '{action}'. Must be one of {', '.join(valid_actions)}")

    if action == 'setup':
        return

    if action == 'save_chat':
        chat_history_json = [
            {"role": "user", "content": msg.content} if isinstance(msg, HumanMessage) else
            {"role": "assistant", "content": msg.content} for msg in chat_history_session_state.messages
        ]
        if chat_history_json:
            with chat_history_file.open('w') as f:
                json.dump(chat_history_json, f, indent=4)
                f.write('\n')
        else:
            st.warning("No chat history to save.")

    if action == 'new_chat':
        chat_history_file = chat_folder / f'chat_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        st.session_state[current_chat_history_file_key] = chat_history_file
        with open(chat_history_file, 'w') as f:
            json.dump([], f, indent=4)
            f.write('\n')
        chat_history_session_state.clear()
        st.rerun()

    if action == 'old_chat':
        try:
            if not chat_history_file.is_file():
                st.error(f"No chat history found for {chat_history_file}. Starting a new chat.")
            with chat_history_file.open("r") as f:
                chat_data = json.load(f)
            if chat_data:
                chat_history_session_state.clear()  # ensure the previous history is cleared before display the new one
                chat_history_session_state.add_messages([
                    HumanMessage(content=item["content"]) if item["role"] == "user" else AIMessage(
                        content=item["content"])
                    for item in chat_data])
            else:
                chat_history_session_state.clear()
        except (FileNotFoundError, json.JSONDecodeError):
            st.error(f"Error loading chat history from {chat_history_file}. Chat history might be corrupted.")


def export_and_download_user_data(user_data_dir: str, username: str) -> tuple[BytesIO, str]:
    """Exports all files within the user data directory as a zip file and provides a download button in Streamlit."""

    if isinstance(user_data_dir, Path):
        user_data_dir = str(user_data_dir)

    # Create a zip file in memory
    zip_buffer = BytesIO()
    zip_filename = f"{username}_user_data.zip"
    with ZipFile(zip_buffer, "w") as zipf:
        for root, _, files in os.walk(user_data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, user_data_dir)
                zipf.write(file_path, arcname)

    zip_buffer.seek(0)
    return zip_buffer, zip_filename
