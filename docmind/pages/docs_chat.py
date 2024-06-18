import logging
import os
from pathlib import Path

import streamlit as st
from langchain_cohere import ChatCohere
from langchain_cohere.embeddings import CohereEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage

from docmind.llm.create_llm_chain import create_llm_with_retriever_chain
from docmind.upload_and_process_files import DocumentProcessor, GetRetriever
from docmind.utils.common import authenticate_user, get_cohere_models, setup_user_directory, setup_page, \
    setup_chat_history
from docmind.utils.helper import rmdir_recursive
from docmind.vectorstore.chromadb import create_userdb, collection_count, delete_user_collection

logger = logging.getLogger(__name__)

# Setup Streamlit page
setup_page("DocumentsChat", "📚")

# User authentication
name, authentication_status, username = authenticate_user()

# Retrieve and cache the list of Cohere models
models = get_cohere_models()

# Set up user directory
user_data_dir = setup_user_directory()

# Initialize Chat History
# TODO limit the chat history to 15 message and show warning
chat_history = StreamlitChatMessageHistory(key="docs_chat_history")
chat_history_folder = Path(user_data_dir) / "docs_chat_history"
current_chat_history_file_key = "current_docs_chat_history_file"

setup_chat_history(chat_history_folder, current_chat_history_file_key=current_chat_history_file_key)

with st.sidebar:
    with st.expander("LLM settings:", expanded=False):
        names = [name for name, model in models.items() if 'generate' in model['endpoints']]
        embedding_names = [name for name, model in models.items() if 'embed' in model['endpoints']]

        st.selectbox("Model name:", names, key="model_name", index=2)
        st.selectbox("Embedding model:", embedding_names, key="embedding_model_name", index=3)

        st.number_input("Temperature:", min_value=0.0, max_value=1.0, value=0.3, step=0.1, key="temperature")

        # Get max tokens based on selected model
        max_tokens = models[st.session_state.model_name]['context_length']
        st.number_input("Max tokens:", value=max_tokens, step=100, key="max_tokens",
                        help="The maximum number of tokens to generate.")

        st.toggle("Streaming:", "True", key="stream_output")

    with st.expander("Chat Management:", expanded=True):
        chat_files = [f for f in os.listdir(chat_history_folder) if f.endswith('.json')]
        if not chat_files:
            chat_history.clear()

        current_file_index = next((index for index, file in enumerate(chat_files) if
                                   file == st.session_state[current_chat_history_file_key].name), 0)
        st.selectbox("Load previous chat:", chat_files,
                     index=current_file_index,
                     key="selected_chat")

        if st.session_state['selected_chat']:
            setup_chat_history(chat_history_folder, action='old_chat',
                               selected_file_to_load_history=st.session_state['selected_chat'],
                               chat_history_session_state=chat_history,
                               current_chat_history_file_key=current_chat_history_file_key)

        new_chat_btn = st.button('New Chat', type="primary", use_container_width=True)
        if new_chat_btn:
            setup_chat_history(chat_history_folder, action='new_chat', chat_history_session_state=chat_history,
                               current_chat_history_file_key=current_chat_history_file_key)
            st.session_state['selected_chat'] = st.session_state[current_chat_history_file_key].name

    # Document Selection
    filter_documents = []  # initialize with empty list if the reference folder not found
    with st.container(border=True):
        reference_dir = Path(user_data_dir) / "reference"
        if reference_dir.is_dir():
            files_list = [file.name for file in reference_dir.iterdir() if file.is_file()]
            filter_documents = st.multiselect(
                "Select documents:", options=files_list,
                help="Select the documents you wish to use for obtaining answers. "
                     "if no selection is made, answers will be derived from all available documents by default."
            )

    # Document Processing
    embedding_func = CohereEmbeddings(model=st.session_state["embedding_model_name"])
    userdb = create_userdb(username, user_data_dir, embedding_func)
    DocumentProcessor(chroma_instance=userdb, user_data_dir=user_data_dir).process_documents()

    # Retriever Logic
    retriever = GetRetriever(chroma_instance=userdb, filter_criteria={
        "source": {"$in": filter_documents}} if filter_documents else {}).get_retriever()

    # Allow the user to destroy your own data
    st.button("Destroy user data", type="primary", use_container_width=True, key="destroy_data_button")
    if st.session_state["destroy_data_button"]:
        if collection_count(userdb) > 0:
            delete_user_collection(userdb, username)
            st.success("User data destroyed successfully.")
        rmdir_recursive(reference_dir)


# Display chat history
def display_chat_message(message, user_type):
    if user_type == "human":
        st.chat_message("human", avatar="👤").write(message.content)
    elif user_type == "ai":
        st.chat_message("ai", avatar="🤖").write(message.content)


for msg in chat_history.messages:
    user_type = "human" if isinstance(msg, HumanMessage) else "ai"
    display_chat_message(msg, user_type)

llm = ChatCohere(
    model_name=st.session_state.model_name,
    temperature=st.session_state.temperature,
    max_tokens=st.session_state.max_tokens,
)

if llm and retriever:
    st.toast("Ready to chat with your document.")

    answer_chain = create_llm_with_retriever_chain(llm, retriever)

    if user_input := st.chat_input(key="input"):
        output_container = st.container()
        output_container.chat_message("human", avatar="👤").write(user_input)

        # Initialize the streaming response container
        answer_container = output_container.chat_message("assistant", avatar="🤖")
        full_response = ""
        msg_placeholder = answer_container.container().markdown(full_response + "▌")

        with st.spinner("Thinking..."):
            for chunk in answer_chain.stream(
                    {"question": user_input, "chat_history": chat_history.messages},
            ):
                full_response += chunk
                msg_placeholder.markdown(full_response + "▌")

        # Finalize the response
        msg_placeholder.markdown(full_response)

        # Update chat history
        chat_history.add_user_message(user_input)
        chat_history.add_ai_message(full_response)
        setup_chat_history(chat_history_folder, action='save_chat', chat_history_session_state=chat_history,
                           current_chat_history_file_key=current_chat_history_file_key)
