import os
from pathlib import Path

import streamlit as st
from langchain_cohere import ChatCohere
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage

from docmind.pages.docs_chat import setup_chat_history
from docmind.utils.common import authenticate_user, get_cohere_models, setup_user_directory, setup_page

# Setup Streamlit page
setup_page("ChatBot", "💬")

# User authentication
name, authentication_status, username = authenticate_user()

# Retrieve and cache the list of Cohere models
models = get_cohere_models()

# Set up user directory
setup_user_directory()

user_data_dir = Path(st.session_state['user_dir'])

# Initialize Chat History
# TODO limit the chat history to 15 message and show warning
chat_history = StreamlitChatMessageHistory(key="chat_history")
chat_history_folder = user_data_dir / "chat_history"
current_chat_history_file_key = "current_chat_history_file"

setup_chat_history(chat_history_folder, current_chat_history_file_key=current_chat_history_file_key)

with st.sidebar:
    with st.expander("LLM settings:", expanded=False):
        names = [name for name, model in models.items() if 'generate' in model['endpoints']]

        st.selectbox("Model name:", names, key="model_name", index=2)

        st.number_input("Temperature:", min_value=0.0, max_value=1.0, value=0.3, step=0.1, key="temperature")

        # Get max tokens based on selected model
        max_tokens = models[st.session_state.model_name]['context_length']
        st.number_input("Max tokens:", value=max_tokens, step=100, key="max_tokens",
                        help="The maximum number of tokens to generate.")

        st.toggle("Streaming:", "True", key="stream_output")

        # cohere internet is enabled by default
        is_connector_enable = st.toggle("🔍 Connectors:", value="True",
                                        help="When specified, the model's reply will be enriched with information"
                                             " found by quering each of the connectors (RAG).",
                                        key="is_connector_enable")
        connectors = None
        prompt_truncate = None
        is_search_enable = False

        if is_connector_enable:
            is_search_enable = st.checkbox("Web Search:", value=True, key="is_search_enable")
            if is_search_enable:
                connectors = [{"id": "web-search"}]
                st.text_input("Search site (Optional):", value="", key="search_site",
                              placeholder="e.g. www.langchain.com",
                              help="Search site (Only support one site!)"
                              )
                if st.session_state.search_site:
                    connectors = [{"id": "web-search", "options": {"site": st.session_state.search_site}}]
                prompt_truncate = st.selectbox("PROMPT TRUNCATION", ["AUTO", "OFF"], key="prompt_truncate",
                                               help="Toggle whether some chat messages and documents will be "
                                                    "automatically dropped to avoid going over the token limit."
                                               )

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


def generate_cited_content_cohere_llm(response):
    """Generates formatted content with accurate in-text citations and a sources list."""

    content = response.content
    citations = response.additional_kwargs['citations']
    documents = response.additional_kwargs['documents']

    doc_id_to_markdown = {doc['id']: f"[{doc['title']}]({doc['url']})" for doc in documents}
    doc_id_to_index = {}  # Map document ID to its citation index
    citation_count = 1  # Track the current citation index

    cited_content = []
    last_cite_end = 0

    for cite in citations:
        cited_content.append(content[last_cite_end: cite.start - 1])

        # Assign citation indices based on first appearance
        indices = []
        for doc_id in cite.document_ids:
            if doc_id not in doc_id_to_index:
                doc_id_to_index[doc_id] = citation_count
                citation_count += 1
            indices.append(str(doc_id_to_index[doc_id]))

        cited_content.append(content[cite.start - 1: cite.end] + f"[{', '.join(indices)}]")
        last_cite_end = cite.end

    cited_content.append(content[last_cite_end:])

    # Create the Sources section with correct citation indices
    sources = [
        f"{index}. {doc_id_to_markdown[doc_id]}" for doc_id, index in doc_id_to_index.items()
    ]

    return "".join(cited_content) + "\n\nSources:\n" + "\n".join(sources)


# Chat input with streaming logic
# TODO display the images from search result
if user_input := st.chat_input(key="input"):
    output_container = st.container()
    output_container.chat_message("human", avatar="👤").write(user_input)

    # Initialize the streaming response container
    answer_container = output_container.chat_message("assistant", avatar="🤖")
    full_response = ""
    msg_placeholder = answer_container.container().markdown(full_response + "▌")

    chunks = []

    # Stream the response from the LLM
    with st.spinner("Thinking..."):
        for chunk in llm.stream(
                user_input,
                prompt_truncation=prompt_truncate,
                connectors=connectors
        ):
            if is_search_enable:
                chunks.append(chunk)
            full_response += chunk.content
            msg_placeholder.markdown(full_response + "▌")

    if is_search_enable:
        content = ""
        additional_kwargs = {}

        for i in chunks:
            content += i.content
            if not additional_kwargs:
                additional_kwargs = i.additional_kwargs

        full_response = AIMessage(content=content, additional_kwargs=additional_kwargs)

    # Finalize the response
    if is_search_enable:
        full_response = generate_cited_content_cohere_llm(full_response)
        msg_placeholder.markdown(full_response)
    else:
        msg_placeholder.markdown(full_response)

    # Update chat history
    chat_history.add_user_message(user_input)
    chat_history.add_ai_message(full_response)
    setup_chat_history(chat_history_folder, action='save_chat', chat_history_session_state=chat_history,
                       current_chat_history_file_key=current_chat_history_file_key)
