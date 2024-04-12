import streamlit as st

from libs.choose_embeddings import SUPPORTED_EMBEDDINGS, choose_embed_function


def set_embeddings():
    with st.sidebar:
        embeddings_choice = st.selectbox("Embeddings", options=SUPPORTED_EMBEDDINGS)

        def choose_embeddings():
            return choose_embed_function(
                embd_func_name=embeddings_choice,
            )

    return choose_embeddings()
