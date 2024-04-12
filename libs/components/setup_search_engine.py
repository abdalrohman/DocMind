import os

import streamlit as st

from libs.constant import ERROR_EMOJI, SEARCH_ENGINES
from libs.search_tools import Search


def set_search_engine_tool() -> Search:
    with st.sidebar.expander("Search engine settings:", expanded=False):
        search_engine_choice = st.sidebar.selectbox(
            "Search Engine",
            options=["DDG", "Google", "Exa", "Tavily"],
        )
        max_num_results = st.sidebar.number_input(
            "Max number of results for search engine",
            value=10,
            step=1,
        )

        if search_engine_choice.lower() in SEARCH_ENGINES:
            for api_key, input_label in zip(
                SEARCH_ENGINES[search_engine_choice.lower()]["api_keys"],
                SEARCH_ENGINES[search_engine_choice.lower()]["input_labels"],
            ):
                if os.environ.get(api_key) is None:
                    key_input = st.sidebar.text_input(input_label, type="password")
                    os.environ[api_key] = key_input
                    st.sidebar.error(
                        f"{ERROR_EMOJI} Please enter an {search_engine_choice} API Key"
                    )
                    st.stop()

    return Search(search_engine=search_engine_choice, max_num_results=max_num_results)
