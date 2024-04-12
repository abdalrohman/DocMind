import logging
import os

import streamlit as st

logger = logging.getLogger(__name__)

api_keys_dict = {
    "Langchain API Keys"           : ["LANGCHAIN_API_KEY", "LANGCHAIN_PROJECT"],
    "Search Engines API Keys"      : [
        "GOOGLE_CSE_ID",
        "GOOGLE_SEARCH_API_KEY",
        "EXA_API_KEY",
        "TAVILY_API_KEY",
        ],
    "LLM Providers API Keys"       : [
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


def set_api_keys():
    api_keys_container = st.sidebar.container(border=True)
    st.session_state.setdefault("submit_env", False)

    api_keys_container.title("API Key Configuration")

    # Set default environment variables
    api_keys_container.checkbox("Enable Langchain Tracing", value=False, key="enable_langchain_tracing")
    os.environ["LANGCHAIN_TRACING_V2"] = f"{str(st.session_state.enable_langchain_tracing).lower()}"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["ANONYMIZED_TELEMETRY"] = "False"

    api_keys_container.markdown(
        """
        **Note:** Set only the keys for the providers you intend to use.
        """
        )

    if not st.session_state["submit_env"]:
        api_keys_container.warning(
            """
            An `.env` file was not detected. Please rename the .env.example file to .env 
            and configure your environmental variables, 
            or provide the necessary information in the sidebar.
            """
            )
        for title, api_keys in api_keys_dict.items():
            with api_keys_container.expander(f"Set your {title}"):
                for i, key in enumerate(api_keys):
                    user_key = st.text_input(
                        f"{key}",
                        type="password",
                        help=f"Set this if not set the the value in .env file for {key}.",
                        key=f"{key.lower()}_{i}",  # Store the value in st.session_state
                        )
                    if user_key:
                        os.environ[key] = user_key

    if api_keys_container.button("Submit", key="submit_env", use_container_width=True):
        st.session_state["submit_env"] = True
        st.sidebar.success("API keys submitted successfully!")


def set_specific_api_key(
        key_name
        ):
    api_keys_container = st.sidebar.container(border=True)
    user_key = api_keys_container.text_input(
        f"{key_name}",
        type="password",
        help=f"Set this if not set the the value in .env file for {key_name}.",
        key=f"{key_name.lower()}",  # Store the value in st.session_state
        )
    if user_key:
        os.environ[key_name] = user_key
        api_keys_container.success(f"{key_name} submitted successfully!")
    else:
        api_keys_container.warning(f"Please provide a {key_name} to submit.")
