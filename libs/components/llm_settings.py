import streamlit as st
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory

from libs.constant import (
    FIREWORKS_MODEL,
    GEMINI_MODEL,
    GROQ_MODEL,
    OPENAI_MODEL,
    SUPPORTED_PROVIDER,
)
from libs.settings import llm_settings as settings


def set_provider():
    return st.selectbox("Provider:", options=SUPPORTED_PROVIDER)


def set_safety_settings():
    if "safety_settings" not in st.session_state:
        st.session_state["safety_settings"] = {
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        }


def set_model_settings():
    settings.openai_model = st.selectbox("OpenAI model:", options=OPENAI_MODEL)

    settings.gemini_model = st.selectbox("Gemini model:", options=GEMINI_MODEL)
    settings.groq_model = st.selectbox("Groq model", options=GROQ_MODEL)
    settings.fireworks_model = st.selectbox("Fireworks model:", options=FIREWORKS_MODEL)

    settings.fireworks_model = st.selectbox(
        "Fireworks model:",
        options=[
            "accounts/fireworks/models/mixtral-8x7b-instruct",
            "accounts/fireworks/models/yi-34b-200k-capybara",
            "accounts/fireworks/models/nous-hermes-2-mixtral-8x7b-dpo-fp8",
            "accounts/fireworks/models/mixtral-8x7b-instruct-hf",
        ],
    )


def set_llm_settings_sidebar():
    with st.sidebar:
        provider = set_provider()
        set_safety_settings()
        with st.expander("Set LLMs settings:", expanded=False):
            set_model_settings()
            # TODO add support max temp to 2 for openai
            settings.temperature = st.number_input(
                "Temperature:",
                value=settings.temperature,
                min_value=0.0,
                max_value=1.0,
                help="The sampling temperature to use.",
            )
            settings.max_tokens = st.number_input(
                "Max tokens:",
                value=settings.max_tokens,
                help="The maximum number of tokens to generate.",
            )
            settings.streaming = st.checkbox(
                "Streaming:",
                value=settings.streaming,
                help="Whether to stream the results or not.",
            )
            settings.max_retries = st.number_input(
                "Max retries:",
                value=settings.max_retries,
                help="The maximum number of retries to make when generating.",
            )
            # TODO add additional settings for openai
            st.header("safety settings for gemini:")
            for category, threshold in st.session_state["safety_settings"].items():
                threshold_value = st.slider(
                    f"{category.name} Threshold",
                    min_value=HarmBlockThreshold.HARM_BLOCK_THRESHOLD_UNSPECIFIED.value,
                    max_value=HarmBlockThreshold.BLOCK_NONE.value,
                    value=threshold,
                )
                st.session_state["safety_settings"][category] = HarmBlockThreshold(
                    threshold_value
                )
    return provider
