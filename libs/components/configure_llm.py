# -- set llm --
import logging
import os

import streamlit as st
from langchain_core.runnables import ConfigurableField
from langchain_fireworks import ChatFireworks
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from libs.components.configure_api_keys import set_specific_api_key
from libs.settings import llm_settings as settings

logger = logging.getLogger(__name__)


def set_llm():
    chat_classes = {
        "FIREWORKS_API_KEY": [ChatFireworks, "fireworks_model", "chat_fireworks"],
        "GOOGLE_API_KEY": [ChatGoogleGenerativeAI, "gemini_model", "gemini_pro"],
        "GROQ_API_KEY": [ChatGroq, "groq_model", "groq"],
        "OPENAI_API_KEY": [ChatOpenAI, "openai_model", "openai"],
    }

    # Initialize an empty dictionary to hold the chat instances
    chat_instances = {}

    # Iterate over the chat classes
    for key, ChatClass in chat_classes.items():
        api_key = os.environ.get(key)
        if api_key:
            # Create an instance of the chat class
            chat_instance = ChatClass[0](
                model=getattr(settings, ChatClass[1]),
                temperature=settings.temperature,
                streaming=settings.streaming,
                max_tokens=settings.max_tokens,
            )
            # Add the instance to the chat_instances dictionary
            chat_instances[ChatClass[2]] = chat_instance
        else:
            logger.error(f"{key} not set")
            set_specific_api_key(key)

    # Check if any chat instances were created
    if chat_instances:
        default_model = None
        for model in ["chat_fireworks", "gemini_pro", "groq", "openai"]:
            if model in chat_instances:
                default_model = chat_instances.pop(model)
                break
        logger.info(f"Default model: {default_model}")
        logger.info(f"Chat instances: {chat_instances}")
        if not default_model:
            st.exception(
                f"Can't continue without a default model. Please provide at least one API key."
            )
            st.stop()
        return default_model.configurable_alternatives(
            ConfigurableField(id="llm"), default_key="chat_fireworks", **chat_instances
        ).with_fallbacks(list(chat_instances.values()))
    else:
        logger.error(
            "No chat instances were created. Please provide at least one API key."
        )
        st.exception(
            "No chat instances were created. Please provide at least one API key."
        )
        st.stop()
