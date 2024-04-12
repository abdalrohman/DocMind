import logging
import os
from typing import List

import streamlit as st
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

SUPPORTED_EMBEDDINGS: List[str] = [
    "VoyageAI",
    "Google",
    "OpenAI",
    "Cloudflare",
    "TogetherAI",
    "HuggingFaceBgeEmbeddings",
    "OllamaEmbeddings",
]


@st.cache_resource
def hug_embedding():
    from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings

    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-base-en",
        model_kwargs={"device": "cpu", "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True},
    )


@st.cache_resource
def ollama_embedding():
    from langchain_community.embeddings.ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model="nomic-embed-text",
    )


def choose_embed_function(
    embd_func_name: str,
) -> Embeddings:
    if embd_func_name not in SUPPORTED_EMBEDDINGS:
        logger.error(
            f"Embeddings {embd_func_name} not supported. "
            f"Supported options are: {SUPPORTED_EMBEDDINGS}"
        )
        raise ValueError(f"Embeddings {embd_func_name} not supported")

    logger.info(f"Using {embd_func_name} embeddings")

    if embd_func_name == "OpenAI":
        if os.environ.get("OPENAI_API_KEY") is None:
            logger.error("OPENAI_API_KEY not set")
            raise ValueError("OPENAI_API_KEY not set")
        from langchain_openai.embeddings import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model="text-embedding-3-small",
        )

    if embd_func_name == "Cloudflare":
        if os.environ.get("CLOUDFLARE_API_KEY") is None:
            logger.error("CLOUDFLARE_API_KEY not set")
            raise ValueError("CLOUDFLARE_API_KEY not set")
        from langchain_community.embeddings.cloudflare_workersai import (
            CloudflareWorkersAIEmbeddings,
        )

        return CloudflareWorkersAIEmbeddings(
            account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
            api_token=os.getenv("CLOUDFLARE_API_KEY"),
            model_name="@cf/baai/bge-large-en-v1.5",
        )

    if embd_func_name == "VoyageAI":
        if os.environ.get("VOYAGE_API_KEY") is None:
            logger.error("VOYAGE_API_KEY not set")
            raise ValueError("VOYAGE_API_KEY not set")
        from langchain_voyageai import VoyageAIEmbeddings

        return VoyageAIEmbeddings(
            model="voyage-large-2",
        )

    if embd_func_name == "TogetherAI":
        if os.environ.get("TOGETHER_API_KEY") is None:
            logger.error("TOGETHER_API_KEY not set")
            raise ValueError("TOGETHER_API_KEY not set")
        from langchain_together.embeddings import TogetherEmbeddings

        return TogetherEmbeddings(
            model="togethercomputer/m2-bert-80M-32k-retrieval",
        )

    if embd_func_name == "Google":
        if os.environ.get("GOOGLE_API_KEY") is None:
            logger.error("GOOGLE_API_KEY not set")
            raise ValueError("GOOGLE_API_KEY not set")
        from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
        )

    if embd_func_name == "HuggingFaceBgeEmbeddings":
        hug_embedding()

    if embd_func_name == "OllamaEmbeddings":
        ollama_embedding()
