import logging
import os
from typing import List

from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

supported_embeddings: List[str] = [
    "OpenAI",
    "Cloudflare",
    "VoyageAI",
    "TogetherAI",
    "Google",
]


def select_embeddings(embeddings_name: str) -> Embeddings:
    if embeddings_name not in supported_embeddings:
        logger.error(
            f"Embeddings {embeddings_name} not supported. "
            f"Supported options are: {supported_embeddings}"
        )
        raise ValueError(f"Embeddings {embeddings_name} not supported")

    if embeddings_name.lower() == "OpenAI".lower():
        from langchain_openai.embeddings import OpenAIEmbeddings

        return OpenAIEmbeddings()

    elif embeddings_name.lower() == "Cloudflare".lower():
        account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
        api_token = os.getenv("CLOUDFLARE_API_KEY")
        if account_id and api_token:
            from langchain_community.embeddings.cloudflare_workersai import (
                CloudflareWorkersAIEmbeddings,
            )

            return CloudflareWorkersAIEmbeddings(
                account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
                api_token=os.getenv("CLOUDFLARE_API_KEY"),
                model_name="@cf/baai/bge-large-en-v1.5",
            )
        else:
            logger.error("Cloudflare account id and api token not provided")
            raise ValueError("Cloudflare account id and api token not provided")

    elif embeddings_name.lower() == "VoyageAI".lower():
        from langchain_voyageai import VoyageAIEmbeddings

        return VoyageAIEmbeddings(model="voyage-large-2")
    elif embeddings_name.lower() == "TogetherAI".lower():
        # # voyage-large-2     16000
        # # voyage-code-2      16000
        # # voyage-2                  4000
        # # voyage-lite-02-instruct   4000
        from langchain_together.embeddings import TogetherEmbeddings

        return TogetherEmbeddings(model="togethercomputer/m2-bert-80M-32k-retrieval")

    elif embeddings_name.lower() == "Google".lower():
        from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")
