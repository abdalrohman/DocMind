import logging
import os
from pathlib import Path
from typing import Union

from chromadb.config import Settings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

# https://docs.trychroma.com/guides#creating,-inspecting,-and-deleting-collections:~:text=Valid%20options%20for%20hnsw%3Aspace%20are%20%22l2%22%2C%20%22ip%2C%20%22or%20%22cosine%22.%20The%20default%20is%20%22l2%22%20which%20is%20the%20squared%20L2%20norm.
DISTANCE_METRIC = ["l2", "ip", "cosine"]
PERSIST = True


def create_userdb(username: str, user_data_dir: Union[Path, str], embedding_func: Embeddings,
                  distance_metric='l2') -> Chroma:
    logger.info("Initializing ChromaDB...")
    if distance_metric not in DISTANCE_METRIC:
        raise ValueError(
            f"distance_metric should be one of {DISTANCE_METRIC}"
        )

    if isinstance(user_data_dir, Path):
        user_data_dir = str(user_data_dir)

    if not Path(user_data_dir).is_dir():
        raise ValueError(
            f"{user_data_dir} is not a valid directory"
        )

    settings = Settings(
        allow_reset=True,
        is_persistent=PERSIST,
        persist_directory=os.path.join(user_data_dir, 'chroma'),
    )

    return Chroma(
        collection_name=f'{username}_collection',
        embedding_function=embedding_func,
        persist_directory=os.path.join(user_data_dir, 'chroma'),
        collection_metadata={"hnsw:space": distance_metric},
        client_settings=settings,
    )


# noinspection PyProtectedMember
def collection_count(chroma_instance: Chroma):
    """Return the number collection."""
    return chroma_instance._client.count_collections()


# noinspection PyProtectedMember
def delete_user_collection(chroma_instance: Chroma, username: str):
    """Delete a collection with the given username."""
    logger.info(f"Deleting collection {username}_collection...")
    chroma_instance._client.delete_collection(name=f'{username}_collection')
