import logging
import re
from pathlib import Path
from typing import List

import chromadb
from chromadb import Settings
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class ChromaDBClient:
    VALID_TYPES = ["persist", "in_memory"]
    VALID_DISTANCE_METRIC = ["l2", "ip", "cosine"]

    def __init__(
            self,
            embeddings_function: Embeddings = None,
            collection_name: str = "abdulrahman",
            chroma_store_type: str = "persist",
            allow_reset: bool = True,
            reset: bool = False,
            delete_collection: bool = False,
            distance_metric: str = "l2",
            path_to_chroma_db: str = None,
            ):
        logger.info("Initializing ChromaDBClient...")
        self.validate_input(chroma_store_type, distance_metric)

        self.collection_name = collection_name
        self.chroma_store_type = chroma_store_type
        self.allow_reset = allow_reset
        self.distance_metric = distance_metric
        self.reset = reset
        self.delete_collection = delete_collection
        self.embeddings_function = embeddings_function
        self.path_to_chroma_db = path_to_chroma_db

        self.client = self.create_client(self.path_to_chroma_db)

        if self.reset:
            logger.info("Resetting client...")
            self.client.reset()

        if self.delete_collection:
            logger.info(f"Deleting collection {collection_name}...")
            self.client.delete_collection(name=collection_name)

        self.collection = self.get_or_create_collection()
        self.db = self.create_chroma_db(self.path_to_chroma_db)
        self.collection_count = self.collection.count()
        logger.info("ChromaDBClient initialized.")

    def validate_input(
            self,
            chroma_store_type,
            distance_metric
            ):
        logger.info("Validating input...")
        if chroma_store_type not in self.VALID_TYPES:
            raise ValueError(f"chroma_store_type should be one of {self.VALID_TYPES}")

        if distance_metric not in self.VALID_DISTANCE_METRIC:
            raise ValueError(
                f"distance_metric should be one of {self.VALID_DISTANCE_METRIC}"
                )

    def create_client(
            self,
            chroma_path
            ):
        logger.info(f"Creating {self.chroma_store_type} client in {chroma_path}...")
        client_class = chromadb.PersistentClient if self.chroma_store_type == "persist" else chromadb.Client
        return client_class(
            path=chroma_path,
            settings=Settings(allow_reset=self.allow_reset),
            )

    def get_or_create_collection(
            self
            ):
        logger.info(f"Getting or creating collection {self.collection_name}...")
        return self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": self.distance_metric}
            )

    def create_chroma_db(
            self,
            chroma_path
            ):
        logger.info(f"Creating Langchain instance of Chroma db in {chroma_path}...")
        return Chroma(
            client=self.client,
            persist_directory=chroma_path,
            collection_name=self.collection_name,
            embedding_function=self.embeddings_function,
            )


def refine_docs(
        docs: List[Document],
        escape_parts: List[str] = None,
        ) -> List[Document]:
    """Remove any empty string from document or add escape parts
    to remove them from the docs.
    this function aim to not produce error when add empty string
    to the embedding function.
    """
    if escape_parts:
        new_docs = [
            doc
            for doc in docs
            if not any(part in doc.page_content for part in escape_parts)
            ]
        new_docs = [doc for doc in new_docs if doc.page_content != ""]
    else:
        new_docs = [doc for doc in docs if doc.page_content != ""]
    return new_docs


def sanitize_file_name(
        path
        ):
    # Get the directory and file name and extension
    directory, file_name, ext = path.parent, path.stem, path.suffix
    # Replace spaces and non-alphanumeric characters in the file name
    sanitized_file_name = re.sub(r"\W+", "_", file_name)
    # Combine the sanitized filename with the directory
    sanitized_path = directory / f"{sanitized_file_name}{ext}"
    return Path(sanitized_path)
