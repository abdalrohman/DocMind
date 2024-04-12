import logging
from pathlib import Path
from typing import Union

import streamlit as st
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from streamlit.runtime.uploaded_file_manager import UploadedFile

from libs.utils import rmdir_recursive
from libs.vectorstore import ChromaDBClient, refine_docs, sanitize_file_name

logger = logging.getLogger(__name__)


class DocumentProcessor:
    supported_files = ("csv", "docx", "epub", "ipynb", "json", "md", "pdf", "ppt", "pptx", "txt")

    def __init__(
            self,
            embeddings: Embeddings,
            chroma_dir: Union[Path, str],
            key: str = "uploaded_files"
            ):
        logger.info("Initializing DocumentProcessor")
        self.embeddings = embeddings
        self.chroma_dir = str(chroma_dir) if isinstance(chroma_dir, Path) else chroma_dir
        self.file_upload_container = st.sidebar.container(border=True)
        self.key = key

    def _get_db(
            self,
            docs: Union[list[Document], None] = None
            ) -> Union[VectorStoreRetriever, None]:
        logger.info("Getting database")
        client = ChromaDBClient(path_to_chroma_db=self.chroma_dir, embeddings_function=self.embeddings)
        db = client.db
        if docs:
            logger.info("Add documents to db")
            db.add_documents(docs)
        return db.as_retriever(search_type="mmr", search_kwargs={"k": 30, "lambda_mult": 0, "fetch_k": 50})

    def upload_documents(
            self,
            ) -> Union[list[UploadedFile], None]:
        logger.info("Uploading documents")
        uploaded_files = self.file_upload_container.file_uploader(
            "Upload your documents",
            type=self.supported_files,
            accept_multiple_files=True,
            key=self.key,
            label_visibility="visible",
            )
        if len(st.session_state[self.key]) > 0:
            self.file_upload_container.success("Success uploaded file!", icon="✅")
            # add process button
            st.sidebar.button("Process docs", key="process_btn", type="primary", use_container_width=True)
            logger.info(f"Uploaded files: {uploaded_files}")
            return uploaded_files
        else:
            self.file_upload_container.warning("No document uploaded", icon="⚠️")
            return None

    def process_files(
            self,
            temp_dir: Union[Path, str],
            ) -> Union[VectorStoreRetriever, None]:
        logger.info("Processing files")
        if st.session_state["process_btn"] and len(st.session_state[self.key]) > 0:
            with self.file_upload_container:
                with st.spinner("Processing documents..."):
                    temp_dir = Path(temp_dir) if isinstance(temp_dir, str) else temp_dir
                    temp_dir.mkdir(exist_ok=True, parents=True)
                    docs = []
                    processed_files = []

                    # save uploaded files to temp directory
                    for file in st.session_state[self.key]:
                        file_path = temp_dir / file.name
                        file_path = sanitize_file_name(file_path)
                        if not file_path.exists():
                            with open(file_path, mode="wb") as tmp_file:
                                tmp_file.write(file.read())
                    for path in temp_dir.glob("*.pdf"):
                        reader = PyMuPDFLoader(str(path))
                        pdf_documents = reader.load()
                        docs.extend(pdf_documents)
                        processed_files.append(path)
                    docs = refine_docs(docs)

                    # Clean up temp directory and clear uploaded files list
                    if len(processed_files) == len(st.session_state[self.key]):
                        logger.info("Clean up temp directory...")
                        st.session_state[self.key].clear()
                        rmdir_recursive(temp_dir)
                    if len(docs) > 0:
                        try:
                            if self._get_db(docs):
                                self.file_upload_container.success("Success processed file!", icon="✅")
                            return self._get_db()
                        except Exception as e:
                            logger.error(e)
                            self.file_upload_container.error(
                                "Consider specifying a different embedding provider, "
                                "or use the same provider that was used to create the previous database. "
                                f"{e}"
                                )
                            return None
                    else:
                        logger.info("Somthing wrong with uploaded files")
                        self.file_upload_container.error("Something wrong with uploaded files!")
                        return None
        else:
            return None

    def get_existing_db(
            self
            ) -> Union[VectorStoreRetriever]:
        if Path(self.chroma_dir).joinpath("chroma.sqlite3").exists():
            client = ChromaDBClient(path_to_chroma_db=self.chroma_dir, embeddings_function=self.embeddings)
            collection_count = client.collection_count
            if collection_count > 0:
                self.file_upload_container.info("Using existing database to retrieve information!")
                logger.info("Getting existing database")
                return self._get_db()
            else:
                logger.info("No existing database found")
                self.file_upload_container.error(
                    "There is no database currently available. "
                    "To initiate a chat with documents, you need to upload a file."
                    )
                st.stop()
