import logging
from pathlib import Path
from typing import List, Tuple, Union

import streamlit as st
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from libs.utils import path_to_str, rmdir_recursive, str_to_path
from libs.vectorstore import ChromaDBClient, refine_docs, sanitize_file_name

logger = logging.getLogger(__name__)


class ProcessRetriever:
    def __init__(
        self,
        embeddings: Embeddings,
        chroma_dir: Union[Path, str],
        docs: List[Document] = None,
    ):
        self.embeddings = embeddings
        self.chroma_dir = path_to_str(chroma_dir)
        self.docs = docs

    def get_retriever(self) -> VectorStoreRetriever:
        sqlite3_path = str_to_path(self.chroma_dir).joinpath("chroma.sqlite3")
        sqlite3_size = sqlite3_path.stat().st_size if sqlite3_path.exists() else 0
        client = ChromaDBClient(
            path_to_chroma_db=self.chroma_dir, embeddings_function=self.embeddings
        )
        db = client.db
        collection_count = client.collection_count
        if self.docs:
            logger.info("Add documents to db")
            db.add_documents(self.docs)
        if sqlite3_path.exists() and sqlite3_size > 0:
            if collection_count < 0:
                logger.info("No existing database found")
                st.sidebar.warning(
                    "There is no collection inside database. "
                    "To initiate a chat with documents, you need to upload a file."
                )
                # st.stop()
        else:
            st.sidebar.warning(
                "Not Initialized vectorstore yet! "
                "To initiate a chat with documents, you need to upload a file."
            )
        if db and collection_count > 0:
            return db.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 30, "lambda_mult": 0, "fetch_k": 50},
            )


class DocumentProcessor:
    supported_files = (
        "csv",
        "docx",
        "epub",
        "ipynb",
        "json",
        "md",
        "pdf",
        "ppt",
        "pptx",
        "txt",
    )

    def __init__(
        self,
        embeddings: Embeddings,
        chroma_dir: Union[Path, str],
        temp_dir: Union[Path, str],
    ):
        logger.info("Initializing DocumentProcessor")
        self.embeddings = embeddings
        self.chroma_dir = path_to_str(chroma_dir)
        self.temp_dir = str_to_path(temp_dir)
        self.file_upload_container = st.sidebar.container(border=True)
        self._initialize_session_state()
        if st.session_state.show_warning == "false":
            self.file_upload_container.empty()

    def _initialize_session_state(self):
        st.session_state.setdefault("uploaded_docs", [])
        st.session_state.setdefault("file_uploader_state", 0)
        st.session_state.setdefault("activate_uploader", True)
        st.session_state.setdefault("retriever", None)
        st.session_state.setdefault("show_warning", True)

    def process_documents(self):
        if st.session_state["activate_uploader"]:
            uploaded_files = self.upload_documents()
            if uploaded_files:
                st.session_state.uploaded_docs.extend(uploaded_files)

        if st.session_state["uploaded_docs"]:
            self.file_upload_container.success("Success uploaded files!", icon="✅")
            st.sidebar.caption(
                "NOTE: Press on the process button to be able to chat with your documents."
            )
            if st.sidebar.button(
                "Process docs",
                key="process_btn",
                type="primary",
                use_container_width=True,
            ):
                self._process_uploaded_docs()
        else:
            if st.session_state.show_warning:
                self.file_upload_container.warning(
                    "There is no document uploaded. "
                    "If you have already processed files before ignore this message.",
                    icon="⚠️",
                )

    def _process_uploaded_docs(self):
        logger.info("Processing files")
        with self.file_upload_container:
            with st.spinner("Processing documents..."):
                self.temp_dir.mkdir(exist_ok=True, parents=True)
                self._save_files_to_disk()
                docs, processed_files = self._process_docs()
                retriever = ProcessRetriever(
                    embeddings=self.embeddings, chroma_dir=self.chroma_dir, docs=docs
                ).get_retriever()
                st.session_state.retriever = retriever
                self._log_and_display_results(docs, processed_files)
                self._cleanup_if_all_files_processed(processed_files)

    def _log_and_display_results(
        self, docs: List[Document], processed_files: List[Path]
    ):
        logger.info(f"Saving {len(processed_files)} processed files to {self.temp_dir}")
        logger.info(f"Processed files {len(st.session_state['uploaded_docs'])}")
        if len(docs) > 0:
            self.file_upload_container.success("Success processed file!", icon="✅")

    def _cleanup_if_all_files_processed(self, processed_files: List[Path]):
        if len(processed_files) == len(st.session_state["uploaded_docs"]):
            logger.info("Clean up temp directory...")
            rmdir_recursive(self.temp_dir)
            st.session_state.uploaded_docs.clear()
            st.session_state.file_uploader_state += 1
            st.session_state["activate_uploader"] = True
            st.rerun()

    def _save_files_to_disk(self) -> None:
        logger.info("Saving files to disk")
        for file in st.session_state["uploaded_docs"]:
            file_path = sanitize_file_name(self.temp_dir / file.name)
            if not file_path.exists():
                with open(file_path, mode="wb") as tmp_file:
                    tmp_file.write(file.read())

    def _process_docs(self) -> Tuple[List[Document], List[Path]]:
        docs = []
        processed_files = []
        for path in self.temp_dir.glob("*.pdf"):
            logger.info(f"Processing {path}")
            reader = PyMuPDFLoader(str(path))
            pdf_documents = reader.load()
            docs.extend(pdf_documents)
            processed_files.append(path)
        return refine_docs(docs), processed_files

    def upload_documents(self):
        uploaded_docs = self.file_upload_container.file_uploader(
            "Upload your documents",
            type=self.supported_files,
            accept_multiple_files=True,
            key=st.session_state.file_uploader_state,
            label_visibility="visible",
        )
        if uploaded_docs:
            logger.info("Uploading documents")
            st.session_state["activate_uploader"] = False
            return uploaded_docs
