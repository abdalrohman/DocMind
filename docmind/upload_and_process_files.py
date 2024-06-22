import logging
from pathlib import Path
from typing import List, Union, Tuple, Optional, Dict

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from docmind.utils.helper import sanitize_file_name, refine_docs, truncate_files_in_folder, move_files, rmdir_recursive

logger = logging.getLogger(__name__)


class GetRetriever:
    def __init__(
            self,
            chroma_instance: Chroma,
            documents: Optional[List[Document]] = None,
            filter_criteria: Optional[Dict] = None
    ):
        """
        Example:
            filter_criteria = {
                 "source": {
                     "$in": ['A.pdf', 'B.pdf']
                 }
             }
        """
        self.chroma_instance = chroma_instance
        self.documents = documents
        self.filter_criteria = filter_criteria

    def get_retriever(self) -> VectorStoreRetriever:
        if self.documents:
            logger.info("Adding documents to the database.")
            self.chroma_instance.add_documents(self.documents)

        search_kwargs = {
            "k": 50,
            "lambda_mult": 0,
            "fetch_k": 50,
        }

        # Use this to filter based on specific book
        if self.filter_criteria:
            search_kwargs['filter'] = self.filter_criteria

        return self.chroma_instance.as_retriever(
            search_type="mmr",
            search_kwargs=search_kwargs,
        )


class DocumentProcessor:
    supported_files = (
        "pdf",
    )

    def __init__(
            self,
            chroma_instance: Chroma,
            user_data_dir: Union[Path, str],
    ):
        logger.info("Initializing DocumentProcessor...")

        self.chroma_instance = chroma_instance

        if isinstance(user_data_dir, str):
            self.user_data_dir = Path(user_data_dir)
        else:
            self.user_data_dir = user_data_dir

        if not Path(self.user_data_dir).is_dir():
            raise ValueError(
                f"{self.user_data_dir} is not a valid directory"
            )

        self.temp_dir = self.user_data_dir / "temp"
        self.reference_dir = self.user_data_dir / "reference"
        self.temp_dir.mkdir(exist_ok=True, parents=True)

        self.file_upload_container = st.sidebar.expander("Documents Processing", expanded=True)
        self._initialize_session_state()

    @staticmethod
    def _initialize_session_state():
        st.session_state.setdefault("uploaded_docs", [])
        st.session_state.setdefault("file_uploader_state", 0)
        st.session_state.setdefault("activate_uploader", True)

    def process_documents(self):
        if st.session_state["activate_uploader"]:
            uploaded_files = self.upload_documents()
            if uploaded_files:
                st.session_state.uploaded_docs.extend(uploaded_files)

        if st.session_state["uploaded_docs"]:
            self.file_upload_container.success("Success uploaded files!", icon="✅")
            self.file_upload_container.caption(
                "NOTE: Press on the process button to be able to chat with your documents."
            )
            if self.file_upload_container.button(
                    "Process docs",
                    key="process_btn",
                    type="primary",
                    use_container_width=True,
            ):
                self._process_uploaded_docs()

    def _process_uploaded_docs(self):
        """Process uploaded documents:
         - Save files to disk.
         - Process PDF files.
         - Add processed documents to the vector store.
         - Display success message and update session state.
        """
        with self.file_upload_container:
            with st.spinner("Processing documents..."):
                self._save_files_to_disk()
                docs, processed_files = self._process_docs()
                GetRetriever(
                    chroma_instance=self.chroma_instance, documents=docs
                ).get_retriever()
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
        """Clean up temporary directory:
         - Move files to the reference directory.
         - Truncate files in the temp directory.
         - Clear session state variables.
        """
        if len(processed_files) == len(st.session_state["uploaded_docs"]):
            logger.info("Cleaning up temp directory...")
            # Instead of delete the files just make it's size zero
            # to use the files as reference for the sources files in database.

            # Truncate files in the temp directory to save space
            truncate_files_in_folder(self.temp_dir)

            # Move processed files to the reference directory
            move_files(self.temp_dir, self.reference_dir)
            rmdir_recursive(self.temp_dir)

            # Clear session state variables
            st.session_state.uploaded_docs.clear()
            st.session_state.file_uploader_state += 1
            st.session_state["activate_uploader"] = True
            st.rerun()

    def _save_files_to_disk(self) -> None:
        """Save uploaded files to the temporary directory."""
        for file in st.session_state["uploaded_docs"]:
            file_path = sanitize_file_name(self.temp_dir / file.name)
            if not file_path.exists():
                with open(file_path, mode="wb") as tmp_file:
                    tmp_file.write(file.read())

    def _process_docs(self) -> Tuple[List[Document], List[Path]]:
        """Process PDF files in the temporary directory."""
        docs = []
        processed_files = []
        for path in self.temp_dir.glob("*.pdf"):
            logger.info(f"Processing PDF file: {path}")

            reader = PyMuPDFLoader(str(path))
            pdf_documents = reader.load()

            # update the metadata to filter the RAG based on the document name
            for doc in pdf_documents:
                doc.metadata["source"] = path.name  # use the file name instead of full path
            # create new docs list to hold the docs update
            docs.extend(pdf_documents)
            processed_files.append(path)
        return refine_docs(docs), processed_files

    def upload_documents(self):
        """Upload documents using the Streamlit file uploader."""
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
