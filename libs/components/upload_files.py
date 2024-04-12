from pathlib import Path
from typing import Union

import streamlit as st
from langchain_community.document_loaders.pdf import PyMuPDFLoader
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from streamlit.runtime.uploaded_file_manager import UploadedFile

from libs.vectorstore import ChromaDBClient, refine_docs, sanitize_file_name


def upload_documents(key: str = "uploaded_files") -> Union[list[UploadedFile], None]:
    uploaded_files = st.sidebar.file_uploader(
        "Upload your documents",
        type=["pdf"],
        accept_multiple_files=True,
        key=key,
        label_visibility="visible",
    )
    if uploaded_files:
        st.sidebar.success("Success uploaded file!", icon="✅")
        return uploaded_files
    else:
        st.sidebar.warning("No document uploaded", icon="⚠️")
        return None


@st.cache_resource(
    experimental_allow_widgets=True, show_spinner="ℹ️ Process files data..."
)
def process_files(
    temp_dir: Union[Path, str],
    chroma_dir: Union[Path, str],
    _embeddings: Embeddings,
    uploaded_files: list[UploadedFile],
) -> VectorStoreRetriever:
    if isinstance(temp_dir, str):
        temp_dir = Path(temp_dir)
    if isinstance(chroma_dir, str):
        chroma_dir = Path(chroma_dir)
    docs = []

    # create temp dir
    temp_dir.mkdir(exist_ok=True, parents=True)
    for file in uploaded_files:
        file_path = temp_dir / file.name
        file_path = sanitize_file_name(file_path)
        if not file_path.exists():
            with open(file_path, mode="wb") as tmp_file:
                tmp_file.write(file.read())

    for path in temp_dir.glob("*.pdf"):
        reader = PyMuPDFLoader(str(path))
        pdf_documents = reader.load()
        docs.extend(pdf_documents)
    docs = refine_docs(docs)
    client = ChromaDBClient(
        path_to_chroma_db=str(chroma_dir), embeddings_function=_embeddings
    )
    db = client.db
    db.add_documents(docs)
    return db.as_retriever(
        search_type="mmr", search_kwargs={"k": 30, "lambda_mult": 0, "fetch_k": 50}
    )


def get_existing_db(
    embeddings: Embeddings,
    chroma_dir: Path,
) -> Union[VectorStoreRetriever, None]:
    if not chroma_dir.exists():
        st.sidebar.error("No existing database found!")
        return None
    else:
        st.sidebar.info("Using existing database to retrieve information!")
    client = ChromaDBClient(
        path_to_chroma_db=str(chroma_dir), embeddings_function=embeddings
    )
    db = client.db
    return db.as_retriever(
        search_type="mmr", search_kwargs={"k": 30, "lambda_mult": 0, "fetch_k": 50}
    )

    # if upload_documents():
    #     retriever = process_files()
    # else:
    #     if chroma_dir.exists():
    #         st.sidebar.info(
    #         "Using existing database to retrieve information!"
    #         )
    #         client = ChromaDBClient(
    #         path_to_chroma_db=str(chroma_dir),
    #         embeddings_function=embeddings
    #         )
    #         db = client.db
    #         retriever = db.as_retriever(
    #         search_type="mmr",
    #         search_kwargs={'k': 30, 'lambda_mult': 0, 'fetch_k': 50}
    #         )
