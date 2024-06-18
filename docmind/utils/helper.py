import re
import shutil
from pathlib import Path
from typing import List

from langchain_core.documents import Document


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


def sanitize_file_name(path):
    # Get the directory and file name and extension
    directory, file_name, ext = path.parent, path.stem, path.suffix
    # Replace spaces and non-alphanumeric characters in the file name
    sanitized_file_name = re.sub(r"\W+", "_", file_name)
    # Combine the sanitized filename with the directory
    sanitized_path = directory / f"{sanitized_file_name}{ext}"
    return Path(sanitized_path)


def rmdir_recursive(dir_path: Path) -> bool:
    """
    Recursively deletes a directory and its contents.
    """

    if not dir_path.exists():
        return True

    if dir_path.is_dir():
        from shutil import rmtree

        rmtree(path=dir_path, ignore_errors=True)
    elif dir_path.is_file():
        dir_path.unlink(missing_ok=True)

    return True if not dir_path.exists() else False


def truncate_files_in_folder(folder_path: Path) -> None:
    """
    Iterates over the files in a given folder and truncates them to zero size.
    """

    if not folder_path.is_dir():
        raise ValueError(f"The provided path: {folder_path} is not a directory.")

    for file_path in folder_path.iterdir():
        if file_path.is_file():
            # Open the file in write mode to truncate it to zero size
            with open(file_path, 'w'):
                pass  # Opening in 'w' mode truncates the file automatically


def move_files(src_dir: Path, dest_dir: Path) -> None:
    """
    Moves files from the source directory to the destination directory.
    """

    if not src_dir.is_dir():
        raise ValueError(f"The source path: {src_dir} is not a directory.")
    if not dest_dir.exists():
        dest_dir.mkdir(parents=True)
    elif not dest_dir.is_dir():
        raise ValueError(f"The destination path: {dest_dir} is not a directory.")

    for file_path in src_dir.iterdir():
        if file_path.is_file():
            shutil.move(str(file_path), str(dest_dir / file_path.name))


def path_to_str(path: Path) -> str:
    """
    Converts a Path object to a string.
    """

    return str(path) if isinstance(path, Path) else path


def str_to_path(path: str) -> Path:
    """
    Converts a string to a Path object.
    """

    return Path(path) if isinstance(path, str) else path
