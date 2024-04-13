from pathlib import Path
from typing import Union


def rmdir_recursive(
        dir_path: Path
        ) -> bool:
    """Deletes dir_path recursively, including all files and dirs in that directory
    Returns True if dir deleted successfully.
    """

    if not dir_path.exists():
        return True

    if dir_path.is_dir():
        from shutil import rmtree

        rmtree(path=dir_path, ignore_errors=True)
    elif dir_path.is_file():
        dir_path.unlink(missing_ok=True)

    return True if not dir_path.exists() else False


def path_to_str(
        path: Union[Path, str]
        ) -> str:
    return str(path) if isinstance(path, Path) else path


def str_to_path(
        path: Union[Path, str]
        ) -> Path:
    return Path(path) if isinstance(path, str) else path
