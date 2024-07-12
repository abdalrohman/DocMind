import logging
import os
from pathlib import Path
from typing import List, Union

from dotenv import load_dotenv

# Configure logger for the module
logger = logging.getLogger(__name__)


class EnvironmentLoader:
    """A class to load and verify environment variables from a .env file."""

    def __init__(self, env_file_path: Union[str, Path] = ".env", required_vars: List[str] = None):
        """
        Initializes the EnvironmentLoader with the given .env file path and required variables.
        """
        self.env_file_path = Path(env_file_path)
        self.required_vars = required_vars or []

    def load_envs(self) -> None:
        """
        Loads environment variables from the .env file and verifies required variables.
        """
        logger.info(f"Loading environment variables from {self.env_file_path}...")
        if not self.env_file_path.is_file():
            raise FileNotFoundError(f"Failed to load .env file at {self.env_file_path}")

        load_dotenv(dotenv_path=self.env_file_path, override=True)
        self._check_required_vars()

    def _check_required_vars(self) -> None:
        """Checks if all required environment variables are set."""
        missing_vars = [var for var in self.required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Required environment variables not found: {missing_vars}")

        # Check for placeholder values in required variables
        placeholder = "your_key"
        invalid_vars = [var for var in self.required_vars if placeholder in (os.getenv(var) or "")]
        if invalid_vars:
            raise ValueError(
                f"Replace the placeholder '{placeholder}' with actual values in these environment variables: {invalid_vars}")
