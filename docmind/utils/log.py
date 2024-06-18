import io
import logging
from pathlib import Path
from typing import List, Union

from pydantic import BaseModel, validator
from rich.console import Console
from rich.logging import RichHandler


# Define a custom filter class for logging
class CustomFilters(logging.Filter):
    def __init__(self, filters: Union[str, List[str]]):
        super().__init__()
        self.filters = [filters] if isinstance(filters, str) else filters

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(f in record.getMessage() for f in self.filters)


# Define a class for log configuration using Pydantic models
class LogConfig(BaseModel):
    log_file_path: Union[Path, io.StringIO, str] = "logging.log"
    level: str = "INFO"
    logger_name: str = "__name__"
    filters: Union[str, List[str]] = ["watchdog.observers.inotify_buffer"]
    format: str = "%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s"
    datefmt: str = "[%X]"
    clean_log_file: bool = False

    # Validators to ensure correct log level and file path
    @validator('level', allow_reuse=True)
    def check_log_level(cls, value):
        valid_levels = logging._nameToLevel.keys()
        if value.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {value}. Valid options are: {list(valid_levels)}")
        return value.upper()

    @validator('log_file_path', pre=True, allow_reuse=True)
    def validate_log_file_path(cls, v):
        return Path(v) if isinstance(v, str) else Path(v.getvalue()) if isinstance(v, io.StringIO) else v

    # Property to create a console for rich logging
    @property
    def console(self) -> Console:
        return Console(force_terminal=True, color_system="auto")

    # Property to create a handler for rich logging
    @property
    def handler(self) -> RichHandler:
        return RichHandler(console=self.console, enable_link_path=False, rich_tracebacks=True,
                           tracebacks_show_locals=True)

    # Method to get a configured logger
    def get_logger(self) -> logging.Logger:
        # Clean the log file if required
        if self.clean_log_file and self.log_file_path.is_file():
            self.log_file_path.unlink()

        # Create a file handler with the specified format and date format
        file_handler = logging.FileHandler(filename=self.log_file_path)
        file_handler.setFormatter(logging.Formatter(self.format, self.datefmt))

        # Configure the basic logging setup
        logging.basicConfig(level=self.level, handlers=[self.handler, file_handler])

        # Get the logger with the specified name
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(self.level)
        logger.addFilter(CustomFilters(self.filters))
        return logger

    class Config:
        arbitrary_types_allowed = True
