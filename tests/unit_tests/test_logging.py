import io
import logging
import unittest
from pathlib import Path
from unittest.mock import patch

from docmind.utils.log import LogConfig


class TestLogConfig(unittest.TestCase):
    def test_initialization(self):
        """Test the initialization of LogConfig."""
        log_config = LogConfig()
        self.assertIsInstance(log_config.log_file_path, str)
        self.assertEqual(log_config.level, "INFO")

    def test_invalid_level(self):
        """Test LogConfig initialization with an invalid log level."""
        with self.assertRaises(ValueError):
            LogConfig(level="INVALID")

    def test_log_file_path_validation(self):
        """Test the log file path validation."""
        log_config = LogConfig(log_file_path="test.log")
        self.assertEqual(log_config.log_file_path, Path("test.log"))

    def test_log_file_path_validation_with_io(self):
        """Test the log file path validation with an io.StringIO object."""
        with io.StringIO("test.log") as file_obj:
            log_config = LogConfig(log_file_path=file_obj)
            self.assertEqual(log_config.log_file_path, Path("test.log"))

    def test_get_logger(self):
        """Test the get_logger method."""
        with patch("logging.FileHandler"):
            log = LogConfig()
            logger = log.get_logger()
            assert isinstance(logger, logging.Logger)

    def test_clean_log_file(self):
        """Test the clean_log_file functionality."""
        temp_log_file = Path("temp.log")
        temp_log_file.touch()
        with patch.object(Path, "exists", return_value=True), patch.object(
                Path, "unlink"
        ) as mock_unlink, patch("logging.FileHandler"):
            log = LogConfig(log_file_path=temp_log_file, clean_log_file=True)
            log.get_logger()
            mock_unlink.assert_called_once()


if __name__ == '__main__':
    unittest.main()
