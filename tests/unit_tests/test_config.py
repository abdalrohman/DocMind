import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from docmind.utils.config import ProjectConfiguration, LOG_PATH


class TestProjectConfiguration(unittest.TestCase):

    def setUp(self):
        # Set up a temporary directory for testing
        self.test_dir = Path(__file__).resolve().parent
        self.test_log_path = self.test_dir / "test_logs"
        self.test_log_path.mkdir(exist_ok=True)
        self.test_config_file = self.test_dir / "test_config.yaml"

    def tearDown(self):
        # Clean up the test directory after tests
        if self.test_log_path.exists():
            for file in self.test_log_path.iterdir():
                file.unlink()
            self.test_log_path.rmdir()
        if self.test_config_file.exists():
            self.test_config_file.unlink()

    def test_log_path_creation(self):
        """Test if the log path is created if it doesn't exist."""
        self.assertTrue(LOG_PATH.exists(), "Log path was not created.")

    def test_default_configuration(self):
        """Test the default configuration values."""
        config = ProjectConfiguration()
        self.assertEqual(config.log_path, str(LOG_PATH))
        self.assertEqual(config.log_level, "INFO")

    def test_load_config(self):
        """Test loading configuration from a YAML file."""
        # Create a test YAML file
        test_config = {'log_path': str(self.test_log_path), 'log_level': 'DEBUG'}
        with self.test_config_file.open('w') as file:
            yaml.dump(test_config, file)

        # Load the configuration
        config = ProjectConfiguration.load_config(self.test_config_file)
        self.assertEqual(config.log_path, self.test_log_path)
        self.assertEqual(config.log_level, 'DEBUG')

    def test_create_default_config_file(self):
        """Test creating a default configuration file."""
        ProjectConfiguration.create_default_config_file(self.test_config_file)
        self.assertTrue(self.test_config_file.exists(), "Default config file was not created.")

        # Load the default configuration
        with self.test_config_file.open('r') as file:
            loaded_config = yaml.safe_load(file)
        self.assertEqual(loaded_config['log_path'], str(LOG_PATH))
        self.assertEqual(loaded_config['log_level'], "INFO")

    def test_invalid_log_level(self):
        """Test setting an invalid log level."""
        with self.assertRaises(ValueError):
            ProjectConfiguration(log_level="INVALID")

    def test_environment_variable_override(self):
        """Test if environment variables override default values."""
        with patch('utils.config.PROJECT_PATH', new_callable=lambda: self.test_dir):
            with patch.dict('os.environ', {'LOG_PATH': str(self.test_log_path), 'LOG_LEVEL': 'WARNING'}):
                config = ProjectConfiguration()
                self.assertEqual(config.log_path, str(LOG_PATH))
                self.assertEqual(config.log_level, 'INFO')


if __name__ == '__main__':
    unittest.main()
