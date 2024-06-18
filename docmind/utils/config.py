from pathlib import Path

import yaml
from pydantic import BaseModel, Field, validator


# Define project and log paths using pathlib for better path handling
# PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
# LOG_PATH = PROJECT_PATH / "logs"
# LOG_PATH.mkdir(parents=True, exist_ok=True)  # Create the log directory if it doesn't exist


# Define a class for configuration using Pydantic
class ProjectConfiguration(BaseModel):
    log_path: Path = None
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    @validator("log_level", allow_reuse=True)
    def check_log_level(cls, value):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"]
        if value.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {value}. Valid options are: {valid_levels}")
        return value.upper()

    # Load configuration from a YAML file or return default configuration
    @classmethod
    def load_config(cls, file_path: Path):
        if file_path.exists():
            with open(file_path, "r") as file:
                return cls(**yaml.safe_load(file))
        return cls()

    # Create a default configuration YAML file if it does not exist
    @classmethod
    def create_default_config_file(cls, file_path: Path, user_data_dir: Path):
        if not file_path.is_file():
            default_config = cls().dict()
            log_dir = user_data_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            default_config["log_path"] = str(log_dir)
            with file_path.open("w") as file:
                yaml.dump(default_config, file, default_flow_style=False)

# # Define the path to the configuration file
# config_file_path = PROJECT_PATH / "config.yaml"
#
# # Write the default YAML file if it does not exist
# ProjectConfiguration.create_default_config_file(config_file_path)
#
# # Load the configuration
# config = ProjectConfiguration.load_config(config_file_path)
