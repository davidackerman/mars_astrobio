"""
Configuration and environment variable management.
"""

import os
from pathlib import Path
from typing import Optional
import logging

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class Config:
    """
    Configuration manager for Mars Biosignature Detection project.

    Loads environment variables from .env file and provides
    convenient access to configuration values.
    """

    def __init__(self, env_file: Optional[Path] = None):
        """
        Initialize configuration.

        Parameters
        ----------
        env_file : Path, optional
            Path to .env file. If None, searches for .env in project root.
        """
        if env_file is None:
            # Search for .env in current directory and parent directories
            current = Path.cwd()
            for parent in [current] + list(current.parents):
                env_path = parent / ".env"
                if env_path.exists():
                    env_file = env_path
                    break

        if env_file and env_file.exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment from {env_file}")
        else:
            logger.warning("No .env file found - using environment variables only")

    # NASA API Configuration
    @property
    def nasa_api_key(self) -> str:
        """Get NASA API key from environment."""
        key = os.getenv("NASA_API_KEY", "")
        if not key or key == "your_nasa_api_key_here":
            raise ValueError(
                "NASA_API_KEY not set! "
                "Please add your API key to the .env file or set the environment variable. "
                "Get a free key at: https://api.nasa.gov/"
            )
        return key

    @property
    def pds_api_url(self) -> str:
        """Get PDS API URL."""
        return os.getenv(
            "PDS_API_URL",
            "https://api.nasa.gov/mars-photos/api/v1"
        )

    @property
    def pds_archive_url(self) -> str:
        """Get PDS Archive URL."""
        return os.getenv(
            "PDS_ARCHIVE_URL",
            "https://pds-geosciences.wustl.edu/missions/mars2020"
        )

    # Data Paths
    @property
    def data_raw_dir(self) -> Path:
        """Get raw data directory."""
        return Path(os.getenv("DATA_RAW_DIR", "data/raw"))

    @property
    def data_processed_dir(self) -> Path:
        """Get processed data directory."""
        return Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))

    @property
    def data_cache_dir(self) -> Path:
        """Get cache directory."""
        return Path(os.getenv("DATA_CACHE_DIR", "data/cache"))

    @property
    def data_annotations_dir(self) -> Path:
        """Get annotations directory."""
        return Path(os.getenv("DATA_ANNOTATIONS_DIR", "data/annotations"))

    # Model Paths
    @property
    def model_checkpoint_dir(self) -> Path:
        """Get model checkpoint directory."""
        return Path(os.getenv("MODEL_CHECKPOINT_DIR", "models/checkpoints"))

    @property
    def model_production_dir(self) -> Path:
        """Get production model directory."""
        return Path(os.getenv("MODEL_PRODUCTION_DIR", "models/production"))

    # Hardware Settings
    @property
    def device(self) -> str:
        """Get compute device (cuda or cpu)."""
        return os.getenv("DEVICE", "cuda")

    @property
    def mixed_precision(self) -> bool:
        """Whether to use mixed precision training."""
        return os.getenv("MIXED_PRECISION", "true").lower() in ("true", "1", "yes")

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get arbitrary environment variable.

        Parameters
        ----------
        key : str
            Environment variable name
        default : str, optional
            Default value if not set

        Returns
        -------
        str or None
            Environment variable value
        """
        return os.getenv(key, default)

    def set(self, key: str, value: str):
        """
        Set environment variable.

        Parameters
        ----------
        key : str
            Environment variable name
        value : str
            Value to set
        """
        os.environ[key] = value

    def validate(self):
        """
        Validate required configuration values.

        Raises
        ------
        ValueError
            If required configuration is missing or invalid
        """
        # Check NASA API key
        try:
            _ = self.nasa_api_key
        except ValueError as e:
            logger.error(str(e))
            raise

        logger.info("Configuration validated successfully")


# Global configuration instance
_config = None


def get_config(reload: bool = False) -> Config:
    """
    Get global configuration instance.

    Parameters
    ----------
    reload : bool
        Whether to reload configuration from .env file

    Returns
    -------
    Config
        Global configuration instance
    """
    global _config
    if _config is None or reload:
        _config = Config()
    return _config
