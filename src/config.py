"""Configuration management for DeepRisk."""

import os
from pathlib import Path


class Settings:
    """Application settings from environment variables."""

    MODEL_DIR: str = os.getenv("MODEL_DIR", "./models")
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

    def get_model_path(self, model_name: str) -> Path:
        """Resolve full path to model file."""
        return Path(self.MODEL_DIR) / model_name
