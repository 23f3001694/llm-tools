"""
Application configuration using Pydantic Settings.
Loads from environment variables with sensible defaults.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Quiz credentials
    email: str
    secret: str

    # API Keys - only Google required for Gemini
    google_api_key: str

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    # Timeouts (in seconds)
    task_timeout_seconds: int = 9000  # 10 minutes for complete quiz chains
    code_execution_timeout_seconds: int = 60
    page_load_timeout_seconds: int = 15
    http_request_timeout_seconds: int = 30

    # Limits
    max_retries_per_question: int = 3
    max_download_size_mb: int = 500
    max_recursion_limit: int = 500

    # Paths
    task_base_dir: str = "/tmp/tds_tasks"

    # Server
    host: str = "0.0.0.0"
    port: int = 7860


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
