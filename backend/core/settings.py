from __future__ import annotations

from functools import lru_cache

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class APISettings(BaseSettings):
    """
    Environment-based backend configuration.
    """

    app_name: str = "ResearchEase AI API"
    api_version: str = "7.0.0"
    api_prefix: str = "/api/v1"

    debug: bool = True

    max_upload_mb: int = 50

    cors_origins: str = (
        "http://localhost:8501,"
        "http://127.0.0.1:8501,"
        "http://localhost:8502,"
        "http://127.0.0.1:8502"
    )

    semantic_scholar_api_key: str = ""
    crossref_mailto: str = ""

    model_config = SettingsConfigDict(
        env_prefix="RESEARCHEASE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def allowed_origins(self) -> list[str]:
        """
        Convert the comma-separated CORS value into a list.
        """

        return [
            origin.strip()
            for origin in self.cors_origins.split(",")
            if origin.strip()
        ]


@lru_cache(maxsize=1)
def get_settings() -> APISettings:
    """
    Return one cached settings object.
    """

    return APISettings()