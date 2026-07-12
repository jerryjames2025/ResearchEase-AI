from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
)


class APISettings(BaseSettings):
    """
    ResearchEase AI application, storage,
    and LLM-provider configuration.
    """

    app_name: str = "ResearchEase AI API"
    api_version: str = "9.0.0"
    api_prefix: str = "/api/v1"

    debug: bool = True
    max_upload_mb: int = 50

    cors_origins: str = (
        "http://localhost:8501,"
        "http://127.0.0.1:8501,"
        "http://localhost:8502,"
        "http://127.0.0.1:8502"
    )

    # -----------------------------------------------------
    # PostgreSQL
    # -----------------------------------------------------

    postgres_url: str = (
        "postgresql+psycopg://"
        "researchease:researchease_dev_password"
        "@127.0.0.1:5432/researchease"
    )

    # -----------------------------------------------------
    # MongoDB
    # -----------------------------------------------------

    mongodb_url: str = (
        "mongodb://"
        "root:researchease_mongo_password"
        "@127.0.0.1:27017/"
        "?authSource=admin"
    )

    mongodb_database: str = "researchease"

    # -----------------------------------------------------
    # Redis
    # -----------------------------------------------------

    redis_url: str = (
        "redis://127.0.0.1:6379/0"
    )

    session_cache_ttl_seconds: int = 3600

    external_search_cache_ttl_seconds: int = (
        1800
    )

    # -----------------------------------------------------
    # FAISS
    # -----------------------------------------------------

    faiss_directory: Path = Path(
        "data/faiss"
    )

    auto_create_tables: bool = False
    strict_storage_startup: bool = False

    # -----------------------------------------------------
    # External academic APIs
    # -----------------------------------------------------

    semantic_scholar_api_key: str = ""
    crossref_mailto: str = ""

    # -----------------------------------------------------
    # Version 9: Ollama
    # -----------------------------------------------------

    ollama_base_url: str = (
        "http://127.0.0.1:11434"
    )

    default_ollama_model: str = (
        "qwen2.5:1.5b"
    )

    # -----------------------------------------------------
    # Version 9: OpenAI
    # -----------------------------------------------------

    openai_api_key: str = ""

    default_openai_model: str = (
        "gpt-5.4-mini"
    )

    # -----------------------------------------------------
    # Version 9: Anthropic
    # -----------------------------------------------------

    anthropic_api_key: str = ""

    default_anthropic_model: str = (
        "claude-sonnet-5"
    )

    # -----------------------------------------------------
    # Version 9: Google Gemini
    # -----------------------------------------------------

    google_api_key: str = ""

    default_google_model: str = (
        "gemini-2.5-flash"
    )

    # -----------------------------------------------------
    # Version 9: Hugging Face
    # -----------------------------------------------------

    huggingface_api_token: str = ""

    default_huggingface_model: str = (
        "microsoft/Phi-3-mini-4k-instruct"
    )

    huggingface_max_new_tokens: int = 1024

    # -----------------------------------------------------
    # Automatic provider fallback
    # -----------------------------------------------------

    llm_fallback_order: str = (
        "ollama,google,openai,"
        "anthropic,huggingface"
    )

    model_config = SettingsConfigDict(
        env_prefix="RESEARCHEASE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def allowed_origins(
        self,
    ) -> list[str]:
        return [
            origin.strip()
            for origin
            in self.cors_origins.split(",")
            if origin.strip()
        ]

    @property
    def fallback_provider_names(
        self,
    ) -> list[str]:
        return [
            provider.strip().lower()
            for provider
            in self.llm_fallback_order.split(",")
            if provider.strip()
        ]


@lru_cache(maxsize=1)
def get_settings() -> APISettings:
    return APISettings()