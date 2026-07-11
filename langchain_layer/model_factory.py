from __future__ import annotations

from functools import lru_cache

from langchain_ollama import ChatOllama

from config import OLLAMA_CHAT_URL


def _ollama_base_url() -> str:
    """
    Convert the configured Ollama chat endpoint
    into Ollama's base URL.
    """

    if "/api/" in OLLAMA_CHAT_URL:
        return OLLAMA_CHAT_URL.split(
            "/api/",
            1,
        )[0]

    return OLLAMA_CHAT_URL.rstrip(
        "/"
    )


@lru_cache(maxsize=16)
def get_ollama_chat_model(
    model_name: str,
    temperature: float = 0.1,
    json_mode: bool = False,
) -> ChatOllama:
    """
    Create and cache a LangChain ChatOllama model.

    A separate model object is created for every
    combination of model, temperature and JSON mode.
    """

    model_name = model_name.strip()

    if not model_name:
        raise ValueError(
            "Ollama model name cannot be empty."
        )

    options = {
        "model": model_name,
        "temperature": float(
            temperature
        ),
        "base_url": _ollama_base_url(),
        "validate_model_on_init": False,
    }

    if json_mode:
        options[
            "format"
        ] = "json"

    return ChatOllama(
        **options
    )