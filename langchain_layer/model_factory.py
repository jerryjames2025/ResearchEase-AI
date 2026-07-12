from __future__ import annotations

from functools import lru_cache

from backend.llm.runtime import (
    get_chat_model_from_spec,
)


@lru_cache(maxsize=128)
def get_ollama_chat_model(
    model_name: str,
    temperature: float = 0.1,
    json_mode: bool = False,
):
    """
    Backward-compatible model factory.

    Version 9 accepts:

    - A normal Ollama model name
    - An encoded multi-provider model specification
    """

    return get_chat_model_from_spec(
        model_spec=model_name,
        temperature=temperature,
        json_mode=json_mode,
    )