from __future__ import annotations

import base64
import importlib.util
import json
from enum import Enum
from functools import lru_cache
from typing import Any

import requests

from langchain_core.messages import BaseMessage
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
)

from backend.core.settings import (
    get_settings,
)


MODEL_SPEC_PREFIX = (
    "researchease-llm-v9:"
)


class LLMProvider(
    str,
    Enum,
):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"


class LLMProviderError(RuntimeError):
    """
    Raised when an LLM provider cannot be used.
    """


def default_model_for(
    provider: LLMProvider,
) -> str:
    settings = get_settings()

    mapping = {
        LLMProvider.OLLAMA: (
            settings.default_ollama_model
        ),
        LLMProvider.OPENAI: (
            settings.default_openai_model
        ),
        LLMProvider.ANTHROPIC: (
            settings.default_anthropic_model
        ),
        LLMProvider.GOOGLE: (
            settings.default_google_model
        ),
        LLMProvider.HUGGINGFACE: (
            settings
            .default_huggingface_model
        ),
    }

    return mapping[provider]


def build_model_spec(
    provider: LLMProvider | str,
    model_name: str,
    enable_fallback: bool,
) -> str:
    """
    Encode provider configuration in a string that
    can travel through the existing Version 6–8
    function signatures.

    Old plain model names remain compatible and
    are interpreted as Ollama model names.
    """

    selected_provider = (
        provider
        if isinstance(
            provider,
            LLMProvider,
        )
        else LLMProvider(
            str(provider).strip().lower()
        )
    )

    payload = {
        "provider": (
            selected_provider.value
        ),
        "model": (
            model_name.strip()
            or default_model_for(
                selected_provider
            )
        ),
        "fallback": bool(
            enable_fallback
        ),
    }

    encoded = (
        base64.urlsafe_b64encode(
            json.dumps(
                payload,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        .decode("ascii")
    )

    return (
        MODEL_SPEC_PREFIX
        + encoded
    )


def parse_model_spec(
    value: str,
) -> tuple[
    LLMProvider,
    str,
    bool,
]:
    """
    Parse a Version 9 model specification.

    Plain values such as qwen2.5:1.5b are treated
    as backward-compatible Ollama model names.
    """

    value = value.strip()

    if not value.startswith(
        MODEL_SPEC_PREFIX
    ):
        return (
            LLMProvider.OLLAMA,
            value or default_model_for(
                LLMProvider.OLLAMA
            ),
            False,
        )

    encoded = value[
        len(MODEL_SPEC_PREFIX):
    ]

    try:
        payload = json.loads(
            base64.urlsafe_b64decode(
                encoded.encode("ascii")
            ).decode("utf-8")
        )

        provider = LLMProvider(
            str(
                payload["provider"]
            ).lower()
        )

        model = (
            str(
                payload.get(
                    "model",
                    "",
                )
            ).strip()
            or default_model_for(
                provider
            )
        )

        fallback = bool(
            payload.get(
                "fallback",
                False,
            )
        )

        return (
            provider,
            model,
            fallback,
        )

    except Exception as exc:
        raise LLMProviderError(
            "Invalid Version 9 LLM model specification."
        ) from exc


def _package_available(
    package_name: str,
) -> bool:
    return (
        importlib.util.find_spec(
            package_name
        )
        is not None
    )


def _ollama_status() -> tuple[
    bool,
    str,
]:
    settings = get_settings()

    try:
        response = requests.get(
            (
                settings
                .ollama_base_url
                .rstrip("/")
                + "/api/tags"
            ),
            timeout=3,
        )

        response.raise_for_status()

        return (
            True,
            "Ollama is connected.",
        )

    except requests.RequestException as exc:
        return (
            False,
            f"Ollama unavailable: {exc}",
        )


def _provider_key_configured(
    provider: LLMProvider,
) -> bool:
    settings = get_settings()

    if provider == LLMProvider.OLLAMA:
        return True

    if provider == LLMProvider.OPENAI:
        return bool(
            settings.openai_api_key.strip()
        )

    if provider == LLMProvider.ANTHROPIC:
        return bool(
            settings
            .anthropic_api_key
            .strip()
        )

    if provider == LLMProvider.GOOGLE:
        return bool(
            settings.google_api_key.strip()
        )

    if provider == LLMProvider.HUGGINGFACE:
        return bool(
            settings
            .huggingface_api_token
            .strip()
        )

    return False


def _required_package(
    provider: LLMProvider,
) -> str:
    mapping = {
        LLMProvider.OLLAMA: (
            "langchain_ollama"
        ),
        LLMProvider.OPENAI: (
            "langchain_openai"
        ),
        LLMProvider.ANTHROPIC: (
            "langchain_anthropic"
        ),
        LLMProvider.GOOGLE: (
            "langchain_google_genai"
        ),
        LLMProvider.HUGGINGFACE: (
            "langchain_huggingface"
        ),
    }

    return mapping[provider]


def provider_catalog() -> list[dict]:
    """
    Return provider configuration without exposing
    any secret API keys.
    """

    ollama_connected, ollama_detail = (
        _ollama_status()
    )

    results: list[dict] = []

    for provider in LLMProvider:
        package_name = (
            _required_package(
                provider
            )
        )

        installed = (
            _package_available(
                package_name
            )
        )

        if provider == LLMProvider.OLLAMA:
            configured = ollama_connected
            detail = ollama_detail

        else:
            configured = (
                _provider_key_configured(
                    provider
                )
            )

            detail = (
                "API key configured."
                if configured
                else "API key not configured."
            )

        if not installed:
            detail = (
                f"Required package "
                f"'{package_name}' is not installed."
            )

        results.append(
            {
                "provider": provider.value,
                "default_model": (
                    default_model_for(
                        provider
                    )
                ),
                "installed": installed,
                "configured": (
                    installed
                    and configured
                ),
                "detail": detail,
            }
        )

    return results


def _create_provider_model(
    provider: LLMProvider,
    model_name: str,
    temperature: float,
    json_mode: bool,
):
    """
    Create one provider-specific LangChain chat model.
    """

    settings = get_settings()

    if not _package_available(
        _required_package(
            provider
        )
    ):
        raise LLMProviderError(
            "The integration package for "
            f"'{provider.value}' is not installed."
        )

    if (
        provider != LLMProvider.OLLAMA
        and not _provider_key_configured(
            provider
        )
    ):
        raise LLMProviderError(
            f"{provider.value} API credentials "
            "are not configured."
        )

    if provider == LLMProvider.OLLAMA:
        from langchain_ollama import (
            ChatOllama,
        )

        options: dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "base_url": (
                settings.ollama_base_url
            ),
            "validate_model_on_init": False,
        }

        if json_mode:
            options["format"] = "json"

        return ChatOllama(
            **options
        )

    if provider == LLMProvider.OPENAI:
        from langchain_openai import (
            ChatOpenAI,
        )

        return ChatOpenAI(
            model=model_name,
            api_key=(
                settings.openai_api_key
            ),
            temperature=temperature,
            max_retries=0,
        )

    if provider == LLMProvider.ANTHROPIC:
        from langchain_anthropic import (
            ChatAnthropic,
        )

        return ChatAnthropic(
            model=model_name,
            api_key=(
                settings.anthropic_api_key
            ),
            temperature=temperature,
            max_retries=0,
        )

    if provider == LLMProvider.GOOGLE:
        from langchain_google_genai import (
            ChatGoogleGenerativeAI,
        )

        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=(
                settings.google_api_key
            ),
            temperature=temperature,
            max_retries=0,
        )

    if provider == LLMProvider.HUGGINGFACE:
        from langchain_huggingface import (
            ChatHuggingFace,
            HuggingFaceEndpoint,
        )

        endpoint_options: dict[str, Any] = {
            "repo_id": model_name,
            "task": "text-generation",
            "provider": "auto",
            "max_new_tokens": (
                settings
                .huggingface_max_new_tokens
            ),
            "huggingfacehub_api_token": (
                settings
                .huggingface_api_token
            ),
            "do_sample": (
                temperature > 0
            ),
        }

        if temperature > 0:
            endpoint_options[
                "temperature"
            ] = temperature

        endpoint = HuggingFaceEndpoint(
            **endpoint_options
        )

        return ChatHuggingFace(
            llm=endpoint
        )

    raise LLMProviderError(
        f"Unsupported provider: {provider.value}"
    )


def _tag_response(
    response,
    provider: LLMProvider,
    model_name: str,
):
    """
    Add ResearchEase provider metadata to an AI message.
    """

    if not isinstance(
        response,
        BaseMessage,
    ):
        return response

    metadata = dict(
        getattr(
            response,
            "response_metadata",
            {},
        )
        or {}
    )

    metadata[
        "researchease_provider"
    ] = provider.value

    metadata[
        "researchease_model"
    ] = model_name

    return response.model_copy(
        update={
            "response_metadata": metadata
        }
    )


def _tagged_model(
    provider: LLMProvider,
    model_name: str,
    temperature: float,
    json_mode: bool,
) -> Runnable:
    """
    Create a model and attach provider metadata.
    """

    model = _create_provider_model(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        json_mode=json_mode,
    )

    return (
        model
        | RunnableLambda(
            lambda response: _tag_response(
                response,
                provider,
                model_name,
            )
        )
    )


def _failure_runnable(
    message: str,
) -> Runnable:
    """
    Create a runnable that raises an error.

    This allows a missing primary provider to trigger
    LangChain fallbacks instead of failing during app
    construction.
    """

    def raise_error(_input):
        raise LLMProviderError(
            message
        )

    return RunnableLambda(
        raise_error
    )


def _safe_provider_runnable(
    provider: LLMProvider,
    model_name: str,
    temperature: float,
    json_mode: bool,
) -> Runnable:
    try:
        return _tagged_model(
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            json_mode=json_mode,
        )

    except Exception as exc:
        return _failure_runnable(
            (
                f"{provider.value} provider "
                f"could not initialize: {exc}"
            )
        )


def _fallback_providers(
    primary: LLMProvider,
) -> list[LLMProvider]:
    settings = get_settings()

    providers: list[
        LLMProvider
    ] = []

    for value in (
        settings.fallback_provider_names
    ):
        try:
            provider = LLMProvider(
                value
            )

        except ValueError:
            continue

        if (
            provider == primary
            or provider in providers
        ):
            continue

        if (
            provider != LLMProvider.OLLAMA
            and not _provider_key_configured(
                provider
            )
        ):
            continue

        providers.append(
            provider
        )

    return providers


@lru_cache(maxsize=128)
def get_chat_model_from_spec(
    model_spec: str,
    temperature: float = 0.1,
    json_mode: bool = False,
) -> Runnable:
    """
    Build a primary model with optional provider fallbacks.
    """

    (
        primary_provider,
        primary_model,
        enable_fallback,
    ) = parse_model_spec(
        model_spec
    )

    primary = _safe_provider_runnable(
        provider=primary_provider,
        model_name=primary_model,
        temperature=temperature,
        json_mode=json_mode,
    )

    if not enable_fallback:
        return primary

    fallbacks: list[
        Runnable
    ] = []

    for provider in _fallback_providers(
        primary_provider
    ):
        fallbacks.append(
            _safe_provider_runnable(
                provider=provider,
                model_name=(
                    default_model_for(
                        provider
                    )
                ),
                temperature=temperature,
                json_mode=json_mode,
            )
        )

    if not fallbacks:
        return primary

    return primary.with_fallbacks(
        fallbacks
    )


def message_text(
    response,
) -> str:
    """
    Convert provider-specific message content into text.
    """

    content = getattr(
        response,
        "content",
        response,
    )

    if isinstance(
        content,
        str,
    ):
        return content

    if isinstance(
        content,
        list,
    ):
        parts: list[str] = []

        for item in content:
            if isinstance(
                item,
                str,
            ):
                parts.append(item)

            elif isinstance(
                item,
                dict,
            ):
                text = item.get(
                    "text"
                )

                if text:
                    parts.append(
                        str(text)
                    )

        return "\n".join(
            parts
        )

    return str(content)