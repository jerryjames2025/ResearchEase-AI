from __future__ import annotations

from dataclasses import dataclass

from fastapi import Header

from backend.core.errors import APIError
from backend.llm.runtime import (
    LLMProvider,
    build_model_spec,
    default_model_for,
)


@dataclass(frozen=True)
class LLMRequestConfig:
    """
    Request-scoped provider configuration.
    """

    provider: LLMProvider
    model: str
    enable_fallback: bool

    def model_spec(
        self,
        fallback_model: str = "",
    ) -> str:
        selected_model = (
            self.model.strip()
            or fallback_model.strip()
            or default_model_for(
                self.provider
            )
        )

        return build_model_spec(
            provider=self.provider,
            model_name=selected_model,
            enable_fallback=(
                self.enable_fallback
            ),
        )


def get_llm_request_config(
    provider_value: str = Header(
        default="ollama",
        alias=(
            "X-ResearchEase-LLM-Provider"
        ),
    ),
    model_value: str = Header(
        default="",
        alias=(
            "X-ResearchEase-LLM-Model"
        ),
    ),
    fallback_value: bool = Header(
        default=False,
        alias=(
            "X-ResearchEase-LLM-Fallback"
        ),
    ),
) -> LLMRequestConfig:
    """
    Read provider configuration from API headers.
    """

    try:
        provider = LLMProvider(
            provider_value
            .strip()
            .lower()
        )

    except ValueError as exc:
        raise APIError(
            status_code=400,
            code="invalid_llm_provider",
            detail=(
                "Supported LLM providers are: "
                "ollama, openai, anthropic, "
                "google, huggingface."
            ),
        ) from exc

    return LLMRequestConfig(
        provider=provider,
        model=model_value.strip(),
        enable_fallback=(
            fallback_value
        ),
    )