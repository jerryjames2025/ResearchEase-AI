from __future__ import annotations

from pydantic import (
    BaseModel,
    Field,
)

from backend.llm.runtime import (
    LLMProvider,
)


class ProviderInfoResponse(BaseModel):
    provider: str
    default_model: str
    installed: bool
    configured: bool
    detail: str


class ProviderListResponse(BaseModel):
    providers: list[
        ProviderInfoResponse
    ]


class ProviderTestRequest(BaseModel):
    provider: LLMProvider = (
        LLMProvider.OLLAMA
    )

    model: str = ""

    enable_fallback: bool = False

    prompt: str = Field(
        default=(
            "Reply with: ResearchEase "
            "provider test successful."
        ),
        min_length=2,
        max_length=1000,
    )


class ProviderTestResponse(BaseModel):
    requested_provider: str
    requested_model: str

    actual_provider: str
    actual_model: str

    fallback_used: bool

    response: str