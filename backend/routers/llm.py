from __future__ import annotations

from fastapi import APIRouter

from langchain_core.messages import (
    HumanMessage,
)

from backend.core.errors import APIError

from backend.llm.api_schemas import (
    ProviderListResponse,
    ProviderTestRequest,
    ProviderTestResponse,
)

from backend.llm.runtime import (
    build_model_spec,
    default_model_for,
    get_chat_model_from_spec,
    message_text,
    provider_catalog,
)


router = APIRouter(
    prefix="/llm",
    tags=["LLM providers"],
)


@router.get(
    "/providers",
    response_model=ProviderListResponse,
)
def list_providers() -> ProviderListResponse:
    """
    Return all supported LLM providers without
    exposing API keys.
    """

    return ProviderListResponse(
        providers=provider_catalog()
    )


@router.post(
    "/test",
    response_model=ProviderTestResponse,
)
def test_provider(
    request: ProviderTestRequest,
) -> ProviderTestResponse:
    """
    Test one selected provider and report whether
    an automatic fallback provider was used.
    """

    selected_model = (
        request.model.strip()
        or default_model_for(
            request.provider
        )
    )

    model_spec = build_model_spec(
        provider=request.provider,
        model_name=selected_model,
        enable_fallback=(
            request.enable_fallback
        ),
    )

    try:
        model = get_chat_model_from_spec(
            model_spec=model_spec,
            temperature=0.0,
            json_mode=False,
        )

        result = model.invoke(
            [
                HumanMessage(
                    content=request.prompt
                )
            ]
        )

    except Exception as exc:
        raw_message = str(exc)
        lowered = raw_message.lower()

    quota_markers = (
        "credit balance is too low",
        "insufficient_quota",
        "quota exceeded",
        "billing",
        "429",
        "too many requests",
    )

    if any(
        marker in lowered
        for marker in quota_markers
    ):
        detail = (
            f"The selected {request.provider.value} provider "
            "has no usable quota, has reached a rate limit, "
            "or requires billing activation. "
            "Select Ollama or enable automatic fallback. "
            f"Provider response: {raw_message}"
        )

        error_code = "llm_quota_unavailable"

    else:
        detail = raw_message
        error_code = "llm_provider_failed"

    raise APIError(
        status_code=503,
        code=error_code,
        detail=detail,
    ) from exc

    metadata = dict(
        getattr(
            result,
            "response_metadata",
            {},
        )
        or {}
    )

    actual_provider = str(
        metadata.get(
            "researchease_provider",
            request.provider.value,
        )
    )

    actual_model = str(
        metadata.get(
            "researchease_model",
            selected_model,
        )
    )

    return ProviderTestResponse(
        requested_provider=(
            request.provider.value
        ),
        requested_model=selected_model,
        actual_provider=actual_provider,
        actual_model=actual_model,
        fallback_used=(
            actual_provider
            != request.provider.value
        ),
        response=message_text(
            result
        ),
    )