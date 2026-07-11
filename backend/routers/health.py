from __future__ import annotations

from fastapi import APIRouter

from backend.core.settings import (
    get_settings,
)
from backend.schemas import (
    HealthResponse,
)
from services.ollama_client import (
    check_ollama,
)


router = APIRouter(
    tags=["health"]
)


@router.get(
    "/health",
    response_model=HealthResponse,
)
def health_check() -> HealthResponse:
    """
    Liveness endpoint.

    This does not require Ollama to be running.
    """

    settings = get_settings()

    return HealthResponse(
        status="healthy",
        version=settings.api_version,
    )


@router.get(
    "/ready",
    response_model=HealthResponse,
)
def readiness_check() -> HealthResponse:
    """
    Readiness endpoint that also verifies Ollama.
    """

    settings = get_settings()

    connected, message = (
        check_ollama()
    )

    return HealthResponse(
        status=(
            "ready"
            if connected
            else "degraded"
        ),
        version=settings.api_version,
        ollama_connected=connected,
        detail=message,
    )