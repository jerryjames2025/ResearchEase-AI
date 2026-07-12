from __future__ import annotations

from fastapi import APIRouter

from backend.core.errors import APIError
from backend.core.settings import (
    get_settings,
)
from services.pinecone_vector_store import (
    PineconeVectorStore,
    pinecone_package_installed,
)


router = APIRouter(
    prefix="/vector",
    tags=["vector databases"],
)


@router.get(
    "/providers",
)
def vector_providers() -> dict:
    settings = get_settings()

    pinecone_key_configured = bool(
        settings.pinecone_api_key.strip()
    )

    return {
        "default_backend": (
            settings
            .default_vector_backend
        ),
        "providers": [
            {
                "provider": "faiss",
                "installed": True,
                "configured": True,
                "persistent": True,
                "location": "local",
                "detail": (
                    "FAISS stores one local index "
                    "file per research session."
                ),
            },
            {
                "provider": "pinecone",
                "installed": (
                    pinecone_package_installed()
                ),
                "configured": (
                    pinecone_package_installed()
                    and pinecone_key_configured
                ),
                "persistent": True,
                "location": "cloud",
                "detail": (
                    "Pinecone API key configured."
                    if pinecone_key_configured
                    else (
                        "Pinecone API key is "
                        "not configured."
                    )
                ),
            },
        ],
    }


@router.get(
    "/health",
)
def vector_health() -> dict:
    pinecone_ok, pinecone_detail = (
        PineconeVectorStore.health()
    )

    return {
        "status": (
            "healthy"
            if pinecone_ok
            else "degraded"
        ),
        "providers": {
            "faiss": {
                "healthy": True,
                "detail": (
                    "Local FAISS is available."
                ),
            },
            "pinecone": {
                "healthy": pinecone_ok,
                "detail": pinecone_detail,
            },
        },
    }


@router.get(
    "/pinecone/stats",
)
def pinecone_stats() -> dict:
    settings = get_settings()

    if not settings.pinecone_api_key.strip():
        raise APIError(
            status_code=400,
            code=(
                "pinecone_not_configured"
            ),
            detail=(
                "Pinecone API key is not configured."
            ),
        )

    try:
        from pinecone import Pinecone

        client = Pinecone(
            api_key=(
                settings.pinecone_api_key
            )
        )

        if not client.has_index(
            settings.pinecone_index_name
        ):
            return {
                "exists": False,
                "index_name": (
                    settings
                    .pinecone_index_name
                ),
            }

        description = (
            client.describe_index(
                name=(
                    settings
                    .pinecone_index_name
                )
            )
        )

        host = getattr(
            description,
            "host",
            "",
        )

        index = client.Index(
            host=host
        )

        statistics = (
            index.describe_index_stats()
        )

        if hasattr(
            statistics,
            "to_dict",
        ):
            stats_payload = (
                statistics.to_dict()
            )

        elif isinstance(
            statistics,
            dict,
        ):
            stats_payload = statistics

        else:
            stats_payload = {
                "value": str(
                    statistics
                )
            }

        return {
            "exists": True,
            "index_name": (
                settings
                .pinecone_index_name
            ),
            "host": host,
            "statistics": (
                stats_payload
            ),
        }

    except Exception as exc:
        raise APIError(
            status_code=503,
            code=(
                "pinecone_stats_failed"
            ),
            detail=str(exc),
        ) from exc