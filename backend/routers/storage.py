from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import (
    StorageHealthResponse,
)

from backend.storage.health import (
    storage_health,
)


router = APIRouter(
    prefix="/storage",
    tags=["storage"],
)


@router.get(
    "/health",
    response_model=(
        StorageHealthResponse
    ),
)
def get_storage_health() -> (
    StorageHealthResponse
):
    return (
        StorageHealthResponse
        .model_validate(
            storage_health()
        )
    )