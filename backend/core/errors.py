from __future__ import annotations

import logging

from fastapi import Request
from fastapi.exceptions import (
    RequestValidationError,
)
from fastapi.responses import JSONResponse

from backend.core.settings import (
    get_settings,
)


logger = logging.getLogger(
    "researchease.api"
)


class APIError(Exception):
    """
    Application error that can safely be returned
    through the REST API.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        detail: str,
    ) -> None:
        super().__init__(detail)

        self.status_code = status_code
        self.code = code
        self.detail = detail


class SessionNotFoundError(APIError):
    """
    Raised when a requested research session
    does not exist.
    """

    def __init__(
        self,
        session_id: str,
    ) -> None:
        super().__init__(
            status_code=404,
            code="session_not_found",
            detail=(
                "No research session was found "
                f"for ID '{session_id}'."
            ),
        )


async def api_error_handler(
    request: Request,
    exc: APIError,
) -> JSONResponse:
    """
    Handle known application errors.
    """

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.code,
            "detail": exc.detail,
            "path": request.url.path,
        },
    )


async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Return consistent validation-error responses.
    """

    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": "Request validation failed.",
            "issues": exc.errors(),
            "path": request.url.path,
        },
    )


async def unexpected_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Prevent internal stack traces from being returned
    to API clients.
    """

    logger.exception(
        "Unhandled API error on %s",
        request.url.path,
    )

    settings = get_settings()

    detail = (
        str(exc)
        if settings.debug
        else "An unexpected server error occurred."
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": detail,
            "path": request.url.path,
        },
    )