from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.exceptions import (
    RequestValidationError,
)
from fastapi.middleware.cors import (
    CORSMiddleware,
)

from backend.core.errors import (
    APIError,
    api_error_handler,
    unexpected_error_handler,
    validation_error_handler,
)
from backend.core.settings import (
    get_settings,
)
from backend.routers import (
    chat,
    health,
    literature,
    mathematics,
    papers,
    research,
)


logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | %(levelname)s | "
        "%(name)s | %(message)s"
    ),
)

logger = logging.getLogger(
    "researchease.api"
)

settings = get_settings()


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Handle API startup and shutdown events.
    """

    logger.info(
        "Starting %s version %s",
        settings.app_name,
        settings.api_version,
    )

    yield

    logger.info(
        "Stopping %s",
        settings.app_name,
    )


app = FastAPI(
    title=settings.app_name,
    version=settings.api_version,
    description=(
        "REST API for research-paper analysis, "
        "citation-grounded RAG, external academic "
        "search, mathematical explanation, and "
        "literature-review generation."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        settings.allowed_origins
    ),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.add_exception_handler(
    APIError,
    api_error_handler,
)

app.add_exception_handler(
    RequestValidationError,
    validation_error_handler,
)

app.add_exception_handler(
    Exception,
    unexpected_error_handler,
)


@app.get(
    "/",
    tags=["root"],
)
def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "version": settings.api_version,
        "documentation": "/docs",
    }


api_prefix = settings.api_prefix

app.include_router(
    health.router,
    prefix=api_prefix,
)

app.include_router(
    papers.router,
    prefix=api_prefix,
)

app.include_router(
    chat.router,
    prefix=api_prefix,
)

app.include_router(
    research.router,
    prefix=api_prefix,
)

app.include_router(
    mathematics.router,
    prefix=api_prefix,
)

app.include_router(
    literature.router,
    prefix=api_prefix,
)