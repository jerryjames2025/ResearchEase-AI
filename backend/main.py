from __future__ import annotations

from contextlib import (
    asynccontextmanager,
)
import logging

from backend.routers import (
    chat,
    evaluation,
    fine_tuning,
    health,
    literature,
    llm,
    mathematics,
    papers,
    research,
    storage,
    vector,
)

from fastapi import FastAPI
from fastapi.exceptions import (
    RequestValidationError,
)
from fastapi.middleware.cors import (
    CORSMiddleware,
)
from backend.routers import (
    chat,
    evaluation,
    health,
    literature,
    llm,
    mathematics,
    papers,
    research,
    storage,
    vector,
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
    llm,
    mathematics,
    papers,
    research,
    storage,
)
from backend.storage.health import (
    storage_health,
)
from backend.storage.mongo import (
    initialize_mongo,
)
from backend.storage.postgres import (
    create_tables,
)
from backend.routers import (
    chat,
    health,
    literature,
    llm,
    mathematics,
    papers,
    research,
    storage,
    vector,
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
    logger.info(
        "Starting %s version %s",
        settings.app_name,
        settings.api_version,
    )

    settings.faiss_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        if settings.auto_create_tables:
            create_tables()

        initialize_mongo()

        status = storage_health()

        for (
            service_name,
            service_status,
        ) in status[
            "services"
        ].items():
            logger.info(
                "Storage %s: %s - %s",
                service_name,
                (
                    "healthy"
                    if service_status[
                        "healthy"
                    ]
                    else "unavailable"
                ),
                service_status["detail"],
            )

        if (
            settings.strict_storage_startup
            and status["status"]
            != "healthy"
        ):
            raise RuntimeError(
                "One or more required storage "
                "services are unavailable."
            )

    except Exception:
        logger.exception(
            "Storage initialization failed."
        )

        if settings.strict_storage_startup:
            raise

    yield

    logger.info(
        "Stopping %s",
        settings.app_name,
    )


app = FastAPI(
    title=settings.app_name,
    version=settings.api_version,
    description=(
        "Persistent multi-provider REST API for "
        "research-paper analysis, RAG, external "
        "research, mathematical explanation and "
        "literature-review generation."
        "Persistent multi-provider research API with "
        "switchable FAISS and Pinecone vector storage."
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
        "storage_health": (
            f"{settings.api_prefix}"
            "/storage/health"
        ),
        "llm_providers": (
            f"{settings.api_prefix}"
            "/llm/providers"
        ),
        "vector_providers": (
            f"{settings.api_prefix}"
            "/vector/providers"),
        
        "evaluation_health": (
            f"{settings.api_prefix}"
            "/evaluation/health"),
        
        "fine_tuning_health": (
            f"{settings.api_prefix}"
            "/fine-tuning/health"),
    }


api_prefix = settings.api_prefix

app.include_router(
    health.router,
    prefix=api_prefix,
)

app.include_router(
    storage.router,
    prefix=api_prefix,
)

app.include_router(
    llm.router,
    prefix=api_prefix,
)
app.include_router(
    vector.router,
    prefix=api_prefix,
)
app.include_router(
    evaluation.router,
    prefix=api_prefix,
)

app.include_router(
    fine_tuning.router,
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