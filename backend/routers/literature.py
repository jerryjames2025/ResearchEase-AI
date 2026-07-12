from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    UploadFile,
)

from backend.core.errors import APIError
from backend.core.files import (
    read_pdf_upload,
)
from backend.core.session_store import (
    session_store,
)
from backend.core.settings import (
    APISettings,
    get_settings,
)
from backend.llm.dependency import (
    LLMRequestConfig,
    get_llm_request_config,
)
from backend.schemas import (
    LiteratureReviewResponse,
)
from config import (
    DEFAULT_OLLAMA_MODEL,
    MAX_LITERATURE_SOURCES,
)
from literature.engine import (
    generate_literature_synthesis,
    reassign_citation_keys,
    summarize_full_text_paper,
)
from services.pdf_parser import (
    extract_pdf,
)


router = APIRouter(
    prefix="/literature",
    tags=["literature review"],
)


@router.post(
    "/review",
    response_model=(
        LiteratureReviewResponse
    ),
)
def create_literature_review(
    files: list[UploadFile] = File(...),
    topic: str = Form(...),
    primary_session_id: str | None = Form(
        default=None
    ),
    ollama_model: str = Form(
        default=DEFAULT_OLLAMA_MODEL
    ),
    explanation_level: str = Form(
        default="Beginner"
    ),
    settings: APISettings = Depends(
        get_settings
    ),
    llm: LLMRequestConfig = Depends(
        get_llm_request_config
    ),
) -> LiteratureReviewResponse:
    cleaned_primary_session_id = (
        primary_session_id.strip()
        if primary_session_id
        else ""
    )

    total_requested_sources = (
        len(files)
        + (
            1
            if cleaned_primary_session_id
            else 0
        )
    )

    if total_requested_sources < 2:
        raise APIError(
            status_code=400,
            code="insufficient_sources",
            detail=(
                "At least two research papers "
                "are required."
            ),
        )

    if (
        total_requested_sources
        > MAX_LITERATURE_SOURCES
    ):
        raise APIError(
            status_code=400,
            code="too_many_sources",
            detail=(
                "The source count exceeds "
                f"{MAX_LITERATURE_SOURCES}."
            ),
        )

    runtime_model = llm.model_spec(
        fallback_model=ollama_model
    )

    summaries = []
    source_number = 1

    if cleaned_primary_session_id:
        primary_session = (
            session_store.get(
                cleaned_primary_session_id
            )
        )

        summaries.append(
            summarize_full_text_paper(
                paper=(
                    primary_session.paper
                ),
                citation_key=(
                    f"P{source_number}"
                ),
                ollama_model=(
                    runtime_model
                ),
            )
        )

        source_number += 1

    for uploaded_file in files:
        upload = read_pdf_upload(
            uploaded_file,
            settings.max_upload_mb,
        )

        extracted_paper = (
            extract_pdf(
                upload
            )
        )

        summaries.append(
            summarize_full_text_paper(
                paper=extracted_paper,
                citation_key=(
                    f"P{source_number}"
                ),
                ollama_model=(
                    runtime_model
                ),
            )
        )

        source_number += 1

    summaries = (
        reassign_citation_keys(
            summaries
        )
    )

    synthesis = (
        generate_literature_synthesis(
            topic=topic,
            summaries=summaries,
            ollama_model=(
                runtime_model
            ),
            explanation_level=(
                explanation_level
            ),
        )
    )

    response = LiteratureReviewResponse(
        topic=topic,
        source_count=len(
            summaries
        ),
        matrix=[
            summary.matrix_row()
            for summary in summaries
        ],
        synthesis=synthesis,
        warnings=[],
    )

    if cleaned_primary_session_id:
        session_store.record_literature_result(
            cleaned_primary_session_id,
            response.model_dump(
                mode="json"
            ),
        )

    return response