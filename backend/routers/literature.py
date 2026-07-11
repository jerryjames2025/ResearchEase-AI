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
    response_model=LiteratureReviewResponse,
)
def create_literature_review(
    files: list[UploadFile] = File(...),
    topic: str = Form(...),
    primary_session_id: str | None = Form(
        None
    ),
    ollama_model: str = Form(
        DEFAULT_OLLAMA_MODEL
    ),
    explanation_level: str = Form(
        "Beginner"
    ),
    settings: APISettings = Depends(
        get_settings
    ),
) -> LiteratureReviewResponse:
    """
    Compare supporting PDFs and an optional
    primary-session paper.
    """

    summaries = []
    source_number = 1

    if primary_session_id:
        primary_session = (
            session_store.get(
                primary_session_id
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
                    ollama_model
                ),
            )
        )

        source_number += 1

    for uploaded_file in files:
        upload = read_pdf_upload(
            uploaded_file,
            settings.max_upload_mb,
        )

        paper = extract_pdf(
            upload
        )

        summaries.append(
            summarize_full_text_paper(
                paper=paper,
                citation_key=(
                    f"P{source_number}"
                ),
                ollama_model=(
                    ollama_model
                ),
            )
        )

        source_number += 1

    if len(summaries) < 2:
        raise APIError(
            status_code=400,
            code="insufficient_sources",
            detail=(
                "At least two papers are required "
                "for a literature review."
            ),
        )

    if (
        len(summaries)
        > MAX_LITERATURE_SOURCES
    ):
        raise APIError(
            status_code=400,
            code="too_many_sources",
            detail=(
                "The number of selected sources "
                "exceeds the configured limit of "
                f"{MAX_LITERATURE_SOURCES}."
            ),
        )

    summaries = reassign_citation_keys(
        summaries
    )

    synthesis = (
        generate_literature_synthesis(
            topic=topic,
            summaries=summaries,
            ollama_model=ollama_model,
            explanation_level=(
                explanation_level
            ),
        )
    )

    return LiteratureReviewResponse(
        topic=topic,
        source_count=len(
            summaries
        ),
        matrix=[
            summary.matrix_row()
            for summary in summaries
        ],
        synthesis=synthesis,
    )