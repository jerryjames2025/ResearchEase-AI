from __future__ import annotations

from backend.schemas import (
    ExternalPaperResponse,
    PaperSourceResponse,
)
from research.models import AcademicPaper
from services.vector_store import (
    SearchResult,
)


def serialize_paper_sources(
    sources: list[SearchResult],
) -> list[PaperSourceResponse]:
    """
    Convert FAISS results into API-safe responses.
    """

    return [
        PaperSourceResponse(
            rank=source.rank,
            page_number=(
                source.chunk.page_number
            ),
            chunk_number=(
                source.chunk
                .chunk_number_on_page
            ),
            score=source.score,
            text=source.chunk.text,
        )
        for source in sources
    ]


def serialize_external_paper(
    paper: AcademicPaper,
) -> ExternalPaperResponse:
    """
    Convert external academic metadata into
    a Pydantic response.
    """

    return ExternalPaperResponse(
        source=paper.source,
        source_id=paper.source_id,
        title=paper.title,
        authors=paper.authors,
        year=paper.year,
        abstract=paper.abstract,
        url=paper.url,
        doi=paper.doi,
        venue=paper.venue,
        citation_count=paper.citation_count,
        publication_date=(
            paper.publication_date
        ),
        pdf_url=paper.pdf_url,
        relevance_score=(
            paper.relevance_score
        ),
    )