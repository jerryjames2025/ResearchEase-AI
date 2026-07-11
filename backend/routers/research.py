from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
)

from backend.core.serializers import (
    serialize_external_paper,
)
from backend.core.settings import (
    APISettings,
    get_settings,
)
from backend.schemas import (
    ResearchSearchRequest,
    ResearchSearchResponse,
)
from research.aggregator import (
    search_academic_sources,
)
from research.ranker import (
    rank_academic_papers,
)
from services.embedding_model import (
    load_embedding_service,
)


router = APIRouter(
    prefix="/research",
    tags=["external research"],
)


@router.post(
    "/search",
    response_model=ResearchSearchResponse,
)
def search_external_research(
    request: ResearchSearchRequest,
    settings: APISettings = Depends(
        get_settings
    ),
) -> ResearchSearchResponse:
    """
    Search Semantic Scholar, arXiv, and Crossref,
    then rank the records using transformer embeddings.
    """

    response = search_academic_sources(
        query=request.query,
        limit_per_source=(
            request.limit_per_source
        ),
        use_semantic_scholar=(
            request.use_semantic_scholar
        ),
        use_arxiv=(
            request.use_arxiv
        ),
        use_crossref=(
            request.use_crossref
        ),
        semantic_scholar_api_key=(
            settings
            .semantic_scholar_api_key
        ),
        crossref_mailto=(
            settings.crossref_mailto
        ),
    )

    embedding_service = (
        load_embedding_service(
            request.embedding_model,
            request.embedding_device,
        )
    )

    ranked = rank_academic_papers(
        query=request.query,
        papers=response.papers,
        embedding_service=(
            embedding_service
        ),
    )

    return ResearchSearchResponse(
        query=request.query,
        results=[
            serialize_external_paper(
                paper
            )
            for paper in ranked
        ],
        warnings=response.warnings,
    )