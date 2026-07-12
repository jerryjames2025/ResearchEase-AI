from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
)

from backend.core.academic_search import (
    search_and_rank_academic_sources,
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


router = APIRouter(
    prefix="/research",
    tags=["external research"],
)


@router.post(
    "/search",
    response_model=(
        ResearchSearchResponse
    ),
)
def search_external_research(
    request: ResearchSearchRequest,
    settings: APISettings = Depends(
        get_settings
    ),
) -> ResearchSearchResponse:

    bundle = (
        search_and_rank_academic_sources(
            query=request.query,
            limit_per_source=(
                request
                .limit_per_source
            ),
            embedding_model=(
                request.embedding_model
            ),
            embedding_device=(
                request.embedding_device
            ),
            use_semantic_scholar=(
                request
                .use_semantic_scholar
            ),
            use_arxiv=(
                request.use_arxiv
            ),
            use_crossref=(
                request.use_crossref
            ),
            settings=settings,
        )
    )

    return ResearchSearchResponse(
        query=request.query,
        results=[
            serialize_external_paper(
                paper
            )
            for paper in bundle.papers
        ],
        warnings=bundle.warnings,
        cache_hit=bundle.cache_hit,
    )