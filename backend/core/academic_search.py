from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
)

from backend.core.settings import (
    APISettings,
)

from backend.storage.redis_cache import (
    redis_cache,
)

from research.aggregator import (
    search_academic_sources,
)

from research.models import (
    AcademicPaper,
)

from research.ranker import (
    rank_academic_papers,
)

from services.embedding_model import (
    load_embedding_service,
)


@dataclass(frozen=True)
class AcademicSearchBundle:
    papers: list[AcademicPaper]
    warnings: list[str]
    cache_hit: bool


def search_and_rank_academic_sources(
    *,
    query: str,
    limit_per_source: int,
    embedding_model: str,
    embedding_device: str,
    use_semantic_scholar: bool,
    use_arxiv: bool,
    use_crossref: bool,
    settings: APISettings,
) -> AcademicSearchBundle:

    cache_payload = {
        "query": (
            query.strip().lower()
        ),
        "limit_per_source": (
            limit_per_source
        ),
        "embedding_model": (
            embedding_model
        ),
        "embedding_device": (
            embedding_device
        ),
        "use_semantic_scholar": (
            use_semantic_scholar
        ),
        "use_arxiv": use_arxiv,
        "use_crossref": use_crossref,
    }

    cache_key = redis_cache.make_key(
        "academic_search",
        cache_payload,
    )

    cached = redis_cache.get_json(
        cache_key
    )

    if isinstance(
        cached,
        dict,
    ):
        papers = [
            AcademicPaper(
                **item
            )
            for item
            in cached.get(
                "papers",
                [],
            )
        ]

        return AcademicSearchBundle(
            papers=papers,
            warnings=list(
                cached.get(
                    "warnings",
                    [],
                )
            ),
            cache_hit=True,
        )

    response = search_academic_sources(
        query=query,
        limit_per_source=(
            limit_per_source
        ),
        use_semantic_scholar=(
            use_semantic_scholar
        ),
        use_arxiv=use_arxiv,
        use_crossref=use_crossref,
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
            embedding_model,
            embedding_device,
        )
    )

    ranked = rank_academic_papers(
        query=query,
        papers=response.papers,
        embedding_service=(
            embedding_service
        ),
    )

    redis_cache.set_json(
        cache_key,
        {
            "papers": [
                asdict(paper)
                for paper in ranked
            ],
            "warnings": (
                response.warnings
            ),
        },
        settings
        .external_search_cache_ttl_seconds,
    )

    return AcademicSearchBundle(
        papers=ranked,
        warnings=response.warnings,
        cache_hit=False,
    )