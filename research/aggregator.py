from __future__ import annotations

from research.arxiv_search import (
    search_arxiv,
)
from research.crossref_search import (
    search_crossref,
)
from research.models import (
    AcademicPaper,
    AcademicSearchResponse,
)
from research.semantic_scholar import (
    AcademicSearchError,
    search_semantic_scholar,
)


def _merge_papers(
    papers: list[AcademicPaper],
) -> list[AcademicPaper]:
    """
    Deduplicate results using DOI or normalized title.

    When duplicate records are found, keep the record
    containing the richer abstract and metadata.
    """

    merged: dict[
        str,
        AcademicPaper,
    ] = {}

    for paper in papers:
        key = paper.identity_key

        existing = merged.get(
            key
        )

        if existing is None:
            merged[key] = paper

            continue

        existing_quality = (
            len(existing.abstract)
            + (
                100
                if existing.doi
                else 0
            )
            + (
                50
                if existing.pdf_url
                else 0
            )
        )

        new_quality = (
            len(paper.abstract)
            + (
                100
                if paper.doi
                else 0
            )
            + (
                50
                if paper.pdf_url
                else 0
            )
        )

        if new_quality > existing_quality:
            merged[key] = paper

    return list(
        merged.values()
    )


def search_academic_sources(
    query: str,
    limit_per_source: int,
    use_semantic_scholar: bool = True,
    use_arxiv: bool = True,
    use_crossref: bool = True,
    semantic_scholar_api_key: str = "",
    crossref_mailto: str = "",
) -> AcademicSearchResponse:
    """
    Search every selected academic provider.
    """

    papers: list[AcademicPaper] = []
    warnings: list[str] = []

    providers = []

    if use_semantic_scholar:
        providers.append(
            (
                "Semantic Scholar",
                lambda: search_semantic_scholar(
                    query=query,
                    limit=limit_per_source,
                    api_key=(
                        semantic_scholar_api_key
                    ),
                ),
            )
        )

    if use_arxiv:
        providers.append(
            (
                "arXiv",
                lambda: search_arxiv(
                    query=query,
                    limit=limit_per_source,
                ),
            )
        )

    if use_crossref:
        providers.append(
            (
                "Crossref",
                lambda: search_crossref(
                    query=query,
                    limit=limit_per_source,
                    mailto=(
                        crossref_mailto
                    ),
                ),
            )
        )

    if not providers:
        return AcademicSearchResponse(
            papers=[],
            warnings=[
                "Select at least one academic "
                "search provider."
            ],
        )

    for (
        provider_name,
        provider_call,
    ) in providers:
        try:
            papers.extend(
                provider_call()
            )

        except AcademicSearchError as exc:
            warnings.append(
                f"{provider_name}: {exc}"
            )

    return AcademicSearchResponse(
        papers=_merge_papers(
            papers
        ),
        warnings=warnings,
    )