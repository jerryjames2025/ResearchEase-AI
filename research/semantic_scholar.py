from __future__ import annotations

import requests

from config import (
    ACADEMIC_REQUEST_TIMEOUT_SECONDS,
    APP_USER_AGENT,
    SEMANTIC_SCHOLAR_URL,
)
from research.models import AcademicPaper
from research.utils import compact_whitespace


class AcademicSearchError(
    RuntimeError
):
    """
    Raised when an academic provider cannot
    complete a search.
    """


def search_semantic_scholar(
    query: str,
    limit: int,
    api_key: str = "",
) -> list[AcademicPaper]:
    """
    Search Semantic Scholar's Academic Graph API.
    """

    query = query.strip()

    if not query:
        return []

    headers = {
        "User-Agent": APP_USER_AGENT,
    }

    if api_key.strip():
        headers[
            "x-api-key"
        ] = api_key.strip()

    params = {
        "query": query,
        "limit": max(
            1,
            min(
                int(limit),
                100,
            ),
        ),
        "fields": (
            "paperId,title,abstract,authors,year,url,"
            "externalIds,citationCount,venue,"
            "publicationDate,openAccessPdf"
        ),
    }

    try:
        response = requests.get(
            SEMANTIC_SCHOLAR_URL,
            params=params,
            headers=headers,
            timeout=(
                ACADEMIC_REQUEST_TIMEOUT_SECONDS
            ),
        )

        if response.status_code == 429:
            raise AcademicSearchError(
                "Semantic Scholar rate limit reached. "
                "Try again later or provide an API key."
            )

        response.raise_for_status()

        payload = response.json()

    except AcademicSearchError:
        raise

    except (
        requests.RequestException,
        ValueError,
    ) as exc:
        raise AcademicSearchError(
            "Semantic Scholar search failed: "
            f"{exc}"
        ) from exc

    papers: list[AcademicPaper] = []

    for item in payload.get(
        "data",
        [],
    ):
        title = compact_whitespace(
            item.get("title")
        )

        if not title:
            continue

        external_ids = (
            item.get(
                "externalIds"
            )
            or {}
        )

        open_access_pdf = (
            item.get(
                "openAccessPdf"
            )
            or {}
        )

        authors = [
            compact_whitespace(
                author.get("name")
            )
            for author
            in (
                item.get("authors")
                or []
            )
            if compact_whitespace(
                author.get("name")
            )
        ]

        papers.append(
            AcademicPaper(
                source=(
                    "Semantic Scholar"
                ),
                source_id=(
                    compact_whitespace(
                        item.get(
                            "paperId"
                        )
                    )
                ),
                title=title,
                authors=authors,
                year=item.get(
                    "year"
                ),
                abstract=(
                    compact_whitespace(
                        item.get(
                            "abstract"
                        )
                    )
                ),
                url=(
                    compact_whitespace(
                        item.get(
                            "url"
                        )
                    )
                ),
                doi=(
                    compact_whitespace(
                        external_ids.get(
                            "DOI"
                        )
                    )
                ),
                venue=(
                    compact_whitespace(
                        item.get(
                            "venue"
                        )
                    )
                ),
                citation_count=(
                    item.get(
                        "citationCount"
                    )
                ),
                publication_date=(
                    compact_whitespace(
                        item.get(
                            "publicationDate"
                        )
                    )
                ),
                pdf_url=(
                    compact_whitespace(
                        open_access_pdf.get(
                            "url"
                        )
                    )
                ),
            )
        )

    return papers