from __future__ import annotations

import requests

from config import (
    ACADEMIC_REQUEST_TIMEOUT_SECONDS,
    APP_USER_AGENT,
    CROSSREF_API_URL,
)
from research.models import AcademicPaper
from research.semantic_scholar import (
    AcademicSearchError,
)
from research.utils import (
    compact_whitespace,
    first_list_value,
    strip_html,
    year_from_date_parts,
)


def search_crossref(
    query: str,
    limit: int,
    mailto: str = "",
) -> list[AcademicPaper]:
    """
    Search the Crossref Works API.
    """

    query = query.strip()

    if not query:
        return []

    params = {
        "query.bibliographic": query,
        "rows": max(
            1,
            min(
                int(limit),
                100,
            ),
        ),
        "select": (
            "DOI,title,author,published,"
            "published-print,published-online,"
            "issued,abstract,URL,container-title,"
            "is-referenced-by-count,type"
        ),
    }

    if mailto.strip():
        params[
            "mailto"
        ] = mailto.strip()

    headers = {
        "User-Agent": APP_USER_AGENT,
    }

    try:
        response = requests.get(
            CROSSREF_API_URL,
            params=params,
            headers=headers,
            timeout=(
                ACADEMIC_REQUEST_TIMEOUT_SECONDS
            ),
        )

        response.raise_for_status()

        payload = response.json()

    except (
        requests.RequestException,
        ValueError,
    ) as exc:
        raise AcademicSearchError(
            f"Crossref search failed: {exc}"
        ) from exc

    papers: list[AcademicPaper] = []

    items = (
        payload.get(
            "message",
            {},
        ).get(
            "items",
            [],
        )
    )

    for item in items:
        title = first_list_value(
            item.get(
                "title"
            )
        )

        if not title:
            continue

        authors: list[str] = []

        for author in (
            item.get("author")
            or []
        ):
            given_name = (
                compact_whitespace(
                    author.get(
                        "given"
                    )
                )
            )

            family_name = (
                compact_whitespace(
                    author.get(
                        "family"
                    )
                )
            )

            name = compact_whitespace(
                f"{given_name} {family_name}"
            )

            if name:
                authors.append(
                    name
                )

        doi = compact_whitespace(
            item.get("DOI")
        )

        papers.append(
            AcademicPaper(
                source="Crossref",
                source_id=(
                    doi
                    or compact_whitespace(
                        item.get(
                            "URL"
                        )
                    )
                ),
                title=title,
                authors=authors,
                year=(
                    year_from_date_parts(
                        item
                    )
                ),
                abstract=strip_html(
                    item.get(
                        "abstract"
                    )
                ),
                url=(
                    compact_whitespace(
                        item.get(
                            "URL"
                        )
                    )
                ),
                doi=doi,
                venue=(
                    first_list_value(
                        item.get(
                            "container-title"
                        )
                    )
                ),
                citation_count=(
                    item.get(
                        "is-referenced-by-count"
                    )
                ),
            )
        )

    return papers