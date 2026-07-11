from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import requests

from config import (
    ACADEMIC_REQUEST_TIMEOUT_SECONDS,
    APP_USER_AGENT,
    ARXIV_API_URL,
)
from research.models import AcademicPaper
from research.semantic_scholar import (
    AcademicSearchError,
)
from research.utils import compact_whitespace


ATOM = (
    "{http://www.w3.org/2005/Atom}"
)

ARXIV = (
    "{http://arxiv.org/schemas/atom}"
)


def _quoted_arxiv_query(
    query: str,
) -> str:
    """
    Convert the user query into an arXiv query.
    """

    cleaned = compact_whitespace(
        query
    ).replace(
        '"',
        " ",
    )

    terms = [
        term
        for term in cleaned.split()
        if term
    ]

    if not terms:
        return ""

    # Limit query size to avoid an excessively long URL.
    return " AND ".join(
        f'all:"{term}"'
        for term in terms[:12]
    )


def _extract_arxiv_id(
    entry_url: str,
) -> str:
    """
    Extract the arXiv identifier from its abstract URL.
    """

    match = re.search(
        r"/abs/([^/?#]+)",
        entry_url,
    )

    if match:
        return match.group(1)

    return entry_url


def search_arxiv(
    query: str,
    limit: int,
) -> list[AcademicPaper]:
    """
    Search arXiv and parse its Atom response.
    """

    search_query = (
        _quoted_arxiv_query(
            query
        )
    )

    if not search_query:
        return []

    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": max(
            1,
            min(
                int(limit),
                50,
            ),
        ),
        "sortBy": "relevance",
        "sortOrder": "descending",
    }

    headers = {
        "User-Agent": APP_USER_AGENT,
    }

    try:
        response = requests.get(
            ARXIV_API_URL,
            params=params,
            headers=headers,
            timeout=(
                ACADEMIC_REQUEST_TIMEOUT_SECONDS
            ),
        )

        response.raise_for_status()

        root = ET.fromstring(
            response.content
        )

    except (
        requests.RequestException,
        ET.ParseError,
    ) as exc:
        raise AcademicSearchError(
            f"arXiv search failed: {exc}"
        ) from exc

    papers: list[AcademicPaper] = []

    for entry in root.findall(
        f"{ATOM}entry"
    ):
        title = compact_whitespace(
            entry.findtext(
                f"{ATOM}title"
            )
        )

        if not title:
            continue

        entry_url = compact_whitespace(
            entry.findtext(
                f"{ATOM}id"
            )
        )

        published = compact_whitespace(
            entry.findtext(
                f"{ATOM}published"
            )
        )

        year = None

        if (
            len(published) >= 4
            and published[:4].isdigit()
        ):
            year = int(
                published[:4]
            )

        authors = [
            compact_whitespace(
                author.findtext(
                    f"{ATOM}name"
                )
            )
            for author
            in entry.findall(
                f"{ATOM}author"
            )
        ]

        authors = [
            author
            for author in authors
            if author
        ]

        doi = compact_whitespace(
            entry.findtext(
                f"{ARXIV}doi"
            )
        )

        journal_reference = (
            compact_whitespace(
                entry.findtext(
                    f"{ARXIV}journal_ref"
                )
            )
        )

        pdf_url = ""

        for link in entry.findall(
            f"{ATOM}link"
        ):
            if (
                link.attrib.get(
                    "title"
                )
                == "pdf"
            ):
                pdf_url = (
                    compact_whitespace(
                        link.attrib.get(
                            "href"
                        )
                    )
                )

                break

        papers.append(
            AcademicPaper(
                source="arXiv",
                source_id=(
                    _extract_arxiv_id(
                        entry_url
                    )
                ),
                title=title,
                authors=authors,
                year=year,
                abstract=(
                    compact_whitespace(
                        entry.findtext(
                            f"{ATOM}summary"
                        )
                    )
                ),
                url=entry_url,
                doi=doi,
                venue=(
                    journal_reference
                    or "arXiv preprint"
                ),
                publication_date=published,
                pdf_url=pdf_url,
            )
        )

    return papers