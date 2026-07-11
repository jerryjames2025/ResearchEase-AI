from __future__ import annotations

from html import unescape
import re


def compact_whitespace(
    text: str | None,
) -> str:
    """
    Convert repeated spaces and line breaks into one space.
    """

    if not text:
        return ""

    return re.sub(
        r"\s+",
        " ",
        unescape(
            str(text)
        ),
    ).strip()


def strip_html(
    text: str | None,
) -> str:
    """
    Remove HTML tags from abstracts returned by Crossref.
    """

    if not text:
        return ""

    without_tags = re.sub(
        r"<[^>]+>",
        " ",
        str(text),
    )

    return compact_whitespace(
        without_tags
    )


def first_list_value(
    value,
) -> str:
    """
    Crossref commonly returns titles and venues as lists.
    """

    if isinstance(
        value,
        list,
    ) and value:
        return compact_whitespace(
            value[0]
        )

    return compact_whitespace(
        value
    )


def year_from_date_parts(
    item: dict,
) -> int | None:
    """
    Extract a publication year from Crossref date fields.
    """

    date_keys = (
        "published-print",
        "published-online",
        "published",
        "issued",
    )

    for key in date_keys:
        date_information = (
            item.get(key)
            or {}
        )

        date_parts = (
            date_information.get(
                "date-parts"
            )
            or []
        )

        if (
            date_parts
            and date_parts[0]
        ):
            try:
                return int(
                    date_parts[0][0]
                )

            except (
                TypeError,
                ValueError,
                IndexError,
            ):
                continue

    return None