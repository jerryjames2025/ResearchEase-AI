from __future__ import annotations

from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
)

from reportlab.lib.units import mm

from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)


def _page_number(
    canvas,
    document,
) -> None:
    canvas.saveState()

    canvas.setFont(
        "Helvetica",
        9,
    )

    canvas.drawRightString(
        A4[0] - 18 * mm,
        12 * mm,
        f"Page {document.page}",
    )

    canvas.restoreState()


def _plain(
    text: str,
) -> str:
    return "".join(
        character
        for character in str(text)
        if (
            character in "\n\t"
            or ord(character) >= 32
        )
    )


def _add_text(
    story: list,
    text: str,
    style,
) -> None:
    cleaned = _plain(
        text
    )

    paragraphs = (
        cleaned.split(
            "\n\n"
        )
        if cleaned
        else [""]
    )

    for paragraph in paragraphs:
        if not paragraph.strip():
            story.append(
                Spacer(
                    1,
                    5,
                )
            )

            continue

        story.append(
            Paragraph(
                escape(
                    paragraph
                ).replace(
                    "\n",
                    "<br/>",
                ),
                style,
            )
        )

        story.append(
            Spacer(
                1,
                4,
            )
        )


def build_text_report(
    filename: str,
    page_count: int,
    explanation_level: str,
    analysis: str,
    chat_history: list[dict],
    external_history: list[dict],
    math_history: list[dict] | None = None,
) -> str:
    """
    Create a combined Version 4 text report.
    """

    math_history = (
        math_history
        or []
    )

    lines = [
        "ResearchEase AI — Version 4 Report",
        "===================================",
        "",
        f"File: {filename}",
        f"Pages: {page_count}",
        (
            "Explanation level: "
            f"{explanation_level}"
        ),
        "",
        "STRUCTURED PAPER ANALYSIS",
        "-------------------------",
        (
            analysis
            or (
                "No complete-paper analysis "
                "was generated."
            )
        ),
        "",
        "RESEARCH CHAT HISTORY",
        "---------------------",
    ]

    if not chat_history:
        lines.append(
            "No research chat questions were asked."
        )

    for message in chat_history:
        role = str(
            message.get(
                "role",
                "",
            )
        ).upper()

        lines.extend(
            [
                f"{role}:",
                str(
                    message.get(
                        "content",
                        "",
                    )
                ),
                "",
            ]
        )

        for source in message.get(
            "paper_sources",
            [],
        ):
            lines.append(
                (
                    "Uploaded paper source: "
                    f"Page {source['page_number']} "
                    "(similarity "
                    f"{source['score']:.3f})"
                )
            )

        for index, paper in enumerate(
            message.get(
                "external_sources",
                [],
            ),
            start=1,
        ):
            lines.append(
                (
                    f"External source {index}: "
                    f"{paper['title']} | "
                    f"{paper['source']} | "
                    f"{paper.get('url', '')}"
                )
            )

        lines.append("")

    lines.extend(
        [
            "",
            "EXTERNAL SEARCH HISTORY",
            "-----------------------",
        ]
    )

    if not external_history:
        lines.append(
            "No separate external searches "
            "were performed."
        )

    for search in external_history:
        lines.append(
            (
                "Query: "
                f"{search.get('query', '')}"
            )
        )

        for index, paper in enumerate(
            search.get(
                "results",
                [],
            ),
            start=1,
        ):
            year = (
                paper.get("year")
                or "Year unavailable"
            )

            lines.append(
                (
                    f"{index}. {paper['title']} "
                    f"({year}) — "
                    f"{paper['source']} — "
                    f"{paper.get('url', '')}"
                )
            )

        lines.append("")

    lines.extend(
        [
            "",
            "MATHEMATICS EXPLANATION HISTORY",
            "-------------------------------",
        ]
    )

    if not math_history:
        lines.append(
            "No mathematical explanations "
            "were generated."
        )

    for index, item in enumerate(
        math_history,
        start=1,
    ):
        lines.append(
            f"Math item {index}"
        )

        lines.append(
            (
                "Mode: "
                f"{item.get('mode', '')}"
            )
        )

        if (
            item.get("mode")
            == "equation"
        ):
            lines.append(
                (
                    "Input: "
                    f"{item.get('original_input', '')}"
                )
            )

            lines.append(
                (
                    "Operation: "
                    f"{item.get('operation', '')}"
                )
            )

            lines.append(
                (
                    "Parsed expression: "
                    f"{item.get('parsed_text', '')}"
                )
            )

            lines.append(
                (
                    "Verified result: "
                    f"{item.get('result_text', '')}"
                )
            )

            if item.get(
                "verification"
            ):
                lines.append(
                    (
                        "Verification: "
                        f"{item['verification']}"
                    )
                )

        else:
            lines.append(
                (
                    "Topic: "
                    f"{item.get('topic', '')}"
                )
            )

        lines.append(
            "Explanation:"
        )

        lines.append(
            str(
                item.get(
                    "explanation",
                    "",
                )
            )
        )

        for source in item.get(
            "paper_sources",
            [],
        ):
            lines.append(
                (
                    "Paper context: "
                    f"Page {source['page_number']} "
                    "(similarity "
                    f"{source['score']:.3f})"
                )
            )

        lines.append("")

    return "\n".join(
        lines
    )


def build_pdf_report(
    filename: str,
    page_count: int,
    explanation_level: str,
    analysis: str,
    chat_history: list[dict],
    external_history: list[dict],
    math_history: list[dict] | None = None,
) -> bytes:
    """
    Create a combined Version 4 PDF report.
    """

    math_history = (
        math_history
        or []
    )

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title=(
            "ResearchEase AI Version 4 Report"
        ),
        author="ResearchEase AI",
    )

    styles = getSampleStyleSheet()

    story: list = [
        Paragraph(
            "ResearchEase AI — Version 4 Report",
            styles["Title"],
        ),
        Paragraph(
            (
                f"File: {escape(filename)}"
                f"<br/>Pages: {page_count}"
                "<br/>Explanation level: "
                f"{escape(explanation_level)}"
            ),
            styles["BodyText"],
        ),
        PageBreak(),
        Paragraph(
            "Structured Paper Analysis",
            styles["Heading1"],
        ),
    ]

    _add_text(
        story,
        (
            analysis
            or (
                "No complete-paper analysis "
                "was generated."
            )
        ),
        styles["BodyText"],
    )

    story.extend(
        [
            PageBreak(),
            Paragraph(
                "Research Chat History",
                styles["Heading1"],
            ),
        ]
    )

    if not chat_history:
        story.append(
            Paragraph(
                (
                    "No research chat "
                    "questions were asked."
                ),
                styles["BodyText"],
            )
        )

    for message in chat_history:
        role = (
            "Question"
            if message.get("role") == "user"
            else "Answer"
        )

        story.append(
            Paragraph(
                role,
                styles["Heading2"],
            )
        )

        _add_text(
            story,
            str(
                message.get(
                    "content",
                    "",
                )
            ),
            styles["BodyText"],
        )

        for source in message.get(
            "paper_sources",
            [],
        ):
            story.append(
                Paragraph(
                    (
                        "Uploaded paper source: "
                        f"Page {source['page_number']} "
                        "(similarity "
                        f"{source['score']:.3f})"
                    ),
                    styles["BodyText"],
                )
            )

        for index, paper in enumerate(
            message.get(
                "external_sources",
                [],
            ),
            start=1,
        ):
            story.append(
                Paragraph(
                    (
                        f"External source {index}: "
                        f"{escape(paper['title'])} — "
                        f"{escape(paper['source'])}"
                    ),
                    styles["BodyText"],
                )
            )

    story.extend(
        [
            PageBreak(),
            Paragraph(
                "External Search History",
                styles["Heading1"],
            ),
        ]
    )

    if not external_history:
        story.append(
            Paragraph(
                (
                    "No external searches "
                    "were performed."
                ),
                styles["BodyText"],
            )
        )

    for search in external_history:
        story.append(
            Paragraph(
                (
                    "Query: "
                    f"{escape(str(search.get('query', '')))}"
                ),
                styles["Heading2"],
            )
        )

        for index, paper in enumerate(
            search.get(
                "results",
                [],
            ),
            start=1,
        ):
            year = (
                paper.get("year")
                or "Year unavailable"
            )

            story.append(
                Paragraph(
                    (
                        f"{index}. "
                        f"{escape(paper['title'])} — "
                        f"{escape(paper['source'])} "
                        f"({year})"
                    ),
                    styles["BodyText"],
                )
            )

    story.extend(
        [
            PageBreak(),
            Paragraph(
                "Mathematics Explanation History",
                styles["Heading1"],
            ),
        ]
    )

    if not math_history:
        story.append(
            Paragraph(
                (
                    "No mathematical explanations "
                    "were generated."
                ),
                styles["BodyText"],
            )
        )

    for index, item in enumerate(
        math_history,
        start=1,
    ):
        label = (
            item.get("topic")
            or item.get(
                "original_input"
            )
            or f"Math item {index}"
        )

        story.append(
            Paragraph(
                (
                    f"Math item {index}: "
                    f"{escape(str(label))}"
                ),
                styles["Heading2"],
            )
        )

        if (
            item.get("mode")
            == "equation"
        ):
            details = (
                "Operation: "
                f"{escape(str(item.get('operation', '')))}"
                "<br/>Parsed expression: "
                f"{escape(str(item.get('parsed_text', '')))}"
                "<br/>Verified result: "
                f"{escape(str(item.get('result_text', '')))}"
            )

            story.append(
                Paragraph(
                    details,
                    styles["BodyText"],
                )
            )

            if item.get(
                "verification"
            ):
                story.append(
                    Paragraph(
                        (
                            "Verification: "
                            f"{escape(str(item['verification']))}"
                        ),
                        styles["BodyText"],
                    )
                )

        _add_text(
            story,
            str(
                item.get(
                    "explanation",
                    "",
                )
            ),
            styles["BodyText"],
        )

        for source in item.get(
            "paper_sources",
            [],
        ):
            story.append(
                Paragraph(
                    (
                        "Paper context: "
                        f"Page {source['page_number']} "
                        "(similarity "
                        f"{source['score']:.3f})"
                    ),
                    styles["BodyText"],
                )
            )

    document.build(
        story,
        onFirstPage=_page_number,
        onLaterPages=_page_number,
    )

    return buffer.getvalue()