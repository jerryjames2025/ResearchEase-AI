from __future__ import annotations

import csv
from io import (
    BytesIO,
    StringIO,
)
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import (
    A4,
    landscape,
)
from reportlab.lib.styles import (
    getSampleStyleSheet,
)
from reportlab.lib.units import mm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from literature.models import (
    LiteraturePaperSummary,
)


def build_literature_csv(
    summaries: list[
        LiteraturePaperSummary
    ],
) -> str:
    """
    Export the comparison matrix as CSV.
    """

    buffer = StringIO(
        newline=""
    )

    rows = [
        item.matrix_row()
        for item in summaries
    ]

    if not rows:
        return ""

    writer = csv.DictWriter(
        buffer,
        fieldnames=list(
            rows[0].keys()
        ),
    )

    writer.writeheader()
    writer.writerows(
        rows
    )

    return buffer.getvalue()


def build_literature_text(
    topic: str,
    summaries: list[
        LiteraturePaperSummary
    ],
    synthesis: str,
) -> str:
    """
    Export the literature review as plain text.
    """

    lines = [
        "ResearchEase AI — Literature Review",
        "====================================",
        "",
        (
            "Topic: "
            f"{topic or 'Not supplied'}"
        ),
        (
            "Papers compared: "
            f"{len(summaries)}"
        ),
        "",
        "LITERATURE MATRIX",
        "-----------------",
    ]

    for item in summaries:
        lines.extend(
            [
                (
                    f"{item.citation_key}: "
                    f"{item.title}"
                ),
                (
                    "Authors: "
                    f"{item.authors}"
                ),
                (
                    "Year: "
                    f"{item.year}"
                ),
                (
                    "Evidence: "
                    f"{item.evidence_scope}"
                ),
                (
                    "Research problem: "
                    f"{item.research_problem}"
                ),
                (
                    "Objective: "
                    f"{item.objective}"
                ),
                (
                    "Methodology: "
                    f"{item.methodology}"
                ),
                (
                    "Dataset: "
                    f"{item.dataset}"
                ),
                (
                    "Models or methods: "
                    f"{item.models_or_methods}"
                ),
                (
                    "Metrics: "
                    f"{item.evaluation_metrics}"
                ),
                (
                    "Main findings: "
                    f"{item.main_findings}"
                ),
                (
                    "Limitations: "
                    f"{item.limitations}"
                ),
                (
                    "Future work: "
                    f"{item.future_work}"
                ),
                (
                    "Keywords: "
                    f"{item.keywords}"
                ),
                "",
            ]
        )

    lines.extend(
        [
            "LITERATURE SYNTHESIS",
            "--------------------",
            (
                synthesis
                or "No synthesis was generated."
            ),
            "",
        ]
    )

    return "\n".join(
        lines
    )


def _page_number(
    canvas,
    document,
) -> None:
    """
    Add page numbers to the landscape PDF.
    """

    canvas.saveState()

    canvas.setFont(
        "Helvetica",
        8,
    )

    canvas.drawRightString(
        landscape(A4)[0]
        - 12 * mm,
        8 * mm,
        f"Page {document.page}",
    )

    canvas.restoreState()


def build_literature_pdf(
    topic: str,
    summaries: list[
        LiteraturePaperSummary
    ],
    synthesis: str,
) -> bytes:
    """
    Export the matrix and synthesis as a landscape PDF.
    """

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(
            A4
        ),
        rightMargin=10 * mm,
        leftMargin=10 * mm,
        topMargin=10 * mm,
        bottomMargin=12 * mm,
        title=(
            "ResearchEase AI "
            "Literature Review"
        ),
        author="ResearchEase AI",
    )

    styles = getSampleStyleSheet()

    small = styles[
        "BodyText"
    ]

    small.fontSize = 7
    small.leading = 8

    story = [
        Paragraph(
            (
                "ResearchEase AI — "
                "Literature Review"
            ),
            styles["Title"],
        ),
        Paragraph(
            (
                "<b>Topic:</b> "
                f"{escape(topic or 'Not supplied')}"
                "<br/><b>Papers compared:</b> "
                f"{len(summaries)}"
            ),
            styles["BodyText"],
        ),
        Spacer(
            1,
            6,
        ),
    ]

    table_headers = [
        "ID",
        "Title",
        "Year",
        "Evidence",
        "Methodology",
        "Dataset",
        "Main findings",
        "Limitations",
    ]

    table_data = [
        [
            Paragraph(
                header,
                small,
            )
            for header
            in table_headers
        ]
    ]

    for item in summaries:
        row = item.matrix_row()

        table_data.append(
            [
                Paragraph(
                    escape(
                        row["ID"]
                    ),
                    small,
                ),
                Paragraph(
                    escape(
                        row["Title"]
                    ),
                    small,
                ),
                Paragraph(
                    escape(
                        row["Year"]
                    ),
                    small,
                ),
                Paragraph(
                    escape(
                        row["Evidence"]
                    ),
                    small,
                ),
                Paragraph(
                    escape(
                        row["Methodology"]
                    ),
                    small,
                ),
                Paragraph(
                    escape(
                        row["Dataset"]
                    ),
                    small,
                ),
                Paragraph(
                    escape(
                        row["Main findings"]
                    ),
                    small,
                ),
                Paragraph(
                    escape(
                        row["Limitations"]
                    ),
                    small,
                ),
            ]
        )

    table = Table(
        table_data,
        repeatRows=1,
        colWidths=[
            12 * mm,
            40 * mm,
            14 * mm,
            30 * mm,
            45 * mm,
            35 * mm,
            48 * mm,
            42 * mm,
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "GRID",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    0.25,
                    colors.grey,
                ),
                (
                    "VALIGN",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    "TOP",
                ),
                (
                    "BACKGROUND",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        0,
                    ),
                    colors.HexColor(
                        "#E8E8E8"
                    ),
                ),
                (
                    "LEFTPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    3,
                ),
                (
                    "RIGHTPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    3,
                ),
                (
                    "TOPPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    3,
                ),
                (
                    "BOTTOMPADDING",
                    (
                        0,
                        0,
                    ),
                    (
                        -1,
                        -1,
                    ),
                    3,
                ),
            ]
        )
    )

    story.extend(
        [
            table,
            PageBreak(),
            Paragraph(
                "Literature Synthesis",
                styles["Heading1"],
            ),
        ]
    )

    synthesis_text = (
        synthesis
        or "No synthesis was generated."
    )

    for paragraph in synthesis_text.split(
        "\n\n"
    ):
        story.append(
            Paragraph(
                escape(
                    paragraph
                ).replace(
                    "\n",
                    "<br/>",
                ),
                styles["BodyText"],
            )
        )

        story.append(
            Spacer(
                1,
                5,
            )
        )

    document.build(
        story,
        onFirstPage=_page_number,
        onLaterPages=_page_number,
    )

    return buffer.getvalue()