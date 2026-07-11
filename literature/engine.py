from __future__ import annotations

from langchain_layer.chains import (
    run_literature_chunk_chain,
    run_literature_record_chain,
    run_literature_synthesis_chain,
)

import json
import re
from dataclasses import replace
from typing import Callable

from config import (
    LITERATURE_CHUNK_OVERLAP,
    LITERATURE_CHUNK_SIZE,
    MAX_LITERATURE_CHUNKS_PER_PAPER,
)
from literature.models import LiteraturePaperSummary
from research.models import AcademicPaper
from services.ollama_client import chat
from services.pdf_parser import (
    ExtractedPaper,
    split_text,
)


class LiteratureReviewError(RuntimeError):
    """
    Raised when a literature-review operation
    cannot be completed.
    """


SUMMARY_FIELDS = (
    "title",
    "authors",
    "year",
    "research_problem",
    "objective",
    "methodology",
    "dataset",
    "models_or_methods",
    "evaluation_metrics",
    "main_findings",
    "limitations",
    "future_work",
    "keywords",
)


def _extract_json_object(
    text: str,
) -> dict:
    """
    Extract a JSON object even when the model wraps
    the response in a Markdown code block.
    """

    cleaned = text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        value = json.loads(
            cleaned
        )

        if isinstance(
            value,
            dict,
        ):
            return value

    except json.JSONDecodeError:
        pass

    start = cleaned.find(
        "{"
    )

    end = cleaned.rfind(
        "}"
    )

    if (
        start == -1
        or end == -1
        or end <= start
    ):
        raise LiteratureReviewError(
            "The model did not return a valid "
            "JSON paper summary."
        )

    try:
        value = json.loads(
            cleaned[
                start:
                end + 1
            ]
        )

    except json.JSONDecodeError as exc:
        raise LiteratureReviewError(
            "Unable to parse the paper summary JSON: "
            f"{exc}"
        ) from exc

    if not isinstance(
        value,
        dict,
    ):
        raise LiteratureReviewError(
            "The paper summary response was not "
            "a JSON object."
        )

    return value


def _clean_field(
    value,
    fallback: str = "Not identified",
) -> str:
    """
    Convert returned values into compact text.
    """

    if value is None:
        return fallback

    if isinstance(
        value,
        list,
    ):
        value = "; ".join(
            str(item)
            for item in value
            if str(item).strip()
        )

    if isinstance(
        value,
        dict,
    ):
        value = "; ".join(
            f"{key}: {item}"
            for key, item in value.items()
            if str(item).strip()
        )

    result = re.sub(
        r"\s+",
        " ",
        str(value),
    ).strip()

    return (
        result
        or fallback
    )


def _summary_from_payload(
    payload: dict,
    citation_key: str,
    source_type: str,
    evidence_scope: str,
    fallback_title: str,
    source_url: str = "",
    doi: str = "",
    notes: list[str] | None = None,
) -> LiteraturePaperSummary:
    """
    Convert model JSON into a normalized paper summary.
    """

    normalized = {
        field: _clean_field(
            payload.get(
                field
            )
        )
        for field in SUMMARY_FIELDS
    }

    if (
        normalized["title"]
        == "Not identified"
    ):
        normalized[
            "title"
        ] = fallback_title

    return LiteraturePaperSummary(
        citation_key=citation_key,
        title=normalized["title"],
        authors=normalized["authors"],
        year=normalized["year"],
        source_type=source_type,
        evidence_scope=evidence_scope,
        research_problem=normalized[
            "research_problem"
        ],
        objective=normalized[
            "objective"
        ],
        methodology=normalized[
            "methodology"
        ],
        dataset=normalized[
            "dataset"
        ],
        models_or_methods=normalized[
            "models_or_methods"
        ],
        evaluation_metrics=normalized[
            "evaluation_metrics"
        ],
        main_findings=normalized[
            "main_findings"
        ],
        limitations=normalized[
            "limitations"
        ],
        future_work=normalized[
            "future_work"
        ],
        keywords=normalized[
            "keywords"
        ],
        source_url=source_url,
        doi=doi,
        notes=notes or [],
    )


def _chunk_notes_prompt(
    text: str,
    chunk_number: int,
    total_chunks: int,
) -> str:
    """
    Prompt for extracting evidence from one paper section.
    """

    return f"""
You are extracting evidence for a literature review.

Analyze section {chunk_number} of {total_chunks}
from one research paper.

Use only this section.

Do not invent missing details.

Return concise evidence notes under these labels
when information is available:

TITLE OR AUTHORS
RESEARCH PROBLEM
OBJECTIVE
METHODOLOGY
DATASET
MODELS OR METHODS
EVALUATION METRICS
MAIN FINDINGS
LIMITATIONS
FUTURE WORK
KEYWORDS
PAGE REFERENCES

PAPER SECTION:

{text}
""".strip()


def _final_summary_prompt(
    filename: str,
    chunk_notes: list[str],
) -> str:
    """
    Prompt for converting section notes into
    a structured JSON paper record.
    """

    joined_notes = "\n\n".join(
        f"SECTION {index}\n{note}"
        for index, note in enumerate(
            chunk_notes,
            start=1,
        )
    )

    return f"""
Create a structured literature-review record
for the paper:

{filename}

Use only the extracted evidence notes below.

Do not invent:

- authors
- publication dates
- datasets
- methodologies
- models
- evaluation metrics
- findings
- limitations
- future work

Return exactly one valid JSON object with
these string fields:

{{
  "title": "",
  "authors": "",
  "year": "",
  "research_problem": "",
  "objective": "",
  "methodology": "",
  "dataset": "",
  "models_or_methods": "",
  "evaluation_metrics": "",
  "main_findings": "",
  "limitations": "",
  "future_work": "",
  "keywords": ""
}}

For unsupported fields, write:

"Not identified"

Do not use Markdown.
Do not use a code fence.
Return JSON only.

EXTRACTED EVIDENCE NOTES:

{joined_notes}
""".strip()


def summarize_full_text_paper(
    paper: ExtractedPaper,
    citation_key: str,
    ollama_model: str,
    progress_callback: (
        Callable[
            [
                int,
                int,
                str,
            ],
            None,
        ]
        | None
    ) = None,
) -> LiteraturePaperSummary:
    """
    Create a structured literature-review record
    from a complete uploaded PDF using LangChain.
    """

    chunks = split_text(
        paper.text,

        chunk_size=(
            LITERATURE_CHUNK_SIZE
        ),

        overlap=(
            LITERATURE_CHUNK_OVERLAP
        ),
    )

    chunks = chunks[
        :MAX_LITERATURE_CHUNKS_PER_PAPER
    ]

    if not chunks:
        raise LiteratureReviewError(
            "No analyzable text was found in "
            f"{paper.filename}."
        )

    notes: list[str] = []

    total_steps = (
        len(chunks)
        + 1
    )

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        if progress_callback:
            progress_callback(
                index - 1,

                total_steps,

                (
                    f"Analyzing {paper.filename}: "
                    f"section {index} "
                    f"of {len(chunks)}"
                ),
            )

        notes.append(
            run_literature_chunk_chain(
                text=chunk,

                chunk_number=index,

                total_chunks=(
                    len(chunks)
                ),

                model_name=(
                    ollama_model
                ),
            )
        )

    if progress_callback:
        progress_callback(
            len(chunks),

            total_steps,

            (
                "Creating the structured "
                "record for "
                f"{paper.filename}"
            ),
        )

    evidence = "\n\n".join(
        f"SECTION {index}\n{note}"
        for index, note in enumerate(
            notes,
            start=1,
        )
    )

    record = run_literature_record_chain(
        source_name=(
            paper.filename
        ),

        evidence_scope=(
            "Full extracted text"
        ),

        evidence=evidence,

        model_name=(
            ollama_model
        ),
    )

    if progress_callback:
        progress_callback(
            total_steps,

            total_steps,

            f"Completed {paper.filename}",
        )

    return _summary_from_payload(
        payload=(
            record.model_dump()
        ),

        citation_key=(
            citation_key
        ),

        source_type=(
            "Uploaded PDF"
        ),

        evidence_scope=(
            "Full extracted text"
        ),

        fallback_title=(
            paper.filename
        ),

        notes=[
            (
                "Extracted from "
                f"{paper.page_count} "
                "PDF pages."
            )
        ],
    )

def summarize_external_paper(
    paper: AcademicPaper,
    citation_key: str,
    ollama_model: str,
) -> LiteraturePaperSummary:
    """
    Create a Pydantic-validated record
    from external metadata and abstract evidence.
    """

    record = run_literature_record_chain(
        source_name=(
            paper.title
        ),

        evidence_scope=(
            "Abstract and metadata only"
        ),

        evidence=(
            paper.evidence_text()
        ),

        model_name=(
            ollama_model
        ),
    )

    return _summary_from_payload(
        payload=(
            record.model_dump()
        ),

        citation_key=(
            citation_key
        ),

        source_type=(
            paper.source
        ),

        evidence_scope=(
            "Abstract and metadata only"
        ),

        fallback_title=(
            paper.title
        ),

        source_url=(
            paper.url
        ),

        doi=paper.doi,

        notes=[
            (
                "The complete external paper "
                "was not analyzed."
            )
        ],
    )

def _literature_context(
    summaries: list[
        LiteraturePaperSummary
    ],
) -> str:
    """
    Format normalized summaries for cross-paper synthesis.
    """

    blocks: list[str] = []

    for item in summaries:
        blocks.append(
            f"""
[{item.citation_key}]

Title:
{item.title}

Authors:
{item.authors}

Year:
{item.year}

Evidence scope:
{item.evidence_scope}

Research problem:
{item.research_problem}

Objective:
{item.objective}

Methodology:
{item.methodology}

Dataset:
{item.dataset}

Models or methods:
{item.models_or_methods}

Evaluation metrics:
{item.evaluation_metrics}

Main findings:
{item.main_findings}

Limitations:
{item.limitations}

Future work:
{item.future_work}

Keywords:
{item.keywords}
""".strip()
        )

    return "\n\n".join(
        blocks
    )


def generate_literature_synthesis(
    topic: str,
    summaries: list[
        LiteraturePaperSummary
    ],
    ollama_model: str,
    explanation_level: str,
) -> str:
    """
    Generate an evidence-grounded cross-paper
    review through LangChain.
    """

    if len(summaries) < 2:
        raise LiteratureReviewError(
            "At least two paper summaries are "
            "required for comparison."
        )

    return run_literature_synthesis_chain(
        topic=topic,

        explanation_level=(
            explanation_level
        ),

        paper_records=(
            _literature_context(
                summaries
            )
        ),

        model_name=(
            ollama_model
        ),
    )

def reassign_citation_keys(
    summaries: list[
        LiteraturePaperSummary
    ],
) -> list[
    LiteraturePaperSummary
]:
    """
    Ensure citation keys remain P1, P2, P3...
    """

    return [
        replace(
            item,
            citation_key=f"P{index}",
        )
        for index, item in enumerate(
            summaries,
            start=1,
        )
    ]