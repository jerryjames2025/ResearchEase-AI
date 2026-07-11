from __future__ import annotations

from dataclasses import dataclass

from config import (
    MAX_EXTERNAL_CONTEXT_CHARACTERS,
)

from langchain_layer.chains import (
    run_expanded_research_chain,
)

from research.models import (
    AcademicPaper,
)

from services.embedding_model import (
    EmbeddingService,
)

from services.vector_store import (
    FaissVectorStore,
    SearchResult,
)


@dataclass(frozen=True)
class ExpandedAnswer:
    """
    Expanded answer and its two evidence categories.
    """

    answer: str
    paper_sources: list[SearchResult]
    external_sources: list[AcademicPaper]


def retrieve_paper_sources(
    question: str,
    vector_store: FaissVectorStore,
    embedding_service: EmbeddingService,
    top_k: int,
    minimum_score: float,
) -> list[SearchResult]:
    """
    Retrieve relevant uploaded-paper sections.
    """

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    query_embedding = (
        embedding_service.encode_query(
            question
        )
    )

    results = vector_store.search(
        query_embedding=(
            query_embedding
        ),
        top_k=top_k,
    )

    return [
        result
        for result in results
        if (
            result.score
            >= minimum_score
        )
    ]


def _build_paper_context(
    sources: list[SearchResult],
) -> str:
    """
    Format uploaded-paper evidence.
    """

    if not sources:
        return (
            "No sufficiently relevant "
            "uploaded-paper sections were found."
        )

    return "\n\n".join(
        (
            "[UPLOADED PAPER | "
            f"PAGE {source.chunk.page_number} | "
            f"SIMILARITY {source.score:.3f}]\n"
            f"{source.chunk.text.strip()}"
        )
        for source in sources
    )


def _build_external_context(
    papers: list[AcademicPaper],
) -> str:
    """
    Format external abstract-level evidence.
    """

    if not papers:
        return (
            "No external academic records "
            "were found."
        )

    blocks: list[str] = []
    used_characters = 0

    for index, paper in enumerate(
        papers,
        start=1,
    ):
        block = (
            f"[EXTERNAL SOURCE {index} | "
            f"{paper.source}]\n"
            f"{paper.evidence_text()}"
        )

        remaining = (
            MAX_EXTERNAL_CONTEXT_CHARACTERS
            - used_characters
        )

        if remaining <= 0:
            break

        if len(block) > remaining:
            block = block[
                :remaining
            ]

        blocks.append(
            block
        )

        used_characters += len(
            block
        )

    return "\n\n".join(
        blocks
    )


def answer_with_external_research(
    question: str,
    paper_sources: list[SearchResult],
    external_sources: list[AcademicPaper],
    ollama_model: str,
    explanation_level: str,
) -> ExpandedAnswer:
    """
    Generate a source-separated answer
    through LangChain.
    """

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    answer = run_expanded_research_chain(
        question=question,

        paper_context=(
            _build_paper_context(
                paper_sources
            )
        ),

        external_context=(
            _build_external_context(
                external_sources
            )
        ),

        explanation_level=(
            explanation_level
        ),

        model_name=ollama_model,
    )

    return ExpandedAnswer(
        answer=answer,

        paper_sources=(
            paper_sources
        ),

        external_sources=(
            external_sources
        ),
    )