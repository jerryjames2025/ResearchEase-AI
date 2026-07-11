from __future__ import annotations

from dataclasses import dataclass

from config import (
    MAX_CONTEXT_CHARACTERS,
)

from langchain_layer.chains import (
    build_paper_rag_chain,
)

from langchain_layer.memory import (
    history_to_messages,
)

from services.embedding_model import (
    EmbeddingService,
)

from services.vector_store import (
    FaissVectorStore,
    SearchResult,
)


@dataclass(frozen=True)
class RagAnswer:
    """
    Paper-only RAG response.
    """

    answer: str
    sources: list[SearchResult]
    retrieval_blocked: bool


def answer_question(
    question: str,
    vector_store: FaissVectorStore,
    embedding_service: EmbeddingService,
    ollama_model: str,
    explanation_level: str,
    chat_history: list[dict],
    top_k: int,
    minimum_score: float,
) -> RagAnswer:
    """
    Answer a paper question through an
    LCEL retrieval chain.

    The function signature remains unchanged,
    so the existing Streamlit app continues working.
    """

    question = question.strip()

    if not question:
        raise ValueError(
            "Question cannot be empty."
        )

    chain = build_paper_rag_chain(
        vector_store=vector_store,

        embedding_service=(
            embedding_service
        ),

        model_name=ollama_model,

        top_k=top_k,

        minimum_score=(
            minimum_score
        ),

        max_context_characters=(
            MAX_CONTEXT_CHARACTERS
        ),
    )

    result = chain.invoke(
        {
            "question": question,

            "explanation_level": (
                explanation_level
            ),

            "history": (
                history_to_messages(
                    chat_history
                )
            ),
        }
    )

    return RagAnswer(
        answer=str(
            result[
                "answer"
            ]
        ),

        sources=list(
            result[
                "sources"
            ]
        ),

        retrieval_blocked=bool(
            result[
                "retrieval_blocked"
            ]
        ),
    )