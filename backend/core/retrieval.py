from __future__ import annotations

from backend.core.errors import APIError
from backend.core.session_store import (
    ResearchSession,
)
from services.embedding_model import (
    load_embedding_service,
)
from services.vector_store import (
    SearchResult,
)


def retrieve_session_context(
    session: ResearchSession,
    query: str,
    top_k: int,
    minimum_score: float,
) -> tuple[
    str,
    list[SearchResult],
]:
    """
    Retrieve page-aware evidence from a session's
    FAISS index.
    """

    if session.vector_store is None:
        raise APIError(
            status_code=409,
            code="index_not_ready",
            detail=(
                "Build the paper RAG index "
                "before requesting paper context."
            ),
        )

    embedding_service = (
        load_embedding_service(
            session.embedding_model_name,
            session.embedding_device,
        )
    )

    query_embedding = (
        embedding_service.encode_query(
            query
        )
    )

    results = session.vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
    )

    accepted = [
        result
        for result in results
        if result.score >= minimum_score
    ]

    context = "\n\n".join(
        (
            "[Uploaded Paper, "
            f"Page {result.chunk.page_number}]\n"
            f"{result.chunk.text}"
        )
        for result in accepted
    )

    return context, accepted