from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
)

from backend.core.errors import (
    APIError,
)
from backend.core.serializers import (
    serialize_external_paper,
    serialize_paper_sources,
)
from backend.core.session_store import (
    session_store,
)
from backend.core.settings import (
    APISettings,
    get_settings,
)
from backend.schemas import (
    AnswerMode,
    ChatRequest,
    ChatResponse,
)
from research.aggregator import (
    search_academic_sources,
)
from research.external_answer import (
    answer_with_external_research,
    retrieve_paper_sources,
)
from research.ranker import (
    rank_academic_papers,
)
from services.embedding_model import (
    load_embedding_service,
)
from services.rag_engine import (
    answer_question,
)


router = APIRouter(
    prefix="/chat",
    tags=["chat"],
)


@router.post(
    "/{session_id}",
    response_model=ChatResponse,
)
def research_chat(
    session_id: str,
    request: ChatRequest,
    settings: APISettings = Depends(
        get_settings
    ),
) -> ChatResponse:
    """
    Ask a paper-only or expanded-research question.
    """

    session = session_store.get(
        session_id
    )

    with session.lock:
        if session.vector_store is None:
            raise APIError(
                status_code=409,
                code="index_not_ready",
                detail=(
                    "Build the paper index before "
                    "using research chat."
                ),
            )

        vector_store = (
            session.vector_store
        )

        embedding_model = (
            session.embedding_model_name
        )

        embedding_device = (
            session.embedding_device
        )

        previous_history = list(
            session.chat_history
        )

    embedding_service = (
        load_embedding_service(
            embedding_model,
            embedding_device,
        )
    )

    warnings: list[str] = []
    external_papers = []

    if (
        request.answer_mode
        == AnswerMode.PAPER_ONLY
    ):
        result = answer_question(
            question=request.question,
            vector_store=vector_store,
            embedding_service=(
                embedding_service
            ),
            ollama_model=(
                request.ollama_model
            ),
            explanation_level=(
                request.explanation_level
            ),
            chat_history=(
                previous_history
            ),
            top_k=request.top_k,
            minimum_score=(
                request.minimum_score
            ),
        )

        answer = result.answer
        paper_sources = result.sources

    else:
        paper_sources = (
            retrieve_paper_sources(
                question=request.question,
                vector_store=vector_store,
                embedding_service=(
                    embedding_service
                ),
                top_k=request.top_k,
                minimum_score=(
                    request.minimum_score
                ),
            )
        )

        search_response = (
            search_academic_sources(
                query=request.question,
                limit_per_source=(
                    request
                    .external_results_per_source
                ),
                use_semantic_scholar=(
                    request
                    .use_semantic_scholar
                ),
                use_arxiv=(
                    request.use_arxiv
                ),
                use_crossref=(
                    request.use_crossref
                ),
                semantic_scholar_api_key=(
                    settings
                    .semantic_scholar_api_key
                ),
                crossref_mailto=(
                    settings.crossref_mailto
                ),
            )
        )

        warnings = (
            search_response.warnings
        )

        external_papers = (
            rank_academic_papers(
                query=request.question,
                papers=(
                    search_response.papers
                ),
                embedding_service=(
                    embedding_service
                ),
            )[:8]
        )

        result = (
            answer_with_external_research(
                question=request.question,
                paper_sources=(
                    paper_sources
                ),
                external_sources=(
                    external_papers
                ),
                ollama_model=(
                    request.ollama_model
                ),
                explanation_level=(
                    request.explanation_level
                ),
            )
        )

        answer = result.answer

    serialized_paper_sources = (
        serialize_paper_sources(
            paper_sources
        )
    )

    serialized_external = [
        serialize_external_paper(
            paper
        )
        for paper in external_papers
    ]

    with session.lock:
        session.chat_history.extend(
            [
                {
                    "role": "user",
                    "content": (
                        request.question
                    ),
                },
                {
                    "role": "assistant",
                    "content": answer,
                    "paper_sources": [
                        source.model_dump()
                        for source
                        in serialized_paper_sources
                    ],
                    "external_sources": [
                        source.model_dump()
                        for source
                        in serialized_external
                    ],
                },
            ]
        )

    return ChatResponse(
        session_id=session_id,
        answer_mode=(
            request.answer_mode
        ),
        answer=answer,
        paper_sources=(
            serialized_paper_sources
        ),
        external_sources=(
            serialized_external
        ),
        warnings=warnings,
    )