from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
)

from backend.core.academic_search import (
    search_and_rank_academic_sources,
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
from backend.llm.dependency import (
    LLMRequestConfig,
    get_llm_request_config,
)

from backend.schemas import (
    AnswerMode,
    ChatHistoryResponse,
    ChatRequest,
    ChatResponse,
    ChatTurnResponse,
    ExternalPaperResponse,
    PaperSourceResponse,
)

from research.external_answer import (
    answer_with_external_research,
    retrieve_paper_sources,
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


@router.get(
    "/{session_id}/history",
    response_model=ChatHistoryResponse,
)
def get_chat_history(
    session_id: str,
) -> ChatHistoryResponse:

    session_store.get(
        session_id
    )

    messages = (
        session_store
        .get_chat_messages(
            session_id
        )
    )

    turns: list[
        ChatTurnResponse
    ] = []

    pending_question = ""

    for message in messages:
        if message.get(
            "role"
        ) == "user":

            pending_question = str(
                message.get(
                    "content",
                    "",
                )
            )

            continue

        if message.get(
            "role"
        ) != "assistant":
            continue

        turns.append(
            ChatTurnResponse(
                question=(
                    pending_question
                ),
                answer=str(
                    message.get(
                        "content",
                        "",
                    )
                ),
                paper_sources=[
                    PaperSourceResponse
                    .model_validate(
                        item
                    )
                    for item
                    in message.get(
                        "paper_sources",
                        [],
                    )
                ],
                external_sources=[
                    ExternalPaperResponse
                    .model_validate(
                        item
                    )
                    for item
                    in message.get(
                        "external_sources",
                        [],
                    )
                ],
                answer_mode=str(
                    message.get(
                        "metadata",
                        {},
                    ).get(
                        "answer_mode",
                        "paper_only",
                    )
                ),
            )
        )

        pending_question = ""

    return ChatHistoryResponse(
        session_id=session_id,
        turns=turns,
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
    llm: LLMRequestConfig = Depends(
        get_llm_request_config
    ),
) -> ChatResponse:

    session = session_store.get(
        session_id
    )

    with session.lock:
        if session.vector_store is None:
            raise APIError(
                status_code=409,
                code="index_not_ready",
                detail=(
                    "Build the paper index "
                    "before using research chat."
                ),
            )

        vector_store = (
            session.vector_store
        )

        embedding_model = (
            session
            .embedding_model_name
        )

        embedding_device = (
            session
            .embedding_device
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
    cache_hit = False
    
    runtime_model = llm.model_spec(
    fallback_model=(
        request.ollama_model
    )
)

    if (
        request.answer_mode
        == AnswerMode.PAPER_ONLY
    ):
        result = answer_question(
            question=request.question,
            vector_store=(
                vector_store
            ),
            embedding_service=(
                embedding_service
            ),
            ollama_model=(
            runtime_model
            ),
            explanation_level=(
                request
                .explanation_level
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
                question=(
                    request.question
                ),
                vector_store=(
                    vector_store
                ),
                embedding_service=(
                    embedding_service
                ),
                top_k=request.top_k,
                minimum_score=(
                    request
                    .minimum_score
                ),
            )
        )

        bundle = (
            search_and_rank_academic_sources(
                query=request.question,
                limit_per_source=(
                    request
                    .external_results_per_source
                ),
                embedding_model=(
                    embedding_model
                ),
                embedding_device=(
                    embedding_device
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
                settings=settings,
            )
        )

        warnings = bundle.warnings
        external_papers = (
            bundle.papers[:8]
        )
        cache_hit = bundle.cache_hit

        expanded = (
            answer_with_external_research(
                question=request.question,
                paper_sources=(
                    paper_sources
                ),
                external_sources=(
                    external_papers
                ),
                ollama_model=(
                runtime_model
                ),
                explanation_level=(
                    request
                    .explanation_level
                ),
            )
        )

        answer = expanded.answer

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

    session_store.append_chat_turn(
        session,
        question=request.question,
        answer=answer,
        paper_sources=[
            source.model_dump(
                mode="json"
            )
            for source
            in serialized_paper_sources
        ],
        external_sources=[
            source.model_dump(
                mode="json"
            )
            for source
            in serialized_external
        ],
        answer_mode=(
            request.answer_mode.value
        ),
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
        cache_hit=cache_hit,
    )