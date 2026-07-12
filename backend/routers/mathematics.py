from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
)

from backend.core.retrieval import (
    retrieve_session_context,
)
from backend.core.serializers import (
    serialize_paper_sources,
)
from backend.core.session_store import (
    session_store,
)
from backend.llm.dependency import (
    LLMRequestConfig,
    get_llm_request_config,
)
from backend.schemas import (
    MathEquationRequest,
    MathResponse,
    MathTopicRequest,
)
from mathematics.engine import (
    compute_math,
)
from mathematics.explainer import (
    explain_computation,
    explain_math_topic,
)


router = APIRouter(
    prefix="/math",
    tags=["mathematics"],
)


def _optional_paper_context(
    session_id: str | None,
    use_paper_context: bool,
    query: str,
    top_k: int,
    minimum_score: float,
):
    if (
        not use_paper_context
        or not session_id
    ):
        return "", []

    session = session_store.get(
        session_id
    )

    return retrieve_session_context(
        session=session,
        query=query,
        top_k=top_k,
        minimum_score=minimum_score,
    )


@router.post(
    "/equation",
    response_model=MathResponse,
)
def explain_equation(
    request: MathEquationRequest,
    llm: LLMRequestConfig = Depends(
        get_llm_request_config
    ),
) -> MathResponse:
    computation = compute_math(
        source_text=(
            request.source_text
        ),
        input_format=(
            request.input_format
        ),
        operation=request.operation,
        variable_name=(
            request.variable_name
        ),
        derivative_order=(
            request.derivative_order
        ),
        lower_bound=(
            request.lower_bound
        ),
        upper_bound=(
            request.upper_bound
        ),
        limit_point=(
            request.limit_point
        ),
        limit_direction=(
            request.limit_direction
        ),
        substitutions_text=(
            request.substitutions_text
        ),
    )

    context_query = " ".join(
        part
        for part in [
            request.source_text,
            request.operation,
            request.user_context,
        ]
        if part.strip()
    )

    paper_context, sources = (
        _optional_paper_context(
            session_id=(
                request.session_id
            ),
            use_paper_context=(
                request.use_paper_context
            ),
            query=context_query,
            top_k=request.top_k,
            minimum_score=(
                request.minimum_score
            ),
        )
    )

    runtime_model = llm.model_spec(
        fallback_model=(
            request.ollama_model
        )
    )

    explanation = explain_computation(
        computation=computation,
        ollama_model=runtime_model,
        explanation_level=(
            request.explanation_level
        ),
        user_context=(
            request.user_context
        ),
        paper_context=paper_context,
    )

    response = MathResponse(
        mode="equation",
        explanation=explanation,
        parsed_text=(
            computation.parsed_text
        ),
        parsed_latex=(
            computation.parsed_latex
        ),
        result_text=(
            computation.result_text
        ),
        result_latex=(
            computation.result_latex
        ),
        verification=(
            computation.verification
        ),
        paper_sources=(
            serialize_paper_sources(
                sources
            )
        ),
    )

    if request.session_id:
        session_store.get(
            request.session_id
        )

        session_store.record_math_result(
            request.session_id,
            response.model_dump(
                mode="json"
            ),
        )

    return response


@router.post(
    "/topic",
    response_model=MathResponse,
)
def explain_topic(
    request: MathTopicRequest,
    llm: LLMRequestConfig = Depends(
        get_llm_request_config
    ),
) -> MathResponse:
    context_query = " ".join(
        part
        for part in [
            request.topic,
            request.user_context,
        ]
        if part.strip()
    )

    paper_context, sources = (
        _optional_paper_context(
            session_id=(
                request.session_id
            ),
            use_paper_context=(
                request.use_paper_context
            ),
            query=context_query,
            top_k=request.top_k,
            minimum_score=(
                request.minimum_score
            ),
        )
    )

    runtime_model = llm.model_spec(
        fallback_model=(
            request.ollama_model
        )
    )

    explanation = explain_math_topic(
        topic=request.topic,
        ollama_model=runtime_model,
        explanation_level=(
            request.explanation_level
        ),
        user_context=(
            request.user_context
        ),
        paper_context=paper_context,
    )

    response = MathResponse(
        mode="topic",
        explanation=explanation,
        paper_sources=(
            serialize_paper_sources(
                sources
            )
        ),
    )

    if request.session_id:
        session_store.get(
            request.session_id
        )

        session_store.record_math_result(
            request.session_id,
            response.model_dump(
                mode="json"
            ),
        )

    return response