from __future__ import annotations

from fastapi import APIRouter

from backend.core.retrieval import (
    retrieve_session_context,
)
from backend.core.serializers import (
    serialize_paper_sources,
)
from backend.core.session_store import (
    session_store,
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
    """
    Retrieve optional uploaded-paper evidence.
    """

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
) -> MathResponse:
    computation = compute_math(
        source_text=request.source_text,
        input_format=request.input_format,
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

    paper_context, sources = (
        _optional_paper_context(
            session_id=request.session_id,
            use_paper_context=(
                request.use_paper_context
            ),
            query=(
                request.source_text
                + " "
                + request.user_context
            ),
            top_k=request.top_k,
            minimum_score=(
                request.minimum_score
            ),
        )
    )

    explanation = explain_computation(
        computation=computation,
        ollama_model=(
            request.ollama_model
        ),
        explanation_level=(
            request.explanation_level
        ),
        user_context=(
            request.user_context
        ),
        paper_context=paper_context,
    )

    return MathResponse(
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


@router.post(
    "/topic",
    response_model=MathResponse,
)
def explain_topic(
    request: MathTopicRequest,
) -> MathResponse:
    paper_context, sources = (
        _optional_paper_context(
            session_id=request.session_id,
            use_paper_context=(
                request.use_paper_context
            ),
            query=(
                request.topic
                + " "
                + request.user_context
            ),
            top_k=request.top_k,
            minimum_score=(
                request.minimum_score
            ),
        )
    )

    explanation = explain_math_topic(
        topic=request.topic,
        ollama_model=(
            request.ollama_model
        ),
        explanation_level=(
            request.explanation_level
        ),
        user_context=(
            request.user_context
        ),
        paper_context=paper_context,
    )

    return MathResponse(
        mode="topic",
        explanation=explanation,
        paper_sources=(
            serialize_paper_sources(
                sources
            )
        ),
    )