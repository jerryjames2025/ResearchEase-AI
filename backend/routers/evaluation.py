from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    Query,
)

from backend.core.errors import APIError
from backend.core.settings import (
    get_settings,
)
from backend.evaluation.mlflow_tracker import (
    mlflow_tracker,
)
from backend.evaluation.runner import (
    rag_evaluation_runner,
)
from backend.evaluation.schemas import (
    EvaluationRunListResponse,
    EvaluationRunSummary,
    EvaluationTemplateResponse,
    MLflowHealthResponse,
    RAGEvaluationRequest,
    RAGEvaluationResponse,
)
from backend.llm.dependency import (
    LLMRequestConfig,
    get_llm_request_config,
)


router = APIRouter(
    prefix="/evaluation",
    tags=["RAG evaluation"],
)


@router.get(
    "/health",
    response_model=(
        MLflowHealthResponse
    ),
)
def evaluation_health() -> (
    MLflowHealthResponse
):
    settings = get_settings()

    connected, detail = (
        mlflow_tracker.health()
    )

    return MLflowHealthResponse(
        status=(
            "healthy"
            if connected
            else "degraded"
        ),
        tracking_uri=(
            settings
            .mlflow_tracking_uri
        ),
        experiment_name=(
            settings
            .mlflow_experiment_name
        ),
        detail=detail,
    )


@router.get(
    "/template",
    response_model=(
        EvaluationTemplateResponse
    ),
)
def evaluation_template() -> (
    EvaluationTemplateResponse
):
    return EvaluationTemplateResponse(
        dataset_name=(
            "research-paper-rag-baseline"
        ),
        cases=[
            {
                "case_id": "methodology",
                "question": (
                    "What methodology does "
                    "the paper use?"
                ),
                "expected_pages": [2, 3],
                "expected_answer": (
                    "Replace this with a concise "
                    "expected methodology summary."
                ),
                "tags": {
                    "category": (
                        "methodology"
                    )
                },
            },
            {
                "case_id": "results",
                "question": (
                    "What are the main results "
                    "reported by the authors?"
                ),
                "expected_pages": [5, 6],
                "expected_answer": (
                    "Replace this with the expected "
                    "main findings."
                ),
                "tags": {
                    "category": "results"
                },
            },
            {
                "case_id": "limitations",
                "question": (
                    "What limitations are "
                    "discussed in the paper?"
                ),
                "expected_pages": [7],
                "expected_answer": (
                    "Replace this with the expected "
                    "limitations."
                ),
                "tags": {
                    "category": (
                        "limitations"
                    )
                },
            },
        ],
    )


@router.post(
    "/run",
    response_model=(
        RAGEvaluationResponse
    ),
)
def run_rag_evaluation(
    request: RAGEvaluationRequest,
    llm: LLMRequestConfig = Depends(
        get_llm_request_config
    ),
) -> RAGEvaluationResponse:
    runtime_model = llm.model_spec(
        fallback_model=(
            request.ollama_model
        )
    )

    return rag_evaluation_runner.run(
        session_id=(
            request.session_id
        ),
        dataset_name=(
            request.dataset_name
        ),
        cases=request.cases,
        top_k=request.top_k,
        minimum_score=(
            request.minimum_score
        ),
        run_generation=(
            request.run_generation
        ),
        use_llm_judge=(
            request.use_llm_judge
        ),
        model_spec=runtime_model,
        explanation_level=(
            request.explanation_level
        ),
    )


@router.get(
    "/runs",
    response_model=(
        EvaluationRunListResponse
    ),
)
def list_evaluation_runs(
    limit: int = Query(
        default=30,
        ge=1,
        le=200,
    ),
) -> EvaluationRunListResponse:
    try:
        runs = mlflow_tracker.list_runs(
            limit=limit
        )

    except Exception as exc:
        raise APIError(
            status_code=503,
            code="mlflow_runs_failed",
            detail=str(exc),
        ) from exc

    return EvaluationRunListResponse(
        runs=[
            EvaluationRunSummary(
                **run
            )
            for run in runs
        ]
    )


@router.get(
    "/runs/{run_id}",
    response_model=(
        EvaluationRunSummary
    ),
)
def get_evaluation_run(
    run_id: str,
) -> EvaluationRunSummary:
    try:
        result = (
            mlflow_tracker.get_run(
                run_id
            )
        )

    except Exception as exc:
        raise APIError(
            status_code=404,
            code=(
                "mlflow_run_not_found"
            ),
            detail=str(exc),
        ) from exc

    return EvaluationRunSummary(
        **result
    )