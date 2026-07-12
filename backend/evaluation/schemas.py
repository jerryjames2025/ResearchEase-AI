from __future__ import annotations

from typing import Any

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from config import DEFAULT_OLLAMA_MODEL


class EvaluationCase(BaseModel):
    """
    One question in a RAG evaluation dataset.

    expected_pages enables retrieval evaluation.
    expected_answer enables lexical answer evaluation.
    """

    case_id: str = ""

    question: str = Field(
        min_length=2,
        max_length=4000,
    )

    expected_pages: list[int] = Field(
        default_factory=list,
    )

    expected_answer: str = Field(
        default="",
        max_length=10000,
    )

    tags: dict[str, str] = Field(
        default_factory=dict,
    )

    @field_validator(
        "expected_pages"
    )
    @classmethod
    def normalize_pages(
        cls,
        pages: list[int],
    ) -> list[int]:
        normalized: list[int] = []

        for page in pages:
            page_number = int(page)

            if (
                page_number > 0
                and page_number
                not in normalized
            ):
                normalized.append(
                    page_number
                )

        return normalized


class RAGEvaluationRequest(BaseModel):
    session_id: str = Field(
        min_length=2,
    )

    dataset_name: str = Field(
        default="manual-rag-evaluation",
        min_length=2,
        max_length=200,
    )

    cases: list[
        EvaluationCase
    ] = Field(
        min_length=1,
        max_length=50,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    minimum_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    run_generation: bool = True

    use_llm_judge: bool = False

    ollama_model: str = Field(
        default=DEFAULT_OLLAMA_MODEL,
        min_length=2,
        max_length=200,
    )

    explanation_level: str = Field(
        default="Intermediate",
        min_length=2,
        max_length=100,
    )


class RetrievedEvaluationSource(
    BaseModel
):
    rank: int
    page_number: int
    score: float
    text: str


class EvaluationJudgeResult(
    BaseModel
):
    relevance: float | None = None
    groundedness: float | None = None
    completeness: float | None = None
    justification: str = ""


class EvaluationCaseResult(
    BaseModel
):
    case_id: str
    question: str

    expected_pages: list[int]
    expected_answer: str

    retrieved_pages: list[int]

    retrieved_sources: list[
        RetrievedEvaluationSource
    ]

    answer: str

    metrics: dict[
        str,
        float | None,
    ]

    judge: EvaluationJudgeResult | None = (
        None
    )

    error: str = ""


class RAGEvaluationResponse(
    BaseModel
):
    status: str

    session_id: str
    dataset_name: str

    mlflow_run_id: str = ""
    mlflow_experiment_name: str = ""

    case_count: int

    aggregate_metrics: dict[
        str,
        float,
    ]

    cases: list[
        EvaluationCaseResult
    ]

    warnings: list[str] = Field(
        default_factory=list
    )


class MLflowHealthResponse(
    BaseModel
):
    status: str
    tracking_uri: str
    experiment_name: str
    detail: str


class EvaluationRunSummary(
    BaseModel
):
    run_id: str
    run_name: str
    status: str

    start_time: str = ""
    end_time: str = ""

    artifact_uri: str = ""

    parameters: dict[
        str,
        str,
    ] = Field(
        default_factory=dict
    )

    metrics: dict[
        str,
        float,
    ] = Field(
        default_factory=dict
    )

    tags: dict[
        str,
        str,
    ] = Field(
        default_factory=dict
    )


class EvaluationRunListResponse(
    BaseModel
):
    runs: list[
        EvaluationRunSummary
    ]


class EvaluationTemplateResponse(
    BaseModel
):
    dataset_name: str
    cases: list[
        dict[str, Any]
    ]