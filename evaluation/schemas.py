from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from config import DEFAULT_OLLAMA_MODEL


class RAGEvaluationCase(BaseModel):
    case_id: str = Field(
        default="",
        max_length=100,
    )

    question: str = Field(
        min_length=2,
        max_length=4000,
    )

    expected_answer: str = Field(
        default="",
        max_length=12000,
    )

    expected_keywords: list[str] = Field(
        default_factory=list,
        max_length=50,
    )

    expected_pages: list[int] = Field(
        default_factory=list,
        max_length=50,
    )


class RAGEvaluationRequest(BaseModel):
    run_name: str = Field(
        default="",
        max_length=250,
    )

    dataset_name: str = Field(
        default="manual-rag-evaluation",
        max_length=250,
    )

    cases: list[RAGEvaluationCase] = Field(
        min_length=1,
        max_length=50,
    )

    ollama_model: str = Field(
        default=DEFAULT_OLLAMA_MODEL,
        min_length=2,
        max_length=250,
    )

    explanation_level: str = Field(
        default="Intermediate",
        max_length=100,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    minimum_score: float = Field(
        default=0.15,
        ge=-1.0,
        le=1.0,
    )

    use_llm_judge: bool = False

    tags: dict[str, str] = Field(
        default_factory=dict
    )


class RAGEvaluationCaseResult(BaseModel):
    case_id: str
    question: str
    answer: str

    retrieved_sources: list[
        dict[str, Any]
    ]

    metrics: dict[str, float]
    judge: dict[str, Any]

    latency_seconds: float
    error: str = ""


class RAGEvaluationResponse(BaseModel):
    status: str
    run_id: str
    experiment_name: str
    artifact_uri: str
    mlflow_ui_url: str

    dataset_name: str
    session_id: str
    case_count: int

    aggregate_metrics: dict[
        str,
        float,
    ]

    results: list[
        RAGEvaluationCaseResult
    ]


class MLflowRunSummary(BaseModel):
    run_id: str
    run_name: str
    status: str

    start_time: str
    end_time: str

    metrics: dict[str, float]
    params: dict[str, str]
    tags: dict[str, str]

    artifact_uri: str


class MLflowRunListResponse(BaseModel):
    experiment_name: str

    runs: list[
        MLflowRunSummary
    ]