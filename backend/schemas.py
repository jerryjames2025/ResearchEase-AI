from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import (
    BaseModel,
    Field,
)

from config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
)


class AnswerMode(
    str,
    Enum,
):
    PAPER_ONLY = "paper_only"

    EXPANDED_RESEARCH = (
        "expanded_research"
    )


class MessageResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    ollama_connected: (
        bool | None
    ) = None
    detail: str = ""


class StorageServiceResponse(
    BaseModel
):
    healthy: bool
    detail: str


class StorageHealthResponse(
    BaseModel
):
    status: str

    services: dict[
        str,
        StorageServiceResponse,
    ]


class PaperUploadResponse(BaseModel):
    session_id: str
    filename: str
    page_count: int
    extracted_characters: int
    created_at: datetime


class SessionResponse(BaseModel):
    session_id: str
    filename: str
    page_count: int
    extracted_characters: int
    created_at: datetime
    updated_at: datetime
    analysis_ready: bool
    index_ready: bool
    chunk_count: int
    chat_message_count: int = 0


class SessionListResponse(BaseModel):
    sessions: list[
        SessionResponse
    ]


class AnalyzeRequest(BaseModel):
    ollama_model: str = Field(
        default=(
            DEFAULT_OLLAMA_MODEL
        ),
        min_length=1,
        max_length=100,
    )

    explanation_level: str = Field(
        default="Beginner",
        min_length=2,
        max_length=50,
    )


class AnalyzeResponse(BaseModel):
    session_id: str
    filename: str
    analysis: str
    sections_processed: int


class IndexRequest(BaseModel):
    embedding_model: str = Field(
        default=(
            DEFAULT_EMBEDDING_MODEL
        ),
        min_length=2,
        max_length=200,
    )

    device: str = Field(
        default="auto",
        pattern=(
            r"^(auto|cpu|cuda)$"
        ),
    )


class IndexResponse(BaseModel):
    session_id: str
    chunk_count: int
    embedding_model: str
    device: str
    index_path: str = ""


class PaperSourceResponse(BaseModel):
    rank: int
    page_number: int
    chunk_number: int
    score: float
    text: str


class ExternalPaperResponse(BaseModel):
    source: str
    source_id: str
    title: str

    authors: list[str] = Field(
        default_factory=list
    )

    year: int | None = None
    abstract: str = ""
    url: str = ""
    doi: str = ""
    venue: str = ""

    citation_count: int | None = None
    publication_date: str = ""
    pdf_url: str = ""

    relevance_score: float | None = None


class ChatRequest(BaseModel):
    question: str = Field(
        min_length=2,
        max_length=4000,
    )

    answer_mode: AnswerMode = (
        AnswerMode.PAPER_ONLY
    )

    ollama_model: str = (
        DEFAULT_OLLAMA_MODEL
    )

    explanation_level: str = (
        "Beginner"
    )

    top_k: int = Field(
        default=5,
        ge=2,
        le=10,
    )

    minimum_score: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
    )

    external_results_per_source: int = (
        Field(
            default=5,
            ge=1,
            le=10,
        )
    )

    use_semantic_scholar: bool = True
    use_arxiv: bool = True
    use_crossref: bool = True


class ChatResponse(BaseModel):
    session_id: str
    answer_mode: AnswerMode
    answer: str

    paper_sources: list[
        PaperSourceResponse
    ] = Field(
        default_factory=list
    )

    external_sources: list[
        ExternalPaperResponse
    ] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(
        default_factory=list
    )

    cache_hit: bool = False


class ChatTurnResponse(BaseModel):
    question: str
    answer: str

    paper_sources: list[
        PaperSourceResponse
    ] = Field(
        default_factory=list
    )

    external_sources: list[
        ExternalPaperResponse
    ] = Field(
        default_factory=list
    )

    answer_mode: str = "paper_only"


class ChatHistoryResponse(BaseModel):
    session_id: str

    turns: list[
        ChatTurnResponse
    ]


class ResearchSearchRequest(BaseModel):
    query: str = Field(
        min_length=2,
        max_length=1000,
    )

    limit_per_source: int = Field(
        default=5,
        ge=1,
        le=10,
    )

    embedding_model: str = (
        DEFAULT_EMBEDDING_MODEL
    )

    embedding_device: str = Field(
        default="auto",
        pattern=(
            r"^(auto|cpu|cuda)$"
        ),
    )

    use_semantic_scholar: bool = True
    use_arxiv: bool = True
    use_crossref: bool = True


class ResearchSearchResponse(
    BaseModel
):
    query: str

    results: list[
        ExternalPaperResponse
    ]

    warnings: list[str] = Field(
        default_factory=list
    )

    cache_hit: bool = False


class MathEquationRequest(BaseModel):
    source_text: str = Field(
        min_length=1,
        max_length=10000,
    )

    input_format: str = (
        "Auto detect"
    )

    operation: str = (
        "Explain and simplify"
    )

    variable_name: str = "x"

    derivative_order: int = Field(
        default=1,
        ge=1,
        le=10,
    )

    lower_bound: str = ""
    upper_bound: str = ""

    limit_point: str = "0"
    limit_direction: str = "+-"

    substitutions_text: str = ""

    ollama_model: str = (
        DEFAULT_OLLAMA_MODEL
    )

    explanation_level: str = (
        "Beginner"
    )

    user_context: str = ""

    session_id: str | None = None
    use_paper_context: bool = False

    top_k: int = Field(
        default=5,
        ge=2,
        le=10,
    )

    minimum_score: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
    )


class MathTopicRequest(BaseModel):
    topic: str = Field(
        min_length=2,
        max_length=1000,
    )

    ollama_model: str = (
        DEFAULT_OLLAMA_MODEL
    )

    explanation_level: str = (
        "Beginner"
    )

    user_context: str = ""

    session_id: str | None = None
    use_paper_context: bool = False

    top_k: int = Field(
        default=5,
        ge=2,
        le=10,
    )

    minimum_score: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
    )


class MathResponse(BaseModel):
    mode: str
    explanation: str

    parsed_text: str = ""
    parsed_latex: str = ""

    result_text: str = ""
    result_latex: str = ""

    verification: str = ""

    paper_sources: list[
        PaperSourceResponse
    ] = Field(
        default_factory=list
    )


class LiteratureReviewResponse(
    BaseModel
):
    topic: str
    source_count: int

    matrix: list[
        dict[str, str]
    ]

    synthesis: str

    warnings: list[str] = Field(
        default_factory=list
    )