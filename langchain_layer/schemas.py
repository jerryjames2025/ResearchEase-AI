from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class RouteDecision(BaseModel):
    """
    Validated route returned by the LangChain router.
    """

    model_config = ConfigDict(
        extra="ignore"
    )

    route: Literal[
        "paper_only",
        "expanded_research",
        "math",
        "literature_review",
    ]

    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
    )

    reason: str = Field(
        default="No reason supplied."
    )


class LiteratureRecord(BaseModel):
    """
    Structured record extracted from one research source.
    """

    model_config = ConfigDict(
        extra="ignore"
    )

    title: str = "Not identified"
    authors: str = "Not identified"
    year: str = "Not identified"

    research_problem: str = "Not identified"
    objective: str = "Not identified"
    methodology: str = "Not identified"
    dataset: str = "Not identified"
    models_or_methods: str = "Not identified"
    evaluation_metrics: str = "Not identified"
    main_findings: str = "Not identified"
    limitations: str = "Not identified"
    future_work: str = "Not identified"
    keywords: str = "Not identified"