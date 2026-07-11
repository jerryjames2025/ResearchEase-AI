from __future__ import annotations

import re

from langchain_core.output_parsers import (
    PydanticOutputParser,
)

from langchain_layer.model_factory import (
    get_ollama_chat_model,
)

from langchain_layer.prompts import (
    router_prompt,
)

from langchain_layer.schemas import (
    RouteDecision,
)


def _heuristic_route(
    question: str,
) -> RouteDecision:
    """
    Deterministic fallback used if the local model
    does not return valid structured JSON.
    """

    lowered = question.lower()

    math_patterns = (
        r"\bsolve\b",
        r"\bequation\b",
        r"\bderivative\b",
        r"\bdifferentiat",
        r"\bintegral\b",
        r"\bintegrat",
        r"\blimit\b",
        r"\bmatrix\b",
        r"\beigen",
        r"\btheorem\b",
        r"\bformula\b",
    )

    literature_patterns = (
        r"\bliterature review\b",
        r"\bcompare (?:these|multiple|all) papers\b",
        r"\bresearch gap",
        r"\bcommon limitation",
        r"\bsynthesis\b",
        r"\bthemes across\b",
    )

    external_patterns = (
        r"\bnewer\b",
        r"\brecent studies\b",
        r"\boutside\b",
        r"\bexternal\b",
        r"\brelated papers\b",
        r"\bother studies\b",
        r"\bcurrent research\b",
        r"\blatest\b",
    )

    if any(
        re.search(
            pattern,
            lowered,
        )
        for pattern
        in math_patterns
    ):
        return RouteDecision(
            route="math",

            confidence=0.65,

            reason=(
                "Fallback keyword routing "
                "detected a mathematical request."
            ),
        )

    if any(
        re.search(
            pattern,
            lowered,
        )
        for pattern
        in literature_patterns
    ):
        return RouteDecision(
            route="literature_review",

            confidence=0.65,

            reason=(
                "Fallback keyword routing "
                "detected a multi-paper "
                "synthesis request."
            ),
        )

    if any(
        re.search(
            pattern,
            lowered,
        )
        for pattern
        in external_patterns
    ):
        return RouteDecision(
            route="expanded_research",

            confidence=0.65,

            reason=(
                "Fallback keyword routing "
                "detected a request for "
                "outside knowledge."
            ),
        )

    return RouteDecision(
        route="paper_only",

        confidence=0.55,

        reason=(
            "Fallback routing selected the "
            "uploaded-paper workflow."
        ),
    )


def route_query(
    *,
    question: str,
    model_name: str,
    has_paper: bool,
    has_external_records: bool,
) -> RouteDecision:
    """
    Route a request with a Pydantic
    structured-output LangChain chain.

    If the local model returns malformed JSON,
    deterministic keyword routing is used.
    """

    question = question.strip()

    if not question:
        raise ValueError(
            "Routing question cannot be empty."
        )

    parser = PydanticOutputParser(
        pydantic_object=(
            RouteDecision
        )
    )

    chain = (
        router_prompt(
            parser.get_format_instructions()
        )
        | get_ollama_chat_model(
            model_name=model_name,
            temperature=0.0,
            json_mode=True,
        )
        | parser
    )

    try:
        return chain.invoke(
            {
                "question": question,

                "has_paper": (
                    has_paper
                ),

                "has_external_records": (
                    has_external_records
                ),
            }
        )

    except Exception:
        return _heuristic_route(
            question
        )