from __future__ import annotations

from langchain_layer.chains import (
    run_math_computation_chain,
    run_math_topic_chain,
)

from mathematics.models import (
    MathComputation,
)


def explain_computation(
    computation: MathComputation,
    ollama_model: str,
    explanation_level: str,
    user_context: str = "",
    paper_context: str = "",
) -> str:
    """
    Explain a SymPy-verified result through
    a LangChain prompt chain.
    """

    return run_math_computation_chain(
        model_name=ollama_model,

        explanation_level=(
            explanation_level
        ),

        operation=(
            computation.operation
        ),

        original_input=(
            computation.original_input
        ),

        input_format=(
            computation.input_format
        ),

        parsed_text=(
            computation.parsed_text
        ),

        result_text=(
            computation.result_text
        ),

        variable=(
            computation.variable
        ),

        verification=(
            computation.verification
        ),

        notes="; ".join(
            computation.notes
        ),

        user_context=(
            user_context
        ),

        paper_context=(
            paper_context
        ),
    )


def explain_math_topic(
    topic: str,
    ollama_model: str,
    explanation_level: str,
    user_context: str = "",
    paper_context: str = "",
) -> str:
    """
    Explain a mathematical concept through LangChain.
    """

    topic = topic.strip()

    if not topic:
        raise ValueError(
            "Mathematical topic cannot be empty."
        )

    return run_math_topic_chain(
        topic=topic,

        model_name=ollama_model,

        explanation_level=(
            explanation_level
        ),

        user_context=(
            user_context
        ),

        paper_context=(
            paper_context
        ),
    )