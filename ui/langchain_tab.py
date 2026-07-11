from __future__ import annotations

import streamlit as st

from langchain_layer.router import (
    route_query,
)


ROUTE_LABELS = {
    "paper_only": (
        "Paper Only RAG"
    ),

    "expanded_research": (
        "Expanded Academic Research"
    ),

    "math": (
        "Mathematical Explainer"
    ),

    "literature_review": (
        "Literature Review"
    ),
}


def render_langchain_tab(
    *,
    ollama_model: str,
    connected: bool,
    has_paper: bool,
    external_results: list[dict],
) -> None:
    """
    Render LangChain workflow diagnostics
    and router testing.
    """

    st.subheader(
        "LangChain Orchestration"
    )

    st.caption(
        "Version 6 uses ChatPromptTemplate, LCEL Runnables, "
        "RunnableBranch, Pydantic structured-output parsers, "
        "ChatOllama and message-based conversational context."
    )

    st.markdown(
        "### Integrated workflow"
    )

    st.code(
        """
Paper question
    → RunnableLambda retrieval
    → FAISS search
    → RunnableBranch
       ├─ no evidence
       │    → grounded refusal
       └─ evidence found
            → ChatPromptTemplate
            → ChatOllama
            → StrOutputParser

Router
    → ChatPromptTemplate
    → ChatOllama JSON mode
    → PydanticOutputParser[RouteDecision]
    → deterministic fallback if parsing fails
""".strip(),
        language="text",
    )

    status_columns = st.columns(
        4
    )

    status_columns[
        0
    ].metric(
        "Ollama",
        (
            "Connected"
            if connected
            else "Offline"
        ),
    )

    status_columns[
        1
    ].metric(
        "Uploaded paper",
        (
            "Available"
            if has_paper
            else "Missing"
        ),
    )

    status_columns[
        2
    ].metric(
        "External records",
        len(
            external_results
        ),
    )

    status_columns[
        3
    ].metric(
        "LangChain model",
        ollama_model,
    )

    st.markdown(
        "### Test the structured router"
    )

    router_question = st.text_area(
        "Enter a request",

        value=(
            "Compare this paper with newer "
            "transformer-based studies."
        ),

        height=100,
    )

    route_clicked = st.button(
        "Route request with LangChain",

        type="primary",

        use_container_width=True,

        disabled=(
            not connected
            or not ollama_model
        ),
    )

    if route_clicked:
        try:
            with st.spinner(
                "Running the structured-output router..."
            ):
                decision = route_query(
                    question=(
                        router_question
                    ),

                    model_name=(
                        ollama_model
                    ),

                    has_paper=(
                        has_paper
                    ),

                    has_external_records=bool(
                        external_results
                    ),
                )

            st.success(
                (
                    "Selected workflow: "
                    f"{ROUTE_LABELS[decision.route]}"
                )
            )

            result_columns = st.columns(
                2
            )

            result_columns[
                0
            ].metric(
                "Route key",
                decision.route,
            )

            result_columns[
                1
            ].metric(
                "Confidence",
                (
                    f"{decision.confidence:.2f}"
                ),
            )

            st.write(
                "**Reason:**",
                decision.reason,
            )

            if decision.route == "math":
                st.info(
                    "Use the Math Explainer tab so "
                    "SymPy can verify the calculation "
                    "before the LLM explains it."
                )

            elif (
                decision.route
                == "literature_review"
            ):
                st.info(
                    "Use the Literature Review tab "
                    "to compare multiple full-text "
                    "and abstract-level sources."
                )

        except Exception as exc:
            st.error(
                (
                    "LangChain routing failed: "
                    f"{exc}"
                )
            )