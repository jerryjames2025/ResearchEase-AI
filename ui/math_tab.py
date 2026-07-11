from __future__ import annotations

from typing import Any

import streamlit as st

from mathematics.engine import (
    SUPPORTED_OPERATIONS,
    compute_math,
)

from mathematics.explainer import (
    explain_computation,
    explain_math_topic,
)

from mathematics.parser import (
    MathInputError,
)

from services.embedding_model import (
    load_embedding_service,
)


def _retrieve_optional_paper_context(
    query: str,
    vector_store,
    embedding_model_name: str,
    embedding_device: str,
    top_k: int,
    minimum_score: float,
) -> tuple[
    str,
    list[dict[str, Any]],
]:
    """
    Retrieve relevant uploaded-paper chunks for a
    mathematical explanation.
    """

    if vector_store is None:
        return "", []

    embedding_service = (
        load_embedding_service(
            embedding_model_name,
            embedding_device,
        )
    )

    query_embedding = (
        embedding_service
        .encode_query(
            query
        )
    )

    results = vector_store.search(
        query_embedding=query_embedding,
        top_k=top_k,
    )

    accepted = [
        result
        for result in results
        if result.score >= minimum_score
    ]

    context_blocks: list[str] = []
    serialized: list[
        dict[str, Any]
    ] = []

    for result in accepted:
        context_blocks.append(
            (
                "[Uploaded Paper, "
                f"Page {result.chunk.page_number}]\n"
                f"{result.chunk.text}"
            )
        )

        serialized.append(
            {
                "page_number": (
                    result.chunk.page_number
                ),
                "chunk_number": (
                    result.chunk
                    .chunk_number_on_page
                ),
                "score": (
                    result.score
                ),
                "text": (
                    result.chunk.text
                ),
            }
        )

    return (
        "\n\n".join(
            context_blocks
        ),
        serialized,
    )


def _render_paper_sources(
    sources: list[
        dict[str, Any]
    ],
) -> None:
    """
    Display paper chunks used in a mathematics explanation.
    """

    if not sources:
        return

    with st.expander(
        "View mathematical paper context"
    ):
        for source in sources:
            st.markdown(
                (
                    f"**Page {source['page_number']} · "
                    f"Chunk {source['chunk_number']} · "
                    "Similarity "
                    f"{source['score']:.3f}**"
                )
            )

            st.write(
                source["text"]
            )

            st.divider()


def render_math_tab(
    *,
    ollama_model: str,
    explanation_level: str,
    connected: bool,
    vector_store,
    embedding_model_name: str,
    embedding_device: str,
    top_k: int,
    minimum_score: float,
) -> None:
    """
    Render the Version 4 mathematics workspace.
    """

    st.subheader(
        "Mathematical Equation and Topic Explainer"
    )

    st.caption(
        "SymPy performs symbolic computation first. "
        "Ollama then explains the result. Typed plain-text "
        "and LaTeX equations are supported."
    )

    if "math_history" not in st.session_state:
        st.session_state.math_history = []

    mode = st.radio(
        "Mathematics mode",
        [
            "Equation or expression",
            "Mathematical topic",
        ],
        horizontal=True,
    )

    use_paper_context = st.checkbox(
        "Use relevant context from the uploaded paper",
        value=(
            vector_store is not None
        ),
        disabled=(
            vector_store is None
        ),
        help=(
            "Build the paper RAG index first to connect "
            "the mathematical explanation to paper pages."
        ),
    )

    user_context = st.text_area(
        (
            "What specifically is confusing or what "
            "should the explanation focus on?"
        ),
        placeholder=(
            "Example: Explain why square-root scaling "
            "is used in attention."
        ),
        height=90,
    )

    # -----------------------------------------------------
    # Mathematical topic mode
    # -----------------------------------------------------

    if mode == "Mathematical topic":
        topic = st.text_input(
            "Mathematical topic",
            placeholder=(
                "Example: Fourier transform, eigenvalues, "
                "Bayes theorem"
            ),
        )

        topic_clicked = st.button(
            "Explain mathematical topic",
            type="primary",
            use_container_width=True,
            disabled=(
                not connected
                or not ollama_model
            ),
        )

        if topic_clicked:
            if not topic.strip():
                st.error(
                    "Enter a mathematical topic."
                )

            else:
                try:
                    paper_context = ""
                    paper_sources: list[
                        dict[str, Any]
                    ] = []

                    if use_paper_context:
                        (
                            paper_context,
                            paper_sources,
                        ) = (
                            _retrieve_optional_paper_context(
                                query=(
                                    f"{topic} "
                                    f"{user_context}"
                                ),
                                vector_store=(
                                    vector_store
                                ),
                                embedding_model_name=(
                                    embedding_model_name
                                ),
                                embedding_device=(
                                    embedding_device
                                ),
                                top_k=top_k,
                                minimum_score=(
                                    minimum_score
                                ),
                            )
                        )

                    with st.spinner(
                        "Preparing the topic explanation..."
                    ):
                        explanation = (
                            explain_math_topic(
                                topic=topic,
                                ollama_model=(
                                    ollama_model
                                ),
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
                        )

                    record = {
                        "mode": "topic",
                        "topic": topic,
                        "user_context": (
                            user_context
                        ),
                        "explanation": (
                            explanation
                        ),
                        "paper_sources": (
                            paper_sources
                        ),
                    }

                    st.session_state.math_history.append(
                        record
                    )

                    st.markdown(
                        explanation
                    )

                    _render_paper_sources(
                        paper_sources
                    )

                except Exception as exc:
                    st.error(
                        (
                            "Unable to explain "
                            f"the topic: {exc}"
                        )
                    )

    # -----------------------------------------------------
    # Equation mode
    # -----------------------------------------------------

    else:
        source_text = st.text_area(
            "Equation or expression",
            value=(
                "x**2 - 5*x + 6 = 0"
            ),
            height=120,
            help=(
                "Plain example: x**2 - 5*x + 6 = 0. "
                r"LaTeX example: \frac{x^2-1}{x-1}."
            ),
        )

        first_column, second_column = (
            st.columns(2)
        )

        with first_column:
            input_format = st.selectbox(
                "Input format",
                [
                    "Auto detect",
                    "SymPy / plain text",
                    "LaTeX",
                ],
            )

            operation = st.selectbox(
                "Operation",
                SUPPORTED_OPERATIONS,
            )

        with second_column:
            variable_name = st.text_input(
                "Variable",
                value="x",
                help=(
                    "Used for solving, calculus, "
                    "and limits."
                ),
            )

            derivative_order = (
                st.number_input(
                    "Derivative order",
                    min_value=1,
                    max_value=10,
                    value=1,
                    disabled=(
                        operation
                        != "Differentiate"
                    ),
                )
            )

        lower_bound = ""
        upper_bound = ""
        limit_point = "0"
        limit_direction = "+-"
        substitutions_text = ""

        if operation == "Integrate":
            bound_columns = st.columns(
                2
            )

            lower_bound = (
                bound_columns[0]
                .text_input(
                    (
                        "Lower bound "
                        "(leave both blank "
                        "for indefinite)"
                    ),
                    value="",
                )
            )

            upper_bound = (
                bound_columns[1]
                .text_input(
                    "Upper bound",
                    value="",
                )
            )

        elif operation == "Calculate limit":
            limit_columns = st.columns(
                2
            )

            limit_point = (
                limit_columns[0]
                .text_input(
                    "Approach value",
                    value="0",
                )
            )

            limit_direction = (
                limit_columns[1]
                .selectbox(
                    "Direction",
                    [
                        "+-",
                        "+",
                        "-",
                    ],
                    help=(
                        "+- means a "
                        "two-sided limit."
                    ),
                )
            )

        elif operation == "Evaluate substitutions":
            substitutions_text = (
                st.text_input(
                    "Substitutions",
                    value="x=2",
                    placeholder=(
                        "x=2, y=3"
                    ),
                )
            )

        calculate_clicked = st.button(
            "Calculate and explain",
            type="primary",
            use_container_width=True,
            disabled=(
                not connected
                or not ollama_model
            ),
        )

        if calculate_clicked:
            try:
                computation = compute_math(
                    source_text=source_text,
                    input_format=(
                        input_format
                    ),
                    operation=operation,
                    variable_name=(
                        variable_name
                    ),
                    derivative_order=int(
                        derivative_order
                    ),
                    lower_bound=(
                        lower_bound
                    ),
                    upper_bound=(
                        upper_bound
                    ),
                    limit_point=(
                        limit_point
                    ),
                    limit_direction=(
                        limit_direction
                    ),
                    substitutions_text=(
                        substitutions_text
                    ),
                )

                st.markdown(
                    "### Parsed expression"
                )

                st.latex(
                    computation.parsed_latex
                )

                st.code(
                    computation.parsed_text,
                    language="text",
                )

                st.markdown(
                    "### SymPy result"
                )

                st.latex(
                    computation.result_latex
                )

                st.code(
                    computation.result_text,
                    language="text",
                )

                if computation.verification:
                    st.info(
                        (
                            "Verification: "
                            f"{computation.verification}"
                        )
                    )

                for note in computation.notes:
                    st.caption(
                        note
                    )

                paper_context = ""
                paper_sources: list[
                    dict[str, Any]
                ] = []

                if use_paper_context:
                    (
                        paper_context,
                        paper_sources,
                    ) = (
                        _retrieve_optional_paper_context(
                            query=(
                                f"{source_text} "
                                f"{user_context}"
                            ),
                            vector_store=(
                                vector_store
                            ),
                            embedding_model_name=(
                                embedding_model_name
                            ),
                            embedding_device=(
                                embedding_device
                            ),
                            top_k=top_k,
                            minimum_score=(
                                minimum_score
                            ),
                        )
                    )

                with st.spinner(
                    "Explaining the verified calculation..."
                ):
                    explanation = (
                        explain_computation(
                            computation=(
                                computation
                            ),
                            ollama_model=(
                                ollama_model
                            ),
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
                    )

                st.markdown(
                    "### AI explanation"
                )

                st.markdown(
                    explanation
                )

                _render_paper_sources(
                    paper_sources
                )

                st.session_state.math_history.append(
                    {
                        "mode": "equation",
                        "original_input": (
                            computation
                            .original_input
                        ),
                        "input_format": (
                            computation
                            .input_format
                        ),
                        "operation": (
                            computation
                            .operation
                        ),
                        "parsed_text": (
                            computation
                            .parsed_text
                        ),
                        "parsed_latex": (
                            computation
                            .parsed_latex
                        ),
                        "result_text": (
                            computation
                            .result_text
                        ),
                        "result_latex": (
                            computation
                            .result_latex
                        ),
                        "variable": (
                            computation
                            .variable
                        ),
                        "verification": (
                            computation
                            .verification
                        ),
                        "notes": (
                            computation.notes
                        ),
                        "substitutions": (
                            computation
                            .substitutions
                        ),
                        "user_context": (
                            user_context
                        ),
                        "explanation": (
                            explanation
                        ),
                        "paper_sources": (
                            paper_sources
                        ),
                    }
                )

            except MathInputError as exc:
                st.error(
                    str(exc)
                )

            except Exception as exc:
                st.error(
                    (
                        "Mathematical analysis "
                        f"failed: {exc}"
                    )
                )

    # -----------------------------------------------------
    # Mathematics history
    # -----------------------------------------------------

    st.divider()

    history_header, clear_column = (
        st.columns(
            [
                4,
                1,
            ]
        )
    )

    history_header.markdown(
        "### Mathematics history"
    )

    clear_clicked = clear_column.button(
        "Clear math history",
        use_container_width=True,
    )

    if clear_clicked:
        st.session_state.math_history = []
        st.rerun()

    if not st.session_state.math_history:
        st.info(
            "No mathematical explanations "
            "have been generated yet."
        )

    else:
        reversed_history = reversed(
            st.session_state.math_history
        )

        for index, item in enumerate(
            reversed_history,
            start=1,
        ):
            label = (
                item.get("topic")
                or item.get(
                    "original_input"
                )
                or f"Math item {index}"
            )

            with st.expander(
                str(label)[:100]
            ):
                if (
                    item.get("mode")
                    == "equation"
                ):
                    st.markdown(
                        (
                            "**Operation:** "
                            f"{item.get('operation', '')}"
                        )
                    )

                    st.latex(
                        item.get(
                            "parsed_latex",
                            "",
                        )
                    )

                    st.markdown(
                        "**Result**"
                    )

                    st.latex(
                        item.get(
                            "result_latex",
                            "",
                        )
                    )

                st.markdown(
                    item.get(
                        "explanation",
                        "",
                    )
                )

                _render_paper_sources(
                    item.get(
                        "paper_sources",
                        [],
                    )
                )