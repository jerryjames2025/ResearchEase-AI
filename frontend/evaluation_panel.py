from __future__ import annotations

import json
import os

import streamlit as st

from frontend.api_client import (
    APIClientError,
    ResearchEaseAPI,
)


DEFAULT_DATASET = {
    "dataset_name": (
        "research-paper-rag-baseline"
    ),
    "cases": [
        {
            "case_id": "methodology",
            "question": (
                "What methodology does "
                "the paper use?"
            ),
            "expected_pages": [2, 3],
            "expected_answer": (
                "Replace this with the expected "
                "methodology summary."
            ),
            "tags": {
                "category": "methodology"
            },
        },
        {
            "case_id": "results",
            "question": (
                "What are the main findings?"
            ),
            "expected_pages": [5, 6],
            "expected_answer": (
                "Replace this with the expected "
                "findings."
            ),
            "tags": {
                "category": "results"
            },
        },
    ],
}


def render_evaluation_panel(
    *,
    api: ResearchEaseAPI,
    session_id: str,
    ollama_model: str,
    explanation_level: str,
) -> None:
    st.subheader(
        "MLflow RAG Evaluation Lab"
    )

    st.caption(
        "Compare retrieval and answer quality across "
        "FAISS/Pinecone, embedding models, Top-K values, "
        "score thresholds, and LLM configurations."
    )

    try:
        health = api.evaluation_health()

        if health.get(
            "status"
        ) == "healthy":
            st.success(
                "MLflow tracking is connected."
            )
        else:
            st.warning(
                health.get(
                    "detail",
                    "MLflow is unavailable.",
                )
            )

    except APIClientError as exc:
        st.warning(
            f"MLflow health check failed: {exc}"
        )

    mlflow_ui_url = os.getenv(
        "RESEARCHEASE_MLFLOW_UI_URL",
        "http://127.0.0.1:5000",
    )

    st.link_button(
        "Open MLflow UI",
        mlflow_ui_url,
    )

    settings_columns = st.columns(
        4
    )

    top_k = settings_columns[
        0
    ].number_input(
        "Top K",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    minimum_score = settings_columns[
        1
    ].number_input(
        "Minimum similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
    )

    run_generation = settings_columns[
        2
    ].checkbox(
        "Generate answers",
        value=True,
    )

    use_llm_judge = settings_columns[
        3
    ].checkbox(
        "Use Ollama judge",
        value=False,
        help=(
            "Adds relevance, groundedness, and "
            "completeness scores. This approximately "
            "doubles LLM calls."
        ),
    )

    if (
        "evaluation_dataset_text"
        not in st.session_state
    ):
        st.session_state[
            "evaluation_dataset_text"
        ] = json.dumps(
            DEFAULT_DATASET,
            indent=2,
        )

    dataset_text = st.text_area(
        "Evaluation dataset JSON",
        key=(
            "evaluation_dataset_text"
        ),
        height=430,
    )

    button_columns = st.columns(
        2
    )

    if button_columns[0].button(
        "Load API template",
        use_container_width=True,
    ):
        try:
            template = (
                api.evaluation_template()
            )

            st.session_state[
                "evaluation_dataset_text"
            ] = json.dumps(
                template,
                indent=2,
            )

            st.rerun()

        except APIClientError as exc:
            st.error(str(exc))

    run_clicked = (
        button_columns[1].button(
            "Run RAG evaluation",
            type="primary",
            use_container_width=True,
        )
    )

    if run_clicked:
        try:
            parsed = json.loads(
                dataset_text
            )

            if isinstance(
                parsed,
                list,
            ):
                dataset_name = (
                    "manual-rag-evaluation"
                )
                cases = parsed

            elif isinstance(
                parsed,
                dict,
            ):
                dataset_name = str(
                    parsed.get(
                        "dataset_name",
                        (
                            "manual-rag-"
                            "evaluation"
                        ),
                    )
                )

                cases = parsed.get(
                    "cases",
                    [],
                )

            else:
                raise ValueError(
                    "Dataset JSON must be an object "
                    "or a list of cases."
                )

            if not cases:
                raise ValueError(
                    "The evaluation dataset "
                    "contains no cases."
                )

            with st.spinner(
                "Running retrieval, generation, "
                "scoring, and MLflow logging..."
            ):
                result = (
                    api.run_rag_evaluation(
                        {
                            "session_id": (
                                session_id
                            ),
                            "dataset_name": (
                                dataset_name
                            ),
                            "cases": cases,
                            "top_k": int(
                                top_k
                            ),
                            "minimum_score": (
                                float(
                                    minimum_score
                                )
                            ),
                            "run_generation": (
                                run_generation
                            ),
                            "use_llm_judge": (
                                use_llm_judge
                            ),
                            "ollama_model": (
                                ollama_model
                            ),
                            "explanation_level": (
                                explanation_level
                            ),
                        }
                    )
                )

            st.session_state[
                "evaluation_result"
            ] = result

            st.success(
                "RAG evaluation completed."
            )

        except json.JSONDecodeError as exc:
            st.error(
                f"Invalid dataset JSON: {exc}"
            )

        except (
            ValueError,
            APIClientError,
        ) as exc:
            st.error(str(exc))

    result = st.session_state.get(
        "evaluation_result",
        {},
    )

    if not result:
        return

    for warning in result.get(
        "warnings",
        [],
    ):
        st.warning(warning)

    run_id = result.get(
        "mlflow_run_id",
        "",
    )

    if run_id:
        st.info(
            f"MLflow run ID: {run_id}"
        )

    aggregate_metrics = result.get(
        "aggregate_metrics",
        {},
    )

    if aggregate_metrics:
        st.markdown(
            "### Aggregate metrics"
        )

        metric_items = list(
            aggregate_metrics.items()
        )

        for start in range(
            0,
            len(metric_items),
            4,
        ):
            columns = st.columns(4)

            for column, (
                metric_name,
                metric_value,
            ) in zip(
                columns,
                metric_items[
                    start:
                    start + 4
                ],
            ):
                column.metric(
                    metric_name.replace(
                        "_",
                        " ",
                    ).title(),
                    f"{metric_value:.4f}",
                )

    case_rows: list[dict] = []

    for case in result.get(
        "cases",
        [],
    ):
        metrics = case.get(
            "metrics",
            {},
        )

        case_rows.append(
            {
                "Case": case.get(
                    "case_id",
                    "",
                ),
                "Question": case.get(
                    "question",
                    "",
                ),
                "Expected pages": (
                    case.get(
                        "expected_pages",
                        [],
                    )
                ),
                "Retrieved pages": (
                    case.get(
                        "retrieved_pages",
                        [],
                    )
                ),
                "Hit rate": metrics.get(
                    (
                        "retrieval_"
                        "hit_rate_at_k"
                    )
                ),
                "MRR": metrics.get(
                    (
                        "retrieval_"
                        "mrr_at_k"
                    )
                ),
                "nDCG": metrics.get(
                    (
                        "retrieval_"
                        "ndcg_at_k"
                    )
                ),
                "Context overlap": (
                    metrics.get(
                        (
                            "answer_context_"
                            "overlap"
                        )
                    )
                ),
                "Error": case.get(
                    "error",
                    "",
                ),
            }
        )

    st.markdown(
        "### Per-case results"
    )

    st.dataframe(
        case_rows,
        use_container_width=True,
        hide_index=True,
    )

    for case in result.get(
        "cases",
        [],
    ):
        with st.expander(
            (
                f"{case.get('case_id', '')}: "
                f"{case.get('question', '')}"
            )
        ):
            if case.get("error"):
                st.error(
                    case["error"]
                )
                continue

            st.write(
                "**Expected pages:**",
                case.get(
                    "expected_pages",
                    [],
                ),
            )

            st.write(
                "**Retrieved pages:**",
                case.get(
                    "retrieved_pages",
                    [],
                ),
            )

            if case.get("answer"):
                st.markdown(
                    "#### Generated answer"
                )

                st.markdown(
                    case["answer"]
                )

            judge = case.get(
                "judge"
            )

            if judge:
                st.markdown(
                    "#### Ollama judge"
                )

                st.json(judge)

            with st.expander(
                "Retrieved evidence"
            ):
                for source in case.get(
                    "retrieved_sources",
                    [],
                ):
                    st.markdown(
                        (
                            f"**Rank "
                            f"{source['rank']} · "
                            f"Page "
                            f"{source['page_number']} · "
                            f"Score "
                            f"{source['score']:.4f}**"
                        )
                    )

                    st.write(
                        source.get(
                            "text",
                            "",
                        )
                    )

    st.download_button(
        "Download evaluation results",
        data=json.dumps(
            result,
            indent=2,
        ),
        file_name=(
            "researchease_"
            "rag_evaluation.json"
        ),
        mime="application/json",
    )