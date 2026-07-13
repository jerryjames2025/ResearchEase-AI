from __future__ import annotations

import json
import os
from frontend.fine_tuning_panel import (
    render_fine_tuning_panel,
)

import streamlit as st
from frontend.evaluation_panel import (render_evaluation_panel,)

from config import DEFAULT_EMBEDDING_MODEL
from frontend.api_client import (
    APIClientError,
    ResearchEaseAPI,
)
from frontend.vector_controls import (
    render_vector_controls,
)


st.set_page_config(
    page_title="ResearchEase AI — Version 14.3",
    page_icon="📚",
    layout="wide",
)

st.title("📚 ResearchEase AI — Version 14.3")

st.caption(
    "FastAPI-powered research assistant with persistent storage, "
    "multi-provider LLM support, and switchable FAISS/Pinecone vector storage."
)


SESSION_DEFAULTS = {
    "api_session_id": "",
    "paper_metadata": {},
    "analysis": "",
    "index_metadata": {},
    "chat_history": [],
    "research_results": [],
    "math_result": {},
    "literature_result": {},
    "available_sessions": [],
    "evaluation_result": {},
}

for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------
# Sidebar configuration
# ---------------------------------------------------------

with st.sidebar:
    st.header("Backend")

    api_url = st.text_input(
        "FastAPI URL",
        value=os.getenv(
            "RESEARCHEASE_API_URL",
            "http://127.0.0.1:8000",
        ),
    )

    llm_provider = st.selectbox(
        "LLM provider",
        [
            "ollama",
            "google",
            "openai",
            "anthropic",
            "huggingface",
        ],
    )

    provider_default_models = {
        "ollama": "qwen2.5:1.5b",
        "google": "gemini-2.5-flash",
        "openai": "gpt-5.4-mini",
        "anthropic": "claude-sonnet-5",
        "huggingface": "microsoft/Phi-3-mini-4k-instruct",
    }

    llm_model = st.text_input(
        "LLM model",
        value=provider_default_models[llm_provider],
        key=f"llm_model_{llm_provider}",
    )

    enable_llm_fallback = st.checkbox(
        "Enable automatic provider fallback",
        value=True,
        help=(
            "When the selected provider fails, ResearchEase "
            "tries another configured provider."
        ),
    )

    embedding_model = st.text_input(
        "Embedding model",
        value=DEFAULT_EMBEDDING_MODEL,
    )

    embedding_device = st.selectbox(
        "Embedding device",
        [
            "auto",
            "cpu",
            "cuda",
        ],
    )

    explanation_level = st.selectbox(
        "Explanation level",
        [
            "Beginner",
            "Intermediate",
            "Advanced",
            "Explain like I am 10",
        ],
    )


# Existing request bodies still use this variable name.
ollama_model = llm_model


# ---------------------------------------------------------
# Create API client before calling API methods
# ---------------------------------------------------------

api = ResearchEaseAPI(
    base_url=api_url,
    llm_provider=llm_provider,
    llm_model=llm_model,
    enable_fallback=enable_llm_fallback,
)


# ---------------------------------------------------------
# Version 10 vector controls
# ---------------------------------------------------------

vector_backend = render_vector_controls(api)


# ---------------------------------------------------------
# Version 9 LLM provider status and testing
# ---------------------------------------------------------

with st.sidebar:
    st.header("LLM provider status")

    try:
        provider_result = api.llm_providers()

        for provider_info in provider_result.get(
            "providers",
            [],
        ):
            label = provider_info.get(
                "provider",
                "unknown",
            )

            configured = bool(
                provider_info.get(
                    "configured",
                    False,
                )
            )

            installed = bool(
                provider_info.get(
                    "installed",
                    False,
                )
            )

            detail = provider_info.get(
                "detail",
                "",
            )

            if configured:
                st.info(f"{label}: configured")
            elif installed:
                st.caption(f"{label}: {detail}")
            else:
                st.warning(f"{label}: {detail}")

        if st.button(
            "Test selected LLM",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    "Testing the selected LLM provider..."
                ):
                    test_result = api.test_llm_provider(
                        provider=llm_provider,
                        model=llm_model,
                        enable_fallback=enable_llm_fallback,
                    )

                actual_provider = test_result.get(
                    "actual_provider",
                    llm_provider,
                )

                actual_model = test_result.get(
                    "actual_model",
                    llm_model,
                )

                st.success("LLM test successful.")

                st.write(
                    "**Actual provider:**",
                    actual_provider,
                )

                st.write(
                    "**Actual model:**",
                    actual_model,
                )

                if test_result.get(
                    "fallback_used",
                    False,
                ):
                    st.warning(
                        "The primary provider failed, "
                        "so an automatic fallback was used."
                    )

                response_text = test_result.get(
                    "response",
                    "",
                )

                if response_text:
                    st.write(response_text)

            except APIClientError as exc:
                st.error(str(exc))

    except APIClientError as exc:
        st.warning(
            "Unable to load LLM provider status."
        )
        st.caption(str(exc))


# ---------------------------------------------------------
# Persistent storage controls
# ---------------------------------------------------------

with st.sidebar:
    st.header("Persistent storage")

    try:
        storage_status = api.storage_health()

        if storage_status.get("status") == "healthy":
            st.success(
                "PostgreSQL, MongoDB, Redis, "
                "and vector storage are ready."
            )
        else:
            st.warning(
                "One or more storage services are unavailable."
            )

        with st.expander("View storage services"):
            for service_name, service_status in (
                storage_status.get("services", {}).items()
            ):
                detail = service_status.get(
                    "detail",
                    "",
                )

                if service_status.get(
                    "healthy",
                    False,
                ):
                    st.success(
                        f"{service_name}: {detail}"
                    )
                else:
                    st.error(
                        f"{service_name}: {detail}"
                    )

    except APIClientError as exc:
        st.error(str(exc))

    if st.button(
        "Refresh saved sessions",
        use_container_width=True,
    ):
        try:
            result = api.list_sessions()

            st.session_state[
                "available_sessions"
            ] = result.get(
                "sessions",
                [],
            )

        except APIClientError as exc:
            st.error(str(exc))

    saved_sessions = st.session_state[
        "available_sessions"
    ]

    session_options = {
        (
            f"{item['filename']} | "
            f"{item['session_id'][:8]} | "
            f"{item['updated_at'][:19]}"
        ): item
        for item in saved_sessions
    }

    selected_session_label = st.selectbox(
        "Saved research sessions",
        options=[
            "",
            *session_options.keys(),
        ],
    )

    if st.button(
        "Load saved session",
        use_container_width=True,
        disabled=not selected_session_label,
    ):
        try:
            selected = session_options[
                selected_session_label
            ]

            session_id_to_load = selected[
                "session_id"
            ]

            session_data = api.get_session(
                session_id_to_load
            )

            st.session_state[
                "api_session_id"
            ] = session_id_to_load

            st.session_state[
                "paper_metadata"
            ] = {
                "session_id": session_id_to_load,
                "filename": session_data[
                    "filename"
                ],
                "page_count": session_data[
                    "page_count"
                ],
                "extracted_characters": session_data[
                    "extracted_characters"
                ],
                "created_at": session_data[
                    "created_at"
                ],
            }

            st.session_state[
                "index_metadata"
            ] = {
                "chunk_count": session_data.get(
                    "chunk_count",
                    0,
                ),
                "index_ready": session_data.get(
                    "index_ready",
                    False,
                ),
                "vector_backend": session_data.get(
                    "vector_backend",
                    "faiss",
                ),
                "vector_index_name": session_data.get(
                    "vector_index_name",
                    "",
                ),
                "vector_namespace": session_data.get(
                    "vector_namespace",
                    "",
                ),
                "vector_dimension": session_data.get(
                    "vector_dimension",
                    0,
                ),
            }

            st.session_state["analysis"] = ""

            if session_data.get(
                "analysis_ready",
                False,
            ):
                analysis_result = api.get_analysis(
                    session_id_to_load
                )

                st.session_state[
                    "analysis"
                ] = analysis_result.get(
                    "analysis",
                    "",
                )

            history_result = api.chat_history(
                session_id_to_load
            )

            st.session_state[
                "chat_history"
            ] = [
                {
                    "question": turn.get(
                        "question",
                        "",
                    ),
                    "response": {
                        "answer": turn.get(
                            "answer",
                            "",
                        ),
                        "paper_sources": turn.get(
                            "paper_sources",
                            [],
                        ),
                        "external_sources": turn.get(
                            "external_sources",
                            [],
                        ),
                    },
                }
                for turn in history_result.get(
                    "turns",
                    [],
                )
            ]

            st.success("Saved session loaded.")
            st.rerun()

        except APIClientError as exc:
            st.error(str(exc))


# ---------------------------------------------------------
# Backend readiness
# ---------------------------------------------------------

try:
    readiness = api.readiness()

    if readiness.get("ollama_connected"):
        st.sidebar.success(
            "FastAPI and Ollama are ready."
        )
    else:
        st.sidebar.warning(
            readiness.get(
                "detail",
                "Ollama is unavailable.",
            )
        )

except APIClientError as exc:
    st.sidebar.error(str(exc))


# ---------------------------------------------------------
# Upload primary paper
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload primary research paper",
    type=["pdf"],
)

if st.button(
    "Upload paper to FastAPI",
    type="primary",
    disabled=uploaded_file is None,
):
    try:
        with st.spinner(
            "Uploading and extracting PDF..."
        ):
            result = api.upload_paper(
                uploaded_file.name,
                uploaded_file.getvalue(),
            )

        st.session_state[
            "api_session_id"
        ] = result["session_id"]

        st.session_state[
            "paper_metadata"
        ] = result

        st.session_state["analysis"] = ""
        st.session_state[
            "index_metadata"
        ] = {}
        st.session_state[
            "chat_history"
        ] = []

        st.success(
            "Paper uploaded. Session ID: "
            f"{result['session_id']}"
        )

    except APIClientError as exc:
        st.error(str(exc))


session_id = st.session_state[
    "api_session_id"
]

if not session_id:
    st.info(
        "Upload a research paper or load a saved "
        "session to continue."
    )
    st.stop()


# ---------------------------------------------------------
# Session overview
# ---------------------------------------------------------

metadata = st.session_state[
    "paper_metadata"
]

metric_columns = st.columns(4)

metric_columns[0].metric(
    "Session",
    session_id[:8],
)

metric_columns[1].metric(
    "File",
    metadata.get(
        "filename",
        "",
    ),
)

metric_columns[2].metric(
    "Pages",
    metadata.get(
        "page_count",
        0,
    ),
)

metric_columns[3].metric(
    "Characters",
    metadata.get(
        "extracted_characters",
        0,
    ),
)


index_metadata = st.session_state.get(
    "index_metadata",
    {},
)

if index_metadata.get(
    "index_ready",
    False,
) or index_metadata.get(
    "chunk_count",
    0,
):
    st.caption(
        "Active vector index: "
        f"{index_metadata.get('vector_backend', 'faiss')} · "
        f"{index_metadata.get('chunk_count', 0)} chunks · "
        f"{index_metadata.get('vector_dimension', 0)} dimensions"
    )


(
    paper_tab,
    chat_tab,
    research_tab,
    math_tab,
    literature_tab,
    evaluation_tab,
    fine_tuning_tab,
) = st.tabs(
    [
        "Paper",
        "Research Chat",
        "External Research",
        "Math",
        "Literature Review",
        "RAG Evaluation",
        "LoRA Fine-Tuning",
    ]
)


# ---------------------------------------------------------
# Paper tab
# ---------------------------------------------------------

with paper_tab:
    st.subheader("Paper processing")

    action_columns = st.columns(2)

    if action_columns[0].button(
        "Analyze paper",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "FastAPI is analyzing the paper..."
            ):
                result = api.analyze_paper(
                    session_id,
                    {
                        "ollama_model": ollama_model,
                        "explanation_level": (
                            explanation_level
                        ),
                    },
                )

            st.session_state[
                "analysis"
            ] = result["analysis"]

            st.success(
                "Paper analysis completed."
            )

        except APIClientError as exc:
            st.error(str(exc))

    if action_columns[1].button(
        "Build RAG index",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                f"Creating embeddings and "
                f"{vector_backend} index..."
            ):
                result = api.build_index(
                    session_id,
                    {
                        "embedding_model": (
                            embedding_model
                        ),
                        "device": (
                            embedding_device
                        ),
                        "vector_backend": (
                            vector_backend
                        ),
                    },
                )

            st.session_state[
                "index_metadata"
            ] = {
                **result,
                "index_ready": True,
            }

            st.success(
                f"{result['vector_backend']} index "
                f"built with {result['chunk_count']} chunks "
                f"and {result['vector_dimension']} dimensions."
            )

            if result.get("index_path"):
                st.caption(
                    "FAISS index: "
                    f"{result['index_path']}"
                )

            if result.get(
                "vector_index_name"
            ):
                st.caption(
                    "Pinecone index: "
                    f"{result['vector_index_name']}"
                )

            if result.get(
                "vector_namespace"
            ):
                st.caption(
                    "Pinecone namespace: "
                    f"{result['vector_namespace']}"
                )

        except APIClientError as exc:
            st.error(
                "Unable to build the vector index: "
                f"{exc}"
            )

        except Exception as exc:
            st.error(
                "Unexpected indexing error: "
                f"{exc}"
            )

    if st.session_state["analysis"]:
        st.markdown(
            st.session_state["analysis"]
        )

        st.download_button(
            "Download analysis",
            data=st.session_state[
                "analysis"
            ],
            file_name=(
                "researchease_api_analysis.md"
            ),
            mime="text/markdown",
        )


# ---------------------------------------------------------
# Research chat tab
# ---------------------------------------------------------

with chat_tab:
    answer_mode = st.radio(
        "Answer mode",
        [
            "paper_only",
            "expanded_research",
        ],
        horizontal=True,
    )

    question = st.text_area(
        "Research question",
        placeholder=(
            "What is the main methodology?"
        ),
    )

    if st.button(
        "Ask research assistant",
        type="primary",
        use_container_width=True,
    ):
        if not question.strip():
            st.warning(
                "Enter a research question."
            )
        else:
            try:
                with st.spinner(
                    "Retrieving evidence and generating answer..."
                ):
                    result = api.chat(
                        session_id,
                        {
                            "question": question,
                            "answer_mode": (
                                answer_mode
                            ),
                            "ollama_model": (
                                ollama_model
                            ),
                            "explanation_level": (
                                explanation_level
                            ),
                            "top_k": 5,
                            "minimum_score": 0.15,
                        },
                    )

                st.session_state[
                    "chat_history"
                ].append(
                    {
                        "question": question,
                        "response": result,
                    }
                )

            except APIClientError as exc:
                st.error(str(exc))

    for item in reversed(
        st.session_state["chat_history"]
    ):
        st.markdown(
            f"### Question\n{item['question']}"
        )

        response = item["response"]

        st.markdown(
            response.get(
                "answer",
                "",
            )
        )

        with st.expander(
            "Paper evidence"
        ):
            paper_sources = response.get(
                "paper_sources",
                [],
            )

            if not paper_sources:
                st.caption(
                    "No paper evidence returned."
                )

            for source in paper_sources:
                st.markdown(
                    f"**Page "
                    f"{source.get('page_number', 0)} · "
                    f"Similarity "
                    f"{float(source.get('score', 0.0)):.3f}**"
                )

                st.write(
                    source.get(
                        "text",
                        "",
                    )
                )

        external_sources = response.get(
            "external_sources",
            [],
        )

        if external_sources:
            with st.expander(
                "External academic evidence"
            ):
                for source in external_sources:
                    st.markdown(
                        f"**{source.get('title', 'Untitled')}** "
                        f"— {source.get('source', 'Unknown')}"
                    )

                    st.write(
                        source.get(
                            "abstract",
                            "",
                        )
                    )


# ---------------------------------------------------------
# External research tab
# ---------------------------------------------------------

with research_tab:
    research_query = st.text_input(
        "External academic query",
        value=(
            "transformer-based research "
            "paper analysis"
        ),
    )

    if st.button(
        "Search academic sources",
        type="primary",
    ):
        if not research_query.strip():
            st.warning(
                "Enter an academic search query."
            )
        else:
            try:
                with st.spinner(
                    "Searching academic providers..."
                ):
                    result = api.research_search(
                        {
                            "query": (
                                research_query
                            ),
                            "limit_per_source": 5,
                            "embedding_model": (
                                embedding_model
                            ),
                            "embedding_device": (
                                embedding_device
                            ),
                        }
                    )

                st.session_state[
                    "research_results"
                ] = result.get(
                    "results",
                    [],
                )

                for warning in result.get(
                    "warnings",
                    [],
                ):
                    st.warning(warning)

            except APIClientError as exc:
                st.error(str(exc))

    for source in st.session_state[
        "research_results"
    ]:
        with st.expander(
            f"{source.get('title', 'Untitled')} "
            f"— {source.get('source', 'Unknown')}"
        ):
            st.write(
                source.get(
                    "abstract",
                    "No abstract available.",
                )
            )

            st.write(
                "Authors:",
                ", ".join(
                    source.get(
                        "authors",
                        [],
                    )
                ),
            )

            st.write(
                "Year:",
                source.get(
                    "year",
                    "Unknown",
                ),
            )


# ---------------------------------------------------------
# Math tab
# ---------------------------------------------------------

with math_tab:
    math_mode = st.radio(
        "Math mode",
        [
            "Equation",
            "Topic",
        ],
        horizontal=True,
    )

    use_paper_context = st.checkbox(
        "Use paper context",
        value=True,
    )

    if math_mode == "Equation":
        expression = st.text_area(
            "Expression",
            value="x**2 - 5*x + 6 = 0",
        )

        operation = st.selectbox(
            "Operation",
            [
                "Explain and simplify",
                "Solve equation",
                "Differentiate",
                "Integrate",
                "Factor",
                "Expand",
                "Evaluate substitutions",
                "Calculate limit",
            ],
        )

        if st.button(
            "Calculate through API",
            type="primary",
        ):
            try:
                result = api.explain_equation(
                    {
                        "source_text": expression,
                        "operation": operation,
                        "variable_name": "x",
                        "ollama_model": (
                            ollama_model
                        ),
                        "explanation_level": (
                            explanation_level
                        ),
                        "session_id": (
                            session_id
                        ),
                        "use_paper_context": (
                            use_paper_context
                        ),
                    }
                )

                st.session_state[
                    "math_result"
                ] = result

            except APIClientError as exc:
                st.error(str(exc))

    else:
        topic = st.text_input(
            "Mathematical topic",
            value=(
                "Scaled dot-product attention"
            ),
        )

        if st.button(
            "Explain topic through API",
            type="primary",
        ):
            try:
                result = api.explain_topic(
                    {
                        "topic": topic,
                        "ollama_model": (
                            ollama_model
                        ),
                        "explanation_level": (
                            explanation_level
                        ),
                        "session_id": (
                            session_id
                        ),
                        "use_paper_context": (
                            use_paper_context
                        ),
                    }
                )

                st.session_state[
                    "math_result"
                ] = result

            except APIClientError as exc:
                st.error(str(exc))

    if st.session_state[
        "math_result"
    ]:
        result = st.session_state[
            "math_result"
        ]

        if result.get("parsed_latex"):
            st.latex(
                result["parsed_latex"]
            )

        if result.get("result_latex"):
            st.markdown(
                "### Verified result"
            )

            st.latex(
                result["result_latex"]
            )

        st.markdown(
            result.get(
                "explanation",
                "",
            )
        )


# ---------------------------------------------------------
# Literature review tab
# ---------------------------------------------------------

with literature_tab:
    literature_topic = st.text_input(
        "Literature-review topic",
        value=(
            "Transformer-powered "
            "academic research assistants"
        ),
    )

    related_files = st.file_uploader(
        "Upload supporting PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    include_primary = st.checkbox(
        "Include primary paper",
        value=True,
    )

    if st.button(
        "Generate literature review through API",
        type="primary",
        disabled=not related_files,
    ):
        try:
            with st.spinner(
                "Analyzing and comparing papers..."
            ):
                result = api.literature_review(
                    files=[
                        (
                            file.name,
                            file.getvalue(),
                        )
                        for file in related_files
                    ],
                    form_data={
                        "topic": (
                            literature_topic
                        ),
                        "primary_session_id": (
                            session_id
                            if include_primary
                            else ""
                        ),
                        "ollama_model": (
                            ollama_model
                        ),
                        "explanation_level": (
                            explanation_level
                        ),
                    },
                )

            st.session_state[
                "literature_result"
            ] = result

        except APIClientError as exc:
            st.error(str(exc))

    if st.session_state[
        "literature_result"
    ]:
        result = st.session_state[
            "literature_result"
        ]

        st.dataframe(
            result.get(
                "matrix",
                [],
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            result.get(
                "synthesis",
                "",
            )
        )

        st.download_button(
            "Download literature JSON",
            data=json.dumps(
                result,
                indent=2,
            ),
            file_name=(
                "researchease_"
                "literature_api.json"
            ),
            mime="application/json",
        )
# ---------------------------------------------------------
# Version 11 RAG evaluation tab
# ---------------------------------------------------------

with evaluation_tab:
    render_evaluation_panel(
        api=api,
        session_id=session_id,
        ollama_model=ollama_model,
        explanation_level=(
            explanation_level
        ),
    )
# ---------------------------------------------------------
# Version 12 LoRA fine-tuning tab
# ---------------------------------------------------------

with fine_tuning_tab:
    render_fine_tuning_panel(
        api
    )