from __future__ import annotations

import json
import os

import streamlit as st

from config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_OLLAMA_MODEL,
)
from frontend.api_client import (
    APIClientError,
    ResearchEaseAPI,
)


st.set_page_config(
    page_title=(
        "ResearchEase AI — Version 7"
    ),
    page_icon="📚",
    layout="wide",
)

st.title(
    "📚 ResearchEase AI — Version 7"
)

st.caption(
    "API-driven Streamlit frontend connected "
    "to a FastAPI backend."
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
}

for key, value in (
    SESSION_DEFAULTS.items()
):
    if key not in st.session_state:
        st.session_state[
            key
        ] = value


with st.sidebar:
    st.header(
        "Backend"
    )

    api_url = st.text_input(
        "FastAPI URL",
        value=os.getenv(
            "RESEARCHEASE_API_URL",
            "http://127.0.0.1:8000",
        ),
    )

    ollama_model = st.text_input(
        "Ollama model",
        value=DEFAULT_OLLAMA_MODEL,
    )

    embedding_model = st.text_input(
        "Embedding model",
        value=(
            DEFAULT_EMBEDDING_MODEL
        ),
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


api = ResearchEaseAPI(
    api_url
)


try:
    readiness = api.readiness()

    if readiness.get(
        "ollama_connected"
    ):
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
    st.sidebar.error(
        str(exc)
    )


uploaded_file = st.file_uploader(
    "Upload primary research paper",
    type=["pdf"],
)


if st.button(
    "Upload paper to FastAPI",
    type="primary",
    disabled=(
        uploaded_file is None
    ),
):
    try:
        with st.spinner(
            "Uploading and extracting PDF..."
        ):
            result = api.upload_paper(
                uploaded_file.name,
                uploaded_file.getvalue(),
            )

        st.session_state.api_session_id = (
            result["session_id"]
        )

        st.session_state.paper_metadata = (
            result
        )

        st.session_state.analysis = ""
        st.session_state.index_metadata = {}
        st.session_state.chat_history = []

        st.success(
            (
                "Paper uploaded. Session ID: "
                f"{result['session_id']}"
            )
        )

    except APIClientError as exc:
        st.error(
            str(exc)
        )


session_id = (
    st.session_state.api_session_id
)


if not session_id:
    st.info(
        "Upload a research paper to create "
        "a FastAPI session."
    )

    st.stop()


metadata = (
    st.session_state.paper_metadata
)

metric_columns = st.columns(
    4
)

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


(
    paper_tab,
    chat_tab,
    research_tab,
    math_tab,
    literature_tab,
) = st.tabs(
    [
        "Paper",
        "Research Chat",
        "External Research",
        "Math",
        "Literature Review",
    ]
)


with paper_tab:
    st.subheader(
        "Paper processing"
    )

    action_columns = st.columns(
        2
    )

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
                        "ollama_model": (
                            ollama_model
                        ),
                        "explanation_level": (
                            explanation_level
                        ),
                    },
                )

            st.session_state.analysis = (
                result["analysis"]
            )

        except APIClientError as exc:
            st.error(
                str(exc)
            )

    if action_columns[1].button(
        "Build RAG index",
        type="primary",
        use_container_width=True,
    ):
        try:
            with st.spinner(
                "Creating embeddings and FAISS index..."
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
                    },
                )

            st.session_state.index_metadata = (
                result
            )

            st.success(
                (
                    "Index built with "
                    f"{result['chunk_count']} chunks."
                )
            )

        except APIClientError as exc:
            st.error(
                str(exc)
            )

    if st.session_state.analysis:
        st.markdown(
            st.session_state.analysis
        )

        st.download_button(
            "Download analysis",
            data=(
                st.session_state.analysis
            ),
            file_name=(
                "researchease_api_analysis.md"
            ),
            mime="text/markdown",
        )


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

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "response": result,
                }
            )

        except APIClientError as exc:
            st.error(
                str(exc)
            )

    for item in reversed(
        st.session_state.chat_history
    ):
        st.markdown(
            f"### Question\n{item['question']}"
        )

        response = item[
            "response"
        ]

        st.markdown(
            response["answer"]
        )

        with st.expander(
            "Paper evidence"
        ):
            for source in response.get(
                "paper_sources",
                [],
            ):
                st.markdown(
                    (
                        f"**Page "
                        f"{source['page_number']} · "
                        f"Similarity "
                        f"{source['score']:.3f}**"
                    )
                )

                st.write(
                    source["text"]
                )

        external_sources = (
            response.get(
                "external_sources",
                [],
            )
        )

        if external_sources:
            with st.expander(
                "External academic evidence"
            ):
                for source in external_sources:
                    st.markdown(
                        (
                            f"**{source['title']}** "
                            f"— {source['source']}"
                        )
                    )

                    st.write(
                        source.get(
                            "abstract",
                            "",
                        )
                    )


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
        try:
            with st.spinner(
                "Searching academic providers..."
            ):
                result = api.research_search(
                    {
                        "query": research_query,
                        "limit_per_source": 5,
                        "embedding_model": (
                            embedding_model
                        ),
                        "embedding_device": (
                            embedding_device
                        ),
                    }
                )

            st.session_state.research_results = (
                result["results"]
            )

            for warning in result.get(
                "warnings",
                [],
            ):
                st.warning(
                    warning
                )

        except APIClientError as exc:
            st.error(
                str(exc)
            )

    for source in (
        st.session_state.research_results
    ):
        with st.expander(
            (
                f"{source['title']} "
                f"— {source['source']}"
            )
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
            value=(
                "x**2 - 5*x + 6 = 0"
            ),
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
                        "source_text": (
                            expression
                        ),
                        "operation": (
                            operation
                        ),
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

                st.session_state.math_result = (
                    result
                )

            except APIClientError as exc:
                st.error(
                    str(exc)
                )

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

                st.session_state.math_result = (
                    result
                )

            except APIClientError as exc:
                st.error(
                    str(exc)
                )

    if st.session_state.math_result:
        result = (
            st.session_state.math_result
        )

        if result.get(
            "parsed_latex"
        ):
            st.latex(
                result["parsed_latex"]
            )

        if result.get(
            "result_latex"
        ):
            st.markdown(
                "### Verified result"
            )

            st.latex(
                result["result_latex"]
            )

        st.markdown(
            result["explanation"]
        )


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
        disabled=(
            not related_files
        ),
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

            st.session_state.literature_result = (
                result
            )

        except APIClientError as exc:
            st.error(
                str(exc)
            )

    if st.session_state.literature_result:
        result = (
            st.session_state
            .literature_result
        )

        st.dataframe(
            result["matrix"],
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            result["synthesis"]
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