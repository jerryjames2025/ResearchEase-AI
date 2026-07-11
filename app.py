from __future__ import annotations

import hashlib
import os
from pathlib import Path
from ui.math_tab import (render_math_tab,)
from ui.literature_tab import (render_literature_tab,)
from ui.langchain_tab import (render_langchain_tab,)

import streamlit as st

from config import (
    APP_TITLE,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_EXTERNAL_RESULTS_PER_SOURCE,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_TOP_K,
    MAX_EXTERNAL_RESULTS_PER_SOURCE,
    MIN_RETRIEVAL_SCORE,
)
from research.aggregator import (
    search_academic_sources,
)
from research.external_answer import (
    answer_with_external_research,
    retrieve_paper_sources,
)
from research.models import AcademicPaper
from research.ranker import (
    rank_academic_papers,
)
from services.embedding_model import (
    load_embedding_service,
)
from services.ollama_client import (
    OllamaError,
    analyze_paper,
    check_ollama,
)
from services.pdf_parser import (
    create_page_chunks,
    extract_pdf,
)
from services.rag_engine import (
    answer_question,
)
from services.report_builder import (
    build_pdf_report,
    build_text_report,
)
from services.vector_store import (
    FaissVectorStore,
)


# ---------------------------------------------------------
# Streamlit page
# ---------------------------------------------------------

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📚",
    layout="wide",
)

st.title(
    "📚 ResearchEase AI — Version 3"
)

st.caption(
    "Upload a paper, chat with page-grounded RAG, "
    "and expand the research through Semantic Scholar, "
    "arXiv, and Crossref."
)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

SESSION_DEFAULTS = {
    "paper": None,
    "paper_hash": None,
    "analysis": "",
    "partial_summaries": [],
    "paper_chunks": [],
    "vector_store": None,
    "index_signature": None,
    "index_device": None,
    "chat_history": [],
    "external_results": [],
    "external_query": "",
    "external_warnings": [],
    "external_history": [],
    "math_history": [],
    "literature_summaries": [],
"literature_synthesis": "",
"literature_topic": "",
}

for key, default_value in (
    SESSION_DEFAULTS.items()
):
    if key not in st.session_state:
        st.session_state[
            key
        ] = default_value


def reset_document_state() -> None:
    """
    Clear the current paper and related results.
    """

    for key, value in (
        SESSION_DEFAULTS.items()
    ):
        st.session_state[
            key
        ] = value


def serialize_paper(
    paper: AcademicPaper,
) -> dict:
    """
    Convert AcademicPaper into session-safe data.
    """

    return {
        "source": paper.source,
        "source_id": paper.source_id,
        "title": paper.title,
        "authors": paper.authors,
        "year": paper.year,
        "abstract": paper.abstract,
        "url": paper.url,
        "doi": paper.doi,
        "venue": paper.venue,
        "citation_count": (
            paper.citation_count
        ),
        "publication_date": (
            paper.publication_date
        ),
        "pdf_url": paper.pdf_url,
        "relevance_score": (
            paper.relevance_score
        ),
    }


def serialize_paper_sources(
    sources,
) -> list[dict]:
    """
    Convert FAISS search results into session-safe data.
    """

    return [
        {
            "rank": source.rank,
            "score": source.score,
            "page_number": (
                source.chunk.page_number
            ),
            "chunk_number": (
                source.chunk
                .chunk_number_on_page
            ),
            "text": source.chunk.text,
        }
        for source in sources
    ]


def render_paper_sources(
    sources: list[dict],
) -> None:
    """
    Display retrieved PDF evidence.
    """

    if not sources:
        return

    with st.expander(
        "View uploaded-paper evidence"
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


def render_external_papers(
    papers: list[dict],
    heading: str = "External sources",
) -> None:
    """
    Display academic paper records.
    """

    if not papers:
        st.info(
            "No external academic records "
            "were returned."
        )

        return

    st.markdown(
        f"### {heading}"
    )

    for index, paper in enumerate(
        papers,
        start=1,
    ):
        score = paper.get(
            "relevance_score"
        )

        score_text = (
            f" · Relevance {score:.3f}"
            if score is not None
            else ""
        )

        year = (
            paper.get("year")
            or "Year unavailable"
        )

        citations = paper.get(
            "citation_count"
        )

        citation_text = (
            f" · Citations {citations}"
            if citations is not None
            else ""
        )

        title = (
            f"{index}. {paper['title']} — "
            f"{paper['source']} ({year})"
        )

        with st.expander(
            title
        ):
            authors = ", ".join(
                paper.get(
                    "authors"
                )
                or []
            )

            st.markdown(
                "**Authors:** "
                f"{authors or 'Not available'}"
            )

            st.markdown(
                "**Venue:** "
                f"{paper.get('venue') or 'Not available'}"
            )

            st.markdown(
                (
                    f"**Source:** {paper['source']}"
                    f"{score_text}"
                    f"{citation_text}"
                )
            )

            if paper.get("doi"):
                st.markdown(
                    f"**DOI:** `{paper['doi']}`"
                )

            st.markdown(
                "**Abstract or metadata summary:**"
            )

            st.write(
                paper.get("abstract")
                or (
                    "No abstract was supplied "
                    "by this provider."
                )
            )

            link_columns = st.columns(
                2
            )

            if paper.get("url"):
                with link_columns[0]:
                    st.link_button(
                        "Open record",
                        paper["url"],
                    )

            if paper.get("pdf_url"):
                with link_columns[1]:
                    st.link_button(
                        "Open available PDF",
                        paper["pdf_url"],
                    )


# Cache identical academic searches for 15 minutes.
@st.cache_data(
    ttl=900,
    show_spinner=False,
)
def cached_academic_search(
    query: str,
    limit_per_source: int,
    use_semantic_scholar: bool,
    use_arxiv: bool,
    use_crossref: bool,
    semantic_scholar_api_key: str,
    crossref_mailto: str,
):
    return search_academic_sources(
        query=query,
        limit_per_source=(
            limit_per_source
        ),
        use_semantic_scholar=(
            use_semantic_scholar
        ),
        use_arxiv=use_arxiv,
        use_crossref=use_crossref,
        semantic_scholar_api_key=(
            semantic_scholar_api_key
        ),
        crossref_mailto=(
            crossref_mailto
        ),
    )


def perform_external_search(
    query: str,
    limit_per_source: int,
    embedding_model_name: str,
    embedding_device: str,
    use_semantic_scholar: bool,
    use_arxiv: bool,
    use_crossref: bool,
    semantic_scholar_api_key: str,
    crossref_mailto: str,
) -> tuple[
    list[AcademicPaper],
    list[str],
]:
    """
    Search providers, merge duplicates and rank results.
    """

    response = cached_academic_search(
        query,
        limit_per_source,
        use_semantic_scholar,
        use_arxiv,
        use_crossref,
        semantic_scholar_api_key,
        crossref_mailto,
    )

    if not response.papers:
        return (
            [],
            response.warnings,
        )

    embedding_service = (
        load_embedding_service(
            embedding_model_name,
            embedding_device,
        )
    )

    ranked = rank_academic_papers(
        query=query,
        papers=response.papers,
        embedding_service=(
            embedding_service
        ),
    )

    return (
        ranked,
        response.warnings,
    )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header(
        "Models"
    )

    ollama_model = st.text_input(
        "Ollama model",
        value=DEFAULT_OLLAMA_MODEL,
    ).strip()

    embedding_model_name = (
        st.text_input(
            "Embedding model",
            value=(
                DEFAULT_EMBEDDING_MODEL
            ),
        ).strip()
    )

    embedding_device = st.selectbox(
        "Embedding device",
        [
            "auto",
            "cpu",
            "cuda",
        ],
        index=0,
    )

    explanation_level = (
        st.selectbox(
            "Explanation level",
            [
                "Beginner",
                "Intermediate",
                "Advanced",
                "Explain like I am 10",
            ],
        )
    )

    st.header(
        "RAG settings"
    )

    top_k = st.slider(
        "Paper chunks",
        min_value=2,
        max_value=10,
        value=DEFAULT_TOP_K,
    )

    minimum_score = st.slider(
        "Minimum paper similarity",
        min_value=0.0,
        max_value=0.6,
        value=float(
            MIN_RETRIEVAL_SCORE
        ),
        step=0.01,
    )

    answer_mode = st.radio(
        "Chat answer mode",
        [
            "Paper Only",
            "Expanded Research",
        ],
    )

    st.header(
        "Academic search"
    )

    use_semantic_scholar = (
        st.checkbox(
            "Semantic Scholar",
            value=True,
        )
    )

    use_arxiv = st.checkbox(
        "arXiv",
        value=True,
    )

    use_crossref = st.checkbox(
        "Crossref",
        value=True,
    )

    external_limit = st.slider(
        "Results per provider",
        min_value=2,
        max_value=(
            MAX_EXTERNAL_RESULTS_PER_SOURCE
        ),
        value=(
            DEFAULT_EXTERNAL_RESULTS_PER_SOURCE
        ),
    )

    semantic_scholar_api_key = (
        st.text_input(
            "Semantic Scholar API key (optional)",
            value=os.getenv(
                "SEMANTIC_SCHOLAR_API_KEY",
                "",
            ),
            type="password",
        ).strip()
    )

    crossref_mailto = st.text_input(
        "Crossref contact email (optional)",
        value=os.getenv(
            "CROSSREF_MAILTO",
            "",
        ),
        help=(
            "Used for Crossref's polite pool "
            "when supplied."
        ),
    ).strip()

    (
        connected,
        connection_message,
    ) = check_ollama()

    if connected:
        st.success(
            connection_message
        )
    else:
        st.warning(
            connection_message
        )

    st.warning(
        "External answers use titles, metadata, "
        "and abstracts returned by academic APIs. "
        "They do not automatically read the complete "
        "external papers."
    )


# ---------------------------------------------------------
# PDF upload
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a text-based research paper",
    type=["pdf"],
)

if uploaded_file is not None:
    current_hash = hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()

    if (
        current_hash
        != st.session_state.paper_hash
    ):
        reset_document_state()

        try:
            with st.spinner(
                "Extracting page-aware PDF text..."
            ):
                st.session_state.paper = (
                    extract_pdf(
                        uploaded_file
                    )
                )

            st.session_state.paper_hash = (
                current_hash
            )

        except ValueError as exc:
            st.error(
                str(exc)
            )


paper = st.session_state.paper

if paper is None:
    st.info(
        "Upload a PDF containing selectable "
        "text to begin."
    )

    st.stop()


# ---------------------------------------------------------
# Invalidate old index when embedding settings change
# ---------------------------------------------------------

current_index_signature = (
    st.session_state.paper_hash,
    embedding_model_name,
    embedding_device,
)

if (
    st.session_state.vector_store
    is not None
    and st.session_state.index_signature
    != current_index_signature
):
    st.session_state.paper_chunks = []
    st.session_state.vector_store = None
    st.session_state.index_signature = None
    st.session_state.index_device = None
    st.session_state.chat_history = []

    st.warning(
        "Embedding settings changed. "
        "Build the paper index again."
    )


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

metrics = st.columns(
    4
)

metrics[0].metric(
    "File",
    paper.filename,
)

metrics[1].metric(
    "Pages",
    paper.page_count,
)

metrics[2].metric(
    "Characters",
    f"{len(paper.text):,}",
)

metrics[3].metric(
    "RAG index",
    (
        f"{st.session_state.vector_store.size} chunks"
        if st.session_state.vector_store
        is not None
        else "Not built"
    ),
)


# ---------------------------------------------------------
# Main tabs
# ---------------------------------------------------------

(
    analysis_tab,
    chat_tab,
    external_tab,
    literature_tab,
    math_tab,
    langchain_tab,
    chunks_tab,
    download_tab,
) = st.tabs(
    [
        "Paper Analysis",
        "Research Chat",
        "External Research",
        "Literature Review",
        "Math Explainer",
        "LangChain Workflow",
        "Indexed Chunks",
        "Download Report",
    ]
)


# ---------------------------------------------------------
# Paper Analysis
# ---------------------------------------------------------

with analysis_tab:
    st.subheader(
        "Structured complete-paper analysis"
    )

    analyze_clicked = st.button(
        "Analyze complete paper",
        type="primary",
        disabled=(
            not connected
            or not ollama_model
        ),
    )

    if analyze_clicked:
        progress = st.progress(
            0.0
        )

        status = st.empty()

        def update_progress(
            completed: int,
            total: int,
            message: str,
        ) -> None:
            fraction = (
                completed / total
                if total
                else 0.0
            )

            progress.progress(
                min(
                    max(
                        fraction,
                        0.0,
                    ),
                    1.0,
                )
            )

            status.write(
                message
            )

        try:
            (
                analysis,
                partial_summaries,
            ) = analyze_paper(
                text=paper.text,
                filename=paper.filename,
                page_count=paper.page_count,
                model=ollama_model,
                explanation_level=(
                    explanation_level
                ),
                progress_callback=(
                    update_progress
                ),
            )

            st.session_state.analysis = (
                analysis
            )

            st.session_state.partial_summaries = (
                partial_summaries
            )

            progress.progress(
                1.0
            )

            status.success(
                "Analysis complete."
            )

        except (
            OllamaError,
            ValueError,
        ) as exc:
            st.error(
                str(exc)
            )

    if st.session_state.analysis:
        st.markdown(
            st.session_state.analysis
        )
    else:
        st.info(
            "Click the button to generate "
            "a structured paper analysis."
        )


# ---------------------------------------------------------
# Research Chat
# ---------------------------------------------------------

with chat_tab:
    st.subheader(
        "Paper and expanded academic research chat"
    )

    if (
        st.session_state.vector_store
        is None
    ):
        st.info(
            "Build the paper index before "
            "using chat."
        )

        build_index = st.button(
            "Build paper RAG index",
            type="primary",
        )

        if build_index:
            try:
                with st.spinner(
                    "Creating page chunks and "
                    "transformer embeddings..."
                ):
                    chunks = (
                        create_page_chunks(
                            paper
                        )
                    )

                    embedding_service = (
                        load_embedding_service(
                            embedding_model_name,
                            embedding_device,
                        )
                    )

                    embeddings = (
                        embedding_service
                        .encode_documents(
                            [
                                chunk.text
                                for chunk
                                in chunks
                            ]
                        )
                    )

                    vector_store = (
                        FaissVectorStore(
                            chunks=chunks,
                            embeddings=embeddings,
                        )
                    )

                    st.session_state.paper_chunks = (
                        chunks
                    )

                    st.session_state.vector_store = (
                        vector_store
                    )

                    st.session_state.index_signature = (
                        current_index_signature
                    )

                    st.session_state.index_device = (
                        embedding_service.device
                    )

                st.success(
                    (
                        "Index created with "
                        f"{vector_store.size} chunks "
                        "on "
                        f"{embedding_service.device}."
                    )
                )

                st.rerun()

            except Exception as exc:
                st.error(
                    (
                        "Could not build the index. "
                        "Select CPU if CUDA failed. "
                        f"Error: {exc}"
                    )
                )

    else:
        status_columns = st.columns(
            [
                4,
                1,
            ]
        )

        status_columns[0].success(
            (
                "Index ready · "
                f"{st.session_state.vector_store.size} "
                "chunks · Mode: "
                f"{answer_mode}"
            )
        )

        if status_columns[1].button(
            "Clear chat"
        ):
            st.session_state.chat_history = []

            st.rerun()

        # Display saved conversation.
        for message in (
            st.session_state.chat_history
        ):
            with st.chat_message(
                message["role"]
            ):
                st.markdown(
                    message["content"]
                )

                if (
                    message["role"]
                    == "assistant"
                ):
                    render_paper_sources(
                        message.get(
                            "paper_sources",
                            [],
                        )
                    )

                    external_sources = (
                        message.get(
                            "external_sources",
                            [],
                        )
                    )

                    if external_sources:
                        render_external_papers(
                            external_sources,
                            "External evidence used",
                        )

        question = st.chat_input(
            (
                "Ask about the paper or request "
                "broader academic knowledge"
            ),
            disabled=(
                not connected
                or not ollama_model
            ),
        )

        if question:
            st.session_state.chat_history.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message(
                "user"
            ):
                st.markdown(
                    question
                )

            with st.chat_message(
                "assistant"
            ):
                try:
                    embedding_service = (
                        load_embedding_service(
                            embedding_model_name,
                            embedding_device,
                        )
                    )

                    if answer_mode == "Paper Only":
                        with st.spinner(
                            "Searching the uploaded paper..."
                        ):
                            result = answer_question(
                                question=question,
                                vector_store=(
                                    st.session_state
                                    .vector_store
                                ),
                                embedding_service=(
                                    embedding_service
                                ),
                                ollama_model=(
                                    ollama_model
                                ),
                                explanation_level=(
                                    explanation_level
                                ),
                                chat_history=(
                                    st.session_state
                                    .chat_history[:-1]
                                ),
                                top_k=top_k,
                                minimum_score=(
                                    minimum_score
                                ),
                            )

                        paper_sources = (
                            serialize_paper_sources(
                                result.sources
                            )
                        )

                        external_sources: list[
                            dict
                        ] = []

                        answer = result.answer

                    else:
                        with st.spinner(
                            "Searching the uploaded paper "
                            "and external academic sources..."
                        ):
                            paper_results = (
                                retrieve_paper_sources(
                                    question=question,
                                    vector_store=(
                                        st.session_state
                                        .vector_store
                                    ),
                                    embedding_service=(
                                        embedding_service
                                    ),
                                    top_k=top_k,
                                    minimum_score=(
                                        minimum_score
                                    ),
                                )
                            )

                            (
                                external_papers,
                                warnings,
                            ) = perform_external_search(
                                query=question,
                                limit_per_source=(
                                    external_limit
                                ),
                                embedding_model_name=(
                                    embedding_model_name
                                ),
                                embedding_device=(
                                    embedding_device
                                ),
                                use_semantic_scholar=(
                                    use_semantic_scholar
                                ),
                                use_arxiv=(
                                    use_arxiv
                                ),
                                use_crossref=(
                                    use_crossref
                                ),
                                semantic_scholar_api_key=(
                                    semantic_scholar_api_key
                                ),
                                crossref_mailto=(
                                    crossref_mailto
                                ),
                            )

                            # Limit prompt size.
                            external_papers = (
                                external_papers[:8]
                            )

                            expanded = (
                                answer_with_external_research(
                                    question=question,
                                    paper_sources=(
                                        paper_results
                                    ),
                                    external_sources=(
                                        external_papers
                                    ),
                                    ollama_model=(
                                        ollama_model
                                    ),
                                    explanation_level=(
                                        explanation_level
                                    ),
                                )
                            )

                        for warning in warnings:
                            st.warning(
                                warning
                            )

                        paper_sources = (
                            serialize_paper_sources(
                                expanded.paper_sources
                            )
                        )

                        external_sources = [
                            serialize_paper(
                                paper_item
                            )
                            for paper_item
                            in expanded.external_sources
                        ]

                        answer = expanded.answer

                    st.markdown(
                        answer
                    )

                    render_paper_sources(
                        paper_sources
                    )

                    if external_sources:
                        render_external_papers(
                            external_sources,
                            "External evidence used",
                        )

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "paper_sources": (
                                paper_sources
                            ),
                            "external_sources": (
                                external_sources
                            ),
                        }
                    )

                except Exception as exc:
                    error_message = (
                        "Unable to answer: "
                        f"{exc}"
                    )

                    st.error(
                        error_message
                    )

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": (
                                error_message
                            ),
                            "paper_sources": [],
                            "external_sources": [],
                        }
                    )


# ---------------------------------------------------------
# External Research tab
# ---------------------------------------------------------

with external_tab:
    st.subheader(
        "Search related papers and external "
        "academic knowledge"
    )

    default_query = (
        Path(
            paper.filename
        )
        .stem
        .replace(
            "_",
            " ",
        )
        .replace(
            "-",
            " ",
        )
    )

    external_query = st.text_input(
        "Academic search query",
        value=(
            st.session_state.external_query
            or default_query
        ),
        placeholder=(
            "Example: transformer traffic forecasting"
        ),
    )

    search_clicked = st.button(
        "Search academic sources",
        type="primary",
    )

    if search_clicked:
        if not external_query.strip():
            st.error(
                "Enter an academic search query."
            )

        else:
            try:
                with st.spinner(
                    "Searching and semantically "
                    "ranking papers..."
                ):
                    (
                        papers,
                        warnings,
                    ) = perform_external_search(
                        query=external_query,
                        limit_per_source=(
                            external_limit
                        ),
                        embedding_model_name=(
                            embedding_model_name
                        ),
                        embedding_device=(
                            embedding_device
                        ),
                        use_semantic_scholar=(
                            use_semantic_scholar
                        ),
                        use_arxiv=(
                            use_arxiv
                        ),
                        use_crossref=(
                            use_crossref
                        ),
                        semantic_scholar_api_key=(
                            semantic_scholar_api_key
                        ),
                        crossref_mailto=(
                            crossref_mailto
                        ),
                    )

                serialized_results = [
                    serialize_paper(
                        paper_item
                    )
                    for paper_item in papers
                ]

                st.session_state.external_query = (
                    external_query
                )

                st.session_state.external_results = (
                    serialized_results
                )

                st.session_state.external_warnings = (
                    warnings
                )

                st.session_state.external_history.append(
                    {
                        "query": external_query,
                        "results": (
                            serialized_results[:15]
                        ),
                        "warnings": warnings,
                    }
                )

            except Exception as exc:
                st.error(
                    (
                        "External search failed: "
                        f"{exc}"
                    )
                )

    for warning in (
        st.session_state.external_warnings
    ):
        st.warning(
            warning
        )

    if st.session_state.external_results:
        st.success(
            (
                "Found "
                f"{len(st.session_state.external_results)} "
                "unique academic records."
            )
        )

        render_external_papers(
            st.session_state.external_results,
            "Semantically ranked related papers",
        )

    else:
        st.info(
            "Enter a topic, method, equation name, "
            "dataset, or research question."
        )
        

# ---------------------------------------------------------
# Version 5: Literature Review
# ---------------------------------------------------------

with literature_tab:
    render_literature_tab(
        primary_paper=paper,
        ollama_model=ollama_model,
        explanation_level=(
            explanation_level
        ),
        connected=connected,
        external_results=(
            st.session_state.external_results
        ),
    )

# ---------------------------------------------------------
# Mathematical Equation and Topic Explainer
# ---------------------------------------------------------

with math_tab:
    render_math_tab(
        ollama_model=ollama_model,
        explanation_level=(
            explanation_level
        ),
        connected=connected,
        vector_store=(
            st.session_state.vector_store
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

# ---------------------------------------------------------
# Version 6: LangChain Orchestration
# ---------------------------------------------------------

with langchain_tab:
    render_langchain_tab(
        ollama_model=(
            ollama_model
        ),

        connected=(
            connected
        ),

        has_paper=(
            paper is not None
        ),

        external_results=(
            st.session_state
            .external_results
        ),
    )

# ---------------------------------------------------------
# Indexed Chunks tab
# ---------------------------------------------------------

with chunks_tab:
    st.subheader(
        "Page-aware paper chunks"
    )

    if not st.session_state.paper_chunks:
        st.info(
            "Build the paper RAG index "
            "to inspect its chunks."
        )

    else:
        available_pages = sorted(
            {
                chunk.page_number
                for chunk
                in st.session_state.paper_chunks
            }
        )

        page_filter = st.selectbox(
            "Page filter",
            [
                "All",
                *available_pages,
            ],
        )

        visible_chunks = (
            st.session_state.paper_chunks
        )

        if page_filter != "All":
            visible_chunks = [
                chunk
                for chunk in visible_chunks
                if (
                    chunk.page_number
                    == page_filter
                )
            ]

        for chunk in visible_chunks:
            with st.expander(
                chunk.source_label
            ):
                st.write(
                    chunk.text
                )


# ---------------------------------------------------------
# Download Report tab
# ---------------------------------------------------------

with download_tab:
    st.subheader(
        "Download the combined research report"
    )

    no_results = (
    not st.session_state.analysis
    and not st.session_state.chat_history
    and not st.session_state.external_history
    and not st.session_state.math_history
)

    if no_results:
        st.info(
            "Generate an analysis, chat answer, "
            "or external search first."
        )

    else:
        text_report = build_text_report(
    filename=paper.filename,
    page_count=paper.page_count,
    explanation_level=(
        explanation_level
    ),
    analysis=(
        st.session_state.analysis
    ),
    chat_history=(
        st.session_state.chat_history
    ),
    external_history=(
        st.session_state.external_history
    ),
    math_history=(
        st.session_state.math_history
    ),
)

        try:
            pdf_report = build_pdf_report(
    filename=paper.filename,
    page_count=paper.page_count,
    explanation_level=(
        explanation_level
    ),
    analysis=(
        st.session_state.analysis
    ),
    chat_history=(
        st.session_state.chat_history
    ),
    external_history=(
        st.session_state.external_history
    ),
    math_history=(
        st.session_state.math_history
    ),
)

        except Exception as exc:
            pdf_report = None

            st.warning(
                (
                    "PDF generation failed: "
                    f"{exc}"
                )
            )

        download_columns = st.columns(
            2
        )

        download_columns[0].download_button(
            "Download TXT report",
            data=text_report,
            file_name=(
                "researchease_v3_report.txt"
            ),
            mime="text/plain",
        )

        if pdf_report is not None:
            download_columns[1].download_button(
                "Download PDF report",
                data=pdf_report,
                file_name=(
                    "researchease_v3_report.pdf"
                ),
                mime="application/pdf",
            )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.divider()

st.caption(
    "Version 6 • LangChain LCEL orchestration • "
    "ChatPromptTemplate • ChatOllama • "
    "RunnableLambda • RunnableBranch • "
    "Pydantic structured output • "
    "Conversation-aware RAG • "
    "External research • Literature review • "
    "SymPy verification • CSV/PDF/TXT export"
)