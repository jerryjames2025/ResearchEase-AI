from __future__ import annotations

import streamlit as st

from config import (
    MAX_LITERATURE_SOURCES,
)
from literature.engine import (
    LiteratureReviewError,
    generate_literature_synthesis,
    reassign_citation_keys,
    summarize_external_paper,
    summarize_full_text_paper,
)
from literature.exporter import (
    build_literature_csv,
    build_literature_pdf,
    build_literature_text,
)
from literature.models import (
    LiteraturePaperSummary,
)
from research.models import (
    AcademicPaper,
)
from services.pdf_parser import (
    extract_pdf,
)


def _summary_from_dict(
    data: dict,
) -> LiteraturePaperSummary:
    """
    Restore a summary from Streamlit session state.
    """

    return LiteraturePaperSummary(
        **data
    )


def _academic_paper_from_dict(
    data: dict,
) -> AcademicPaper:
    """
    Restore an AcademicPaper from Version 3
    external-search data.
    """

    return AcademicPaper(
        source=data.get(
            "source",
            "",
        ),
        source_id=data.get(
            "source_id",
            "",
        ),
        title=data.get(
            "title",
            "",
        ),
        authors=data.get(
            "authors",
            [],
        ) or [],
        year=data.get(
            "year"
        ),
        abstract=data.get(
            "abstract",
            "",
        ),
        url=data.get(
            "url",
            "",
        ),
        doi=data.get(
            "doi",
            "",
        ),
        venue=data.get(
            "venue",
            "",
        ),
        citation_count=data.get(
            "citation_count"
        ),
        publication_date=data.get(
            "publication_date",
            "",
        ),
        pdf_url=data.get(
            "pdf_url",
            "",
        ),
        relevance_score=data.get(
            "relevance_score"
        ),
    )


def _matrix_rows(
    summaries: list[
        LiteraturePaperSummary
    ],
) -> list[
    dict[str, str]
]:
    """
    Create the rows displayed by Streamlit.
    """

    return [
        item.matrix_row()
        for item in summaries
    ]


def render_literature_tab(
    *,
    primary_paper,
    ollama_model: str,
    explanation_level: str,
    connected: bool,
    external_results: list[dict],
) -> None:
    """
    Render the Version 5 literature-review workspace.
    """

    defaults = {
        "literature_summaries": [],
        "literature_synthesis": "",
        "literature_topic": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[
                key
            ] = value

    st.subheader(
        "Literature Review and "
        "Multi-Paper Comparison"
    )

    st.caption(
        "Compare the primary paper with related PDFs "
        "and external academic records. External records "
        "remain abstract-level evidence."
    )

    topic = st.text_input(
        "Literature-review topic",
        value=(
            st.session_state
            .literature_topic
        ),
        placeholder=(
            "Example: transformer-based "
            "traffic forecasting"
        ),
    )

    include_primary = st.checkbox(
        "Include the main uploaded paper",
        value=True,
        disabled=(
            primary_paper is None
        ),
    )

    supporting_files = st.file_uploader(
        "Upload related research papers",
        type=[
            "pdf"
        ],
        accept_multiple_files=True,
        help=(
            "Use text-based PDFs. Start with "
            "two to five papers on your laptop."
        ),
    )

    # -----------------------------------------------------
    # External paper selection
    # -----------------------------------------------------

    external_options: dict[
        str,
        dict,
    ] = {}

    for index, paper in enumerate(
        external_results
    ):
        label = (
            f"{paper.get('title', 'Untitled')} "
            f"({paper.get('year') or 'year unavailable'}) "
            "— "
            f"{paper.get('source', 'external')}"
        )

        external_options[
            f"{index}: {label}"
        ] = paper

    selected_external_labels = (
        st.multiselect(
            (
                "Include external records from "
                "the External Research tab"
            ),
            options=list(
                external_options.keys()
            ),
            help=(
                "These sources use metadata and "
                "abstracts only. Perform an external "
                "search first if the list is empty."
            ),
        )
    )

    selected_external = [
        external_options[
            label
        ]
        for label
        in selected_external_labels
    ]

    total_requested = (
        (
            1
            if (
                include_primary
                and primary_paper
                is not None
            )
            else 0
        )
        + len(
            supporting_files
            or []
        )
        + len(
            selected_external
        )
    )

    source_count_columns = st.columns(
        3
    )

    source_count_columns[0].metric(
        "Selected sources",
        total_requested,
    )

    source_count_columns[1].metric(
        "Full-text PDFs",
        (
            (
                1
                if (
                    include_primary
                    and primary_paper
                    is not None
                )
                else 0
            )
            + len(
                supporting_files
                or []
            )
        ),
    )

    source_count_columns[2].metric(
        "Abstract records",
        len(
            selected_external
        ),
    )

    if (
        total_requested
        > MAX_LITERATURE_SOURCES
    ):
        st.warning(
            "For a stable local run, analyze at most "
            f"{MAX_LITERATURE_SOURCES} sources at one time."
        )

    if total_requested < 2:
        st.info(
            "Select at least two papers or academic "
            "records to create a comparison."
        )

    analyze_clicked = st.button(
        "Build literature-review matrix",
        type="primary",
        use_container_width=True,
        disabled=(
            not connected
            or not ollama_model
            or total_requested < 2
            or total_requested
            > MAX_LITERATURE_SOURCES
        ),
    )

    # -----------------------------------------------------
    # Build matrix
    # -----------------------------------------------------

    if analyze_clicked:
        summaries: list[
            LiteraturePaperSummary
        ] = []

        progress = st.progress(
            0.0
        )

        status = st.empty()

        try:
            source_index = 1
            completed_sources = 0

            if (
                include_primary
                and primary_paper
                is not None
            ):

                def primary_progress(
                    done: int,
                    total: int,
                    message: str,
                ) -> None:
                    base_fraction = (
                        completed_sources
                        / total_requested
                    )

                    source_fraction = (
                        (
                            done / total
                        )
                        / total_requested
                        if total
                        else 0
                    )

                    progress.progress(
                        min(
                            base_fraction
                            + source_fraction,
                            1.0,
                        )
                    )

                    status.write(
                        message
                    )

                summaries.append(
                    summarize_full_text_paper(
                        paper=primary_paper,
                        citation_key=(
                            f"P{source_index}"
                        ),
                        ollama_model=(
                            ollama_model
                        ),
                        progress_callback=(
                            primary_progress
                        ),
                    )
                )

                source_index += 1
                completed_sources += 1

            for uploaded_file in (
                supporting_files
                or []
            ):
                status.write(
                    (
                        "Extracting "
                        f"{uploaded_file.name}"
                    )
                )

                extracted = extract_pdf(
                    uploaded_file
                )

                def support_progress(
                    done: int,
                    total: int,
                    message: str,
                ) -> None:
                    base_fraction = (
                        completed_sources
                        / total_requested
                    )

                    source_fraction = (
                        (
                            done / total
                        )
                        / total_requested
                        if total
                        else 0
                    )

                    progress.progress(
                        min(
                            base_fraction
                            + source_fraction,
                            1.0,
                        )
                    )

                    status.write(
                        message
                    )

                summaries.append(
                    summarize_full_text_paper(
                        paper=extracted,
                        citation_key=(
                            f"P{source_index}"
                        ),
                        ollama_model=(
                            ollama_model
                        ),
                        progress_callback=(
                            support_progress
                        ),
                    )
                )

                source_index += 1
                completed_sources += 1

            for external_data in (
                selected_external
            ):
                external_paper = (
                    _academic_paper_from_dict(
                        external_data
                    )
                )

                status.write(
                    (
                        "Summarizing external record: "
                        f"{external_paper.title}"
                    )
                )

                summaries.append(
                    summarize_external_paper(
                        paper=external_paper,
                        citation_key=(
                            f"P{source_index}"
                        ),
                        ollama_model=(
                            ollama_model
                        ),
                    )
                )

                source_index += 1
                completed_sources += 1

                progress.progress(
                    completed_sources
                    / total_requested
                )

            summaries = (
                reassign_citation_keys(
                    summaries
                )
            )

            st.session_state[
                "literature_summaries"
            ] = [
                item.to_dict()
                for item in summaries
            ]

            st.session_state[
                "literature_synthesis"
            ] = ""

            st.session_state[
                "literature_topic"
            ] = topic

            progress.progress(
                1.0
            )

            status.success(
                (
                    "Created a literature matrix for "
                    f"{len(summaries)} sources."
                )
            )

        except Exception as exc:
            st.error(
                (
                    "Literature matrix generation "
                    f"failed: {exc}"
                )
            )

    summaries = [
        _summary_from_dict(
            item
        )
        for item
        in st.session_state[
            "literature_summaries"
        ]
    ]

    # -----------------------------------------------------
    # Display matrix
    # -----------------------------------------------------

    if summaries:
        st.markdown(
            "### Literature-review matrix"
        )

        st.dataframe(
            _matrix_rows(
                summaries
            ),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander(
            "Inspect individual paper records"
        ):
            for item in summaries:
                st.markdown(
                    (
                        f"#### {item.citation_key}: "
                        f"{item.title}"
                    )
                )

                st.write(
                    (
                        "**Authors:** "
                        f"{item.authors}"
                    )
                )

                st.write(
                    (
                        "**Year:** "
                        f"{item.year}"
                    )
                )

                st.write(
                    (
                        "**Evidence:** "
                        f"{item.evidence_scope}"
                    )
                )

                st.write(
                    (
                        "**Research problem:** "
                        f"{item.research_problem}"
                    )
                )

                st.write(
                    (
                        "**Objective:** "
                        f"{item.objective}"
                    )
                )

                st.write(
                    (
                        "**Methodology:** "
                        f"{item.methodology}"
                    )
                )

                st.write(
                    (
                        "**Dataset:** "
                        f"{item.dataset}"
                    )
                )

                st.write(
                    (
                        "**Models or methods:** "
                        f"{item.models_or_methods}"
                    )
                )

                st.write(
                    (
                        "**Metrics:** "
                        f"{item.evaluation_metrics}"
                    )
                )

                st.write(
                    (
                        "**Findings:** "
                        f"{item.main_findings}"
                    )
                )

                st.write(
                    (
                        "**Limitations:** "
                        f"{item.limitations}"
                    )
                )

                st.write(
                    (
                        "**Future work:** "
                        f"{item.future_work}"
                    )
                )

                if item.source_url:
                    st.link_button(
                        (
                            "Open source for "
                            f"{item.citation_key}"
                        ),
                        item.source_url,
                    )

                st.divider()

        # -------------------------------------------------
        # Generate cross-paper synthesis
        # -------------------------------------------------

        synthesis_clicked = st.button(
            (
                "Generate literature synthesis "
                "and research gaps"
            ),
            type="primary",
            use_container_width=True,
            disabled=(
                not connected
                or not ollama_model
            ),
        )

        if synthesis_clicked:
            try:
                with st.spinner(
                    (
                        "Comparing methodologies, datasets, "
                        "findings, limitations, and gaps..."
                    )
                ):
                    st.session_state[
                        "literature_synthesis"
                    ] = generate_literature_synthesis(
                        topic=topic,
                        summaries=summaries,
                        ollama_model=(
                            ollama_model
                        ),
                        explanation_level=(
                            explanation_level
                        ),
                    )

                    st.session_state[
                        "literature_topic"
                    ] = topic

            except LiteratureReviewError as exc:
                st.error(
                    str(exc)
                )

            except Exception as exc:
                st.error(
                    (
                        "Literature synthesis failed: "
                        f"{exc}"
                    )
                )

        if st.session_state[
            "literature_synthesis"
        ]:
            st.markdown(
                "### Generated literature review"
            )

            st.markdown(
                st.session_state[
                    "literature_synthesis"
                ]
            )

        # -------------------------------------------------
        # Downloads
        # -------------------------------------------------

        st.markdown(
            "### Download literature-review outputs"
        )

        csv_data = build_literature_csv(
            summaries
        )

        text_data = build_literature_text(
            topic=topic,
            summaries=summaries,
            synthesis=(
                st.session_state[
                    "literature_synthesis"
                ]
            ),
        )

        try:
            pdf_data = build_literature_pdf(
                topic=topic,
                summaries=summaries,
                synthesis=(
                    st.session_state[
                        "literature_synthesis"
                    ]
                ),
            )

        except Exception as exc:
            pdf_data = None

            st.warning(
                (
                    "Literature PDF generation "
                    f"failed: {exc}"
                )
            )

        download_columns = st.columns(
            3
        )

        download_columns[
            0
        ].download_button(
            "Download matrix CSV",
            data=csv_data,
            file_name=(
                "researchease_"
                "literature_matrix.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

        download_columns[
            1
        ].download_button(
            "Download literature TXT",
            data=text_data,
            file_name=(
                "researchease_"
                "literature_review.txt"
            ),
            mime="text/plain",
            use_container_width=True,
        )

        if pdf_data is not None:
            download_columns[
                2
            ].download_button(
                "Download literature PDF",
                data=pdf_data,
                file_name=(
                    "researchease_"
                    "literature_review.pdf"
                ),
                mime="application/pdf",
                use_container_width=True,
            )

        if st.button(
            "Clear literature workspace",
            use_container_width=True,
        ):
            st.session_state[
                "literature_summaries"
            ] = []

            st.session_state[
                "literature_synthesis"
            ] = ""

            st.session_state[
                "literature_topic"
            ] = ""

            st.rerun()

    else:
        st.info(
            "Select at least two sources and build "
            "the literature-review matrix."
        )