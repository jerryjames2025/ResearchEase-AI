from __future__ import annotations

import streamlit as st

from frontend.api_client import (
    APIClientError,
    ResearchEaseAPI,
)


def render_vector_controls(
    api: ResearchEaseAPI,
) -> str:
    """
    Display the vector backend selector and
    return the selected backend key.
    """

    with st.sidebar:
        st.header(
            "Vector database"
        )

        providers = {}

        try:
            result = (
                api.vector_providers()
            )

            providers = {
                item["provider"]: item
                for item in result.get(
                    "providers",
                    [],
                )
            }

        except APIClientError as exc:
            st.warning(
                "Unable to load vector providers."
            )

            st.caption(
                str(exc)
            )

        selected = st.selectbox(
            "Vector backend",
            [
                "faiss",
                "pinecone",
            ],
            index=0,
            help=(
                "FAISS stores vectors locally. "
                "Pinecone stores them in a managed "
                "cloud vector database."
            ),
        )

        selected_info = providers.get(
            selected,
            {},
        )

        if selected == "faiss":
            st.success(
                "FAISS local storage selected."
            )

        elif selected_info.get(
            "configured",
            False,
        ):
            st.success(
                "Pinecone is configured."
            )

        else:
            st.warning(
                "Pinecone is not configured. "
                "Add RESEARCHEASE_PINECONE_API_KEY "
                "or select FAISS."
            )

        if st.button(
            "Check vector health",
            use_container_width=True,
        ):
            try:
                health = (
                    api.vector_health()
                )

                for name, item in health.get(
                    "providers",
                    {},
                ).items():
                    if item.get(
                        "healthy",
                        False,
                    ):
                        st.success(
                            f"{name}: {item['detail']}"
                        )

                    else:
                        st.warning(
                            f"{name}: {item['detail']}"
                        )

            except APIClientError as exc:
                st.error(
                    str(exc)
                )

        return selected