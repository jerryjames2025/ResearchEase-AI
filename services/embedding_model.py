from __future__ import annotations

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_BATCH_SIZE


class EmbeddingService:
    """
    Loads a Sentence Transformer and creates normalized embeddings.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.requested_device = device

        model_kwargs = {}

        if device != "auto":
            model_kwargs["device"] = device

        self.model = SentenceTransformer(
            model_name,
            **model_kwargs,
        )

        self.device = str(
            self.model.device
        )

    def encode_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        """
        Convert paper chunks into normalized vectors.
        """

        if not texts:
            raise ValueError(
                "No document text was supplied "
                "for embedding."
            )

        embeddings = self.model.encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

    def encode_query(
        self,
        text: str,
    ) -> np.ndarray:
        """
        Convert one user question into a normalized vector.
        """

        if not text.strip():
            raise ValueError(
                "The search query cannot be empty."
            )

        embedding = self.model.encode(
            [text],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.ascontiguousarray(
            embedding,
            dtype=np.float32,
        )


@st.cache_resource(
    show_spinner=False,
)
def load_embedding_service(
    model_name: str,
    device: str = "auto",
) -> EmbeddingService:
    """
    Load the embedding transformer once and reuse it
    across Streamlit reruns.
    """

    return EmbeddingService(
        model_name=model_name,
        device=device,
    )