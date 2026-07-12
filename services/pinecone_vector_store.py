from __future__ import annotations

import importlib.util
import logging
import re
import time
from typing import Any

import numpy as np

from backend.core.settings import (
    get_settings,
)
from services.pdf_parser import (
    PaperChunk,
)
from services.vector_store import (
    SearchResult,
)


logger = logging.getLogger(
    "researchease.pinecone"
)


class PineconeVectorStoreError(
    RuntimeError
):
    """
    Raised when a Pinecone vector operation fails.
    """


def _get_value(
    value: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Read a value from either a dictionary or
    an SDK response object.
    """

    if isinstance(
        value,
        dict,
    ):
        return value.get(
            key,
            default,
        )

    return getattr(
        value,
        key,
        default,
    )


def _sanitize_name(
    value: str,
    fallback: str,
    max_length: int,
) -> str:
    """
    Convert a value into a safe Pinecone name.
    """

    cleaned = re.sub(
        r"[^a-z0-9-]+",
        "-",
        value.strip().lower(),
    )

    cleaned = re.sub(
        r"-+",
        "-",
        cleaned,
    ).strip("-")

    return (
        cleaned[:max_length]
        or fallback
    )


def pinecone_package_installed() -> bool:
    return (
        importlib.util.find_spec(
            "pinecone"
        )
        is not None
    )


class PineconeVectorStore:
    """
    Pinecone implementation matching the same search
    interface used by the local FAISS vector store.

    ResearchEase creates embeddings externally with
    Sentence Transformers and sends dense vectors to
    Pinecone.
    """

    def __init__(
        self,
        *,
        index_name: str,
        namespace: str,
        dimension: int,
        expected_count: int = 0,
    ) -> None:
        if not pinecone_package_installed():
            raise PineconeVectorStoreError(
                "The 'pinecone' package is not installed."
            )

        if dimension <= 0:
            raise PineconeVectorStoreError(
                "Pinecone vector dimension must be positive."
            )

        self.settings = get_settings()

        if not self.settings.pinecone_api_key.strip():
            raise PineconeVectorStoreError(
                "Pinecone API key is not configured."
            )

        self.index_name = _sanitize_name(
            index_name,
            fallback="researchease-vectors",
            max_length=45,
        )

        self.namespace = _sanitize_name(
            namespace,
            fallback="research-session",
            max_length=180,
        )

        self.dimension = int(
            dimension
        )

        self.expected_count = max(
            0,
            int(expected_count),
        )

        self._client_instance = None
        self._index_instance = None
        self._host = ""

    # -----------------------------------------------------
    # Client and index lifecycle
    # -----------------------------------------------------

    def _client(self):
        if self._client_instance is None:
            from pinecone import Pinecone

            self._client_instance = Pinecone(
                api_key=(
                    self.settings
                    .pinecone_api_key
                )
            )

        return self._client_instance

    @staticmethod
    def _description_host(
        description: Any,
    ) -> str:
        host = _get_value(
            description,
            "host",
            "",
        )

        return str(
            host or ""
        )

    @staticmethod
    def _description_ready(
        description: Any,
    ) -> bool:
        status = _get_value(
            description,
            "status",
            {},
        )

        ready = _get_value(
            status,
            "ready",
            False,
        )

        return bool(
            ready
        )

    @staticmethod
    def _description_dimension(
        description: Any,
    ) -> int:
        value = _get_value(
            description,
            "dimension",
            0,
        )

        return int(
            value or 0
        )

    @staticmethod
    def _description_metric(
        description: Any,
    ) -> str:
        return str(
            _get_value(
                description,
                "metric",
                "",
            )
            or ""
        ).lower()

    def _wait_until_ready(
        self,
        timeout_seconds: int = 120,
    ) -> Any:
        deadline = (
            time.monotonic()
            + timeout_seconds
        )

        latest = None

        while (
            time.monotonic()
            < deadline
        ):
            latest = (
                self._client()
                .describe_index(
                    name=self.index_name
                )
            )

            if self._description_ready(
                latest
            ):
                return latest

            time.sleep(2)

        raise PineconeVectorStoreError(
            "Pinecone index did not become ready "
            "within the configured timeout."
        )

    def _ensure_index(self) -> Any:
        from pinecone import (
            ServerlessSpec,
        )

        client = self._client()

        try:
            exists = bool(
                client.has_index(
                    self.index_name
                )
            )

        except Exception:
            indexes = (
                client.list_indexes()
            )

            if hasattr(
                indexes,
                "names",
            ):
                exists = (
                    self.index_name
                    in indexes.names()
                )

            else:
                exists = False

                for item in indexes:
                    item_name = _get_value(
                        item,
                        "name",
                        "",
                    )

                    if item_name == self.index_name:
                        exists = True
                        break

        if not exists:
            if not (
                self.settings
                .pinecone_auto_create_index
            ):
                raise PineconeVectorStoreError(
                    "The configured Pinecone index "
                    "does not exist and automatic index "
                    "creation is disabled."
                )

            client.create_index(
                name=self.index_name,
                vector_type="dense",
                dimension=self.dimension,
                metric=(
                    self.settings
                    .pinecone_metric
                ),
                spec=ServerlessSpec(
                    cloud=(
                        self.settings
                        .pinecone_cloud
                    ),
                    region=(
                        self.settings
                        .pinecone_region
                    ),
                ),
                deletion_protection=(
                    "disabled"
                ),
                tags={
                    "application": (
                        "researchease-ai"
                    ),
                    "environment": (
                        "development"
                    ),
                },
            )

        description = (
            self._wait_until_ready()
        )

        actual_dimension = (
            self._description_dimension(
                description
            )
        )

        if (
            actual_dimension
            and actual_dimension
            != self.dimension
        ):
            raise PineconeVectorStoreError(
                "Pinecone index dimension mismatch. "
                f"Index '{self.index_name}' uses "
                f"{actual_dimension} dimensions, but the "
                f"selected embedding model produced "
                f"{self.dimension}. Use a different index "
                "name or rebuild the index with the correct "
                "dimension."
            )

        configured_metric = (
            self.settings
            .pinecone_metric
            .lower()
        )

        actual_metric = (
            self._description_metric(
                description
            )
        )

        if (
            actual_metric
            and actual_metric
            != configured_metric
        ):
            raise PineconeVectorStoreError(
                "Pinecone metric mismatch. "
                f"Index uses '{actual_metric}', while "
                f"ResearchEase is configured for "
                f"'{configured_metric}'."
            )

        return description

    def _index(self):
        if self._index_instance is not None:
            return self._index_instance

        description = (
            self._ensure_index()
        )

        host = self._description_host(
            description
        )

        if not host:
            raise PineconeVectorStoreError(
                "Pinecone did not return an index host."
            )

        self._host = host

        # Resolve the host once and target it directly.
        self._index_instance = (
            self._client().Index(
                host=host
            )
        )

        return self._index_instance

    # -----------------------------------------------------
    # Build and ingest
    # -----------------------------------------------------

    @classmethod
    def from_embeddings(
        cls,
        *,
        chunks: list[PaperChunk],
        embeddings: np.ndarray,
        session_id: str,
    ) -> "PineconeVectorStore":
        if not chunks:
            raise PineconeVectorStoreError(
                "Cannot create a Pinecone store "
                "without paper chunks."
            )

        embeddings = np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
        )

        if embeddings.ndim != 2:
            raise PineconeVectorStoreError(
                "Embeddings must be a "
                "two-dimensional matrix."
            )

        if (
            len(chunks)
            != embeddings.shape[0]
        ):
            raise PineconeVectorStoreError(
                "Chunk count and embedding count "
                "do not match."
            )

        settings = get_settings()

        namespace = (
            f"{settings.pinecone_namespace_prefix}-"
            f"{session_id}"
        )

        store = cls(
            index_name=(
                settings.pinecone_index_name
            ),
            namespace=namespace,
            dimension=int(
                embeddings.shape[1]
            ),
            expected_count=len(
                chunks
            ),
        )

        index = store._index()

        # Rebuilding the same session replaces its
        # previous namespace contents.
        try:
            index.delete(
                delete_all=True,
                namespace=store.namespace,
            )

        except Exception:
            # A missing namespace is acceptable.
            pass

        batch_size = max(
            1,
            int(
                settings
                .pinecone_upsert_batch_size
            ),
        )

        records: list[dict] = []

        for chunk, vector in zip(
            chunks,
            embeddings,
        ):
            records.append(
                {
                    "id": (
                        f"chunk-{chunk.chunk_id}"
                    ),
                    "values": (
                        vector.tolist()
                    ),
                    "metadata": {
                        "session_id": (
                            session_id
                        ),
                        "chunk_id": int(
                            chunk.chunk_id
                        ),
                        "page_number": int(
                            chunk.page_number
                        ),
                        "chunk_number_on_page": int(
                            chunk
                            .chunk_number_on_page
                        ),
                        "text": chunk.text,
                    },
                }
            )

        for start in range(
            0,
            len(records),
            batch_size,
        ):
            batch = records[
                start:
                start + batch_size
            ]

            index.upsert(
                vectors=batch,
                namespace=store.namespace,
            )

        store._wait_for_vector_count(
            expected=len(records)
        )

        return store

    def _namespace_vector_count(
        self,
    ) -> int:
        stats = (
            self._index()
            .describe_index_stats()
        )

        namespaces = _get_value(
            stats,
            "namespaces",
            {},
        ) or {}

        namespace_stats = None

        if isinstance(
            namespaces,
            dict,
        ):
            namespace_stats = (
                namespaces.get(
                    self.namespace
                )
            )

        if namespace_stats is None:
            return 0

        count = _get_value(
            namespace_stats,
            "vector_count",
            None,
        )

        if count is None:
            count = _get_value(
                namespace_stats,
                "vectorCount",
                0,
            )

        return int(
            count or 0
        )

    def _wait_for_vector_count(
        self,
        expected: int,
    ) -> None:
        timeout = max(
            5,
            int(
                self.settings
                .pinecone_freshness_timeout_seconds
            ),
        )

        deadline = (
            time.monotonic()
            + timeout
        )

        while (
            time.monotonic()
            < deadline
        ):
            try:
                count = (
                    self._namespace_vector_count()
                )

                if count >= expected:
                    return

            except Exception:
                pass

            time.sleep(2)

        logger.warning(
            (
                "Pinecone namespace '%s' did not "
                "report the expected vector count "
                "before the freshness timeout."
            ),
            self.namespace,
        )

    # -----------------------------------------------------
    # Search interface
    # -----------------------------------------------------

    @property
    def size(self) -> int:
        if self.expected_count:
            return self.expected_count

        try:
            return (
                self._namespace_vector_count()
            )

        except Exception:
            return 0

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        query_embedding = (
            np.ascontiguousarray(
                query_embedding,
                dtype=np.float32,
            )
        )

        if (
            query_embedding.ndim != 2
            or query_embedding.shape[0] != 1
        ):
            raise PineconeVectorStoreError(
                "Query embedding must have "
                "shape (1, dimension)."
            )

        if (
            query_embedding.shape[1]
            != self.dimension
        ):
            raise PineconeVectorStoreError(
                "Query embedding dimension does not "
                "match the Pinecone index dimension."
            )

        response = (
            self._index().query(
                namespace=self.namespace,
                vector=(
                    query_embedding[
                        0
                    ].tolist()
                ),
                top_k=max(
                    1,
                    int(top_k),
                ),
                include_values=False,
                include_metadata=True,
            )
        )

        matches = _get_value(
            response,
            "matches",
            [],
        ) or []

        results: list[
            SearchResult
        ] = []

        for rank, match in enumerate(
            matches,
            start=1,
        ):
            metadata = _get_value(
                match,
                "metadata",
                {},
            ) or {}

            score = float(
                _get_value(
                    match,
                    "score",
                    0.0,
                )
                or 0.0
            )

            chunk = PaperChunk(
                chunk_id=int(
                    metadata.get(
                        "chunk_id",
                        rank - 1,
                    )
                ),
                page_number=int(
                    metadata.get(
                        "page_number",
                        0,
                    )
                ),
                chunk_number_on_page=int(
                    metadata.get(
                        "chunk_number_on_page",
                        0,
                    )
                ),
                text=str(
                    metadata.get(
                        "text",
                        "",
                    )
                ),
            )

            results.append(
                SearchResult(
                    rank=rank,
                    score=score,
                    chunk=chunk,
                )
            )

        return results

    # -----------------------------------------------------
    # Cleanup and diagnostics
    # -----------------------------------------------------

    def delete_namespace(
        self,
    ) -> None:
        self._index().delete(
            delete_all=True,
            namespace=self.namespace,
        )

    def stats(self) -> dict:
        return {
            "index_name": (
                self.index_name
            ),
            "namespace": (
                self.namespace
            ),
            "dimension": (
                self.dimension
            ),
            "vector_count": (
                self._namespace_vector_count()
            ),
            "host": self._host,
        }

    @classmethod
    def health(cls) -> tuple[
        bool,
        str,
    ]:
        if not pinecone_package_installed():
            return (
                False,
                "The Pinecone SDK is not installed.",
            )

        settings = get_settings()

        if not settings.pinecone_api_key.strip():
            return (
                False,
                "Pinecone API key is not configured.",
            )

        try:
            from pinecone import Pinecone

            client = Pinecone(
                api_key=(
                    settings
                    .pinecone_api_key
                )
            )

            client.list_indexes()

            return (
                True,
                "Pinecone is connected.",
            )

        except Exception as exc:
            return (
                False,
                f"Pinecone connection failed: {exc}",
            )