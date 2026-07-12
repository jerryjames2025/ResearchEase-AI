from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from services.pdf_parser import (
    PaperChunk,
)


@dataclass(frozen=True)
class SearchResult:
    rank: int
    score: float
    chunk: PaperChunk


class FaissVectorStore:
    """
    Exact semantic search using FAISS IndexFlatIP.

    Normalized embeddings make inner-product
    similarity behave like cosine similarity.
    """

    def __init__(
        self,
        chunks: list[PaperChunk],
        embeddings: np.ndarray,
    ) -> None:

        if not chunks:
            raise ValueError(
                "Cannot build an index "
                "without chunks."
            )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a "
                "two-dimensional matrix."
            )

        if (
            len(chunks)
            != embeddings.shape[0]
        ):
            raise ValueError(
                "The number of chunks and "
                "embedding rows must match."
            )

        self.chunks = chunks

        self.embeddings = (
            np.ascontiguousarray(
                embeddings,
                dtype=np.float32,
            )
        )

        self.dimension = int(
            self.embeddings.shape[1]
        )

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.index.add(
            self.embeddings
        )

    @classmethod
    def from_existing_index(
        cls,
        chunks: list[PaperChunk],
        index: faiss.Index,
    ) -> "FaissVectorStore":

        if not chunks:
            raise ValueError(
                "Cannot restore an index "
                "without chunks."
            )

        if int(
            index.ntotal
        ) != len(chunks):
            raise ValueError(
                "Stored FAISS vectors and "
                "MongoDB chunks do not match."
            )

        instance = cls.__new__(
            cls
        )

        instance.chunks = chunks
        instance.embeddings = None
        instance.dimension = int(
            index.d
        )
        instance.index = index

        return instance

    @classmethod
    def load(
        cls,
        chunks: list[PaperChunk],
        path: str | Path,
    ) -> "FaissVectorStore":

        index = faiss.read_index(
            str(path)
        )

        return cls.from_existing_index(
            chunks,
            index,
        )

    def save(
        self,
        path: str | Path,
    ) -> None:

        destination = Path(
            path
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.index,
            str(destination),
        )

    @property
    def size(self) -> int:
        return int(
            self.index.ntotal
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:

        if self.size == 0:
            return []

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
            raise ValueError(
                "Query embedding must have "
                "shape (1, dimension)."
            )

        if (
            query_embedding.shape[1]
            != self.dimension
        ):
            raise ValueError(
                "Query and document embedding "
                "dimensions differ."
            )

        requested_k = max(
            1,
            min(
                int(top_k),
                self.size,
            ),
        )

        scores, indices = (
            self.index.search(
                query_embedding,
                requested_k,
            )
        )

        results: list[
            SearchResult
        ] = []

        for rank, (
            score,
            index_id,
        ) in enumerate(
            zip(
                scores[0],
                indices[0],
            ),
            start=1,
        ):
            if index_id < 0:
                continue

            results.append(
                SearchResult(
                    rank=rank,
                    score=float(score),
                    chunk=self.chunks[
                        int(index_id)
                    ],
                )
            )

        return results