from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from services.pdf_parser import PaperChunk


@dataclass(frozen=True)
class SearchResult:
    """
    One semantic search result.
    """

    rank: int
    score: float
    chunk: PaperChunk


class FaissVectorStore:
    """
    Exact semantic search using FAISS IndexFlatIP.

    Because the embeddings are normalized, inner-product
    similarity behaves like cosine similarity.
    """

    def __init__(
        self,
        chunks: list[PaperChunk],
        embeddings: np.ndarray,
    ) -> None:
        if not chunks:
            raise ValueError(
                "Cannot build an index without chunks."
            )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a "
                "two-dimensional matrix."
            )

        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                "The number of chunks and embedding "
                "rows must match."
            )

        self.chunks = chunks

        self.embeddings = np.ascontiguousarray(
            embeddings,
            dtype=np.float32,
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

    @property
    def size(self) -> int:
        """
        Number of vectors in the FAISS index.
        """

        return int(
            self.index.ntotal
        )

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
    ) -> list[SearchResult]:
        """
        Search for the chunks that are most similar
        to the question embedding.
        """

        if self.size == 0:
            return []

        query_embedding = np.ascontiguousarray(
            query_embedding,
            dtype=np.float32,
        )

        if (
            query_embedding.ndim != 2
            or query_embedding.shape[0] != 1
        ):
            raise ValueError(
                "Query embedding must have shape "
                "(1, dimension)."
            )

        if query_embedding.shape[1] != self.dimension:
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

        scores, indices = self.index.search(
            query_embedding,
            requested_k,
        )

        results: list[SearchResult] = []

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