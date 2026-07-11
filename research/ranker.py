from __future__ import annotations

from dataclasses import replace

import numpy as np

from research.models import AcademicPaper
from services.embedding_model import (
    EmbeddingService,
)


def rank_academic_papers(
    query: str,
    papers: list[AcademicPaper],
    embedding_service: EmbeddingService,
) -> list[AcademicPaper]:
    """
    Rank papers using normalized Sentence Transformer
    embeddings.

    Since embeddings are normalized, the dot product
    operates as cosine similarity.
    """

    if not papers:
        return []

    searchable_texts = [
        (
            f"{paper.title}. "
            f"{paper.abstract or paper.venue}"
        )
        for paper in papers
    ]

    query_embedding = (
        embedding_service
        .encode_query(
            query
        )[0]
    )

    paper_embeddings = (
        embedding_service
        .encode_documents(
            searchable_texts
        )
    )

    scores = np.dot(
        paper_embeddings,
        query_embedding,
    )

    ranked = [
        replace(
            paper,
            relevance_score=float(
                score
            ),
        )
        for paper, score
        in zip(
            papers,
            scores,
        )
    ]

    ranked.sort(
        key=lambda paper: (
            paper.relevance_score
            if paper.relevance_score
            is not None
            else -1.0
        ),
        reverse=True,
    )

    return ranked