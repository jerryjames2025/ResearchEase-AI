from __future__ import annotations

import math
import re
from collections.abc import Iterable


STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "among",
    "because",
    "been",
    "before",
    "being",
    "between",
    "both",
    "could",
    "does",
    "from",
    "have",
    "into",
    "more",
    "most",
    "other",
    "over",
    "should",
    "such",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "through",
    "under",
    "using",
    "very",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}


def tokenize(
    text: str,
) -> set[str]:
    """
    Create a lightweight normalized token set.
    """

    tokens = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower(),
    )

    return {
        token
        for token in tokens
        if (
            len(token) >= 3
            and token not in STOPWORDS
        )
    }


def unique_in_order(
    values: Iterable[int],
) -> list[int]:
    result: list[int] = []

    for value in values:
        normalized = int(value)

        if normalized not in result:
            result.append(normalized)

    return result


def retrieval_page_metrics(
    *,
    expected_pages: list[int],
    retrieved_pages: list[int],
    top_k: int,
) -> dict[str, float | None]:
    """
    Calculate page-level retrieval metrics.

    Results are None when no expected-page labels
    were provided.
    """

    expected = set(
        unique_in_order(
            expected_pages
        )
    )

    retrieved = unique_in_order(
        retrieved_pages
    )[:top_k]

    if not expected:
        return {
            "retrieval_hit_rate_at_k": None,
            "retrieval_precision_at_k": None,
            "retrieval_recall_at_k": None,
            "retrieval_mrr_at_k": None,
            "retrieval_ndcg_at_k": None,
        }

    relevant_flags = [
        1
        if page in expected
        else 0
        for page in retrieved
    ]

    relevant_count = sum(
        relevant_flags
    )

    hit_rate = (
        1.0
        if relevant_count > 0
        else 0.0
    )

    precision = (
        relevant_count
        / len(retrieved)
        if retrieved
        else 0.0
    )

    recall = (
        relevant_count
        / len(expected)
    )

    reciprocal_rank = 0.0

    for rank, is_relevant in enumerate(
        relevant_flags,
        start=1,
    ):
        if is_relevant:
            reciprocal_rank = (
                1.0 / rank
            )
            break

    dcg = sum(
        relevance
        / math.log2(rank + 1)
        for rank, relevance
        in enumerate(
            relevant_flags,
            start=1,
        )
    )

    ideal_relevant_count = min(
        len(expected),
        max(
            1,
            min(
                top_k,
                len(retrieved)
                if retrieved
                else top_k,
            ),
        ),
    )

    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(
            1,
            ideal_relevant_count + 1,
        )
    )

    ndcg = (
        dcg / ideal_dcg
        if ideal_dcg > 0
        else 0.0
    )

    return {
        "retrieval_hit_rate_at_k": (
            float(hit_rate)
        ),
        "retrieval_precision_at_k": (
            float(precision)
        ),
        "retrieval_recall_at_k": (
            float(recall)
        ),
        "retrieval_mrr_at_k": (
            float(reciprocal_rank)
        ),
        "retrieval_ndcg_at_k": (
            float(ndcg)
        ),
    }


def answer_keyword_coverage(
    *,
    expected_answer: str,
    answer: str,
) -> float | None:
    """
    Measure how many important expected-answer
    terms appear in the generated answer.
    """

    expected_tokens = tokenize(
        expected_answer
    )

    if not expected_tokens:
        return None

    answer_tokens = tokenize(
        answer
    )

    covered = (
        expected_tokens
        & answer_tokens
    )

    return float(
        len(covered)
        / len(expected_tokens)
    )


def answer_context_overlap(
    *,
    answer: str,
    context: str,
) -> float | None:
    """
    Lightweight deterministic groundedness proxy.

    It measures the percentage of answer content
    tokens found in retrieved context.
    """

    answer_tokens = tokenize(
        answer
    )

    if not answer_tokens:
        return None

    context_tokens = tokenize(
        context
    )

    supported = (
        answer_tokens
        & context_tokens
    )

    return float(
        len(supported)
        / len(answer_tokens)
    )


def safe_mean(
    values: Iterable[
        float | None
    ],
) -> float | None:
    valid_values = [
        float(value)
        for value in values
        if value is not None
        and math.isfinite(
            float(value)
        )
    ]

    if not valid_values:
        return None

    return float(
        sum(valid_values)
        / len(valid_values)
    )