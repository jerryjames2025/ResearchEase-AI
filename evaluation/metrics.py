from __future__ import annotations

import math
import re
from collections.abc import Iterable


_TOKEN_PATTERN = re.compile(
    r"[a-zA-Z0-9]+"
)

_PAGE_CITATION_PATTERN = re.compile(
    r"\[(?:page|p\.?)[\s:#-]*(\d+)\]",
    re.IGNORECASE,
)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "hers",
    "him",
    "his",
    "i",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "ours",
    "she",
    "that",
    "the",
    "their",
    "theirs",
    "them",
    "they",
    "this",
    "those",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "you",
    "your",
}


def _tokens(
    text: str,
    *,
    remove_stopwords: bool = True,
) -> list[str]:
    values = [
        match.group(0).lower()
        for match
        in _TOKEN_PATTERN.finditer(
            text or ""
        )
    ]

    if not remove_stopwords:
        return values

    return [
        token
        for token in values
        if token not in _STOPWORDS
        and len(token) > 1
    ]


def _safe_ratio(
    numerator: float,
    denominator: float,
) -> float:
    if denominator <= 0:
        return 0.0

    return float(
        numerator / denominator
    )


def lexical_f1(
    reference: str,
    prediction: str,
) -> float:
    reference_tokens = set(
        _tokens(reference)
    )

    prediction_tokens = set(
        _tokens(prediction)
    )

    if (
        not reference_tokens
        or not prediction_tokens
    ):
        return 0.0

    overlap = len(
        reference_tokens
        & prediction_tokens
    )

    precision = _safe_ratio(
        overlap,
        len(prediction_tokens),
    )

    recall = _safe_ratio(
        overlap,
        len(reference_tokens),
    )

    if precision + recall == 0:
        return 0.0

    return (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )


def keyword_recall(
    answer: str,
    expected_keywords: Iterable[str],
) -> float:
    keywords = [
        item.strip().lower()
        for item in expected_keywords
        if item.strip()
    ]

    if not keywords:
        return 0.0

    normalized_answer = " ".join(
        _tokens(
            answer,
            remove_stopwords=False,
        )
    )

    matched = 0

    for keyword in keywords:
        normalized_keyword = " ".join(
            _tokens(
                keyword,
                remove_stopwords=False,
            )
        )

        if (
            normalized_keyword
            and normalized_keyword
            in normalized_answer
        ):
            matched += 1

    return _safe_ratio(
        matched,
        len(keywords),
    )


def retrieval_page_recall(
    retrieved_pages: list[int],
    expected_pages: list[int],
) -> float:
    expected = {
        int(page)
        for page in expected_pages
        if int(page) > 0
    }

    if not expected:
        return 0.0

    retrieved = {
        int(page)
        for page in retrieved_pages
        if int(page) > 0
    }

    return _safe_ratio(
        len(
            expected
            & retrieved
        ),
        len(expected),
    )


def retrieval_precision(
    retrieved_pages: list[int],
    expected_pages: list[int],
) -> float:
    expected = {
        int(page)
        for page in expected_pages
        if int(page) > 0
    }

    if (
        not expected
        or not retrieved_pages
    ):
        return 0.0

    relevant = sum(
        1
        for page in retrieved_pages
        if int(page) in expected
    )

    return _safe_ratio(
        relevant,
        len(retrieved_pages),
    )


def mean_reciprocal_rank(
    retrieved_pages: list[int],
    expected_pages: list[int],
) -> float:
    expected = {
        int(page)
        for page in expected_pages
        if int(page) > 0
    }

    if not expected:
        return 0.0

    for rank, page in enumerate(
        retrieved_pages,
        start=1,
    ):
        if int(page) in expected:
            return 1.0 / rank

    return 0.0


def context_relevance(
    question: str,
    contexts: list[str],
) -> float:
    question_tokens = set(
        _tokens(question)
    )

    if (
        not question_tokens
        or not contexts
    ):
        return 0.0

    values: list[float] = []

    for context in contexts:
        context_tokens = set(
            _tokens(context)
        )

        overlap = len(
            question_tokens
            & context_tokens
        )

        values.append(
            _safe_ratio(
                overlap,
                len(
                    question_tokens
                ),
            )
        )

    return (
        sum(values)
        / len(values)
    )


def groundedness_proxy(
    answer: str,
    contexts: list[str],
) -> float:
    answer_tokens = set(
        _tokens(answer)
    )

    context_tokens = set(
        _tokens(
            "\n".join(contexts)
        )
    )

    if (
        not answer_tokens
        or not context_tokens
    ):
        return 0.0

    return _safe_ratio(
        len(
            answer_tokens
            & context_tokens
        ),
        len(answer_tokens),
    )


def citation_coverage(
    answer: str,
    retrieved_pages: list[int],
) -> float:
    retrieved = {
        int(page)
        for page in retrieved_pages
        if int(page) > 0
    }

    if not retrieved:
        return 0.0

    cited = {
        int(value)
        for value
        in _PAGE_CITATION_PATTERN.findall(
            answer or ""
        )
    }

    return _safe_ratio(
        len(
            cited
            & retrieved
        ),
        len(retrieved),
    )


def evaluate_case_metrics(
    *,
    question: str,
    answer: str,
    contexts: list[str],
    retrieved_pages: list[int],
    expected_answer: str,
    expected_keywords: list[str],
    expected_pages: list[int],
    latency_seconds: float,
) -> dict[str, float]:
    metrics: dict[
        str,
        float,
    ] = {
        "context_relevance": (
            context_relevance(
                question,
                contexts,
            )
        ),
        "groundedness_proxy": (
            groundedness_proxy(
                answer,
                contexts,
            )
        ),
        "citation_coverage": (
            citation_coverage(
                answer,
                retrieved_pages,
            )
        ),
        "answer_length_words": float(
            len(
                _tokens(
                    answer,
                    remove_stopwords=False,
                )
            )
        ),
        "retrieved_count": float(
            len(contexts)
        ),
        "latency_seconds": float(
            max(
                0.0,
                latency_seconds,
            )
        ),
    }

    if expected_answer.strip():
        metrics[
            "answer_reference_f1"
        ] = lexical_f1(
            expected_answer,
            answer,
        )

    if expected_keywords:
        metrics[
            "keyword_recall"
        ] = keyword_recall(
            answer,
            expected_keywords,
        )

    if expected_pages:
        metrics[
            "retrieval_page_recall"
        ] = retrieval_page_recall(
            retrieved_pages,
            expected_pages,
        )

        metrics[
            "retrieval_precision"
        ] = retrieval_precision(
            retrieved_pages,
            expected_pages,
        )

        metrics[
            "retrieval_mrr"
        ] = mean_reciprocal_rank(
            retrieved_pages,
            expected_pages,
        )

    return {
        key: round(
            float(value),
            6,
        )
        for key, value
        in metrics.items()
        if math.isfinite(value)
    }


def aggregate_metrics(
    results: list[dict],
) -> dict[str, float]:
    buckets: dict[
        str,
        list[float],
    ] = {}

    for result in results:
        for key, value in (
            result.get(
                "metrics",
                {},
            ).items()
        ):
            if (
                isinstance(
                    value,
                    (
                        int,
                        float,
                    ),
                )
                and math.isfinite(
                    float(value)
                )
            ):
                buckets.setdefault(
                    key,
                    [],
                ).append(
                    float(value)
                )

        judge = result.get(
            "judge",
            {},
        )

        if judge.get(
            "available"
        ):
            for key in (
                "answer_relevance",
                "groundedness",
                "completeness",
            ):
                value = judge.get(key)

                if (
                    isinstance(
                        value,
                        (
                            int,
                            float,
                        ),
                    )
                    and math.isfinite(
                        float(value)
                    )
                ):
                    buckets.setdefault(
                        f"judge_{key}",
                        [],
                    ).append(
                        float(value)
                    )

    aggregates = {
        f"mean_{key}": round(
            sum(values)
            / len(values),
            6,
        )
        for key, values
        in buckets.items()
        if values
    }

    quality_keys = [
        key
        for key in aggregates
        if key.startswith("mean_")
        and key not in {
            "mean_answer_length_words",
            "mean_retrieved_count",
            "mean_latency_seconds",
        }
    ]

    if quality_keys:
        aggregates[
            "overall_quality_score"
        ] = round(
            sum(
                aggregates[key]
                for key
                in quality_keys
            )
            / len(quality_keys),
            6,
        )

    aggregates[
        "evaluated_cases"
    ] = float(
        len(results)
    )

    aggregates[
        "successful_cases"
    ] = float(
        sum(
            1
            for item in results
            if not item.get(
                "error"
            )
        )
    )

    return aggregates