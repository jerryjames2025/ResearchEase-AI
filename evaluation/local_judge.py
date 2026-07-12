from __future__ import annotations

import json
import re
from typing import Any

from services.ollama_client import chat


_JSON_PATTERN = re.compile(
    r"\{.*\}",
    re.DOTALL,
)


def _extract_json(
    text: str,
) -> dict[str, Any]:
    cleaned = (
        text or ""
    ).strip()

    try:
        payload = json.loads(
            cleaned
        )

        if isinstance(
            payload,
            dict,
        ):
            return payload

    except json.JSONDecodeError:
        pass

    match = _JSON_PATTERN.search(
        cleaned
    )

    if not match:
        raise ValueError(
            "The judge did not return "
            "a JSON object."
        )

    payload = json.loads(
        match.group(0)
    )

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError(
            "The judge response was "
            "not a JSON object."
        )

    return payload


def _normalize_score(
    value: Any,
) -> float:
    try:
        number = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if number > 1.0:
        number = (
            number / 5.0
        )

    return round(
        min(
            1.0,
            max(
                0.0,
                number,
            ),
        ),
        6,
    )


def evaluate_with_local_judge(
    *,
    question: str,
    answer: str,
    context: str,
    expected_answer: str,
    model: str,
) -> dict[str, Any]:
    prompt = f"""
You are evaluating a retrieval-augmented generation answer.

Return ONLY valid JSON with this structure:

{{
  "answer_relevance": 1,
  "groundedness": 1,
  "completeness": 1,
  "reason": "brief explanation"
}}

Use integer scores from 1 to 5.

Scoring rules:

- answer_relevance:
  Does the answer directly address the question?

- groundedness:
  Are the claims supported by the supplied context?

- completeness:
  Does the answer cover the important information available?

Question:

{question}

Retrieved context:

{context[:12000]}

Candidate answer:

{answer[:8000]}

Reference answer, when available:

{
    expected_answer[:4000]
    if expected_answer
    else "No reference answer supplied."
}
""".strip()

    try:
        raw = chat(
            model=model,
            prompt=prompt,
            system_prompt=(
                "You are a strict RAG evaluator. "
                "Return JSON only and do not add markdown."
            ),
            temperature=0.0,
        )

        payload = _extract_json(
            raw
        )

        return {
            "available": True,

            "answer_relevance": (
                _normalize_score(
                    payload.get(
                        "answer_relevance"
                    )
                )
            ),

            "groundedness": (
                _normalize_score(
                    payload.get(
                        "groundedness"
                    )
                )
            ),

            "completeness": (
                _normalize_score(
                    payload.get(
                        "completeness"
                    )
                )
            ),

            "reason": str(
                payload.get(
                    "reason",
                    "",
                )
            )[:2000],

            "raw": raw[:4000],
        }

    except Exception as exc:
        return {
            "available": False,
            "answer_relevance": 0.0,
            "groundedness": 0.0,
            "completeness": 0.0,
            "reason": "",
            "error": str(exc),
        }