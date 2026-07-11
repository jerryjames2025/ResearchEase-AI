from __future__ import annotations

from typing import Callable

import requests

from config import (
    MAX_SUMMARY_CHUNKS,
    OLLAMA_CHAT_URL,
    REQUEST_TIMEOUT_SECONDS,
    SUMMARY_CHUNK_OVERLAP,
    SUMMARY_CHUNK_SIZE,
)
from services.pdf_parser import split_text
from services.prompts import (
    chunk_summary_prompt,
    final_analysis_prompt,
)


class OllamaError(RuntimeError):
    """
    Error raised when Ollama cannot complete a request.
    """


def check_ollama() -> tuple[bool, str]:
    """
    Check whether the local Ollama server is running.
    """

    try:
        base_url = OLLAMA_CHAT_URL.rsplit(
            "/api/",
            1,
        )[0]

        response = requests.get(
            f"{base_url}/api/tags",
            timeout=10,
        )

        response.raise_for_status()

        return True, "Ollama is connected."

    except requests.RequestException as exc:
        return False, (
            "Ollama is not reachable at "
            "http://localhost:11434. "
            "Start Ollama and confirm that the selected "
            "model is installed. "
            f"Technical detail: {exc}"
        )


def chat(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.1,
) -> str:
    """
    Send a non-streaming chat request to Ollama.
    """

    messages: list[dict[str, str]] = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

        response.raise_for_status()

    except requests.Timeout as exc:
        raise OllamaError(
            "Ollama took too long to respond. "
            "Try a smaller model or a shorter paper."
        ) from exc

    except requests.RequestException as exc:
        raise OllamaError(
            "Could not communicate with Ollama. "
            "Confirm that Ollama is running and that "
            "the selected model has been downloaded."
        ) from exc

    try:
        response_data = response.json()

        content = response_data[
            "message"
        ][
            "content"
        ].strip()

    except (KeyError, TypeError, ValueError) as exc:
        raise OllamaError(
            "Ollama returned an unexpected response."
        ) from exc

    if not content:
        raise OllamaError(
            "Ollama returned an empty response."
        )

    return content


def analyze_paper(
    text: str,
    filename: str,
    page_count: int,
    model: str,
    explanation_level: str,
    progress_callback: (
        Callable[[int, int, str], None] | None
    ) = None,
) -> tuple[str, list[str]]:
    """
    Summarize a complete paper using hierarchical summarization.

    First, each large section is summarized.
    Then, the partial summaries are combined into one report.
    """

    chunks = split_text(
        text=text,
        chunk_size=SUMMARY_CHUNK_SIZE,
        overlap=SUMMARY_CHUNK_OVERLAP,
    )

    # Prevent a huge number of local LLM calls.
    chunks = chunks[:MAX_SUMMARY_CHUNKS]

    if not chunks:
        raise ValueError(
            "No text was available for analysis."
        )

    partial_summaries: list[str] = []

    total_steps = len(chunks) + 1

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        if progress_callback:
            progress_callback(
                index - 1,
                total_steps,
                (
                    f"Analyzing section {index} "
                    f"of {len(chunks)}..."
                ),
            )

        partial_summary = chat(
            model=model,
            prompt=chunk_summary_prompt(
                chunk=chunk,
                index=index,
                total=len(chunks),
            ),
            system_prompt=(
                "Analyze academic text faithfully. "
                "Never fabricate missing information."
            ),
            temperature=0.1,
        )

        partial_summaries.append(
            partial_summary
        )

    if progress_callback:
        progress_callback(
            len(chunks),
            total_steps,
            (
                "Combining the sections into the "
                "final report..."
            ),
        )

    final_report = chat(
        model=model,
        prompt=final_analysis_prompt(
            filename=filename,
            page_count=page_count,
            partial_summaries=partial_summaries,
            level=explanation_level,
        ),
        system_prompt=(
            "Be precise, evidence-aware, and honest "
            "about missing information."
        ),
        temperature=0.15,
    )

    if progress_callback:
        progress_callback(
            total_steps,
            total_steps,
            "Analysis complete.",
        )

    return final_report, partial_summaries