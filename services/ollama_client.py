from __future__ import annotations

from collections.abc import Callable
from typing import Any

import requests

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from config import (
    MAX_SUMMARY_CHUNKS,
    OLLAMA_CHAT_URL,
    REQUEST_TIMEOUT_SECONDS,
    SUMMARY_CHUNK_OVERLAP,
    SUMMARY_CHUNK_SIZE,
)

from backend.core.settings import (
    get_settings,
)
from backend.llm.runtime import (
    message_text,
)
from langchain_layer.model_factory import (
    get_ollama_chat_model,
)
from services.pdf_parser import (
    split_text,
)
from services.prompts import (
    chunk_summary_prompt,
    final_analysis_prompt,
)


class OllamaError(RuntimeError):
    """
    Backward-compatible generation error.

    The name remains OllamaError so existing UI code
    and exception handlers continue working.
    """


def check_ollama() -> tuple[
    bool,
    str,
]:
    """
    Check the local Ollama service.
    """

    settings = get_settings()

    try:
        response = requests.get(
            (
                settings
                .ollama_base_url
                .rstrip("/")
                + "/api/tags"
            ),
            timeout=5,
        )

        response.raise_for_status()

        return (
            True,
            "Ollama is connected.",
        )

    except requests.RequestException as exc:
        return (
            False,
            f"Ollama is unavailable: {exc}",
        )


def chat(
    model: str,
    prompt: str,
    system_prompt: str = "",
    temperature: float = 0.1,
) -> str:
    """
    Generate through the selected Version 9 provider.

    Plain model names still use Ollama.
    """

    if not prompt.strip():
        raise ValueError(
            "Prompt cannot be empty."
        )

    messages = []

    if system_prompt.strip():
        messages.append(
            SystemMessage(
                content=system_prompt
            )
        )

    messages.append(
        HumanMessage(
            content=prompt
        )
    )

    try:
        llm = get_ollama_chat_model(
            model_name=model,
            temperature=temperature,
            json_mode=False,
        )

        result = llm.invoke(
            messages
        )

        return message_text(
            result
        ).strip()

    except Exception as exc:
        raise OllamaError(
            f"LLM generation failed: {exc}"
        ) from exc


def analyze_paper(
    text: str,
    filename: str,
    page_count: int,
    model: str,
    explanation_level: str,
    progress_callback: (
        Callable[
            [
                int,
                int,
                str,
            ],
            None,
        ]
        | None
    ) = None,
) -> tuple[
    str,
    list[str],
]:
    """
    Analyze a complete paper through any Version 9
    LLM provider.
    """

    if not text.strip():
        raise ValueError(
            "Paper text cannot be empty."
        )

    chunks = split_text(
        text,
        chunk_size=(
            SUMMARY_CHUNK_SIZE
        ),
        overlap=(
            SUMMARY_CHUNK_OVERLAP
        ),
    )

    chunks = chunks[
        :MAX_SUMMARY_CHUNKS
    ]

    if not chunks:
        raise ValueError(
            "No paper chunks were created."
        )

    partial_summaries: list[
        str
    ] = []

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
                    "Analyzing paper section "
                    f"{index} of {len(chunks)}"
                ),
            )

        summary = chat(
            model=model,
            prompt=chunk_summary_prompt(
                chunk=chunk,
                index=index,
                total=len(chunks),
            ),
            system_prompt=(
                "Analyze academic evidence carefully. "
                "Do not invent unsupported details."
            ),
            temperature=0.05,
        )

        partial_summaries.append(
            summary
        )

    if progress_callback:
        progress_callback(
            len(chunks),
            total_steps,
            "Creating final paper analysis",
        )

    final_analysis = chat(
        model=model,
        prompt=final_analysis_prompt(
            filename=filename,
            page_count=page_count,
            partial_summaries=(
                partial_summaries
            ),
            level=explanation_level,
        ),
        system_prompt=(
            "Produce a faithful academic paper analysis "
            "using only the supplied evidence notes."
        ),
        temperature=0.05,
    )

    if progress_callback:
        progress_callback(
            total_steps,
            total_steps,
            "Paper analysis completed",
        )

    return (
        final_analysis,
        partial_summaries,
    )