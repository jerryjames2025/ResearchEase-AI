from __future__ import annotations

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)

from config import (
    MAX_CHAT_HISTORY_TURNS,
)


def history_to_messages(
    chat_history: list[dict],
    max_turns: int = MAX_CHAT_HISTORY_TURNS,
) -> list[BaseMessage]:
    """
    Convert Streamlit chat dictionaries into
    LangChain HumanMessage and AIMessage objects.

    Retrieved evidence is deliberately excluded from
    memory because previous evidence is not proof for
    a new question.
    """

    if not chat_history:
        return []

    recent_messages = chat_history[
        -(max(
            1,
            int(max_turns),
        ) * 2):
    ]

    messages: list[
        BaseMessage
    ] = []

    for item in recent_messages:
        role = str(
            item.get(
                "role",
                "",
            )
        ).strip().lower()

        content = str(
            item.get(
                "content",
                "",
            )
        ).strip()

        if not content:
            continue

        if role == "user":
            messages.append(
                HumanMessage(
                    content=content
                )
            )

        elif role == "assistant":
            messages.append(
                AIMessage(
                    content=content
                )
            )

    return messages