from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import RLock
from typing import Any
from uuid import uuid4

from backend.core.errors import (
    SessionNotFoundError,
)
from services.pdf_parser import (
    ExtractedPaper,
    PaperChunk,
)


@dataclass
class ResearchSession:
    """
    Temporary research session stored in memory.
    """

    session_id: str
    paper: ExtractedPaper
    created_at: datetime

    analysis: str = ""
    partial_summaries: list[str] = field(
        default_factory=list
    )

    paper_chunks: list[PaperChunk] = field(
        default_factory=list
    )

    vector_store: Any = None

    embedding_model_name: str = ""
    embedding_device: str = ""

    chat_history: list[dict] = field(
        default_factory=list
    )

    lock: RLock = field(
        default_factory=RLock,
        repr=False,
    )

    @property
    def index_ready(self) -> bool:
        return self.vector_store is not None


class SessionStore:
    """
    Thread-safe in-memory research-session store.
    """

    def __init__(self) -> None:
        self._sessions: dict[
            str,
            ResearchSession,
        ] = {}

        self._lock = RLock()

    def create(
        self,
        paper: ExtractedPaper,
    ) -> ResearchSession:
        session = ResearchSession(
            session_id=str(
                uuid4()
            ),
            paper=paper,
            created_at=datetime.now(
                timezone.utc
            ),
        )

        with self._lock:
            self._sessions[
                session.session_id
            ] = session

        return session

    def get(
        self,
        session_id: str,
    ) -> ResearchSession:
        with self._lock:
            session = self._sessions.get(
                session_id
            )

        if session is None:
            raise SessionNotFoundError(
                session_id
            )

        return session

    def delete(
        self,
        session_id: str,
    ) -> None:
        with self._lock:
            if (
                session_id
                not in self._sessions
            ):
                raise SessionNotFoundError(
                    session_id
                )

            del self._sessions[
                session_id
            ]


session_store = SessionStore()