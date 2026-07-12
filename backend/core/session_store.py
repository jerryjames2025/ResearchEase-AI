from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)

from datetime import datetime
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4

from backend.core.errors import (
    SessionNotFoundError,
)

from backend.core.settings import (
    get_settings,
)

from backend.storage.mongo import (
    mongo_repository,
)

from backend.storage.postgres import (
    ResearchSessionModel,
    postgres_repository,
)

from backend.storage.redis_cache import (
    redis_cache,
)

from services.pdf_parser import (
    ExtractedPaper,
    PaperChunk,
)

from services.vector_store import (
    FaissVectorStore,
)


@dataclass
class ResearchSession:
    session_id: str
    paper: ExtractedPaper
    created_at: datetime
    updated_at: datetime

    analysis: str = ""

    partial_summaries: list[str] = (
        field(
            default_factory=list
        )
    )

    paper_chunks: list[
        PaperChunk
    ] = field(
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


class PersistentSessionStore:
    """
    PostgreSQL stores metadata and chat messages.

    MongoDB stores extracted content, pages,
    chunks and flexible histories.

    FAISS indexes are persisted on disk.

    Redis caches lightweight session summaries.
    """

    def __init__(self) -> None:
        self._runtime_sessions: dict[
            str,
            ResearchSession,
        ] = {}

        self._lock = RLock()
        self._settings = get_settings()

    def _cache_key(
        self,
        session_id: str,
    ) -> str:
        return (
            "researchease:"
            f"session:{session_id}"
        )

    def _cache_summary(
        self,
        row: ResearchSessionModel,
    ) -> None:

        redis_cache.set_json(
            self._cache_key(
                row.session_id
            ),
            self._row_to_summary(
                row
            ),
            self._settings
            .session_cache_ttl_seconds,
        )

    @staticmethod
    def _row_to_summary(
        row: ResearchSessionModel,
    ) -> dict:

        return {
            "session_id": row.session_id,
            "filename": row.filename,
            "page_count": row.page_count,
            "extracted_characters": (
                row.extracted_characters
            ),
            "created_at": row.created_at,
            "updated_at": row.updated_at,
            "analysis_ready": (
                row.analysis_ready
            ),
            "index_ready": (
                row.index_ready
            ),
            "chunk_count": (
                row.chunk_count
            ),
            "embedding_model_name": (
                row.embedding_model_name
            ),
            "embedding_device": (
                row.embedding_device
            ),
        }

    def create(
        self,
        paper: ExtractedPaper,
        raw_pdf: bytes,
    ) -> ResearchSession:

        session_id = str(
            uuid4()
        )

        mongo_document_id = (
            mongo_repository.store_paper(
                session_id=session_id,
                paper=paper,
                pdf_bytes=raw_pdf,
            )
        )

        try:
            row = (
                postgres_repository
                .create_session(
                    session_id=session_id,
                    filename=paper.filename,
                    page_count=(
                        paper.page_count
                    ),
                    extracted_characters=len(
                        paper.text
                    ),
                    mongo_document_id=(
                        mongo_document_id
                    ),
                )
            )

        except Exception:
            mongo_repository.delete_session(
                session_id
            )

            raise

        runtime = ResearchSession(
            session_id=session_id,
            paper=paper,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )

        with self._lock:
            self._runtime_sessions[
                session_id
            ] = runtime

        self._cache_summary(row)

        return runtime

    def _hydrate(
        self,
        row: ResearchSessionModel,
    ) -> ResearchSession:

        loaded = (
            mongo_repository
            .load_paper(
                row.session_id
            )
        )

        if loaded is None:
            raise SessionNotFoundError(
                row.session_id
            )

        (
            paper,
            mongo_metadata,
        ) = loaded

        chunks: list[
            PaperChunk
        ] = []

        vector_store = None

        if (
            row.index_ready
            and row.faiss_index_path
        ):
            index_path = Path(
                row.faiss_index_path
            )

            chunks = (
                mongo_repository
                .load_chunks(
                    row.session_id
                )
            )

            if (
                index_path.exists()
                and chunks
            ):
                vector_store = (
                    FaissVectorStore.load(
                        chunks,
                        index_path,
                    )
                )

        return ResearchSession(
            session_id=row.session_id,
            paper=paper,
            created_at=row.created_at,
            updated_at=row.updated_at,
            analysis=mongo_metadata.get(
                "analysis",
                "",
            ),
            partial_summaries=(
                mongo_metadata.get(
                    "partial_summaries",
                    [],
                )
            ),
            paper_chunks=chunks,
            vector_store=vector_store,
            embedding_model_name=(
                row.embedding_model_name
            ),
            embedding_device=(
                row.embedding_device
            ),
            chat_history=(
                postgres_repository
                .get_chat_messages(
                    row.session_id
                )
            ),
        )

    def get(
        self,
        session_id: str,
    ) -> ResearchSession:

        with self._lock:
            runtime = (
                self._runtime_sessions
                .get(session_id)
            )

        if runtime is not None:
            return runtime

        row = (
            postgres_repository
            .get_session(
                session_id
            )
        )

        if row is None:
            raise SessionNotFoundError(
                session_id
            )

        runtime = self._hydrate(
            row
        )

        with self._lock:
            self._runtime_sessions[
                session_id
            ] = runtime

        self._cache_summary(row)

        return runtime

    def list(
        self,
        limit: int = 100,
    ) -> list[dict]:

        return [
            self._row_to_summary(
                row
            )
            for row
            in postgres_repository
            .list_sessions(
                limit=limit
            )
        ]

    def save_analysis(
        self,
        session: ResearchSession,
        analysis: str,
        partial_summaries: list[str],
    ) -> None:

        mongo_repository.save_analysis(
            session.session_id,
            analysis,
            partial_summaries,
        )

        postgres_repository.update_analysis(
            session.session_id
        )

        with session.lock:
            session.analysis = analysis
            session.partial_summaries = (
                partial_summaries
            )

        row = (
            postgres_repository
            .get_session(
                session.session_id
            )
        )

        if row is not None:
            session.updated_at = (
                row.updated_at
            )

            self._cache_summary(row)

    def save_index(
        self,
        session: ResearchSession,
        chunks: list[PaperChunk],
        vector_store: FaissVectorStore,
        embedding_model_name: str,
        embedding_device: str,
    ) -> str:

        self._settings.faiss_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        index_path = (
            self._settings
            .faiss_directory
            / (
                f"{session.session_id}"
                ".faiss"
            )
        )

        vector_store.save(
            index_path
        )

        mongo_repository.save_chunks(
            session.session_id,
            chunks,
        )

        postgres_repository.update_index(
            session_id=(
                session.session_id
            ),
            chunk_count=(
                vector_store.size
            ),
            embedding_model_name=(
                embedding_model_name
            ),
            embedding_device=(
                embedding_device
            ),
            faiss_index_path=str(
                index_path.resolve()
            ),
        )

        with session.lock:
            session.paper_chunks = chunks
            session.vector_store = (
                vector_store
            )
            session.embedding_model_name = (
                embedding_model_name
            )
            session.embedding_device = (
                embedding_device
            )
            session.chat_history = []

        row = (
            postgres_repository
            .get_session(
                session.session_id
            )
        )

        if row is not None:
            session.updated_at = (
                row.updated_at
            )

            self._cache_summary(row)

        return str(
            index_path.resolve()
        )

    def append_chat_turn(
        self,
        session: ResearchSession,
        *,
        question: str,
        answer: str,
        paper_sources: list[dict],
        external_sources: list[dict],
        answer_mode: str,
    ) -> None:

        messages = [
            {
                "role": "user",
                "content": question,
            },
            {
                "role": "assistant",
                "content": answer,
                "paper_sources": (
                    paper_sources
                ),
                "external_sources": (
                    external_sources
                ),
                "metadata": {
                    "answer_mode": (
                        answer_mode
                    )
                },
            },
        ]

        postgres_repository.append_chat_messages(
            session.session_id,
            messages,
        )

        with session.lock:
            session.chat_history.extend(
                messages
            )

    def get_chat_messages(
        self,
        session_id: str,
    ) -> list[dict]:

        return (
            postgres_repository
            .get_chat_messages(
                session_id
            )
        )

    def record_math_result(
        self,
        session_id: str,
        payload: dict,
    ) -> None:

        mongo_repository.append_math_history(
            session_id,
            payload,
        )

    def record_literature_result(
        self,
        session_id: str,
        payload: dict,
    ) -> None:

        mongo_repository.append_literature_history(
            session_id,
            payload,
        )

    def delete(
        self,
        session_id: str,
    ) -> None:

        row = (
            postgres_repository
            .delete_session(
                session_id
            )
        )

        if row is None:
            raise SessionNotFoundError(
                session_id
            )

        mongo_repository.delete_session(
            session_id
        )

        if row.faiss_index_path:
            Path(
                row.faiss_index_path
            ).unlink(
                missing_ok=True
            )

        redis_cache.delete(
            self._cache_key(
                session_id
            )
        )

        with self._lock:
            self._runtime_sessions.pop(
                session_id,
                None,
            )


session_store = (
    PersistentSessionStore()
)