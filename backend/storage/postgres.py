from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    create_engine,
    delete,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    sessionmaker,
)

from backend.core.settings import (
    get_settings,
)


def utcnow() -> datetime:
    return datetime.now(
        timezone.utc
    )


class Base(DeclarativeBase):
    pass


class ResearchSessionModel(Base):
    __tablename__ = (
        "research_sessions"
    )

    session_id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )

    filename: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    page_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    extracted_characters: Mapped[int] = (
        mapped_column(
            Integer,
            nullable=False,
        )
    )

    created_at: Mapped[datetime] = (
        mapped_column(
            DateTime(timezone=True),
            default=utcnow,
            nullable=False,
        )
    )

    updated_at: Mapped[datetime] = (
        mapped_column(
            DateTime(timezone=True),
            default=utcnow,
            onupdate=utcnow,
            nullable=False,
            index=True,
        )
    )

    analysis_ready: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=False,
            nullable=False,
        )
    )

    index_ready: Mapped[bool] = (
        mapped_column(
            Boolean,
            default=False,
            nullable=False,
        )
    )

    chunk_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
    )

    embedding_model_name: Mapped[str] = (
        mapped_column(
            String(300),
            default="",
            nullable=False,
        )
    )

    embedding_device: Mapped[str] = (
        mapped_column(
            String(50),
            default="",
            nullable=False,
        )
    )

    faiss_index_path: Mapped[str] = (
        mapped_column(
            Text,
            default="",
            nullable=False,
        )
    )

    mongo_document_id: Mapped[str] = (
        mapped_column(
            String(50),
            default="",
            nullable=False,
        )
    )

    # -----------------------------------------------------
    # Version 10 vector-backend fields
    # -----------------------------------------------------

    vector_backend: Mapped[str] = (
        mapped_column(
            String(20),
            default="faiss",
            nullable=False,
        )
    )

    vector_index_name: Mapped[str] = (
        mapped_column(
            String(200),
            default="",
            nullable=False,
        )
    )

    vector_namespace: Mapped[str] = (
        mapped_column(
            String(200),
            default="",
            nullable=False,
        )
    )

    vector_dimension: Mapped[int] = (
        mapped_column(
            Integer,
            default=0,
            nullable=False,
        )
    )


class ChatMessageModel(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "research_sessions.session_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    role: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    paper_sources: Mapped[list] = (
        mapped_column(
            JSON,
            default=list,
            nullable=False,
        )
    )

    external_sources: Mapped[list] = (
        mapped_column(
            JSON,
            default=list,
            nullable=False,
        )
    )

    message_metadata: Mapped[dict] = (
        mapped_column(
            "metadata_json",
            JSON,
            default=dict,
            nullable=False,
        )
    )

    created_at: Mapped[datetime] = (
        mapped_column(
            DateTime(timezone=True),
            default=utcnow,
            nullable=False,
        )
    )


class GeneratedReportModel(Base):
    __tablename__ = (
        "generated_reports"
    )

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
        autoincrement=True,
    )

    session_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey(
            "research_sessions.session_id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    report_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )

    file_path: Mapped[str] = mapped_column(
        Text,
        default="",
        nullable=False,
    )

    report_metadata: Mapped[dict] = (
        mapped_column(
            "metadata_json",
            JSON,
            default=dict,
            nullable=False,
        )
    )

    created_at: Mapped[datetime] = (
        mapped_column(
            DateTime(timezone=True),
            default=utcnow,
            nullable=False,
        )
    )


_settings = get_settings()

engine = create_engine(
    _settings.postgres_url,
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=Session,
)


@contextmanager
def session_scope() -> Iterator[Session]:
    database = SessionLocal()

    try:
        yield database
        database.commit()

    except Exception:
        database.rollback()
        raise

    finally:
        database.close()


def create_tables() -> None:
    Base.metadata.create_all(
        bind=engine
    )


def ping_postgres() -> tuple[
    bool,
    str,
]:
    try:
        with engine.connect() as connection:
            connection.exec_driver_sql(
                "SELECT 1"
            )

        return (
            True,
            "PostgreSQL is connected.",
        )

    except Exception as exc:
        return (
            False,
            (
                "PostgreSQL connection "
                f"failed: {exc}"
            ),
        )


class PostgresRepository:
    def create_session(
        self,
        *,
        session_id: str,
        filename: str,
        page_count: int,
        extracted_characters: int,
        mongo_document_id: str,
    ) -> ResearchSessionModel:
        row = ResearchSessionModel(
            session_id=session_id,
            filename=filename,
            page_count=page_count,
            extracted_characters=(
                extracted_characters
            ),
            mongo_document_id=(
                mongo_document_id
            ),
        )

        with session_scope() as database:
            database.add(row)

        return row

    def get_session(
        self,
        session_id: str,
    ) -> ResearchSessionModel | None:
        with session_scope() as database:
            return database.get(
                ResearchSessionModel,
                session_id,
            )

    def list_sessions(
        self,
        limit: int = 100,
    ) -> list[ResearchSessionModel]:
        with session_scope() as database:
            statement = (
                select(
                    ResearchSessionModel
                )
                .order_by(
                    ResearchSessionModel
                    .updated_at.desc()
                )
                .limit(
                    max(
                        1,
                        min(
                            int(limit),
                            500,
                        ),
                    )
                )
            )

            return list(
                database
                .scalars(statement)
                .all()
            )

    def update_analysis(
        self,
        session_id: str,
    ) -> None:
        with session_scope() as database:
            row = database.get(
                ResearchSessionModel,
                session_id,
            )

            if row is None:
                return

            row.analysis_ready = True
            row.updated_at = utcnow()

    def update_index(
        self,
        *,
        session_id: str,
        chunk_count: int,
        embedding_model_name: str,
        embedding_device: str,
        faiss_index_path: str,
        vector_backend: str,
        vector_index_name: str,
        vector_namespace: str,
        vector_dimension: int,
    ) -> None:
        with session_scope() as database:
            row = database.get(
                ResearchSessionModel,
                session_id,
            )

            if row is None:
                return

            row.index_ready = True
            row.chunk_count = int(
                chunk_count
            )

            row.embedding_model_name = (
                embedding_model_name
            )

            row.embedding_device = (
                embedding_device
            )

            row.faiss_index_path = (
                faiss_index_path
            )

            row.vector_backend = (
                vector_backend
            )

            row.vector_index_name = (
                vector_index_name
            )

            row.vector_namespace = (
                vector_namespace
            )

            row.vector_dimension = int(
                vector_dimension
            )

            row.updated_at = utcnow()

    def append_chat_messages(
        self,
        session_id: str,
        messages: list[dict],
    ) -> None:
        with session_scope() as database:
            for message in messages:
                database.add(
                    ChatMessageModel(
                        session_id=session_id,
                        role=str(
                            message.get(
                                "role",
                                "",
                            )
                        ),
                        content=str(
                            message.get(
                                "content",
                                "",
                            )
                        ),
                        paper_sources=list(
                            message.get(
                                "paper_sources",
                                [],
                            )
                        ),
                        external_sources=list(
                            message.get(
                                "external_sources",
                                [],
                            )
                        ),
                        message_metadata=dict(
                            message.get(
                                "metadata",
                                {},
                            )
                        ),
                    )
                )

            row = database.get(
                ResearchSessionModel,
                session_id,
            )

            if row is not None:
                row.updated_at = utcnow()

    def get_chat_messages(
        self,
        session_id: str,
    ) -> list[dict]:
        with session_scope() as database:
            statement = (
                select(
                    ChatMessageModel
                )
                .where(
                    ChatMessageModel.session_id
                    == session_id
                )
                .order_by(
                    ChatMessageModel.id.asc()
                )
            )

            rows = list(
                database
                .scalars(statement)
                .all()
            )

        return [
            {
                "role": row.role,
                "content": row.content,
                "paper_sources": (
                    row.paper_sources
                    or []
                ),
                "external_sources": (
                    row.external_sources
                    or []
                ),
                "metadata": (
                    row.message_metadata
                    or {}
                ),
                "created_at": (
                    row.created_at
                ),
            }
            for row in rows
        ]

    def delete_session(
        self,
        session_id: str,
    ) -> ResearchSessionModel | None:
        with session_scope() as database:
            row = database.get(
                ResearchSessionModel,
                session_id,
            )

            if row is None:
                return None

            snapshot = ResearchSessionModel(
                session_id=row.session_id,
                filename=row.filename,
                page_count=row.page_count,
                extracted_characters=(
                    row.extracted_characters
                ),
                created_at=row.created_at,
                updated_at=row.updated_at,
                analysis_ready=(
                    row.analysis_ready
                ),
                index_ready=row.index_ready,
                chunk_count=row.chunk_count,
                embedding_model_name=(
                    row.embedding_model_name
                ),
                embedding_device=(
                    row.embedding_device
                ),
                faiss_index_path=(
                    row.faiss_index_path
                ),
                mongo_document_id=(
                    row.mongo_document_id
                ),
                vector_backend=(
                    row.vector_backend
                ),
                vector_index_name=(
                    row.vector_index_name
                ),
                vector_namespace=(
                    row.vector_namespace
                ),
                vector_dimension=(
                    row.vector_dimension
                ),
            )

            database.execute(
                delete(
                    ChatMessageModel
                ).where(
                    ChatMessageModel.session_id
                    == session_id
                )
            )

            database.execute(
                delete(
                    GeneratedReportModel
                ).where(
                    GeneratedReportModel.session_id
                    == session_id
                )
            )

            database.delete(row)

            return snapshot


postgres_repository = (
    PostgresRepository()
)