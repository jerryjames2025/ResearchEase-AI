from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from io import BytesIO

from bson import ObjectId
from gridfs import GridFSBucket
from pymongo import (
    ASCENDING,
    MongoClient,
)

from pymongo.database import Database

from backend.core.settings import (
    get_settings,
)

from services.pdf_parser import (
    ExtractedPaper,
    PageText,
    PaperChunk,
)


def utcnow() -> datetime:
    return datetime.now(
        timezone.utc
    )


@lru_cache(maxsize=1)
def get_mongo_client() -> MongoClient:
    settings = get_settings()

    return MongoClient(
        settings.mongodb_url,
        serverSelectionTimeoutMS=3000,
        connectTimeoutMS=3000,
        socketTimeoutMS=10000,
    )


@lru_cache(maxsize=1)
def get_mongo_database() -> Database:
    settings = get_settings()

    return get_mongo_client()[
        settings.mongodb_database
    ]


@lru_cache(maxsize=1)
def get_gridfs_bucket() -> GridFSBucket:
    return GridFSBucket(
        get_mongo_database(),
        bucket_name="paper_files",
    )


def ping_mongo() -> tuple[
    bool,
    str,
]:
    try:
        get_mongo_client().admin.command(
            "ping"
        )

        return (
            True,
            "MongoDB is connected.",
        )

    except Exception as exc:
        return (
            False,
            (
                "MongoDB connection "
                f"failed: {exc}"
            ),
        )


def initialize_mongo() -> None:
    database = get_mongo_database()

    database.papers.create_index(
        "session_id",
        unique=True,
    )

    database.paper_pages.create_index(
        [
            (
                "session_id",
                ASCENDING,
            ),
            (
                "page_number",
                ASCENDING,
            ),
        ],
        unique=True,
    )

    database.paper_chunks.create_index(
        [
            (
                "session_id",
                ASCENDING,
            ),
            (
                "chunk_id",
                ASCENDING,
            ),
        ],
        unique=True,
    )

    database.math_history.create_index(
        [
            (
                "session_id",
                ASCENDING,
            ),
            (
                "created_at",
                ASCENDING,
            ),
        ]
    )

    database.literature_history.create_index(
        [
            (
                "session_id",
                ASCENDING,
            ),
            (
                "created_at",
                ASCENDING,
            ),
        ]
    )


class MongoPaperRepository:
    def store_paper(
        self,
        *,
        session_id: str,
        paper: ExtractedPaper,
        pdf_bytes: bytes,
    ) -> str:

        database = get_mongo_database()
        bucket = get_gridfs_bucket()

        file_id = bucket.upload_from_stream(
            paper.filename,
            BytesIO(pdf_bytes),
            metadata={
                "session_id": session_id,
                "content_type": (
                    "application/pdf"
                ),
            },
        )

        document = {
            "session_id": session_id,
            "filename": paper.filename,
            "page_count": paper.page_count,
            "pdf_file_id": file_id,
            "analysis": "",
            "partial_summaries": [],
            "created_at": utcnow(),
            "updated_at": utcnow(),
        }

        try:
            result = (
                database.papers.insert_one(
                    document
                )
            )

            if paper.pages:
                database.paper_pages.insert_many(
                    [
                        {
                            "session_id": (
                                session_id
                            ),
                            "page_number": (
                                page.page_number
                            ),
                            "text": page.text,
                        }
                        for page in paper.pages
                    ],
                    ordered=True,
                )

        except Exception:
            database.papers.delete_one(
                {
                    "session_id": (
                        session_id
                    )
                }
            )

            database.paper_pages.delete_many(
                {
                    "session_id": (
                        session_id
                    )
                }
            )

            bucket.delete(file_id)
            raise

        return str(
            result.inserted_id
        )

    def load_paper(
        self,
        session_id: str,
    ) -> tuple[
        ExtractedPaper,
        dict,
    ] | None:

        database = get_mongo_database()

        document = (
            database.papers.find_one(
                {
                    "session_id": (
                        session_id
                    )
                }
            )
        )

        if document is None:
            return None

        page_documents = list(
            database.paper_pages.find(
                {
                    "session_id": (
                        session_id
                    )
                }
            ).sort(
                "page_number",
                ASCENDING,
            )
        )

        pages = [
            PageText(
                page_number=int(
                    page["page_number"]
                ),
                text=str(
                    page.get(
                        "text",
                        "",
                    )
                ),
            )
            for page in page_documents
        ]

        combined_pages = [
            (
                (
                    f"\n--- PAGE "
                    f"{page.page_number} ---\n"
                    f"{page.text}"
                )
                if page.text
                else (
                    f"\n--- PAGE "
                    f"{page.page_number} ---\n"
                    "[No selectable text was "
                    "found on this page.]"
                )
            )
            for page in pages
        ]

        paper = ExtractedPaper(
            filename=str(
                document["filename"]
            ),
            page_count=int(
                document["page_count"]
            ),
            text="\n".join(
                combined_pages
            ).strip(),
            pages=pages,
        )

        metadata = {
            "mongo_document_id": str(
                document["_id"]
            ),
            "analysis": str(
                document.get(
                    "analysis",
                    "",
                )
            ),
            "partial_summaries": list(
                document.get(
                    "partial_summaries",
                    [],
                )
            ),
        }

        return paper, metadata

    def save_analysis(
        self,
        session_id: str,
        analysis: str,
        partial_summaries: list[str],
    ) -> None:

        get_mongo_database().papers.update_one(
            {
                "session_id": (
                    session_id
                )
            },
            {
                "$set": {
                    "analysis": analysis,
                    "partial_summaries": (
                        partial_summaries
                    ),
                    "updated_at": utcnow(),
                }
            },
        )

    def save_chunks(
        self,
        session_id: str,
        chunks: list[PaperChunk],
    ) -> None:

        collection = (
            get_mongo_database()
            .paper_chunks
        )

        collection.delete_many(
            {
                "session_id": (
                    session_id
                )
            }
        )

        if not chunks:
            return

        collection.insert_many(
            [
                {
                    "session_id": (
                        session_id
                    ),
                    "chunk_id": (
                        chunk.chunk_id
                    ),
                    "page_number": (
                        chunk.page_number
                    ),
                    "chunk_number_on_page": (
                        chunk
                        .chunk_number_on_page
                    ),
                    "text": chunk.text,
                }
                for chunk in chunks
            ],
            ordered=True,
        )

    def load_chunks(
        self,
        session_id: str,
    ) -> list[PaperChunk]:

        cursor = (
            get_mongo_database()
            .paper_chunks
            .find(
                {
                    "session_id": (
                        session_id
                    )
                }
            )
            .sort(
                "chunk_id",
                ASCENDING,
            )
        )

        return [
            PaperChunk(
                chunk_id=int(
                    document["chunk_id"]
                ),
                page_number=int(
                    document[
                        "page_number"
                    ]
                ),
                chunk_number_on_page=int(
                    document[
                        "chunk_number_on_page"
                    ]
                ),
                text=str(
                    document["text"]
                ),
            )
            for document in cursor
        ]

    def append_math_history(
        self,
        session_id: str,
        payload: dict,
    ) -> None:

        get_mongo_database().math_history.insert_one(
            {
                "session_id": session_id,
                "payload": payload,
                "created_at": utcnow(),
            }
        )

    def append_literature_history(
        self,
        session_id: str,
        payload: dict,
    ) -> None:

        get_mongo_database().literature_history.insert_one(
            {
                "session_id": session_id,
                "payload": payload,
                "created_at": utcnow(),
            }
        )

    def delete_session(
        self,
        session_id: str,
    ) -> None:

        database = get_mongo_database()

        document = (
            database.papers.find_one(
                {
                    "session_id": (
                        session_id
                    )
                }
            )
        )

        if document is not None:
            file_id = document.get(
                "pdf_file_id"
            )

            if isinstance(
                file_id,
                ObjectId,
            ):
                try:
                    get_gridfs_bucket().delete(
                        file_id
                    )

                except Exception:
                    pass

        database.papers.delete_one(
            {
                "session_id": (
                    session_id
                )
            }
        )

        database.paper_pages.delete_many(
            {
                "session_id": (
                    session_id
                )
            }
        )

        database.paper_chunks.delete_many(
            {
                "session_id": (
                    session_id
                )
            }
        )

        database.math_history.delete_many(
            {
                "session_id": (
                    session_id
                )
            }
        )

        database.literature_history.delete_many(
            {
                "session_id": (
                    session_id
                )
            }
        )


mongo_repository = (
    MongoPaperRepository()
)