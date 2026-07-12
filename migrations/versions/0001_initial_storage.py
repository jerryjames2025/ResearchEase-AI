"""
Create persistent ResearchEase storage tables.

Revision ID: 0001_initial_storage
Revises:
Create Date: 2026-07-11
"""

from typing import (
    Sequence,
    Union,
)

from alembic import op
import sqlalchemy as sa


revision: str = (
    "0001_initial_storage"
)

down_revision: Union[
    str,
    None,
] = None

branch_labels: Union[
    str,
    Sequence[str],
    None,
] = None

depends_on: Union[
    str,
    Sequence[str],
    None,
] = None


def upgrade() -> None:
    op.create_table(
        "research_sessions",

        sa.Column(
            "session_id",
            sa.String(length=36),
            nullable=False,
        ),

        sa.Column(
            "filename",
            sa.String(length=500),
            nullable=False,
        ),

        sa.Column(
            "page_count",
            sa.Integer(),
            nullable=False,
        ),

        sa.Column(
            "extracted_characters",
            sa.Integer(),
            nullable=False,
        ),

        sa.Column(
            "created_at",
            sa.DateTime(
                timezone=True
            ),
            nullable=False,
        ),

        sa.Column(
            "updated_at",
            sa.DateTime(
                timezone=True
            ),
            nullable=False,
        ),

        sa.Column(
            "analysis_ready",
            sa.Boolean(),
            nullable=False,
        ),

        sa.Column(
            "index_ready",
            sa.Boolean(),
            nullable=False,
        ),

        sa.Column(
            "chunk_count",
            sa.Integer(),
            nullable=False,
        ),

        sa.Column(
            "embedding_model_name",
            sa.String(length=300),
            nullable=False,
        ),

        sa.Column(
            "embedding_device",
            sa.String(length=50),
            nullable=False,
        ),

        sa.Column(
            "faiss_index_path",
            sa.Text(),
            nullable=False,
        ),

        sa.Column(
            "mongo_document_id",
            sa.String(length=50),
            nullable=False,
        ),

        sa.PrimaryKeyConstraint(
            "session_id"
        ),
    )

    op.create_index(
        (
            "ix_research_sessions_"
            "updated_at"
        ),
        "research_sessions",
        ["updated_at"],
        unique=False,
    )


    op.create_table(
        "chat_messages",

        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
        ),

        sa.Column(
            "session_id",
            sa.String(length=36),
            nullable=False,
        ),

        sa.Column(
            "role",
            sa.String(length=20),
            nullable=False,
        ),

        sa.Column(
            "content",
            sa.Text(),
            nullable=False,
        ),

        sa.Column(
            "paper_sources",
            sa.JSON(),
            nullable=False,
        ),

        sa.Column(
            "external_sources",
            sa.JSON(),
            nullable=False,
        ),

        sa.Column(
            "metadata_json",
            sa.JSON(),
            nullable=False,
        ),

        sa.Column(
            "created_at",
            sa.DateTime(
                timezone=True
            ),
            nullable=False,
        ),

        sa.ForeignKeyConstraint(
            ["session_id"],
            [
                "research_sessions."
                "session_id"
            ],
            ondelete="CASCADE",
        ),

        sa.PrimaryKeyConstraint(
            "id"
        ),
    )

    op.create_index(
        (
            "ix_chat_messages_"
            "session_id"
        ),
        "chat_messages",
        ["session_id"],
        unique=False,
    )


    op.create_table(
        "generated_reports",

        sa.Column(
            "id",
            sa.BigInteger(),
            autoincrement=True,
            nullable=False,
        ),

        sa.Column(
            "session_id",
            sa.String(length=36),
            nullable=False,
        ),

        sa.Column(
            "report_type",
            sa.String(length=50),
            nullable=False,
        ),

        sa.Column(
            "file_path",
            sa.Text(),
            nullable=False,
        ),

        sa.Column(
            "metadata_json",
            sa.JSON(),
            nullable=False,
        ),

        sa.Column(
            "created_at",
            sa.DateTime(
                timezone=True
            ),
            nullable=False,
        ),

        sa.ForeignKeyConstraint(
            ["session_id"],
            [
                "research_sessions."
                "session_id"
            ],
            ondelete="CASCADE",
        ),

        sa.PrimaryKeyConstraint(
            "id"
        ),
    )

    op.create_index(
        (
            "ix_generated_reports_"
            "session_id"
        ),
        "generated_reports",
        ["session_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        (
            "ix_generated_reports_"
            "session_id"
        ),
        table_name=(
            "generated_reports"
        ),
    )

    op.drop_table(
        "generated_reports"
    )

    op.drop_index(
        (
            "ix_chat_messages_"
            "session_id"
        ),
        table_name=(
            "chat_messages"
        ),
    )

    op.drop_table(
        "chat_messages"
    )

    op.drop_index(
        (
            "ix_research_sessions_"
            "updated_at"
        ),
        table_name=(
            "research_sessions"
        ),
    )

    op.drop_table(
        "research_sessions"
    )