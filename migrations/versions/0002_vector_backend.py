"""
Add configurable vector backend metadata.

Revision ID: 0002_vector_backend
Revises: 0001_initial_storage
Create Date: 2026-07-12
"""

from typing import (
    Sequence,
    Union,
)

from alembic import op
import sqlalchemy as sa


revision: str = "0002_vector_backend"

down_revision: Union[
    str,
    None,
] = "0001_initial_storage"

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
    op.add_column(
        "research_sessions",
        sa.Column(
            "vector_backend",
            sa.String(length=20),
            nullable=False,
            server_default="faiss",
        ),
    )

    op.add_column(
        "research_sessions",
        sa.Column(
            "vector_index_name",
            sa.String(length=200),
            nullable=False,
            server_default="",
        ),
    )

    op.add_column(
        "research_sessions",
        sa.Column(
            "vector_namespace",
            sa.String(length=200),
            nullable=False,
            server_default="",
        ),
    )

    op.add_column(
        "research_sessions",
        sa.Column(
            "vector_dimension",
            sa.Integer(),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column(
        "research_sessions",
        "vector_dimension",
    )

    op.drop_column(
        "research_sessions",
        "vector_namespace",
    )

    op.drop_column(
        "research_sessions",
        "vector_index_name",
    )

    op.drop_column(
        "research_sessions",
        "vector_backend",
    )