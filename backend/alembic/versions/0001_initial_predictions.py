"""Initial predictions table migration.

Revision ID: 0001_initial_predictions
Revises:
Create Date: 2026-10-25 07:35:00
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "0001_initial_predictions"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", sa.String(length=36), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("original_filename", sa.String(length=255), nullable=False),
        sa.Column("file_sha256", sa.String(length=64), nullable=False),
        sa.Column("detected_item", sa.String(length=100), nullable=False),
        sa.Column("category", sa.String(length=50), nullable=False),
        sa.Column("recyclability_class", sa.String(length=50), nullable=False),
        sa.Column("bin", sa.String(length=150), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("confidence_level", sa.String(length=20), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column("top_k", sa.JSON(), nullable=False),
        sa.Column("quality_warnings", sa.JSON(), nullable=False),
        sa.Column("model_version", sa.String(length=120), nullable=False),
        sa.Column("inference_ms", sa.Float(), nullable=False),
        sa.Column("image_width", sa.Integer(), nullable=False),
        sa.Column("image_height", sa.Integer(), nullable=False),
        sa.Column("thumbnail_path", sa.String(length=255), nullable=True),
        sa.Column("is_demo", sa.Boolean(), nullable=False, server_default=sa.text("0")),
        sa.Column("corrected_category", sa.String(length=50), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_predictions_created_at", "predictions", ["created_at"], unique=False)
    op.create_index("ix_predictions_category", "predictions", ["category"], unique=False)
    op.create_index("ix_predictions_created_at_desc", "predictions", ["created_at"], unique=False)
    op.create_index("ix_predictions_category_status", "predictions", ["category", "status"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_predictions_category_status", table_name="predictions")
    op.drop_index("ix_predictions_created_at_desc", table_name="predictions")
    op.drop_index("ix_predictions_category", table_name="predictions")
    op.drop_index("ix_predictions_created_at", table_name="predictions")
    op.drop_table("predictions")
