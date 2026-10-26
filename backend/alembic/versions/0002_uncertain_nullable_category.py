"""Make category, detected_item, and bin nullable for uncertain predictions and add predicted_top1.

Revision ID: 0002_uncertain_nullable_category
Revises: 0001_initial_predictions
Create Date: 2026-10-25 09:12:00
"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "0002_uncertain_nullable_category"
down_revision: Union[str, None] = "0001_initial_predictions"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("predictions") as batch_op:
        batch_op.add_column(sa.Column("predicted_top1", sa.String(length=50), nullable=True))
        batch_op.alter_column("detected_item", existing_type=sa.String(length=100), nullable=True)
        batch_op.alter_column("category", existing_type=sa.String(length=50), nullable=True)
        batch_op.alter_column("bin", existing_type=sa.String(length=150), nullable=True)

    op.execute("UPDATE predictions SET predicted_top1 = category WHERE predicted_top1 IS NULL")
    op.execute(
        "UPDATE predictions SET category = NULL, detected_item = NULL, bin = NULL, "
        "recyclability_class = 'Unknown' WHERE status = 'uncertain' AND corrected_category IS NULL"
    )


def downgrade() -> None:
    op.execute(
        "UPDATE predictions SET category = COALESCE(predicted_top1, 'plastic'), "
        "detected_item = COALESCE(predicted_top1, 'plastic'), "
        "bin = 'General Waste Bin' WHERE category IS NULL"
    )
    with op.batch_alter_table("predictions") as batch_op:
        batch_op.alter_column("bin", existing_type=sa.String(length=150), nullable=False)
        batch_op.alter_column("category", existing_type=sa.String(length=50), nullable=False)
        batch_op.alter_column("detected_item", existing_type=sa.String(length=100), nullable=False)
        batch_op.drop_column("predicted_top1")
