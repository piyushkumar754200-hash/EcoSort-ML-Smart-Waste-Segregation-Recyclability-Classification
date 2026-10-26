"""SQLAlchemy 2.x ORM model for waste classification predictions."""
from datetime import datetime, timezone
from typing import Any, List, Optional
import uuid
from sqlalchemy import Boolean, DateTime, Float, Index, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, nullable=False, index=True)
    original_filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    detected_item: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)
    predicted_top1: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    recyclability_class: Mapped[str] = mapped_column(String(50), nullable=False)
    bin: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    confidence_level: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    top_k: Mapped[List[Any]] = mapped_column(JSON, nullable=False, default=list)
    quality_warnings: Mapped[List[str]] = mapped_column(JSON, nullable=False, default=list)
    model_version: Mapped[str] = mapped_column(String(120), nullable=False)
    inference_ms: Mapped[float] = mapped_column(Float, nullable=False)
    image_width: Mapped[int] = mapped_column(Integer, nullable=False)
    image_height: Mapped[int] = mapped_column(Integer, nullable=False)
    thumbnail_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    corrected_category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    __table_args__ = (
        Index("ix_predictions_created_at_desc", "created_at"),
        Index("ix_predictions_category_status", "category", "status"),
    )
