"""SQLAlchemy 2.x database engine, session factory, and SQLite WAL pragma configuration."""
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine, event
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import settings


class Base(DeclarativeBase):
    pass


def _build_engine(db_url: str):
    is_sqlite = db_url.startswith("sqlite")
    if is_sqlite and db_url.startswith("sqlite:///") and ":memory:" not in db_url:
        db_file = Path(db_url.replace("sqlite:///", "", 1))
        db_file.parent.mkdir(parents=True, exist_ok=True)

    connect_args = {"check_same_thread": False, "timeout": 30} if is_sqlite else {}
    eng = create_engine(db_url, connect_args=connect_args, future=True)

    if is_sqlite:
        @event.listens_for(eng, "connect")
        def _set_sqlite_pragma(dbapi_connection, _connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.close()

    return eng


engine = _build_engine(settings.DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
