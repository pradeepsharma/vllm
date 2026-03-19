"""Database engine, session factory, and dependency injection helper."""

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./team_test.db")

# SQLite requires check_same_thread=False for use with FastAPI/Starlette
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session and closes it afterwards.

    Usage::

        @app.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables defined in the ORM models.

    Imports ``team_test.database.models`` to ensure the model classes are
    registered with ``Base.metadata`` before ``create_all`` is called.
    """
    from team_test.database import models  # noqa: F401 – side-effect import

    Base.metadata.create_all(bind=engine)
