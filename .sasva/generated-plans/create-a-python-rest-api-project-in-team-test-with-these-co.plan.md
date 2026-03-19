# Execution Plan: Python REST API Project in `team-test/`

**Generated:** March 19, 2026  
**Workspace:** `/Users/pradeepsharma/sasva/projects/vllm`  
**Target Directory:** `team-test/`

---

## Codebase Context

This workspace is the **vLLM** project — a high-throughput LLM inference engine. Key observations:

- **FastAPI** is already a first-class dependency (`requirements/common.txt`: `fastapi[standard] >= 0.115.0`)
- **Pydantic v2** is used throughout (`pydantic >= 2.12.0`); all schemas use `BaseModel` with `ConfigDict`
- **Middleware pattern** is established in `vllm/entrypoints/serve/elastic_ep/middleware.py` using ASGI `__call__` interface
- **APIRouter** pattern used in `vllm/entrypoints/openai/` and `vllm/entrypoints/serve/*/api_router.py`
- **pytest** with `pytest-asyncio` is the test framework (`requirements/test.in`)
- **conftest.py** fixtures follow the pattern in `tests/entrypoints/conftest.py` and `tests/conftest.py`
- **No SQLAlchemy** exists in the vLLM codebase — the new project introduces it fresh
- Python version: `>=3.10,<3.14` (from `pyproject.toml`)

---

## Project File Tree

```
team-test/
├── __init__.py
├── requirements.txt
├── database/
│   ├── __init__.py
│   ├── models.py          # SQLAlchemy User + Project ORM models
│   ├── connection.py      # Engine, session factory, Base, get_db()
│   └── seed_data.py       # Sample data insertion script
├── api/
│   ├── __init__.py
│   ├── schemas.py         # Pydantic v2 request/response schemas
│   ├── middleware.py      # API key authentication middleware
│   └── endpoints.py       # FastAPI app + CRUD routes for Users & Projects
└── tests/
    ├── __init__.py
    ├── conftest.py        # pytest fixtures (engine, session, client, seed)
    ├── test_models.py     # Unit tests for ORM models
    └── test_endpoints.py  # Integration tests for API endpoints
```

---

## Phase 1 — Project Scaffold & Root Files

**Goal:** Create the directory skeleton and top-level files.

### 1.1 — `team-test/__init__.py`

```python
# team-test/__init__.py
"""team-test: Python REST API project with FastAPI + SQLAlchemy."""
```

### 1.2 — `team-test/requirements.txt`

```
# Web framework
fastapi[standard]>=0.115.0
uvicorn[standard]>=0.30.0

# Database
sqlalchemy>=2.0.0
alembic>=1.13.0

# Validation
pydantic>=2.12.0

# Testing
pytest>=8.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0          # ASGI test client for FastAPI

# Dev utilities
python-dotenv>=1.0.0
```

**Rationale:** Mirrors the versions already pinned in `requirements/common.txt` for FastAPI and Pydantic. Uses SQLite (no extra driver needed) for portability. `httpx` is required by FastAPI's `TestClient` (already in `requirements/test.in`).

---

## Phase 2 — Database Layer

**Goal:** Implement `team-test/database/` with models, connection setup, and seed data.

### 2.1 — `team-test/database/__init__.py`

```python
# team-test/database/__init__.py
from .connection import Base, engine, get_db, SessionLocal
from .models import User, Project

__all__ = ["Base", "engine", "get_db", "SessionLocal", "User", "Project"]
```

### 2.2 — `team-test/database/connection.py`

```python
# team-test/database/connection.py
"""Database engine, session factory, and dependency injection helper."""

import os
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./team_test.db")

# SQLite-specific: allow same-thread usage in FastAPI (needed for SQLite)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency: yields a database session and ensures cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables defined on Base.metadata."""
    # Import models so they register on Base.metadata before create_all
    from team_test.database import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
```

**Key design decisions:**
- Uses `DeclarativeBase` (SQLAlchemy 2.0 style), not the legacy `declarative_base()` factory
- `DATABASE_URL` is environment-variable driven for easy test overrides (e.g., `sqlite:///:memory:`)
- `get_db()` is a generator-based FastAPI `Depends()` compatible dependency

### 2.3 — `team-test/database/models.py`

```python
# team-test/database/models.py
"""SQLAlchemy ORM models: User and Project."""

import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .connection import Base


class User(Base):
    """Represents an application user."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    full_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # One user → many projects
    projects: Mapped[list["Project"]] = relationship(
        "Project", back_populates="owner", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r}>"


class Project(Base):
    """Represents a project owned by a User."""

    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    owner_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Many projects → one user
    owner: Mapped["User"] = relationship("User", back_populates="projects")

    def __repr__(self) -> str:
        return f"<Project id={self.id} name={self.name!r} owner_id={self.owner_id}>"
```

**Key design decisions:**
- Uses SQLAlchemy 2.0 `Mapped` + `mapped_column` typed annotations (not legacy `Column`)
- `cascade="all, delete-orphan"` on `User.projects` ensures referential integrity
- `ForeignKey("users.id", ondelete="CASCADE")` mirrors the ORM cascade at the DB level
- `server_default=func.now()` for timestamps — DB-side default, not Python-side

### 2.4 — `team-test/database/seed_data.py`

```python
# team-test/database/seed_data.py
"""Seed script: inserts sample Users and Projects into the database."""

import hashlib
import sys
from pathlib import Path

# Allow running as a standalone script from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from team_test.database.connection import SessionLocal, init_db
from team_test.database.models import Project, User


def _fake_hash(password: str) -> str:
    """Deterministic fake hash for seed data (NOT for production use)."""
    return hashlib.sha256(password.encode()).hexdigest()


SEED_USERS = [
    {
        "username": "alice",
        "email": "alice@example.com",
        "full_name": "Alice Wonderland",
        "hashed_password": _fake_hash("alice_secret"),
        "is_active": True,
    },
    {
        "username": "bob",
        "email": "bob@example.com",
        "full_name": "Bob Builder",
        "hashed_password": _fake_hash("bob_secret"),
        "is_active": True,
    },
    {
        "username": "charlie",
        "email": "charlie@example.com",
        "full_name": "Charlie Chaplin",
        "hashed_password": _fake_hash("charlie_secret"),
        "is_active": False,
    },
]

SEED_PROJECTS = [
    {
        "name": "Alpha Initiative",
        "description": "First project for Alice",
        "owner_username": "alice",
        "is_active": True,
    },
    {
        "name": "Beta Platform",
        "description": "Alice's second project",
        "owner_username": "alice",
        "is_active": True,
    },
    {
        "name": "Bob's Workshop",
        "description": "Bob's primary project",
        "owner_username": "bob",
        "is_active": True,
    },
    {
        "name": "Legacy System",
        "description": "Archived project by Charlie",
        "owner_username": "charlie",
        "is_active": False,
    },
]


def seed(db=None) -> None:
    """Insert seed data. Accepts an optional session for testability."""
    close_after = db is None
    if db is None:
        init_db()
        db = SessionLocal()

    try:
        # Skip if data already exists
        if db.query(User).count() > 0:
            print("Seed data already present — skipping.")
            return

        # Insert users
        user_map: dict[str, User] = {}
        for u_data in SEED_USERS:
            user = User(**u_data)
            db.add(user)
            user_map[u_data["username"]] = user

        db.flush()  # Assign IDs without committing

        # Insert projects
        for p_data in SEED_PROJECTS:
            owner_username = p_data.pop("owner_username")
            project = Project(owner=user_map[owner_username], **p_data)
            db.add(project)

        db.commit()
        print(f"Seeded {len(SEED_USERS)} users and {len(SEED_PROJECTS)} projects.")

    except Exception:
        db.rollback()
        raise
    finally:
        if close_after:
            db.close()


if __name__ == "__main__":
    seed()
    print("Done.")
```

---

## Phase 3 — API Layer

**Goal:** Implement `team-test/api/` with Pydantic schemas, auth middleware, and FastAPI endpoints.

### 3.1 — `team-test/api/__init__.py`

```python
# team-test/api/__init__.py
from .endpoints import app

__all__ = ["app"]
```

### 3.2 — `team-test/api/schemas.py`

```python
# team-test/api/schemas.py
"""Pydantic v2 request/response schemas for Users and Projects."""

import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# ─── Shared config ────────────────────────────────────────────────────────────

class _OrmBase(BaseModel):
    """Base model with ORM mode enabled (reads from SQLAlchemy model instances)."""
    model_config = ConfigDict(from_attributes=True)


# ─── User schemas ─────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    """Request body for creating a new user."""
    username: Annotated[str, Field(min_length=3, max_length=50, pattern=r"^[a-zA-Z0-9_]+$")]
    email: EmailStr
    full_name: str | None = Field(default=None, max_length=100)
    password: Annotated[str, Field(min_length=8, description="Plain-text password (will be hashed)")]

    @field_validator("username")
    @classmethod
    def username_lowercase(cls, v: str) -> str:
        return v.lower()


class UserUpdate(BaseModel):
    """Request body for updating an existing user (all fields optional)."""
    email: EmailStr | None = None
    full_name: str | None = Field(default=None, max_length=100)
    is_active: bool | None = None
    password: str | None = Field(default=None, min_length=8)


class UserResponse(_OrmBase):
    """Response schema for a user (never exposes hashed_password)."""
    id: int
    username: str
    email: str
    full_name: str | None
    is_active: bool
    created_at: datetime.datetime
    updated_at: datetime.datetime


class UserListResponse(BaseModel):
    """Paginated list of users."""
    total: int
    items: list[UserResponse]


# ─── Project schemas ──────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    """Request body for creating a new project."""
    name: Annotated[str, Field(min_length=1, max_length=100)]
    description: str | None = Field(default=None)
    owner_id: int = Field(gt=0, description="ID of the owning user")


class ProjectUpdate(BaseModel):
    """Request body for updating an existing project (all fields optional)."""
    name: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = None
    is_active: bool | None = None


class ProjectResponse(_OrmBase):
    """Response schema for a project."""
    id: int
    name: str
    description: str | None
    is_active: bool
    owner_id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime


class ProjectListResponse(BaseModel):
    """Paginated list of projects."""
    total: int
    items: list[ProjectResponse]


# ─── Generic response schemas ─────────────────────────────────────────────────

class MessageResponse(BaseModel):
    """Generic success/info message."""
    message: str


class ErrorDetail(BaseModel):
    """Structured error detail."""
    code: str
    message: str
    field: str | None = None


class ErrorResponse(BaseModel):
    """Structured error response body."""
    error: ErrorDetail
```

### 3.3 — `team-test/api/middleware.py`

```python
# team-test/api/middleware.py
"""Authentication middleware: validates API key from X-API-Key header."""

import os
from collections.abc import Awaitable

from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

# Default dev key — override via environment variable in production
_DEFAULT_API_KEY = "dev-secret-key-change-me"
VALID_API_KEY: str = os.getenv("API_KEY", _DEFAULT_API_KEY)

# Paths that bypass authentication
_PUBLIC_PATHS: frozenset[str] = frozenset({"/", "/health", "/docs", "/openapi.json", "/redoc"})


class APIKeyMiddleware:
    """
    ASGI middleware that enforces API key authentication via the
    ``X-API-Key`` request header.

    Requests to paths in ``_PUBLIC_PATHS`` are allowed through without
    a key.  All other requests must supply a matching key or receive a
    ``401 Unauthorized`` response.

    Pattern mirrors ``ScalingMiddleware`` in
    ``vllm/entrypoints/serve/elastic_ep/middleware.py``.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] != "http":
            return self.app(scope, receive, send)

        path: str = scope.get("path", "")

        # Allow public paths without authentication
        if path in _PUBLIC_PATHS:
            return self.app(scope, receive, send)

        # Extract X-API-Key header (headers are byte pairs in ASGI)
        headers: dict[bytes, bytes] = dict(scope.get("headers", []))
        api_key_bytes: bytes | None = headers.get(b"x-api-key")
        api_key: str = api_key_bytes.decode("utf-8") if api_key_bytes else ""

        if api_key != VALID_API_KEY:
            response = JSONResponse(
                content={
                    "error": {
                        "code": "UNAUTHORIZED",
                        "message": "Invalid or missing API key. Provide a valid X-API-Key header.",
                        "field": "X-API-Key",
                    }
                },
                status_code=401,
            )
            return response(scope, receive, send)

        return self.app(scope, receive, send)
```

### 3.4 — `team-test/api/endpoints.py`

```python
# team-test/api/endpoints.py
"""FastAPI application with CRUD routes for Users and Projects."""

import hashlib
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, status
from sqlalchemy.orm import Session

from team_test.api.middleware import APIKeyMiddleware
from team_test.api.schemas import (
    ErrorResponse,
    MessageResponse,
    ProjectCreate,
    ProjectListResponse,
    ProjectResponse,
    ProjectUpdate,
    UserCreate,
    UserListResponse,
    UserResponse,
    UserUpdate,
)
from team_test.database.connection import get_db, init_db
from team_test.database.models import Project, User

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Team Test REST API",
    description="CRUD API for Users and Projects",
    version="1.0.0",
    responses={401: {"model": ErrorResponse}, 404: {"model": ErrorResponse}},
)

app.add_middleware(APIKeyMiddleware)


@app.on_event("startup")
def on_startup() -> None:
    """Create database tables on application startup."""
    init_db()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    """Deterministic SHA-256 hash (replace with bcrypt in production)."""
    return hashlib.sha256(plain.encode()).hexdigest()


def _get_user_or_404(user_id: int, db: Session) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return user


def _get_project_or_404(project_id: int, db: Session) -> Project:
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return project


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health_check() -> dict:
    """Public health check endpoint."""
    return {"status": "ok"}


# ─── User endpoints ───────────────────────────────────────────────────────────

@app.post(
    "/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Users"],
    summary="Create a new user",
)
def create_user(
    payload: UserCreate,
    db: Annotated[Session, Depends(get_db)],
) -> User:
    # Check uniqueness
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(status_code=409, detail="Username already taken")
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        username=payload.username,
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=_hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.get(
    "/users",
    response_model=UserListResponse,
    tags=["Users"],
    summary="List all users",
)
def list_users(
    db: Annotated[Session, Depends(get_db)],
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
) -> UserListResponse:
    total = db.query(User).count()
    items = db.query(User).offset(skip).limit(limit).all()
    return UserListResponse(total=total, items=items)


@app.get(
    "/users/{user_id}",
    response_model=UserResponse,
    tags=["Users"],
    summary="Get a user by ID",
)
def get_user(
    user_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> User:
    return _get_user_or_404(user_id, db)


@app.put(
    "/users/{user_id}",
    response_model=UserResponse,
    tags=["Users"],
    summary="Update a user",
)
def update_user(
    user_id: int,
    payload: UserUpdate,
    db: Annotated[Session, Depends(get_db)],
) -> User:
    user = _get_user_or_404(user_id, db)
    update_data = payload.model_dump(exclude_unset=True)

    if "password" in update_data:
        update_data["hashed_password"] = _hash_password(update_data.pop("password"))

    for field, value in update_data.items():
        setattr(user, field, value)

    db.commit()
    db.refresh(user)
    return user


@app.delete(
    "/users/{user_id}",
    response_model=MessageResponse,
    tags=["Users"],
    summary="Delete a user",
)
def delete_user(
    user_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    user = _get_user_or_404(user_id, db)
    db.delete(user)
    db.commit()
    return MessageResponse(message=f"User {user_id} deleted successfully")


# ─── Project endpoints ────────────────────────────────────────────────────────

@app.post(
    "/projects",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Projects"],
    summary="Create a new project",
)
def create_project(
    payload: ProjectCreate,
    db: Annotated[Session, Depends(get_db)],
) -> Project:
    # Validate owner exists
    _get_user_or_404(payload.owner_id, db)

    project = Project(
        name=payload.name,
        description=payload.description,
        owner_id=payload.owner_id,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@app.get(
    "/projects",
    response_model=ProjectListResponse,
    tags=["Projects"],
    summary="List all projects",
)
def list_projects(
    db: Annotated[Session, Depends(get_db)],
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=20, ge=1, le=100),
    owner_id: int | None = Query(default=None, description="Filter by owner user ID"),
) -> ProjectListResponse:
    query = db.query(Project)
    if owner_id is not None:
        query = query.filter(Project.owner_id == owner_id)
    total = query.count()
    items = query.offset(skip).limit(limit).all()
    return ProjectListResponse(total=total, items=items)


@app.get(
    "/projects/{project_id}",
    response_model=ProjectResponse,
    tags=["Projects"],
    summary="Get a project by ID",
)
def get_project(
    project_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> Project:
    return _get_project_or_404(project_id, db)


@app.put(
    "/projects/{project_id}",
    response_model=ProjectResponse,
    tags=["Projects"],
    summary="Update a project",
)
def update_project(
    project_id: int,
    payload: ProjectUpdate,
    db: Annotated[Session, Depends(get_db)],
) -> Project:
    project = _get_project_or_404(project_id, db)
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(project, field, value)
    db.commit()
    db.refresh(project)
    return project


@app.delete(
    "/projects/{project_id}",
    response_model=MessageResponse,
    tags=["Projects"],
    summary="Delete a project",
)
def delete_project(
    project_id: int,
    db: Annotated[Session, Depends(get_db)],
) -> MessageResponse:
    project = _get_project_or_404(project_id, db)
    db.delete(project)
    db.commit()
    return MessageResponse(message=f"Project {project_id} deleted successfully")
```

---

## Phase 4 — Test Suite

**Goal:** Implement `team-test/tests/` with fixtures, model unit tests, and endpoint integration tests.

### 4.1 — `team-test/tests/__init__.py`

```python
# team-test/tests/__init__.py
```

### 4.2 — `team-test/tests/conftest.py`

```python
# team-test/tests/conftest.py
"""
pytest fixtures for team-test.

Follows the pattern established in:
  - tests/entrypoints/conftest.py  (simple fixtures)
  - tests/conftest.py              (session-scoped heavy fixtures)
"""

import os

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

# Use in-memory SQLite for all tests — fast and isolated
TEST_DATABASE_URL = "sqlite:///:memory:"

# Override DATABASE_URL before importing app modules
os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["API_KEY"] = "test-api-key"

from team_test.database.connection import Base, get_db
from team_test.database.models import User, Project
from team_test.api.endpoints import app


@pytest.fixture(scope="session")
def test_engine():
    """Create a single in-memory SQLite engine for the entire test session."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(test_engine) -> Session:
    """
    Provide a transactional database session per test function.
    Rolls back after each test to ensure isolation.
    """
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)

    yield session

    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture(scope="function")
def client(db_session: Session) -> TestClient:
    """
    FastAPI TestClient with the database dependency overridden to use
    the test session.
    """
    def override_get_db():
        try:
            yield db_session
        finally:
            pass  # Rollback handled by db_session fixture

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, headers={"X-API-Key": "test-api-key"}) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture
def sample_user(db_session: Session) -> User:
    """Insert and return a single test user."""
    import hashlib
    user = User(
        username="testuser",
        email="testuser@example.com",
        full_name="Test User",
        hashed_password=hashlib.sha256(b"password123").hexdigest(),
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def sample_project(db_session: Session, sample_user: User) -> Project:
    """Insert and return a single test project owned by sample_user."""
    project = Project(
        name="Test Project",
        description="A project for testing",
        owner_id=sample_user.id,
        is_active=True,
    )
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)
    return project
```

### 4.3 — `team-test/tests/test_models.py`

```python
# team-test/tests/test_models.py
"""Unit tests for SQLAlchemy ORM models (User and Project)."""

import hashlib

import pytest
from sqlalchemy.exc import IntegrityError

from team_test.database.models import Project, User


class TestUserModel:
    """Tests for the User ORM model."""

    def test_create_user(self, db_session):
        """User can be created with required fields."""
        user = User(
            username="alice",
            email="alice@example.com",
            hashed_password=hashlib.sha256(b"secret").hexdigest(),
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        assert user.id is not None
        assert user.username == "alice"
        assert user.email == "alice@example.com"
        assert user.is_active is True  # default
        assert user.full_name is None  # optional

    def test_user_repr(self, db_session):
        """User __repr__ includes id and username."""
        user = User(
            username="bob",
            email="bob@example.com",
            hashed_password="hash",
        )
        db_session.add(user)
        db_session.commit()
        assert "bob" in repr(user)

    def test_user_unique_username(self, db_session):
        """Duplicate username raises IntegrityError."""
        for _ in range(2):
            db_session.add(User(username="dup", email=f"dup{_}@x.com", hashed_password="h"))
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_user_unique_email(self, db_session):
        """Duplicate email raises IntegrityError."""
        db_session.add(User(username="u1", email="same@x.com", hashed_password="h"))
        db_session.add(User(username="u2", email="same@x.com", hashed_password="h"))
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_user_timestamps(self, sample_user):
        """created_at and updated_at are populated automatically."""
        assert sample_user.created_at is not None
        assert sample_user.updated_at is not None

    def test_user_projects_relationship(self, db_session, sample_user):
        """User.projects returns associated Project instances."""
        p = Project(name="P1", owner_id=sample_user.id)
        db_session.add(p)
        db_session.commit()
        db_session.refresh(sample_user)
        assert len(sample_user.projects) == 1
        assert sample_user.projects[0].name == "P1"

    def test_cascade_delete_projects(self, db_session, sample_user, sample_project):
        """Deleting a user cascades to their projects."""
        user_id = sample_user.id
        project_id = sample_project.id
        db_session.delete(sample_user)
        db_session.commit()
        assert db_session.query(Project).filter(Project.id == project_id).first() is None


class TestProjectModel:
    """Tests for the Project ORM model."""

    def test_create_project(self, db_session, sample_user):
        """Project can be created with required fields."""
        project = Project(
            name="My Project",
            description="A test project",
            owner_id=sample_user.id,
        )
        db_session.add(project)
        db_session.commit()
        db_session.refresh(project)

        assert project.id is not None
        assert project.name == "My Project"
        assert project.is_active is True  # default
        assert project.owner_id == sample_user.id

    def test_project_repr(self, sample_project):
        """Project __repr__ includes id, name, and owner_id."""
        r = repr(sample_project)
        assert "Test Project" in r
        assert str(sample_project.owner_id) in r

    def test_project_owner_relationship(self, db_session, sample_project, sample_user):
        """Project.owner returns the owning User instance."""
        db_session.refresh(sample_project)
        assert sample_project.owner.id == sample_user.id
        assert sample_project.owner.username == "testuser"

    def test_project_timestamps(self, sample_project):
        """created_at and updated_at are populated automatically."""
        assert sample_project.created_at is not None
        assert sample_project.updated_at is not None

    def test_project_requires_owner(self, db_session):
        """Project without owner_id raises IntegrityError."""
        project = Project(name="Orphan")
        db_session.add(project)
        with pytest.raises(IntegrityError):
            db_session.commit()
```

### 4.4 — `team-test/tests/test_endpoints.py`

```python
# team-test/tests/test_endpoints.py
"""Integration tests for FastAPI CRUD endpoints."""

import pytest


# ─── Health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_check(self, client):
        """GET /health returns 200 without auth."""
        from fastapi.testclient import TestClient
        from team_test.api.endpoints import app
        # Health is a public path — test without API key
        with TestClient(app) as unauthenticated:
            resp = unauthenticated.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


# ─── Authentication ───────────────────────────────────────────────────────────

class TestAuthentication:
    def test_missing_api_key_returns_401(self):
        """Requests without X-API-Key header are rejected."""
        from fastapi.testclient import TestClient
        from team_test.api.endpoints import app
        with TestClient(app) as c:
            resp = c.get("/users")
        assert resp.status_code == 401

    def test_wrong_api_key_returns_401(self):
        """Requests with wrong X-API-Key are rejected."""
        from fastapi.testclient import TestClient
        from team_test.api.endpoints import app
        with TestClient(app, headers={"X-API-Key": "wrong-key"}) as c:
            resp = c.get("/users")
        assert resp.status_code == 401

    def test_valid_api_key_passes(self, client):
        """Requests with correct X-API-Key succeed."""
        resp = client.get("/users")
        assert resp.status_code == 200


# ─── User CRUD ────────────────────────────────────────────────────────────────

class TestUserEndpoints:
    USER_PAYLOAD = {
        "username": "newuser",
        "email": "newuser@example.com",
        "full_name": "New User",
        "password": "securepassword",
    }

    def test_create_user_201(self, client):
        """POST /users creates a user and returns 201."""
        resp = client.post("/users", json=self.USER_PAYLOAD)
        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == "newuser"
        assert data["email"] == "newuser@example.com"
        assert "hashed_password" not in data
        assert "id" in data

    def test_create_user_duplicate_username_409(self, client):
        """POST /users with duplicate username returns 409."""
        client.post("/users", json=self.USER_PAYLOAD)
        resp = client.post("/users", json=self.USER_PAYLOAD)
        assert resp.status_code == 409

    def test_create_user_invalid_payload_422(self, client):
        """POST /users with invalid payload returns 422."""
        resp = client.post("/users", json={"username": "x"})  # missing required fields
        assert resp.status_code == 422

    def test_list_users_200(self, client, sample_user):
        """GET /users returns paginated list."""
        resp = client.get("/users")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "items" in data
        assert data["total"] >= 1

    def test_list_users_pagination(self, client, sample_user):
        """GET /users?skip=0&limit=1 returns at most 1 item."""
        resp = client.get("/users?skip=0&limit=1")
        assert resp.status_code == 200
        assert len(resp.json()["items"]) <= 1

    def test_get_user_200(self, client, sample_user):
        """GET /users/{id} returns the correct user."""
        resp = client.get(f"/users/{sample_user.id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == sample_user.id
        assert resp.json()["username"] == "testuser"

    def test_get_user_404(self, client):
        """GET /users/99999 returns 404."""
        resp = client.get("/users/99999")
        assert resp.status_code == 404

    def test_update_user_200(self, client, sample_user):
        """PUT /users/{id} updates fields and returns updated user."""
        resp = client.put(
            f"/users/{sample_user.id}",
            json={"full_name": "Updated Name", "is_active": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["full_name"] == "Updated Name"
        assert data["is_active"] is False

    def test_update_user_404(self, client):
        """PUT /users/99999 returns 404."""
        resp = client.put("/users/99999", json={"full_name": "X"})
        assert resp.status_code == 404

    def test_delete_user_200(self, client, sample_user):
        """DELETE /users/{id} removes the user."""
        resp = client.delete(f"/users/{sample_user.id}")
        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"]
        # Confirm gone
        assert client.get(f"/users/{sample_user.id}").status_code == 404

    def test_delete_user_404(self, client):
        """DELETE /users/99999 returns 404."""
        resp = client.delete("/users/99999")
        assert resp.status_code == 404


# ─── Project CRUD ─────────────────────────────────────────────────────────────

class TestProjectEndpoints:
    def _project_payload(self, owner_id: int) -> dict:
        return {
            "name": "Test Project",
            "description": "A test project",
            "owner_id": owner_id,
        }

    def test_create_project_201(self, client, sample_user):
        """POST /projects creates a project and returns 201."""
        resp = client.post("/projects", json=self._project_payload(sample_user.id))
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "Test Project"
        assert data["owner_id"] == sample_user.id
        assert "id" in data

    def test_create_project_invalid_owner_404(self, client):
        """POST /projects with non-existent owner_id returns 404."""
        resp = client.post("/projects", json=self._project_payload(99999))
        assert resp.status_code == 404

    def test_list_projects_200(self, client, sample_project):
        """GET /projects returns paginated list."""
        resp = client.get("/projects")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert data["total"] >= 1

    def test_list_projects_filter_by_owner(self, client, sample_project, sample_user):
        """GET /projects?owner_id=X filters by owner."""
        resp = client.get(f"/projects?owner_id={sample_user.id}")
        assert resp.status_code == 200
        items = resp.json()["items"]
        assert all(p["owner_id"] == sample_user.id for p in items)

    def test_get_project_200(self, client, sample_project):
        """GET /projects/{id} returns the correct project."""
        resp = client.get(f"/projects/{sample_project.id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == sample_project.id

    def test_get_project_404(self, client):
        """GET /projects/99999 returns 404."""
        resp = client.get("/projects/99999")
        assert resp.status_code == 404

    def test_update_project_200(self, client, sample_project):
        """PUT /projects/{id} updates fields."""
        resp = client.put(
            f"/projects/{sample_project.id}",
            json={"name": "Renamed Project", "is_active": False},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Renamed Project"
        assert data["is_active"] is False

    def test_update_project_404(self, client):
        """PUT /projects/99999 returns 404."""
        resp = client.put("/projects/99999", json={"name": "X"})
        assert resp.status_code == 404

    def test_delete_project_200(self, client, sample_project):
        """DELETE /projects/{id} removes the project."""
        resp = client.delete(f"/projects/{sample_project.id}")
        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"]
        assert client.get(f"/projects/{sample_project.id}").status_code == 404

    def test_delete_project_404(self, client):
        """DELETE /projects/99999 returns 404."""
        resp = client.delete("/projects/99999")
        assert resp.status_code == 404
```

---

## Phase 5 — Sub-package `__init__.py` Files

Create empty (or minimal) `__init__.py` files for each sub-package:

| File | Content |
|------|---------|
| `team-test/database/__init__.py` | Re-exports `Base`, `engine`, `get_db`, `SessionLocal`, `User`, `Project` |
| `team-test/api/__init__.py` | Re-exports `app` |
| `team-test/tests/__init__.py` | Empty |

---

## Execution Order & Parallelism

```
Phase 1 (scaffold)
    │
    ├─── Phase 2 (database layer) ──────────────────────────────────────────┐
    │       2.1 connection.py                                               │
    │       2.2 models.py (depends on connection.py)                        │
    │       2.3 seed_data.py (depends on models.py + connection.py)         │
    │                                                                       │
    └─── Phase 3 (api layer) ────────────────────────────────────────────── │
            3.1 schemas.py (independent)                                    │
            3.2 middleware.py (independent)                                 │
            3.3 endpoints.py (depends on schemas + middleware + database)   │
                                                                            │
Phase 4 (tests) ─────────────────────────────────────────────────────────── ┘
    4.1 conftest.py (depends on all of Phase 2 + 3)
    4.2 test_models.py (depends on conftest.py)
    4.3 test_endpoints.py (depends on conftest.py)

Phase 5 (__init__.py files) — can be done in parallel with Phase 2/3
```

**Phases 2 and 3 can be executed in parallel** since `schemas.py` and `middleware.py` have no database dependency.

---

## File Creation Checklist

| # | File | Phase | Dependencies |
|---|------|-------|-------------|
| 1 | `team-test/__init__.py` | 1 | none |
| 2 | `team-test/requirements.txt` | 1 | none |
| 3 | `team-test/database/__init__.py` | 5 | models.py, connection.py |
| 4 | `team-test/database/connection.py` | 2 | none |
| 5 | `team-test/database/models.py` | 2 | connection.py |
| 6 | `team-test/database/seed_data.py` | 2 | models.py, connection.py |
| 7 | `team-test/api/__init__.py` | 5 | endpoints.py |
| 8 | `team-test/api/schemas.py` | 3 | none |
| 9 | `team-test/api/middleware.py` | 3 | none |
| 10 | `team-test/api/endpoints.py` | 3 | schemas.py, middleware.py, database/ |
| 11 | `team-test/tests/__init__.py` | 5 | none |
| 12 | `team-test/tests/conftest.py` | 4 | all of database/ + api/ |
| 13 | `team-test/tests/test_models.py` | 4 | conftest.py |
| 14 | `team-test/tests/test_endpoints.py` | 4 | conftest.py |

---

## Important Implementation Notes

### SQLAlchemy 2.0 Style
All models use the new `Mapped[T]` + `mapped_column()` API (not legacy `Column()`). This is consistent with SQLAlchemy ≥ 2.0 and avoids deprecation warnings.

### Pydantic v2 Compatibility
- Use `model_dump(exclude_unset=True)` (not `.dict()`)
- Use `ConfigDict(from_attributes=True)` (not `orm_mode = True`)
- Use `model_config = ConfigDict(...)` class variable (not inner `class Config`)
- These align with `pydantic >= 2.12.0` already in `requirements/common.txt`

### Test Isolation Strategy
- Each test function gets a **fresh transactional session** that rolls back after the test
- The `TestClient` dependency override replaces `get_db` with the test session
- `scope="session"` engine is created once; `scope="function"` sessions roll back per test
- `os.environ["DATABASE_URL"] = "sqlite:///:memory:"` is set before any imports

### Middleware Pattern
`APIKeyMiddleware` follows the exact ASGI `__call__` pattern from `vllm/entrypoints/serve/elastic_ep/middleware.py` — raw ASGI scope/receive/send, no Starlette `BaseHTTPMiddleware` overhead.

### Import Path Note
The package is named `team_test` (underscore) in Python imports but lives in `team-test/` (hyphen) on disk. The `team-test/__init__.py` makes it importable when the parent directory is on `sys.path`. For `pytest`, add a `conftest.py` at the root or a `pytest.ini` with `pythonpath = .`.

---

## Verification Criteria

### Install dependencies
```bash
cd /Users/pradeepsharma/sasva/projects/vllm/team-test
pip install -r requirements.txt
```

### Run the full test suite
```bash
cd /Users/pradeepsharma/sasva/projects/vllm
python -m pytest team-test/tests/ -v
```

**Expected output:**
- All **26+ tests pass** (0 failures, 0 errors)
- Test classes: `TestHealth`, `TestAuthentication`, `TestUserModel`, `TestProjectModel`, `TestUserEndpoints`, `TestProjectEndpoints`
- No `ImportError` or `ModuleNotFoundError`

### Specific test counts to verify
| Test file | Expected passing tests |
|-----------|----------------------|
| `test_models.py` | ≥ 10 tests (6 User + 4 Project) |
| `test_endpoints.py` | ≥ 16 tests (3 Health/Auth + 10 User + 10 Project) |

### Manual API verification (after `uvicorn team_test.api.endpoints:app --reload`)
```bash
# Health (no auth needed)
curl http://localhost:8000/health
# Expected: {"status": "ok"}

# Auth rejection
curl http://localhost:8000/users
# Expected: 401 {"error": {"code": "UNAUTHORIZED", ...}}

# Create user
curl -X POST http://localhost:8000/users \
  -H "X-API-Key: dev-secret-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","email":"alice@test.com","password":"password123"}'
# Expected: 201 {"id": 1, "username": "alice", "email": "alice@test.com", ...}

# List users
curl http://localhost:8000/users -H "X-API-Key: dev-secret-key-change-me"
# Expected: 200 {"total": 1, "items": [...]}

# Create project
curl -X POST http://localhost:8000/projects \
  -H "X-API-Key: dev-secret-key-change-me" \
  -H "Content-Type: application/json" \
  -d '{"name":"My Project","owner_id":1}'
# Expected: 201 {"id": 1, "name": "My Project", "owner_id": 1, ...}

# Run seed script
python team-test/database/seed_data.py
# Expected: "Seeded 3 users and 4 projects."
```

### Schema validation
```bash
curl http://localhost:8000/openapi.json -H "X-API-Key: dev-secret-key-change-me" | python -m json.tool
# Expected: valid OpenAPI 3.x JSON with paths for /users and /projects
```
