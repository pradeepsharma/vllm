# Plan: Create a Python REST API Project in `team-test/`

**Generated:** 2026-03-19  
**Workspace:** `/Users/pradeepsharma/sasva/projects/vllm`  
**Target directory:** `team-test/`

---

## Context & Existing State

From codebase exploration, the following already exists in `team-test/`:

| File | Status | Notes |
|------|--------|-------|
| `team-test/__init__.py` | ✅ Exists | Package docstring present |
| `team-test/requirements.txt` | ✅ Exists | fastapi, uvicorn, sqlalchemy, alembic, pydantic, pytest, pytest-asyncio, httpx, python-dotenv |
| `team-test/database/__init__.py` | ✅ Exists | Exports `Base`, `engine`, `get_db`, `SessionLocal`, `User`, `Project` |
| `team-test/database/connection.py` | ✅ Exists | SQLite default via `DATABASE_URL` env var; `get_db()` generator; `init_db()` |
| `team-test/database/models.py` | ✅ Exists | `User` (id, username, email, full_name, is_active, hashed_password, timestamps) + `Project` (id, name, description, is_active, owner_id FK→users, timestamps) with bidirectional relationship |
| `team-test/database/seed_data.py` | ✅ Exists | 3 seed users (alice, bob, charlie) + 4 seed projects; `seed()` function + `__main__` entry |

**Missing (must be created):**
- `team-test/api/__init__.py`
- `team-test/api/schemas.py`
- `team-test/api/middleware.py`
- `team-test/api/endpoints.py`
- `team-test/tests/__init__.py`
- `team-test/tests/conftest.py`
- `team-test/tests/test_models.py`
- `team-test/tests/test_endpoints.py`

**Key design decisions from existing code:**
- Import path uses `team_test` (underscore) as the Python package name (see `connection.py` line: `from team_test.database import models`)
- Database URL defaults to `sqlite:///./team_test.db`
- SQLAlchemy 2.x `Mapped`/`mapped_column` style
- Pydantic v2 (`pydantic>=2.12.0`)
- FastAPI with `Depends` for DB injection via `get_db()`
- Auth pattern: Bearer token in `Authorization` header (mirrors `vllm/entrypoints/openai/server_utils.py` `AuthenticationMiddleware`)

---

## Execution Phases

### Phase 1 — API Schemas (`team-test/api/schemas.py`)

**File:** `team-test/api/schemas.py`

Create Pydantic v2 request/response schemas for both `User` and `Project` resources.

```python
# team-test/api/schemas.py
"""Pydantic v2 request/response schemas for Users and Projects."""

from __future__ import annotations

import datetime
from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ── User schemas ──────────────────────────────────────────────────────────────

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: str | None = Field(None, max_length=100)
    is_active: bool = True


class UserCreate(UserBase):
    """Used for POST /users — includes plain-text password."""
    password: str = Field(..., min_length=6)


class UserUpdate(BaseModel):
    """Used for PATCH /users/{id} — all fields optional."""
    full_name: str | None = Field(None, max_length=100)
    is_active: bool | None = None
    password: str | None = Field(None, min_length=6)


class UserResponse(UserBase):
    """Returned by all user endpoints — never exposes hashed_password."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime


# ── Project schemas ───────────────────────────────────────────────────────────

class ProjectBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    is_active: bool = True


class ProjectCreate(ProjectBase):
    """Used for POST /projects — owner_id must reference an existing user."""
    owner_id: int


class ProjectUpdate(BaseModel):
    """Used for PATCH /projects/{id} — all fields optional."""
    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = None
    is_active: bool | None = None


class ProjectResponse(ProjectBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    owner_id: int
    created_at: datetime.datetime
    updated_at: datetime.datetime


# ── Generic wrappers ──────────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    message: str
```

**Key decisions:**
- `UserCreate` has `password` (plain text); endpoints hash it before storing in `hashed_password`
- `UserResponse` never exposes `hashed_password`
- `model_config = ConfigDict(from_attributes=True)` enables ORM → Pydantic conversion
- `EmailStr` requires `pydantic[email]` — add `email-validator>=2.0` to requirements if not present (it's bundled with `pydantic[standard]` which is already in requirements)

---

### Phase 2 — Authentication Middleware (`team-test/api/middleware.py`)

**File:** `team-test/api/middleware.py`

Implement a lightweight Bearer-token auth middleware following the same ASGI pattern used in `vllm/entrypoints/openai/server_utils.py` (`AuthenticationMiddleware`).

```python
# team-test/api/middleware.py
"""
Authentication middleware for the team-test REST API.

Design mirrors vllm/entrypoints/openai/server_utils.py::AuthenticationMiddleware:
  - Pure ASGI middleware (no Starlette BaseHTTPMiddleware overhead)
  - Checks Authorization: Bearer <token> header
  - Skips auth for: OPTIONS requests, /health, /docs, /openapi.json, /redoc
  - Token is read from API_TOKEN env var (default: "dev-token" for local dev)
  - Returns 401 JSON on failure
"""

from __future__ import annotations

import hashlib
import os
import secrets
from collections.abc import Awaitable

from fastapi.responses import JSONResponse
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Receive, Scope, Send

# Public paths that bypass authentication
_PUBLIC_PATHS: frozenset[str] = frozenset({
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
})

_DEFAULT_TOKEN = "dev-token"


class BearerAuthMiddleware:
    """
    Pure ASGI middleware: validates Bearer token on every protected request.

    Configuration
    -------------
    Set the ``API_TOKEN`` environment variable to the expected token.
    Multiple tokens can be separated by commas:
        API_TOKEN=token1,token2

    Skipped for
    -----------
    - HTTP OPTIONS (pre-flight)
    - Paths in _PUBLIC_PATHS (/health, /docs, /openapi.json, /redoc)
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        raw = os.getenv("API_TOKEN", _DEFAULT_TOKEN)
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        # Store SHA-256 digests to avoid timing attacks on plain-text comparison
        self._token_hashes: list[bytes] = [
            hashlib.sha256(t.encode()).digest() for t in tokens
        ]

    def _is_valid_token(self, headers: Headers) -> bool:
        auth = headers.get("Authorization", "")
        scheme, _, param = auth.partition(" ")
        if scheme.lower() != "bearer" or not param:
            return False
        param_hash = hashlib.sha256(param.encode()).digest()
        return any(
            secrets.compare_digest(param_hash, stored)
            for stored in self._token_hashes
        )

    def __call__(self, scope: Scope, receive: Receive, send: Send) -> Awaitable[None]:
        if scope["type"] not in ("http", "websocket"):
            # lifespan / startup events — pass through
            return self.app(scope, receive, send)

        method = scope.get("method", "")
        if method == "OPTIONS":
            return self.app(scope, receive, send)

        path: str = scope.get("path", "")
        if path in _PUBLIC_PATHS:
            return self.app(scope, receive, send)

        headers = Headers(scope=scope)
        if not self._is_valid_token(headers):
            response = JSONResponse(
                content={"detail": "Unauthorized — valid Bearer token required"},
                status_code=401,
            )
            return response(scope, receive, send)

        return self.app(scope, receive, send)
```

**Key decisions:**
- Token(s) from `API_TOKEN` env var; comma-separated for multi-token support
- SHA-256 + `secrets.compare_digest` prevents timing attacks
- Public paths: `/health`, `/docs`, `/openapi.json`, `/redoc`
- Pure ASGI (no `BaseHTTPMiddleware`) — same pattern as vLLM's own middleware

---

### Phase 3 — API Endpoints (`team-test/api/endpoints.py`)

**File:** `team-test/api/endpoints.py`

Full CRUD for `User` and `Project` using FastAPI `APIRouter`. Depends on `get_db()` from `team-test/database/connection.py`.

```python
# team-test/api/endpoints.py
"""
FastAPI CRUD routes for Users and Projects.

Routers
-------
users_router  — prefix /users
projects_router — prefix /projects

Both routers are mounted on a top-level `app` FastAPI instance with
BearerAuthMiddleware applied.

User CRUD
---------
POST   /users/           — create user (hashes password with hashlib.sha256)
GET    /users/           — list all users (supports ?skip=&limit=)
GET    /users/{user_id}  — get single user
PATCH  /users/{user_id}  — partial update (full_name, is_active, password)
DELETE /users/{user_id}  — delete user (cascades to projects)

Project CRUD
------------
POST   /projects/              — create project (validates owner_id exists)
GET    /projects/              — list all projects (?skip=&limit=)
GET    /projects/{project_id}  — get single project
PATCH  /projects/{project_id}  — partial update
DELETE /projects/{project_id}  — delete project

Health
------
GET /health  — returns {"status": "ok"} — no auth required
"""

from __future__ import annotations

import hashlib
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, status
from sqlalchemy.orm import Session

from team_test.api.middleware import BearerAuthMiddleware
from team_test.api.schemas import (
    MessageResponse,
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
    UserCreate,
    UserResponse,
    UserUpdate,
)
from team_test.database.connection import get_db, init_db
from team_test.database.models import Project, User

# ── Helpers ───────────────────────────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    """Deterministic SHA-256 hash (replace with bcrypt in production)."""
    return hashlib.sha256(plain.encode()).hexdigest()


def _get_user_or_404(user_id: int, db: Session) -> User:
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")
    return user


def _get_project_or_404(project_id: int, db: Session) -> Project:
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id} not found")
    return project


# ── Users router ──────────────────────────────────────────────────────────────

users_router = APIRouter(prefix="/users", tags=["users"])


@users_router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate, db: Session = Depends(get_db)) -> User:
    # Uniqueness checks
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(status_code=409, detail="Username already taken")
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    user = User(
        username=payload.username,
        email=payload.email,
        full_name=payload.full_name,
        is_active=payload.is_active,
        hashed_password=_hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@users_router.get("/", response_model=list[UserResponse])
def list_users(
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    db: Session = Depends(get_db),
) -> list[User]:
    return db.query(User).offset(skip).limit(limit).all()


@users_router.get("/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)) -> User:
    return _get_user_or_404(user_id, db)


@users_router.patch("/{user_id}", response_model=UserResponse)
def update_user(user_id: int, payload: UserUpdate, db: Session = Depends(get_db)) -> User:
    user = _get_user_or_404(user_id, db)
    if payload.full_name is not None:
        user.full_name = payload.full_name
    if payload.is_active is not None:
        user.is_active = payload.is_active
    if payload.password is not None:
        user.hashed_password = _hash_password(payload.password)
    db.commit()
    db.refresh(user)
    return user


@users_router.delete("/{user_id}", response_model=MessageResponse)
def delete_user(user_id: int, db: Session = Depends(get_db)) -> dict:
    user = _get_user_or_404(user_id, db)
    db.delete(user)
    db.commit()
    return {"message": f"User {user_id} deleted"}


# ── Projects router ───────────────────────────────────────────────────────────

projects_router = APIRouter(prefix="/projects", tags=["projects"])


@projects_router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> Project:
    # Validate owner exists
    if db.get(User, payload.owner_id) is None:
        raise HTTPException(status_code=404, detail=f"Owner user {payload.owner_id} not found")
    project = Project(
        name=payload.name,
        description=payload.description,
        is_active=payload.is_active,
        owner_id=payload.owner_id,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@projects_router.get("/", response_model=list[ProjectResponse])
def list_projects(
    skip: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=200)] = 100,
    db: Session = Depends(get_db),
) -> list[Project]:
    return db.query(Project).offset(skip).limit(limit).all()


@projects_router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: int, db: Session = Depends(get_db)) -> Project:
    return _get_project_or_404(project_id, db)


@projects_router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(project_id: int, payload: ProjectUpdate, db: Session = Depends(get_db)) -> Project:
    project = _get_project_or_404(project_id, db)
    if payload.name is not None:
        project.name = payload.name
    if payload.description is not None:
        project.description = payload.description
    if payload.is_active is not None:
        project.is_active = payload.is_active
    db.commit()
    db.refresh(project)
    return project


@projects_router.delete("/{project_id}", response_model=MessageResponse)
def delete_project(project_id: int, db: Session = Depends(get_db)) -> dict:
    project = _get_project_or_404(project_id, db)
    db.delete(project)
    db.commit()
    return {"message": f"Project {project_id} deleted"}


# ── App assembly ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Factory function — creates and configures the FastAPI application."""
    application = FastAPI(
        title="team-test REST API",
        description="CRUD API for Users and Projects",
        version="0.1.0",
    )
    application.add_middleware(BearerAuthMiddleware)
    application.include_router(users_router)
    application.include_router(projects_router)

    @application.get("/health", tags=["health"])
    def health() -> dict:
        return {"status": "ok"}

    @application.on_event("startup")
    def on_startup() -> None:
        init_db()

    return application


app = create_app()
```

**Key decisions:**
- `create_app()` factory pattern — makes testing easy (override `get_db` dependency)
- `_hash_password` uses SHA-256 (same as `seed_data.py`); comment notes bcrypt for production
- 409 Conflict on duplicate username/email
- 404 on missing user/project or missing owner
- `on_event("startup")` calls `init_db()` to auto-create tables

---

### Phase 4 — API `__init__.py` (`team-test/api/__init__.py`)

**File:** `team-test/api/__init__.py`

```python
# team-test/api/__init__.py
"""team-test API layer: FastAPI app, routers, middleware, and schemas."""

from .endpoints import app, create_app, projects_router, users_router
from .middleware import BearerAuthMiddleware
from .schemas import (
    MessageResponse,
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
    UserCreate,
    UserResponse,
    UserUpdate,
)

__all__ = [
    "app",
    "create_app",
    "users_router",
    "projects_router",
    "BearerAuthMiddleware",
    "UserCreate",
    "UserResponse",
    "UserUpdate",
    "ProjectCreate",
    "ProjectResponse",
    "ProjectUpdate",
    "MessageResponse",
]
```

---

### Phase 5 — Test Fixtures (`team-test/tests/conftest.py`)

**File:** `team-test/tests/conftest.py`

Provides isolated in-memory SQLite database and a pre-configured `TestClient` for all tests.

```python
# team-test/tests/conftest.py
"""
Pytest fixtures for team-test tests.

Fixtures
--------
engine_fixture   — in-memory SQLite engine (function scope)
db_session       — SQLAlchemy Session bound to in-memory engine
client           — FastAPI TestClient with get_db overridden
seeded_db        — db_session with alice/bob users + 2 projects pre-inserted
"""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from team_test.api.endpoints import create_app
from team_test.database.connection import Base, get_db
from team_test.database.models import Project, User

# ── Database fixtures ─────────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def engine_fixture():
    """Fresh in-memory SQLite engine per test function."""
    eng = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=eng)
    yield eng
    Base.metadata.drop_all(bind=eng)
    eng.dispose()


@pytest.fixture(scope="function")
def db_session(engine_fixture) -> Session:
    """Transactional session that rolls back after each test."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=engine_fixture
    )
    session = TestingSessionLocal()
    yield session
    session.close()


# ── FastAPI test client ───────────────────────────────────────────────────────

@pytest.fixture(scope="function")
def client(engine_fixture) -> TestClient:
    """
    TestClient with:
    - get_db dependency overridden to use in-memory engine
    - API_TOKEN set to 'test-token' via default BearerAuthMiddleware behaviour
    """
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=engine_fixture
    )

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    application = create_app()
    application.dependency_overrides[get_db] = override_get_db

    # Patch middleware token for tests
    import os
    os.environ.setdefault("API_TOKEN", "test-token")

    with TestClient(application, headers={"Authorization": "Bearer test-token"}) as c:
        yield c


# ── Seeded data fixture ───────────────────────────────────────────────────────

def _fake_hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


@pytest.fixture(scope="function")
def seeded_db(db_session: Session) -> Session:
    """db_session pre-populated with 2 users and 2 projects."""
    alice = User(
        username="alice",
        email="alice@example.com",
        full_name="Alice Wonderland",
        hashed_password=_fake_hash("alice_secret"),
        is_active=True,
    )
    bob = User(
        username="bob",
        email="bob@example.com",
        full_name="Bob Builder",
        hashed_password=_fake_hash("bob_secret"),
        is_active=True,
    )
    db_session.add_all([alice, bob])
    db_session.flush()

    p1 = Project(name="Alpha", description="Alice project", owner_id=alice.id, is_active=True)
    p2 = Project(name="Beta", description="Bob project", owner_id=bob.id, is_active=True)
    db_session.add_all([p1, p2])
    db_session.commit()
    return db_session


@pytest.fixture(scope="function")
def seeded_client(engine_fixture) -> TestClient:
    """TestClient pre-seeded with alice, bob, and 2 projects."""
    import os
    os.environ.setdefault("API_TOKEN", "test-token")

    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=engine_fixture
    )

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    # Seed data
    seed_db = TestingSessionLocal()
    alice = User(
        username="alice", email="alice@example.com",
        full_name="Alice Wonderland",
        hashed_password=_fake_hash("alice_secret"), is_active=True,
    )
    bob = User(
        username="bob", email="bob@example.com",
        full_name="Bob Builder",
        hashed_password=_fake_hash("bob_secret"), is_active=True,
    )
    seed_db.add_all([alice, bob])
    seed_db.flush()
    seed_db.add_all([
        Project(name="Alpha", description="Alice project", owner_id=alice.id, is_active=True),
        Project(name="Beta", description="Bob project", owner_id=bob.id, is_active=True),
    ])
    seed_db.commit()
    seed_db.close()

    application = create_app()
    application.dependency_overrides[get_db] = override_get_db

    with TestClient(application, headers={"Authorization": "Bearer test-token"}) as c:
        yield c
```

---

### Phase 6 — Model Unit Tests (`team-test/tests/test_models.py`)

**File:** `team-test/tests/test_models.py`

```python
# team-test/tests/test_models.py
"""Unit tests for SQLAlchemy ORM models (User and Project)."""

import hashlib
import pytest
from sqlalchemy.exc import IntegrityError

from team_test.database.models import Project, User


def _hash(pw: str) -> str:
    return hashlib.sha256(pw.encode()).hexdigest()


class TestUserModel:
    def test_create_user(self, db_session):
        user = User(
            username="testuser",
            email="test@example.com",
            hashed_password=_hash("secret"),
        )
        db_session.add(user)
        db_session.commit()
        db_session.refresh(user)

        assert user.id is not None
        assert user.username == "testuser"
        assert user.email == "test@example.com"
        assert user.is_active is True  # default
        assert user.full_name is None  # nullable
        assert user.created_at is not None
        assert user.updated_at is not None

    def test_user_repr(self, db_session):
        user = User(username="repr_user", email="r@example.com", hashed_password=_hash("x"))
        db_session.add(user)
        db_session.commit()
        assert "repr_user" in repr(user)

    def test_unique_username_constraint(self, db_session):
        u1 = User(username="dup", email="dup1@example.com", hashed_password=_hash("a"))
        u2 = User(username="dup", email="dup2@example.com", hashed_password=_hash("b"))
        db_session.add(u1)
        db_session.commit()
        db_session.add(u2)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_unique_email_constraint(self, db_session):
        u1 = User(username="u1", email="same@example.com", hashed_password=_hash("a"))
        u2 = User(username="u2", email="same@example.com", hashed_password=_hash("b"))
        db_session.add(u1)
        db_session.commit()
        db_session.add(u2)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_user_is_active_default_true(self, db_session):
        user = User(username="active_default", email="ad@example.com", hashed_password=_hash("x"))
        db_session.add(user)
        db_session.commit()
        assert user.is_active is True

    def test_user_can_be_inactive(self, db_session):
        user = User(
            username="inactive", email="inactive@example.com",
            hashed_password=_hash("x"), is_active=False,
        )
        db_session.add(user)
        db_session.commit()
        assert user.is_active is False


class TestProjectModel:
    def test_create_project(self, seeded_db):
        user = seeded_db.query(User).filter_by(username="alice").first()
        project = Project(
            name="New Project",
            description="A test project",
            owner_id=user.id,
        )
        seeded_db.add(project)
        seeded_db.commit()
        seeded_db.refresh(project)

        assert project.id is not None
        assert project.name == "New Project"
        assert project.owner_id == user.id
        assert project.is_active is True

    def test_project_repr(self, seeded_db):
        project = seeded_db.query(Project).first()
        assert "Project" in repr(project)
        assert str(project.id) in repr(project)

    def test_project_owner_relationship(self, seeded_db):
        project = seeded_db.query(Project).filter_by(name="Alpha").first()
        assert project.owner is not None
        assert project.owner.username == "alice"

    def test_user_projects_relationship(self, seeded_db):
        alice = seeded_db.query(User).filter_by(username="alice").first()
        assert len(alice.projects) >= 1
        assert any(p.name == "Alpha" for p in alice.projects)

    def test_cascade_delete_user_deletes_projects(self, seeded_db):
        alice = seeded_db.query(User).filter_by(username="alice").first()
        alice_id = alice.id
        seeded_db.delete(alice)
        seeded_db.commit()
        remaining = seeded_db.query(Project).filter_by(owner_id=alice_id).all()
        assert remaining == []

    def test_project_nullable_description(self, seeded_db):
        user = seeded_db.query(User).filter_by(username="bob").first()
        project = Project(name="No Desc", owner_id=user.id)
        seeded_db.add(project)
        seeded_db.commit()
        assert project.description is None


class TestSeedData:
    def test_seed_inserts_users_and_projects(self, db_session):
        from team_test.database.seed_data import seed
        seed(db=db_session)
        assert db_session.query(User).count() == 3
        assert db_session.query(Project).count() == 4

    def test_seed_is_idempotent(self, db_session):
        from team_test.database.seed_data import seed
        seed(db=db_session)
        seed(db=db_session)  # second call should be a no-op
        assert db_session.query(User).count() == 3
```

---

### Phase 7 — Endpoint Integration Tests (`team-test/tests/test_endpoints.py`)

**File:** `team-test/tests/test_endpoints.py`

```python
# team-test/tests/test_endpoints.py
"""Integration tests for FastAPI CRUD endpoints (Users and Projects)."""

import pytest


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_no_auth(self, client):
        # /health is public — remove auth header for this test
        from fastapi.testclient import TestClient
        resp = client.get("/health", headers={})
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


# ── Auth middleware ───────────────────────────────────────────────────────────

class TestAuth:
    def test_missing_token_returns_401(self, client):
        resp = client.get("/users/", headers={})
        assert resp.status_code == 401

    def test_wrong_token_returns_401(self, client):
        resp = client.get("/users/", headers={"Authorization": "Bearer wrong-token"})
        assert resp.status_code == 401

    def test_valid_token_passes(self, client):
        resp = client.get("/users/")  # client fixture includes valid token
        assert resp.status_code == 200


# ── User CRUD ─────────────────────────────────────────────────────────────────

class TestUserCreate:
    def test_create_user_success(self, client):
        payload = {
            "username": "newuser",
            "email": "newuser@example.com",
            "password": "securepass",
        }
        resp = client.post("/users/", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == "newuser"
        assert data["email"] == "newuser@example.com"
        assert "hashed_password" not in data
        assert "id" in data

    def test_create_user_duplicate_username(self, seeded_client):
        payload = {"username": "alice", "email": "other@example.com", "password": "pass123"}
        resp = seeded_client.post("/users/", json=payload)
        assert resp.status_code == 409

    def test_create_user_duplicate_email(self, seeded_client):
        payload = {"username": "newname", "email": "alice@example.com", "password": "pass123"}
        resp = seeded_client.post("/users/", json=payload)
        assert resp.status_code == 409

    def test_create_user_invalid_email(self, client):
        payload = {"username": "baduser", "email": "not-an-email", "password": "pass123"}
        resp = client.post("/users/", json=payload)
        assert resp.status_code == 422

    def test_create_user_short_password(self, client):
        payload = {"username": "shortpw", "email": "short@example.com", "password": "abc"}
        resp = client.post("/users/", json=payload)
        assert resp.status_code == 422


class TestUserRead:
    def test_list_users_empty(self, client):
        resp = client.get("/users/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_users_with_data(self, seeded_client):
        resp = seeded_client.get("/users/")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_get_user_by_id(self, seeded_client):
        users = seeded_client.get("/users/").json()
        user_id = users[0]["id"]
        resp = seeded_client.get(f"/users/{user_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == user_id

    def test_get_user_not_found(self, client):
        resp = client.get("/users/9999")
        assert resp.status_code == 404

    def test_list_users_pagination(self, seeded_client):
        resp = seeded_client.get("/users/?skip=0&limit=1")
        assert resp.status_code == 200
        assert len(resp.json()) == 1


class TestUserUpdate:
    def test_update_user_full_name(self, seeded_client):
        users = seeded_client.get("/users/").json()
        user_id = users[0]["id"]
        resp = seeded_client.patch(f"/users/{user_id}", json={"full_name": "Updated Name"})
        assert resp.status_code == 200
        assert resp.json()["full_name"] == "Updated Name"

    def test_update_user_deactivate(self, seeded_client):
        users = seeded_client.get("/users/").json()
        user_id = users[0]["id"]
        resp = seeded_client.patch(f"/users/{user_id}", json={"is_active": False})
        assert resp.status_code == 200
        assert resp.json()["is_active"] is False

    def test_update_user_not_found(self, client):
        resp = client.patch("/users/9999", json={"full_name": "Ghost"})
        assert resp.status_code == 404


class TestUserDelete:
    def test_delete_user(self, seeded_client):
        users = seeded_client.get("/users/").json()
        user_id = users[0]["id"]
        resp = seeded_client.delete(f"/users/{user_id}")
        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"]
        # Confirm gone
        assert seeded_client.get(f"/users/{user_id}").status_code == 404

    def test_delete_user_not_found(self, client):
        resp = client.delete("/users/9999")
        assert resp.status_code == 404


# ── Project CRUD ──────────────────────────────────────────────────────────────

class TestProjectCreate:
    def test_create_project_success(self, seeded_client):
        users = seeded_client.get("/users/").json()
        owner_id = users[0]["id"]
        payload = {"name": "New Project", "description": "Test", "owner_id": owner_id}
        resp = seeded_client.post("/projects/", json=payload)
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "New Project"
        assert data["owner_id"] == owner_id

    def test_create_project_invalid_owner(self, client):
        payload = {"name": "Orphan", "owner_id": 9999}
        resp = client.post("/projects/", json=payload)
        assert resp.status_code == 404

    def test_create_project_missing_name(self, seeded_client):
        users = seeded_client.get("/users/").json()
        resp = seeded_client.post("/projects/", json={"owner_id": users[0]["id"]})
        assert resp.status_code == 422


class TestProjectRead:
    def test_list_projects_empty(self, client):
        resp = client.get("/projects/")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_list_projects_with_data(self, seeded_client):
        resp = seeded_client.get("/projects/")
        assert resp.status_code == 200
        assert len(resp.json()) == 2

    def test_get_project_by_id(self, seeded_client):
        projects = seeded_client.get("/projects/").json()
        pid = projects[0]["id"]
        resp = seeded_client.get(f"/projects/{pid}")
        assert resp.status_code == 200
        assert resp.json()["id"] == pid

    def test_get_project_not_found(self, client):
        resp = client.get("/projects/9999")
        assert resp.status_code == 404


class TestProjectUpdate:
    def test_update_project_name(self, seeded_client):
        projects = seeded_client.get("/projects/").json()
        pid = projects[0]["id"]
        resp = seeded_client.patch(f"/projects/{pid}", json={"name": "Renamed"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "Renamed"

    def test_update_project_deactivate(self, seeded_client):
        projects = seeded_client.get("/projects/").json()
        pid = projects[0]["id"]
        resp = seeded_client.patch(f"/projects/{pid}", json={"is_active": False})
        assert resp.status_code == 200
        assert resp.json()["is_active"] is False

    def test_update_project_not_found(self, client):
        resp = client.patch("/projects/9999", json={"name": "Ghost"})
        assert resp.status_code == 404


class TestProjectDelete:
    def test_delete_project(self, seeded_client):
        projects = seeded_client.get("/projects/").json()
        pid = projects[0]["id"]
        resp = seeded_client.delete(f"/projects/{pid}")
        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"]
        assert seeded_client.get(f"/projects/{pid}").status_code == 404

    def test_delete_project_not_found(self, client):
        resp = client.delete("/projects/9999")
        assert resp.status_code == 404


# ── Cascade behaviour ─────────────────────────────────────────────────────────

class TestCascade:
    def test_delete_user_cascades_to_projects(self, seeded_client):
        """Deleting a user must also delete their projects (CASCADE)."""
        users = seeded_client.get("/users/").json()
        alice = next(u for u in users if u["username"] == "alice")
        alice_id = alice["id"]

        # Confirm alice has projects
        projects_before = [
            p for p in seeded_client.get("/projects/").json()
            if p["owner_id"] == alice_id
        ]
        assert len(projects_before) >= 1

        # Delete alice
        seeded_client.delete(f"/users/{alice_id}")

        # Alice's projects should be gone
        projects_after = [
            p for p in seeded_client.get("/projects/").json()
            if p["owner_id"] == alice_id
        ]
        assert projects_after == []
```

---

### Phase 8 — Tests `__init__.py` (`team-test/tests/__init__.py`)

**File:** `team-test/tests/__init__.py`

```python
# team-test/tests/__init__.py
"""Test suite for team-test REST API."""
```

---

## File Creation Order (Dependency Graph)

```
Phase 1: team-test/api/schemas.py          (no internal deps)
Phase 2: team-test/api/middleware.py       (no internal deps)
Phase 3: team-test/api/endpoints.py        (depends on schemas, middleware, database/)
Phase 4: team-test/api/__init__.py         (depends on endpoints, middleware, schemas)
Phase 5: team-test/tests/conftest.py       (depends on api/endpoints, database/)
Phase 6: team-test/tests/test_models.py    (depends on conftest, database/models, seed_data)
Phase 7: team-test/tests/test_endpoints.py (depends on conftest, api/endpoints)
Phase 8: team-test/tests/__init__.py       (no deps)
```

Phases 1 & 2 can run in parallel. Phase 3 must follow 1 & 2. Phase 4 follows 3. Phases 5 & 8 can run in parallel after 4. Phases 6 & 7 follow 5.

---

## Import Path Note

The existing `connection.py` uses `from team_test.database import models` (underscore). This means the project must be installed or the workspace root must be on `sys.path`. For running tests:

```bash
cd /Users/pradeepsharma/sasva/projects/vllm
pip install -e team-test/   # if setup.py/pyproject.toml exists in team-test/
# OR
PYTHONPATH=. pytest team-test/tests/ -v
```

Since there is no `setup.py` in `team-test/`, use `PYTHONPATH=.` approach.

---

## Verification Criteria

### Setup
```bash
cd /Users/pradeepsharma/sasva/projects/vllm
pip install -r team-test/requirements.txt
```

### Run all tests
```bash
PYTHONPATH=. pytest team-test/tests/ -v
```

**Expected output:**
- All tests pass (0 failures, 0 errors)
- Approximate test count: **~40 tests** across `test_models.py` and `test_endpoints.py`
- `test_models.py`: ~12 tests covering User/Project model constraints, relationships, cascade delete, seed data
- `test_endpoints.py`: ~28 tests covering health, auth, full CRUD for users and projects, cascade behaviour

### Specific test expectations

| Test | Expected result |
|------|----------------|
| `TestHealth::test_health_no_auth` | HTTP 200, `{"status": "ok"}` |
| `TestAuth::test_missing_token_returns_401` | HTTP 401 |
| `TestAuth::test_valid_token_passes` | HTTP 200 |
| `TestUserCreate::test_create_user_success` | HTTP 201, body has `id`, `username`, no `hashed_password` |
| `TestUserCreate::test_create_user_duplicate_username` | HTTP 409 |
| `TestUserCreate::test_create_user_invalid_email` | HTTP 422 |
| `TestUserRead::test_list_users_with_data` | HTTP 200, list of 2 users |
| `TestUserDelete::test_delete_user` | HTTP 200, subsequent GET returns 404 |
| `TestProjectCreate::test_create_project_invalid_owner` | HTTP 404 |
| `TestCascade::test_delete_user_cascades_to_projects` | HTTP 200 on delete, alice's projects gone |
| `TestSeedData::test_seed_inserts_users_and_projects` | 3 users, 4 projects in DB |
| `TestSeedData::test_seed_is_idempotent` | Still 3 users after second seed call |

### Manual API smoke test
```bash
# Start server
PYTHONPATH=. API_TOKEN=mytoken uvicorn team_test.api.endpoints:app --reload

# Health (no auth)
curl http://localhost:8000/health
# Expected: {"status":"ok"}

# List users (with auth)
curl -H "Authorization: Bearer mytoken" http://localhost:8000/users/
# Expected: []

# Create user
curl -X POST http://localhost:8000/users/ \
  -H "Authorization: Bearer mytoken" \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","email":"alice@example.com","password":"secret123"}'
# Expected: HTTP 201, JSON with id, username, email, no hashed_password

# Create project
curl -X POST http://localhost:8000/projects/ \
  -H "Authorization: Bearer mytoken" \
  -H "Content-Type: application/json" \
  -d '{"name":"My Project","owner_id":1}'
# Expected: HTTP 201, JSON with id, name, owner_id

# Unauthorized request
curl http://localhost:8000/users/
# Expected: HTTP 401, {"detail":"Unauthorized — valid Bearer token required"}
```

### Seed script
```bash
PYTHONPATH=. python team-test/database/seed_data.py
# Expected: "Seeded 3 users and 4 projects."
```
