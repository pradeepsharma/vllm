# Execution Plan: Python REST API Project in `team-test/`

## Project Overview

Create a fully functional Python REST API project under `team-test/` using FastAPI, SQLAlchemy, and Pydantic — consistent with the patterns already used in this vLLM workspace (see `vllm/entrypoints/api_server.py`, `vllm/entrypoints/openai/api_server.py`, and `vllm/entrypoints/openai/engine/protocol.py`).

**Target directory structure:**
```
team-test/
├── __init__.py
├── requirements.txt
├── database/
│   ├── __init__.py
│   ├── models.py
│   ├── connection.py
│   └── seed_data.py
├── api/
│   ├── __init__.py
│   ├── schemas.py
│   ├── middleware.py
│   └── endpoints.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_models.py
    └── test_endpoints.py
```

---

## Phase 1: Project Scaffold & Requirements

**Goal:** Create the top-level package files and `requirements.txt`.

### Files to create:

#### `team-test/__init__.py`
```python
# team-test package
```

#### `team-test/requirements.txt`
```
fastapi[standard]>=0.115.0
uvicorn[standard]>=0.30.0
sqlalchemy>=2.0.0
pydantic>=2.0.0
pytest>=7.0.0
pytest-asyncio>=0.23.0
httpx>=0.27.0
aiosqlite>=0.20.0
greenlet>=3.0.0
```

**Rationale:** Mirrors the workspace's own `requirements/common.txt` which already pins `fastapi[standard] >= 0.115.0` and `pydantic >= 2.12.0`. Uses SQLite (via `aiosqlite`) for zero-config local development. `httpx` is used for `TestClient` in FastAPI integration tests (same pattern as `requirements/test.in`).

---

## Phase 2: Database Layer

**Goal:** Create SQLAlchemy models, database connection setup, and seed data script.

### File: `team-test/database/__init__.py`
```python
# database package
```

### File: `team-test/database/connection.py`

```python
"""Database connection and session management."""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = "sqlite:///./team_test.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # Required for SQLite
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables defined in models."""
    from team_test.database import models  # noqa: F401 — import triggers table registration
    Base.metadata.create_all(bind=engine)
```

**Key design decisions:**
- Uses `DeclarativeBase` (SQLAlchemy 2.x style), consistent with modern Python projects.
- `check_same_thread=False` is required for SQLite with FastAPI's async request handling.
- `get_db()` is a FastAPI `Depends`-compatible generator.

### File: `team-test/database/models.py`

```python
"""SQLAlchemy ORM models for User and Project."""
import datetime
from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from team_test.database.connection import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    # Relationship: one user owns many projects
    projects: Mapped[list["Project"]] = relationship(
        "Project", back_populates="owner", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r}>"


class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    owner_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )

    # Relationship: many projects belong to one user
    owner: Mapped["User"] = relationship("User", back_populates="projects")

    def __repr__(self) -> str:
        return f"<Project id={self.id} name={self.name!r} owner_id={self.owner_id}>"
```

**Key design decisions:**
- Uses SQLAlchemy 2.x `Mapped` + `mapped_column` typed annotations (modern, type-safe).
- `User` → `Project` is a one-to-many relationship with cascade delete.
- `hashed_password` stores a bcrypt/plain hash (seeded with a placeholder in `seed_data.py`).

### File: `team-test/database/seed_data.py`

```python
"""Seed the database with sample Users and Projects."""
import hashlib

from team_test.database.connection import SessionLocal, init_db
from team_test.database.models import Project, User


def hash_password(password: str) -> str:
    """Simple SHA-256 hash for demo purposes (use bcrypt in production)."""
    return hashlib.sha256(password.encode()).hexdigest()


SAMPLE_USERS = [
    {"username": "alice", "email": "alice@example.com", "password": "secret123"},
    {"username": "bob",   "email": "bob@example.com",   "password": "password456"},
    {"username": "carol", "email": "carol@example.com", "password": "carol789"},
]

SAMPLE_PROJECTS = [
    {"name": "Alpha",   "description": "First project",  "owner_username": "alice"},
    {"name": "Beta",    "description": "Second project", "owner_username": "alice"},
    {"name": "Gamma",   "description": "Third project",  "owner_username": "bob"},
    {"name": "Delta",   "description": "Fourth project", "owner_username": "carol"},
]


def seed():
    init_db()
    db = SessionLocal()
    try:
        # Skip if data already exists
        if db.query(User).count() > 0:
            print("Database already seeded. Skipping.")
            return

        users: dict[str, User] = {}
        for u in SAMPLE_USERS:
            user = User(
                username=u["username"],
                email=u["email"],
                hashed_password=hash_password(u["password"]),
            )
            db.add(user)
            users[u["username"]] = user

        db.flush()  # Assign IDs before creating projects

        for p in SAMPLE_PROJECTS:
            project = Project(
                name=p["name"],
                description=p["description"],
                owner_id=users[p["owner_username"]].id,
            )
            db.add(project)

        db.commit()
        print(f"Seeded {len(SAMPLE_USERS)} users and {len(SAMPLE_PROJECTS)} projects.")
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()


if __name__ == "__main__":
    seed()
```

---

## Phase 3: API Layer — Schemas

**Goal:** Define Pydantic request/response schemas, mirroring the pattern in `vllm/entrypoints/openai/engine/protocol.py` where `OpenAIBaseModel(BaseModel)` is used with `ConfigDict`.

### File: `team-test/api/__init__.py`
```python
# api package
```

### File: `team-test/api/schemas.py`

```python
"""Pydantic schemas for request validation and response serialization."""
import datetime
from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ─── User Schemas ────────────────────────────────────────────────────────────

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, examples=["alice"])
    email: EmailStr = Field(..., examples=["alice@example.com"])


class UserCreate(UserBase):
    password: str = Field(..., min_length=6, examples=["secret123"])


class UserUpdate(BaseModel):
    username: str | None = Field(None, min_length=3, max_length=50)
    email: EmailStr | None = None
    is_active: bool | None = None


class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    is_active: bool
    created_at: datetime.datetime


class UserWithProjectsResponse(UserResponse):
    projects: list["ProjectResponse"] = []


# ─── Project Schemas ──────────────────────────────────────────────────────────

class ProjectBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, examples=["Alpha"])
    description: str | None = Field(None, examples=["First project"])


class ProjectCreate(ProjectBase):
    owner_id: int = Field(..., gt=0, examples=[1])


class ProjectUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = None
    is_active: bool | None = None


class ProjectResponse(ProjectBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    is_active: bool
    created_at: datetime.datetime
    owner_id: int


# ─── Generic Response Schemas ─────────────────────────────────────────────────

class MessageResponse(BaseModel):
    message: str


class ErrorResponse(BaseModel):
    detail: str


# Resolve forward references
UserWithProjectsResponse.model_rebuild()
```

**Key design decisions:**
- `ConfigDict(from_attributes=True)` enables ORM-mode (replaces `orm_mode = True` from Pydantic v1).
- Separate `Create`, `Update`, and `Response` schemas for each entity (standard REST pattern).
- `EmailStr` validates email format at the Pydantic level.

---

## Phase 4: API Layer — Middleware

**Goal:** Create authentication middleware using the same `BaseHTTPMiddleware` / ASGI pattern seen in `vllm/entrypoints/serve/elastic_ep/middleware.py` and `vllm/entrypoints/openai/api_server.py` (`AuthenticationMiddleware`).

### File: `team-test/api/middleware.py`

```python
"""Authentication and logging middleware."""
import time
import logging
from collections.abc import Awaitable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Paths that do not require authentication
PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}

# Simple static API key store (replace with DB lookup in production)
VALID_API_KEYS = {"dev-secret-key-123", "test-api-key-456"}


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Token-based authentication middleware.

    Checks for an `X-API-Key` header on all non-public routes.
    Returns HTTP 401 if the key is missing or invalid.

    Mirrors the pattern in vllm/entrypoints/openai/api_server.py
    where AuthenticationMiddleware is added via app.add_middleware().
    """

    def __init__(self, app: ASGIApp, api_keys: set[str] | None = None) -> None:
        super().__init__(app)
        self.api_keys = api_keys or VALID_API_KEYS

    async def dispatch(self, request: Request, call_next) -> Awaitable:
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key not in self.api_keys:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. Provide X-API-Key header."},
            )

        response = await call_next(request)
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs method, path, status code, and duration for every request.
    """

    async def dispatch(self, request: Request, call_next) -> Awaitable:
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s → %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
```

---

## Phase 5: API Layer — Endpoints

**Goal:** Create FastAPI CRUD routes for Users and Projects, using `Depends(get_db)` for session injection — the same dependency-injection pattern used throughout the vLLM OpenAI server.

### File: `team-test/api/endpoints.py`

```python
"""FastAPI CRUD endpoints for Users and Projects."""
from fastapi import APIRouter, Depends, FastAPI, HTTPException, status
from sqlalchemy.orm import Session

from team_test.api.middleware import AuthenticationMiddleware, RequestLoggingMiddleware
from team_test.api.schemas import (
    MessageResponse,
    ProjectCreate,
    ProjectResponse,
    ProjectUpdate,
    UserCreate,
    UserResponse,
    UserUpdate,
    UserWithProjectsResponse,
)
from team_test.database.connection import get_db, init_db
from team_test.database.models import Project, User

import hashlib

# ─── Routers ─────────────────────────────────────────────────────────────────

users_router = APIRouter(prefix="/users", tags=["Users"])
projects_router = APIRouter(prefix="/projects", tags=["Projects"])


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


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


# ─── User Endpoints ───────────────────────────────────────────────────────────

@users_router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    """Create a new user."""
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(status_code=409, detail="Username already taken")
    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=_hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@users_router.get("/", response_model=list[UserResponse])
def list_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List all users with pagination."""
    return db.query(User).offset(skip).limit(limit).all()


@users_router.get("/{user_id}", response_model=UserWithProjectsResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get a single user with their projects."""
    return _get_user_or_404(user_id, db)


@users_router.patch("/{user_id}", response_model=UserResponse)
def update_user(user_id: int, payload: UserUpdate, db: Session = Depends(get_db)):
    """Partially update a user."""
    user = _get_user_or_404(user_id, db)
    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(user, field, value)
    db.commit()
    db.refresh(user)
    return user


@users_router.delete("/{user_id}", response_model=MessageResponse)
def delete_user(user_id: int, db: Session = Depends(get_db)):
    """Delete a user (cascades to their projects)."""
    user = _get_user_or_404(user_id, db)
    db.delete(user)
    db.commit()
    return {"message": f"User {user_id} deleted successfully"}


# ─── Project Endpoints ────────────────────────────────────────────────────────

@projects_router.post("/", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)):
    """Create a new project."""
    _get_user_or_404(payload.owner_id, db)  # Validate owner exists
    project = Project(
        name=payload.name,
        description=payload.description,
        owner_id=payload.owner_id,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@projects_router.get("/", response_model=list[ProjectResponse])
def list_projects(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List all projects with pagination."""
    return db.query(Project).offset(skip).limit(limit).all()


@projects_router.get("/{project_id}", response_model=ProjectResponse)
def get_project(project_id: int, db: Session = Depends(get_db)):
    """Get a single project by ID."""
    return _get_project_or_404(project_id, db)


@projects_router.patch("/{project_id}", response_model=ProjectResponse)
def update_project(project_id: int, payload: ProjectUpdate, db: Session = Depends(get_db)):
    """Partially update a project."""
    project = _get_project_or_404(project_id, db)
    update_data = payload.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(project, field, value)
    db.commit()
    db.refresh(project)
    return project


@projects_router.delete("/{project_id}", response_model=MessageResponse)
def delete_project(project_id: int, db: Session = Depends(get_db)):
    """Delete a project."""
    project = _get_project_or_404(project_id, db)
    db.delete(project)
    db.commit()
    return {"message": f"Project {project_id} deleted successfully"}


# ─── App Factory ──────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    Registers middleware and routers.
    """
    app = FastAPI(
        title="Team Test REST API",
        description="CRUD API for Users and Projects",
        version="1.0.0",
    )

    # Middleware (order matters — outermost runs first)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(AuthenticationMiddleware)

    # Routers
    app.include_router(users_router)
    app.include_router(projects_router)

    @app.on_event("startup")
    def on_startup():
        init_db()

    @app.get("/health", tags=["Health"])
    def health():
        return {"status": "ok"}

    @app.get("/", tags=["Root"])
    def root():
        return {"message": "Team Test API is running"}

    return app


app = create_app()
```

---

## Phase 6: Tests

**Goal:** Create pytest fixtures (`conftest.py`), unit tests for models (`test_models.py`), and integration tests for endpoints (`test_endpoints.py`).

### File: `team-test/tests/__init__.py`
```python
# tests package
```

### File: `team-test/tests/conftest.py`

```python
"""
Pytest fixtures for team-test.

Uses an in-memory SQLite database so tests are isolated and fast.
Pattern mirrors tests/entrypoints/conftest.py in the vLLM workspace.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from team_test.api.endpoints import create_app
from team_test.database.connection import Base, get_db
from team_test.database.models import Project, User

TEST_DATABASE_URL = "sqlite:///:memory:"

TEST_API_KEY = "dev-secret-key-123"
AUTH_HEADERS = {"X-API-Key": TEST_API_KEY}


@pytest.fixture(scope="function")
def db_engine():
    """Create a fresh in-memory SQLite engine per test function."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture(scope="function")
def db_session(db_engine):
    """Provide a transactional database session that rolls back after each test."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture(scope="function")
def client(db_session):
    """
    FastAPI TestClient with the real app but an overridden in-memory DB session.
    Uses dependency_overrides to inject the test session.
    """
    app = create_app()

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_user(db_session) -> User:
    """Insert and return a sample User."""
    import hashlib
    user = User(
        username="testuser",
        email="testuser@example.com",
        hashed_password=hashlib.sha256(b"password").hexdigest(),
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def sample_project(db_session, sample_user) -> Project:
    """Insert and return a sample Project owned by sample_user."""
    project = Project(
        name="Test Project",
        description="A project for testing",
        owner_id=sample_user.id,
    )
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)
    return project
```

### File: `team-test/tests/test_models.py`

```python
"""Unit tests for SQLAlchemy User and Project models."""
import hashlib
import pytest
from sqlalchemy.exc import IntegrityError

from team_test.database.models import Project, User


class TestUserModel:
    def test_create_user(self, db_session):
        user = User(
            username="alice",
            email="alice@example.com",
            hashed_password=hashlib.sha256(b"secret").hexdigest(),
        )
        db_session.add(user)
        db_session.commit()
        assert user.id is not None
        assert user.is_active is True
        assert user.created_at is not None

    def test_user_repr(self, db_session, sample_user):
        assert "testuser" in repr(sample_user)

    def test_unique_email_constraint(self, db_session, sample_user):
        duplicate = User(
            username="other",
            email=sample_user.email,  # same email
            hashed_password="x",
        )
        db_session.add(duplicate)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_unique_username_constraint(self, db_session, sample_user):
        duplicate = User(
            username=sample_user.username,  # same username
            email="other@example.com",
            hashed_password="x",
        )
        db_session.add(duplicate)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_user_projects_relationship(self, db_session, sample_user, sample_project):
        db_session.refresh(sample_user)
        assert len(sample_user.projects) == 1
        assert sample_user.projects[0].name == "Test Project"


class TestProjectModel:
    def test_create_project(self, db_session, sample_user):
        project = Project(
            name="My Project",
            description="Description here",
            owner_id=sample_user.id,
        )
        db_session.add(project)
        db_session.commit()
        assert project.id is not None
        assert project.is_active is True
        assert project.owner_id == sample_user.id

    def test_project_repr(self, db_session, sample_project):
        assert "Test Project" in repr(sample_project)

    def test_project_owner_relationship(self, db_session, sample_project):
        db_session.refresh(sample_project)
        assert sample_project.owner.username == "testuser"

    def test_cascade_delete(self, db_session, sample_user, sample_project):
        """Deleting a user should cascade-delete their projects."""
        project_id = sample_project.id
        db_session.delete(sample_user)
        db_session.commit()
        deleted = db_session.query(Project).filter(Project.id == project_id).first()
        assert deleted is None

    def test_project_nullable_description(self, db_session, sample_user):
        project = Project(name="No Desc", owner_id=sample_user.id)
        db_session.add(project)
        db_session.commit()
        assert project.description is None
```

### File: `team-test/tests/test_endpoints.py`

```python
"""Integration tests for User and Project API endpoints."""
import pytest

AUTH = {"X-API-Key": "dev-secret-key-123"}


class TestHealthEndpoint:
    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_root(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "running" in resp.json()["message"]


class TestAuthMiddleware:
    def test_missing_api_key_returns_401(self, client):
        resp = client.get("/users/")
        assert resp.status_code == 401

    def test_invalid_api_key_returns_401(self, client):
        resp = client.get("/users/", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_valid_api_key_passes(self, client):
        resp = client.get("/users/", headers=AUTH)
        assert resp.status_code == 200


class TestUserEndpoints:
    def test_create_user(self, client):
        payload = {"username": "newuser", "email": "new@example.com", "password": "pass123"}
        resp = client.post("/users/", json=payload, headers=AUTH)
        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == "newuser"
        assert data["email"] == "new@example.com"
        assert "id" in data
        assert "hashed_password" not in data  # Never expose password hash

    def test_create_user_duplicate_email(self, client, sample_user):
        payload = {"username": "other", "email": sample_user.email, "password": "pass123"}
        resp = client.post("/users/", json=payload, headers=AUTH)
        assert resp.status_code == 409

    def test_create_user_duplicate_username(self, client, sample_user):
        payload = {"username": sample_user.username, "email": "other@x.com", "password": "pass123"}
        resp = client.post("/users/", json=payload, headers=AUTH)
        assert resp.status_code == 409

    def test_list_users(self, client, sample_user):
        resp = client.get("/users/", headers=AUTH)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) >= 1

    def test_get_user(self, client, sample_user):
        resp = client.get(f"/users/{sample_user.id}", headers=AUTH)
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == sample_user.id
        assert "projects" in data

    def test_get_user_not_found(self, client):
        resp = client.get("/users/99999", headers=AUTH)
        assert resp.status_code == 404

    def test_update_user(self, client, sample_user):
        resp = client.patch(
            f"/users/{sample_user.id}",
            json={"username": "updated_name"},
            headers=AUTH,
        )
        assert resp.status_code == 200
        assert resp.json()["username"] == "updated_name"

    def test_delete_user(self, client, sample_user):
        resp = client.delete(f"/users/{sample_user.id}", headers=AUTH)
        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"]
        # Confirm gone
        resp2 = client.get(f"/users/{sample_user.id}", headers=AUTH)
        assert resp2.status_code == 404


class TestProjectEndpoints:
    def test_create_project(self, client, sample_user):
        payload = {"name": "New Project", "description": "Desc", "owner_id": sample_user.id}
        resp = client.post("/projects/", json=payload, headers=AUTH)
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "New Project"
        assert data["owner_id"] == sample_user.id

    def test_create_project_invalid_owner(self, client):
        payload = {"name": "Orphan", "owner_id": 99999}
        resp = client.post("/projects/", json=payload, headers=AUTH)
        assert resp.status_code == 404

    def test_list_projects(self, client, sample_project):
        resp = client.get("/projects/", headers=AUTH)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) >= 1

    def test_get_project(self, client, sample_project):
        resp = client.get(f"/projects/{sample_project.id}", headers=AUTH)
        assert resp.status_code == 200
        assert resp.json()["id"] == sample_project.id

    def test_get_project_not_found(self, client):
        resp = client.get("/projects/99999", headers=AUTH)
        assert resp.status_code == 404

    def test_update_project(self, client, sample_project):
        resp = client.patch(
            f"/projects/{sample_project.id}",
            json={"name": "Renamed", "is_active": False},
            headers=AUTH,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Renamed"
        assert data["is_active"] is False

    def test_delete_project(self, client, sample_project):
        resp = client.delete(f"/projects/{sample_project.id}", headers=AUTH)
        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"]
        resp2 = client.get(f"/projects/{sample_project.id}", headers=AUTH)
        assert resp2.status_code == 404
```

---

## Execution Order & Parallelism

| Phase | Files | Can Parallelize With |
|-------|-------|----------------------|
| 1 | `__init__.py`, `requirements.txt` | — (foundation) |
| 2 | `database/__init__.py`, `connection.py`, `models.py`, `seed_data.py` | — (depends on Phase 1) |
| 3 | `api/schemas.py` | Can run in parallel with Phase 2 |
| 4 | `api/middleware.py` | Can run in parallel with Phase 2 |
| 5 | `api/endpoints.py` | Depends on Phases 2, 3, 4 |
| 6 | `tests/conftest.py`, `test_models.py`, `test_endpoints.py` | Depends on Phases 2–5 |

**Recommended parallel execution:**
- **Batch A** (parallel): Phase 1 scaffold
- **Batch B** (parallel): Phase 2 database layer + Phase 3 schemas + Phase 4 middleware
- **Batch C** (sequential): Phase 5 endpoints (imports from B)
- **Batch D** (sequential): Phase 6 tests (imports from all)

---

## Verification Criteria

### Installation
```bash
cd team-test
pip install -r requirements.txt
```
Expected: All packages install without errors.

### Database Seeding
```bash
cd team-test
python -m team_test.database.seed_data
```
Expected output:
```
Seeded 3 users and 4 projects.
```

### Run the API Server
```bash
cd team-test
uvicorn team_test.api.endpoints:app --reload --port 8080
```

### Endpoint Verification (curl)

**Health check (no auth required):**
```bash
curl -s http://localhost:8080/health
# Expected: {"status":"ok"}
```

**Auth rejection:**
```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/users/
# Expected: 401
```

**Create user:**
```bash
curl -s -X POST http://localhost:8080/users/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret-key-123" \
  -d '{"username":"testuser","email":"test@example.com","password":"pass123"}'
# Expected: HTTP 201, JSON with id, username, email, is_active, created_at
```

**List users:**
```bash
curl -s http://localhost:8080/users/ -H "X-API-Key: dev-secret-key-123"
# Expected: HTTP 200, JSON array of user objects
```

**Create project:**
```bash
curl -s -X POST http://localhost:8080/projects/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-secret-key-123" \
  -d '{"name":"My Project","description":"Test","owner_id":1}'
# Expected: HTTP 201, JSON with id, name, description, owner_id, is_active, created_at
```

**Get user with projects:**
```bash
curl -s http://localhost:8080/users/1 -H "X-API-Key: dev-secret-key-123"
# Expected: HTTP 200, JSON with "projects" array field
```

**404 for missing resource:**
```bash
curl -s -o /dev/null -w "%{http_code}" \
  http://localhost:8080/users/99999 -H "X-API-Key: dev-secret-key-123"
# Expected: 404
```

### Run Tests
```bash
cd team-test
pytest tests/ -v
```

**Expected output:**
```
tests/test_models.py::TestUserModel::test_create_user PASSED
tests/test_models.py::TestUserModel::test_user_repr PASSED
tests/test_models.py::TestUserModel::test_unique_email_constraint PASSED
tests/test_models.py::TestUserModel::test_unique_username_constraint PASSED
tests/test_models.py::TestUserModel::test_user_projects_relationship PASSED
tests/test_models.py::TestProjectModel::test_create_project PASSED
tests/test_models.py::TestProjectModel::test_project_repr PASSED
tests/test_models.py::TestProjectModel::test_project_owner_relationship PASSED
tests/test_models.py::TestProjectModel::test_cascade_delete PASSED
tests/test_models.py::TestProjectModel::test_project_nullable_description PASSED
tests/test_endpoints.py::TestHealthEndpoint::test_health_check PASSED
tests/test_endpoints.py::TestHealthEndpoint::test_root PASSED
tests/test_endpoints.py::TestAuthMiddleware::test_missing_api_key_returns_401 PASSED
tests/test_endpoints.py::TestAuthMiddleware::test_invalid_api_key_returns_401 PASSED
tests/test_endpoints.py::TestAuthMiddleware::test_valid_api_key_passes PASSED
tests/test_endpoints.py::TestUserEndpoints::test_create_user PASSED
tests/test_endpoints.py::TestUserEndpoints::test_create_user_duplicate_email PASSED
tests/test_endpoints.py::TestUserEndpoints::test_create_user_duplicate_username PASSED
tests/test_endpoints.py::TestUserEndpoints::test_list_users PASSED
tests/test_endpoints.py::TestUserEndpoints::test_get_user PASSED
tests/test_endpoints.py::TestUserEndpoints::test_get_user_not_found PASSED
tests/test_endpoints.py::TestUserEndpoints::test_update_user PASSED
tests/test_endpoints.py::TestUserEndpoints::test_delete_user PASSED
tests/test_endpoints.py::TestProjectEndpoints::test_create_project PASSED
tests/test_endpoints.py::TestProjectEndpoints::test_create_project_invalid_owner PASSED
tests/test_endpoints.py::TestProjectEndpoints::test_list_projects PASSED
tests/test_endpoints.py::TestProjectEndpoints::test_get_project PASSED
tests/test_endpoints.py::TestProjectEndpoints::test_get_project_not_found PASSED
tests/test_endpoints.py::TestProjectEndpoints::test_update_project PASSED
tests/test_endpoints.py::TestProjectEndpoints::test_delete_project PASSED

============================== 30 passed in <5s ==============================
```

**All 30 tests must pass. Zero failures, zero errors.**

### OpenAPI Docs
Navigate to `http://localhost:8080/docs` — the Swagger UI should display:
- `/health` (GET)
- `/users/` (GET, POST)
- `/users/{user_id}` (GET, PATCH, DELETE)
- `/projects/` (GET, POST)
- `/projects/{project_id}` (GET, PATCH, DELETE)

---

## File Reference Map

| File | Key Symbols |
|------|-------------|
| `team-test/database/connection.py` | `Base`, `engine`, `SessionLocal`, `get_db()`, `init_db()` |
| `team-test/database/models.py` | `User`, `Project` (SQLAlchemy ORM, `Mapped` annotations) |
| `team-test/database/seed_data.py` | `seed()`, `SAMPLE_USERS`, `SAMPLE_PROJECTS` |
| `team-test/api/schemas.py` | `UserCreate`, `UserResponse`, `UserWithProjectsResponse`, `ProjectCreate`, `ProjectResponse`, `ProjectUpdate`, `UserUpdate` |
| `team-test/api/middleware.py` | `AuthenticationMiddleware`, `RequestLoggingMiddleware` |
| `team-test/api/endpoints.py` | `users_router`, `projects_router`, `create_app()`, `app` |
| `team-test/tests/conftest.py` | `db_engine`, `db_session`, `client`, `sample_user`, `sample_project` |
| `team-test/tests/test_models.py` | `TestUserModel`, `TestProjectModel` |
| `team-test/tests/test_endpoints.py` | `TestHealthEndpoint`, `TestAuthMiddleware`, `TestUserEndpoints`, `TestProjectEndpoints` |

---

## Workspace Alignment Notes

- **FastAPI pattern**: Matches `vllm/entrypoints/api_server.py` (`FastAPI()`, `@app.get`, `@app.post`) and `vllm/entrypoints/openai/api_server.py` (`APIRouter`, `app.add_middleware`, `app.include_router`).
- **Pydantic schemas**: Matches `vllm/entrypoints/openai/engine/protocol.py` (`BaseModel`, `ConfigDict`, `Field`).
- **Middleware pattern**: Matches `vllm/entrypoints/serve/elastic_ep/middleware.py` (ASGI middleware class with `__call__`) and `vllm/entrypoints/openai/api_server.py` (`AuthenticationMiddleware`, `app.add_middleware()`).
- **Test fixtures**: Matches `tests/entrypoints/conftest.py` (`@pytest.fixture`, `scope="function"`, session-scoped fixtures).
- **Requirements**: Pins `fastapi[standard]>=0.115.0` and `pydantic>=2.0.0` consistent with `requirements/common.txt`.
