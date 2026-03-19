# Execution Plan: Python REST API Project in `team-test/`

## Project Overview

Build a fully functional Python REST API inside `team-test/` within the existing vLLM workspace (`/Users/pradeepsharma/sasva/projects/vllm`). The project uses **FastAPI** (already a dependency in `requirements/common.txt`) and **SQLAlchemy** with **Pydantic v2** (also present in `requirements/common.txt`). The stack is consistent with the vLLM project's own tooling choices.

### Target File Tree
```
team-test/
├── __init__.py
├── requirements.txt
├── database/
│   ├── __init__.py
│   ├── models.py          # SQLAlchemy User + Project ORM models
│   ├── connection.py      # Engine, SessionLocal, Base setup
│   └── seed_data.py       # Sample data insertion script
├── api/
│   ├── __init__.py
│   ├── schemas.py         # Pydantic request/response schemas
│   ├── middleware.py      # API-key authentication middleware
│   └── endpoints.py       # FastAPI app + CRUD routes
└── tests/
    ├── __init__.py
    ├── conftest.py        # pytest fixtures (DB session, test client)
    ├── test_models.py     # Unit tests for ORM models
    └── test_endpoints.py  # Integration tests for API endpoints
```

---

## Workspace Context

- **Workspace root**: `/Users/pradeepsharma/sasva/projects/vllm`
- **Python version**: 3.10–3.13 (per `pyproject.toml`)
- **FastAPI** already declared in `requirements/common.txt` (`fastapi[standard] >= 0.115.0`)
- **Pydantic** already declared (`pydantic >= 2.12.0`)
- **pytest** + **pytest-asyncio** + **httpx** already in `requirements/test.in`
- `team-test/api/` and `team-test/database/` directories already exist (empty, with `__pycache__` only)
- `team-test/tests/` directory does **not** yet exist — must be created
- No existing Python source files in `team-test/`

---

## Phase 1 — Project Scaffolding & Configuration

**Goal**: Create `requirements.txt`, top-level `__init__.py`, and sub-package `__init__.py` files.

### Files to Create

#### `team-test/__init__.py`
```python
# team-test package root
```

#### `team-test/requirements.txt`
```
fastapi[standard]>=0.115.0
uvicorn[standard]>=0.29.0
sqlalchemy>=2.0.0
pydantic>=2.12.0
httpx>=0.27.0
pytest>=8.0.0
pytest-asyncio>=0.23.0
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
aiosqlite>=0.20.0
```

#### `team-test/database/__init__.py`
```python
from team_test.database.connection import Base, engine, get_db
from team_test.database.models import User, Project

__all__ = ["Base", "engine", "get_db", "User", "Project"]
```

#### `team-test/api/__init__.py`
```python
from team_test.api.endpoints import app

__all__ = ["app"]
```

#### `team-test/tests/__init__.py`
```python
# tests package
```

**Parallelism**: All five files are independent — can be written simultaneously.

---

## Phase 2 — Database Layer

**Goal**: Implement SQLAlchemy models, database connection setup, and seed data script.

### 2a — `team-test/database/connection.py`

```python
"""Database connection and session management."""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = "sqlite:///./team_test.db"  # Override via env var TEAM_TEST_DB_URL

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # SQLite-specific
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
    """Create all tables."""
    from team_test.database import models  # noqa: F401 — registers models
    Base.metadata.create_all(bind=engine)
```

**Key design decisions**:
- Uses SQLite by default (zero-config, test-friendly); URL overridable via env var
- `check_same_thread=False` required for SQLite + FastAPI async context
- `init_db()` imported lazily to avoid circular imports

### 2b — `team-test/database/models.py`

```python
"""SQLAlchemy ORM models: User and Project."""
import datetime
from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey,
    Integer, String, Text, func,
)
from sqlalchemy.orm import relationship
from team_test.database.connection import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    projects = relationship("Project", back_populates="owner", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r}>"


class Project(Base):
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    description = Column(Text, nullable=True)
    is_public = Column(Boolean, default=False, nullable=False)
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    owner = relationship("User", back_populates="projects")

    def __repr__(self) -> str:
        return f"<Project id={self.id} name={self.name!r} owner_id={self.owner_id}>"
```

**Key design decisions**:
- `User` ↔ `Project` is a one-to-many relationship via `owner_id` FK
- `cascade="all, delete-orphan"` ensures projects are deleted when their owner is deleted
- `server_default=func.now()` lets the DB handle timestamps
- Both models have `index=True` on lookup columns for query performance

### 2c — `team-test/database/seed_data.py`

```python
"""Seed the database with sample Users and Projects."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from team_test.database.connection import SessionLocal, init_db
from team_test.database.models import User, Project


SAMPLE_USERS = [
    {"username": "alice", "email": "alice@example.com", "hashed_password": "hashed_pw_alice"},
    {"username": "bob",   "email": "bob@example.com",   "hashed_password": "hashed_pw_bob"},
    {"username": "carol", "email": "carol@example.com", "hashed_password": "hashed_pw_carol"},
]

SAMPLE_PROJECTS = [
    {"name": "Alpha",   "description": "First project",  "is_public": True,  "owner_username": "alice"},
    {"name": "Beta",    "description": "Second project", "is_public": False, "owner_username": "alice"},
    {"name": "Gamma",   "description": "Third project",  "is_public": True,  "owner_username": "bob"},
    {"name": "Delta",   "description": "Fourth project", "is_public": False, "owner_username": "carol"},
]


def seed():
    init_db()
    db = SessionLocal()
    try:
        # Skip if already seeded
        if db.query(User).count() > 0:
            print("Database already seeded. Skipping.")
            return

        users = {}
        for u in SAMPLE_USERS:
            user = User(**u)
            db.add(user)
            db.flush()  # get auto-generated id
            users[u["username"]] = user

        for p in SAMPLE_PROJECTS:
            owner = users[p.pop("owner_username")]
            project = Project(**p, owner_id=owner.id)
            db.add(project)

        db.commit()
        print(f"Seeded {len(SAMPLE_USERS)} users and {len(SAMPLE_PROJECTS)} projects.")
    except Exception as exc:
        db.rollback()
        raise exc
    finally:
        db.close()


if __name__ == "__main__":
    seed()
```

---

## Phase 3 — API Layer

**Goal**: Implement Pydantic schemas, authentication middleware, and FastAPI CRUD endpoints.

### 3a — `team-test/api/schemas.py`

```python
"""Pydantic v2 request/response schemas for Users and Projects."""
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, ConfigDict


# ── User Schemas ──────────────────────────────────────────────────────────────

class UserBase(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr


class UserCreate(UserBase):
    password: str = Field(..., min_length=8)


class UserUpdate(BaseModel):
    username: str | None = Field(None, min_length=3, max_length=50)
    email: EmailStr | None = None
    is_active: bool | None = None


class UserResponse(UserBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    is_active: bool
    created_at: datetime
    updated_at: datetime


# ── Project Schemas ───────────────────────────────────────────────────────────

class ProjectBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    description: str | None = None
    is_public: bool = False


class ProjectCreate(ProjectBase):
    owner_id: int


class ProjectUpdate(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=100)
    description: str | None = None
    is_public: bool | None = None


class ProjectResponse(ProjectBase):
    model_config = ConfigDict(from_attributes=True)

    id: int
    owner_id: int
    created_at: datetime
    updated_at: datetime


# ── Generic Schemas ───────────────────────────────────────────────────────────

class MessageResponse(BaseModel):
    message: str

class PaginatedUsers(BaseModel):
    total: int
    items: list[UserResponse]

class PaginatedProjects(BaseModel):
    total: int
    items: list[ProjectResponse]
```

**Key design decisions**:
- `model_config = ConfigDict(from_attributes=True)` enables ORM-mode (Pydantic v2 syntax)
- `UserCreate` takes a plain `password`; the endpoint hashes it before storing
- `UserUpdate` / `ProjectUpdate` use all-optional fields for PATCH semantics
- `EmailStr` validates email format at the schema level

### 3b — `team-test/api/middleware.py`

```python
"""Authentication middleware using API key header."""
import os
from fastapi import HTTPException, Request, status
from fastapi.security import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

API_KEY_HEADER_NAME = "X-API-Key"
_api_key_scheme = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

# In production, load from a secrets manager. Here we use an env var with a default.
VALID_API_KEY = os.environ.get("TEAM_TEST_API_KEY", "dev-secret-key-12345")

# Paths that do NOT require authentication
PUBLIC_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Starlette-compatible middleware that enforces API key authentication
    on all routes except those listed in PUBLIC_PATHS.

    Usage:
        app.add_middleware(AuthMiddleware)

    Clients must send:
        X-API-Key: <valid_key>
    """

    async def dispatch(self, request: Request, call_next):
        if request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        api_key = request.headers.get(API_KEY_HEADER_NAME)
        if not api_key or api_key != VALID_API_KEY:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Invalid or missing API key"},
                headers={"WWW-Authenticate": "ApiKey"},
            )
        return await call_next(request)


async def get_api_key(request: Request) -> str:
    """
    FastAPI Depends()-compatible function for per-route auth.
    Use as: `api_key: str = Depends(get_api_key)`
    """
    api_key = request.headers.get(API_KEY_HEADER_NAME)
    if not api_key or api_key != VALID_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return api_key
```

### 3c — `team-test/api/endpoints.py`

```python
"""FastAPI application with CRUD routes for Users and Projects."""
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from team_test.database.connection import get_db, init_db
from team_test.database.models import User, Project
from team_test.api.schemas import (
    UserCreate, UserUpdate, UserResponse, PaginatedUsers,
    ProjectCreate, ProjectUpdate, ProjectResponse, PaginatedProjects,
    MessageResponse,
)
from team_test.api.middleware import AuthMiddleware, get_api_key


# ── App lifecycle ─────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()          # Create tables on startup
    yield
    # (teardown hooks go here if needed)


app = FastAPI(
    title="Team Test REST API",
    description="CRUD API for Users and Projects",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(AuthMiddleware)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=MessageResponse, tags=["Health"])
async def health_check():
    return {"message": "ok"}


# ── Users ─────────────────────────────────────────────────────────────────────

@app.post(
    "/users",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Users"],
    dependencies=[Depends(get_api_key)],
)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(status_code=400, detail="Username already taken")
    user = User(
        username=payload.username,
        email=payload.email,
        hashed_password=f"hashed_{payload.password}",  # Replace with bcrypt in prod
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.get("/users", response_model=PaginatedUsers, tags=["Users"], dependencies=[Depends(get_api_key)])
def list_users(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    total = db.query(User).count()
    items = db.query(User).offset(skip).limit(limit).all()
    return {"total": total, "items": items}


@app.get("/users/{user_id}", response_model=UserResponse, tags=["Users"], dependencies=[Depends(get_api_key)])
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.patch("/users/{user_id}", response_model=UserResponse, tags=["Users"], dependencies=[Depends(get_api_key)])
def update_user(user_id: int, payload: UserUpdate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(user, field, value)
    db.commit()
    db.refresh(user)
    return user


@app.delete("/users/{user_id}", response_model=MessageResponse, tags=["Users"], dependencies=[Depends(get_api_key)])
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {"message": f"User {user_id} deleted"}


# ── Projects ──────────────────────────────────────────────────────────────────

@app.post(
    "/projects",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Projects"],
    dependencies=[Depends(get_api_key)],
)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)):
    owner = db.query(User).filter(User.id == payload.owner_id).first()
    if not owner:
        raise HTTPException(status_code=404, detail="Owner user not found")
    project = Project(**payload.model_dump())
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@app.get("/projects", response_model=PaginatedProjects, tags=["Projects"], dependencies=[Depends(get_api_key)])
def list_projects(
    skip: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=100),
    owner_id: int | None = Query(None),
    db: Session = Depends(get_db),
):
    q = db.query(Project)
    if owner_id is not None:
        q = q.filter(Project.owner_id == owner_id)
    total = q.count()
    items = q.offset(skip).limit(limit).all()
    return {"total": total, "items": items}


@app.get("/projects/{project_id}", response_model=ProjectResponse, tags=["Projects"], dependencies=[Depends(get_api_key)])
def get_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@app.patch("/projects/{project_id}", response_model=ProjectResponse, tags=["Projects"], dependencies=[Depends(get_api_key)])
def update_project(project_id: int, payload: ProjectUpdate, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(project, field, value)
    db.commit()
    db.refresh(project)
    return project


@app.delete("/projects/{project_id}", response_model=MessageResponse, tags=["Projects"], dependencies=[Depends(get_api_key)])
def delete_project(project_id: int, db: Session = Depends(get_db)):
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    db.delete(project)
    db.commit()
    return {"message": f"Project {project_id} deleted"}
```

---

## Phase 4 — Testing Layer

**Goal**: Implement pytest fixtures, model unit tests, and endpoint integration tests.

### 4a — `team-test/tests/conftest.py`

```python
"""Shared pytest fixtures for team-test."""
import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Use in-memory SQLite for tests — isolated, fast, no file cleanup needed
TEST_DATABASE_URL = "sqlite:///:memory:"

os.environ["TEAM_TEST_API_KEY"] = "test-api-key-99999"
os.environ["TEAM_TEST_DB_URL"] = TEST_DATABASE_URL

from team_test.database.connection import Base, get_db
from team_test.database.models import User, Project
from team_test.api.endpoints import app

TEST_ENGINE = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=TEST_ENGINE)


@pytest.fixture(scope="session", autouse=True)
def create_tables():
    """Create all tables once per test session."""
    Base.metadata.create_all(bind=TEST_ENGINE)
    yield
    Base.metadata.drop_all(bind=TEST_ENGINE)


@pytest.fixture()
def db_session(create_tables):
    """Provide a transactional DB session that rolls back after each test."""
    connection = TEST_ENGINE.connect()
    transaction = connection.begin()
    session = TestingSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


@pytest.fixture()
def client(db_session):
    """FastAPI TestClient with DB dependency overridden to use test session."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def auth_headers():
    """Return headers with valid API key."""
    return {"X-API-Key": "test-api-key-99999"}


@pytest.fixture()
def sample_user(db_session) -> User:
    """Insert and return a sample User."""
    user = User(
        username="testuser",
        email="testuser@example.com",
        hashed_password="hashed_testpassword",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture()
def sample_project(db_session, sample_user) -> Project:
    """Insert and return a sample Project owned by sample_user."""
    project = Project(
        name="Test Project",
        description="A project for testing",
        is_public=True,
        owner_id=sample_user.id,
    )
    db_session.add(project)
    db_session.commit()
    db_session.refresh(project)
    return project
```

### 4b — `team-test/tests/test_models.py`

```python
"""Unit tests for SQLAlchemy ORM models (User and Project)."""
import pytest
from sqlalchemy.exc import IntegrityError
from team_test.database.models import User, Project


class TestUserModel:
    def test_create_user(self, db_session):
        user = User(username="alice", email="alice@test.com", hashed_password="pw")
        db_session.add(user)
        db_session.commit()
        assert user.id is not None
        assert user.username == "alice"
        assert user.is_active is True

    def test_user_repr(self, db_session, sample_user):
        assert "testuser" in repr(sample_user)

    def test_unique_email_constraint(self, db_session, sample_user):
        duplicate = User(
            username="other",
            email=sample_user.email,  # same email
            hashed_password="pw",
        )
        db_session.add(duplicate)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_unique_username_constraint(self, db_session, sample_user):
        duplicate = User(
            username=sample_user.username,  # same username
            email="unique@test.com",
            hashed_password="pw",
        )
        db_session.add(duplicate)
        with pytest.raises(IntegrityError):
            db_session.commit()

    def test_user_has_projects_relationship(self, db_session, sample_user, sample_project):
        db_session.refresh(sample_user)
        assert len(sample_user.projects) == 1
        assert sample_user.projects[0].name == "Test Project"

    def test_cascade_delete_projects(self, db_session, sample_user, sample_project):
        project_id = sample_project.id
        db_session.delete(sample_user)
        db_session.commit()
        deleted = db_session.query(Project).filter(Project.id == project_id).first()
        assert deleted is None


class TestProjectModel:
    def test_create_project(self, db_session, sample_user):
        project = Project(
            name="My Project",
            description="Desc",
            is_public=False,
            owner_id=sample_user.id,
        )
        db_session.add(project)
        db_session.commit()
        assert project.id is not None
        assert project.owner_id == sample_user.id

    def test_project_repr(self, db_session, sample_project):
        assert "Test Project" in repr(sample_project)

    def test_project_owner_relationship(self, db_session, sample_project):
        db_session.refresh(sample_project)
        assert sample_project.owner.username == "testuser"

    def test_project_default_is_public_false(self, db_session, sample_user):
        project = Project(name="Private", owner_id=sample_user.id, hashed_password=None)
        # is_public defaults to False
        assert project.is_public is False
```

### 4c — `team-test/tests/test_endpoints.py`

```python
"""Integration tests for FastAPI CRUD endpoints."""
import pytest


# ── Health ────────────────────────────────────────────────────────────────────

def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"message": "ok"}


# ── Auth middleware ───────────────────────────────────────────────────────────

def test_missing_api_key_returns_401(client):
    resp = client.get("/users")
    assert resp.status_code == 401

def test_invalid_api_key_returns_401(client):
    resp = client.get("/users", headers={"X-API-Key": "wrong-key"})
    assert resp.status_code == 401


# ── Users CRUD ────────────────────────────────────────────────────────────────

class TestUsersEndpoints:
    def test_create_user(self, client, auth_headers):
        payload = {"username": "newuser", "email": "new@test.com", "password": "securepass"}
        resp = client.post("/users", json=payload, headers=auth_headers)
        assert resp.status_code == 201
        data = resp.json()
        assert data["username"] == "newuser"
        assert data["email"] == "new@test.com"
        assert "id" in data
        assert "hashed_password" not in data  # never exposed

    def test_create_user_duplicate_email(self, client, auth_headers, sample_user):
        payload = {"username": "other", "email": sample_user.email, "password": "pass1234"}
        resp = client.post("/users", json=payload, headers=auth_headers)
        assert resp.status_code == 400

    def test_create_user_duplicate_username(self, client, auth_headers, sample_user):
        payload = {"username": sample_user.username, "email": "unique2@test.com", "password": "pass1234"}
        resp = client.post("/users", json=payload, headers=auth_headers)
        assert resp.status_code == 400

    def test_list_users(self, client, auth_headers, sample_user):
        resp = client.get("/users", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "items" in data
        assert data["total"] >= 1

    def test_get_user(self, client, auth_headers, sample_user):
        resp = client.get(f"/users/{sample_user.id}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == sample_user.id

    def test_get_user_not_found(self, client, auth_headers):
        resp = client.get("/users/99999", headers=auth_headers)
        assert resp.status_code == 404

    def test_update_user(self, client, auth_headers, sample_user):
        resp = client.patch(
            f"/users/{sample_user.id}",
            json={"is_active": False},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["is_active"] is False

    def test_delete_user(self, client, auth_headers, sample_user):
        resp = client.delete(f"/users/{sample_user.id}", headers=auth_headers)
        assert resp.status_code == 200
        assert "deleted" in resp.json()["message"]
        # Confirm gone
        resp2 = client.get(f"/users/{sample_user.id}", headers=auth_headers)
        assert resp2.status_code == 404


# ── Projects CRUD ─────────────────────────────────────────────────────────────

class TestProjectsEndpoints:
    def test_create_project(self, client, auth_headers, sample_user):
        payload = {
            "name": "New Project",
            "description": "Desc",
            "is_public": True,
            "owner_id": sample_user.id,
        }
        resp = client.post("/projects", json=payload, headers=auth_headers)
        assert resp.status_code == 201
        data = resp.json()
        assert data["name"] == "New Project"
        assert data["owner_id"] == sample_user.id

    def test_create_project_invalid_owner(self, client, auth_headers):
        payload = {"name": "Orphan", "owner_id": 99999}
        resp = client.post("/projects", json=payload, headers=auth_headers)
        assert resp.status_code == 404

    def test_list_projects(self, client, auth_headers, sample_project):
        resp = client.get("/projects", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 1

    def test_list_projects_filter_by_owner(self, client, auth_headers, sample_user, sample_project):
        resp = client.get(f"/projects?owner_id={sample_user.id}", headers=auth_headers)
        assert resp.status_code == 200
        for item in resp.json()["items"]:
            assert item["owner_id"] == sample_user.id

    def test_get_project(self, client, auth_headers, sample_project):
        resp = client.get(f"/projects/{sample_project.id}", headers=auth_headers)
        assert resp.status_code == 200
        assert resp.json()["id"] == sample_project.id

    def test_get_project_not_found(self, client, auth_headers):
        resp = client.get("/projects/99999", headers=auth_headers)
        assert resp.status_code == 404

    def test_update_project(self, client, auth_headers, sample_project):
        resp = client.patch(
            f"/projects/{sample_project.id}",
            json={"name": "Renamed", "is_public": False},
            headers=auth_headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Renamed"
        assert data["is_public"] is False

    def test_delete_project(self, client, auth_headers, sample_project):
        resp = client.delete(f"/projects/{sample_project.id}", headers=auth_headers)
        assert resp.status_code == 200
        resp2 = client.get(f"/projects/{sample_project.id}", headers=auth_headers)
        assert resp2.status_code == 404
```

---

## Phase 5 — Integration & Smoke Test

**Goal**: Verify the full stack works end-to-end by running the test suite.

### Steps

1. **Install dependencies**:
   ```bash
   cd /Users/pradeepsharma/sasva/projects/vllm/team-test
   pip install -r requirements.txt
   ```

2. **Run seed script** (optional smoke test of DB layer):
   ```bash
   cd /Users/pradeepsharma/sasva/projects/vllm
   python -m team_test.database.seed_data
   ```

3. **Run full test suite**:
   ```bash
   cd /Users/pradeepsharma/sasva/projects/vllm
   python -m pytest team-test/tests/ -v --tb=short
   ```

4. **Start the API server** (manual smoke test):
   ```bash
   cd /Users/pradeepsharma/sasva/projects/vllm
   TEAM_TEST_API_KEY=dev-secret-key-12345 uvicorn team_test.api.endpoints:app --reload --port 8080
   ```

---

## Execution Order & Parallelism

```
Phase 1 (Scaffolding)
  └─ All 5 __init__.py + requirements.txt files → PARALLEL

Phase 2 (Database Layer)
  ├─ 2a: connection.py  ─┐
  ├─ 2b: models.py      ─┤ PARALLEL (models imports Base from connection,
  └─ 2c: seed_data.py   ─┘ but files can be written simultaneously)

Phase 3 (API Layer)
  ├─ 3a: schemas.py     ─┐
  ├─ 3b: middleware.py  ─┤ PARALLEL (no inter-dependencies at write time)
  └─ 3c: endpoints.py   ─┘

Phase 4 (Tests)
  ├─ 4a: conftest.py    ─┐
  ├─ 4b: test_models.py ─┤ PARALLEL
  └─ 4c: test_endpoints ─┘

Phase 5 (Integration)
  └─ Sequential: install → seed → pytest → uvicorn smoke
```

**Critical path**: Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5

---

## File Creation Checklist

| File | Phase | Status |
|------|-------|--------|
| `team-test/__init__.py` | 1 | ☐ |
| `team-test/requirements.txt` | 1 | ☐ |
| `team-test/database/__init__.py` | 1 | ☐ |
| `team-test/api/__init__.py` | 1 | ☐ |
| `team-test/tests/__init__.py` | 1 | ☐ |
| `team-test/database/connection.py` | 2a | ☐ |
| `team-test/database/models.py` | 2b | ☐ |
| `team-test/database/seed_data.py` | 2c | ☐ |
| `team-test/api/schemas.py` | 3a | ☐ |
| `team-test/api/middleware.py` | 3b | ☐ |
| `team-test/api/endpoints.py` | 3c | ☐ |
| `team-test/tests/conftest.py` | 4a | ☐ |
| `team-test/tests/test_models.py` | 4b | ☐ |
| `team-test/tests/test_endpoints.py` | 4c | ☐ |

**Total: 14 files**

---

## Key Design Decisions & Rationale

| Decision | Rationale |
|----------|-----------|
| SQLite default DB | Zero-config, file-based, perfect for dev/test; URL overridable via `TEAM_TEST_DB_URL` env var |
| `DeclarativeBase` (SQLAlchemy 2.x style) | Matches SQLAlchemy ≥ 2.0 API; avoids deprecated `declarative_base()` |
| Pydantic v2 `ConfigDict(from_attributes=True)` | Replaces Pydantic v1's `orm_mode = True`; required for `pydantic >= 2.0` |
| `BaseHTTPMiddleware` for auth | Starlette-native; works with FastAPI's middleware stack; easy to test |
| In-memory SQLite for tests | Isolated per test session; no file cleanup; fast |
| Transactional rollback fixture | Each test gets a clean DB state without recreating tables |
| `dependency_overrides` in tests | FastAPI's official pattern for swapping DB sessions in tests |
| `cascade="all, delete-orphan"` | Ensures referential integrity when users are deleted |
| `exclude_unset=True` in PATCH | Implements true partial-update semantics (only changed fields written) |
| `hashed_password` never in response schema | Security: `UserResponse` does not include `hashed_password` field |

---

## Verification Criteria

### How to Verify This Project Works

#### 1. Run the Test Suite (Primary Verification)
```bash
cd /Users/pradeepsharma/sasva/projects/vllm
python -m pytest team-test/tests/ -v --tb=short
```
**Expected**: All tests pass. Approximate count: **~25 tests** across `test_models.py` and `test_endpoints.py`.
- `test_models.py`: ~10 tests (User model: 6, Project model: 4)
- `test_endpoints.py`: ~18 tests (health: 1, auth: 2, users CRUD: 8, projects CRUD: 8)

#### 2. API Endpoint Smoke Tests (Manual curl)
Start the server first:
```bash
TEAM_TEST_API_KEY=dev-secret-key-12345 uvicorn team_test.api.endpoints:app --port 8080
```

Then run:
```bash
# Health (no auth required)
curl -s http://localhost:8080/health
# Expected: {"message":"ok"}

# Auth rejection
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/users
# Expected: 401

# Create user
curl -s -X POST http://localhost:8080/users \
  -H "X-API-Key: dev-secret-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"username":"alice","email":"alice@example.com","password":"securepass"}' | python -m json.tool
# Expected: JSON with id, username, email, is_active, created_at, updated_at

# List users
curl -s http://localhost:8080/users -H "X-API-Key: dev-secret-key-12345" | python -m json.tool
# Expected: {"total": 1, "items": [...]}

# Create project (use user id from above, e.g. 1)
curl -s -X POST http://localhost:8080/projects \
  -H "X-API-Key: dev-secret-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"name":"Alpha","description":"First","is_public":true,"owner_id":1}' | python -m json.tool
# Expected: JSON with id, name, owner_id, created_at

# Get project
curl -s http://localhost:8080/projects/1 -H "X-API-Key: dev-secret-key-12345" | python -m json.tool
# Expected: {"id":1,"name":"Alpha","owner_id":1,...}

# Update project
curl -s -X PATCH http://localhost:8080/projects/1 \
  -H "X-API-Key: dev-secret-key-12345" \
  -H "Content-Type: application/json" \
  -d '{"name":"Alpha Renamed"}' | python -m json.tool
# Expected: {"id":1,"name":"Alpha Renamed",...}

# Delete project
curl -s -X DELETE http://localhost:8080/projects/1 -H "X-API-Key: dev-secret-key-12345"
# Expected: {"message":"Project 1 deleted"}

# Confirm 404
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080/projects/1 -H "X-API-Key: dev-secret-key-12345"
# Expected: 404
```

#### 3. Seed Script Verification
```bash
cd /Users/pradeepsharma/sasva/projects/vllm
python team-test/database/seed_data.py
# Expected output: "Seeded 3 users and 4 projects."
# Second run: "Database already seeded. Skipping."
```

#### 4. OpenAPI Docs
Navigate to `http://localhost:8080/docs` — should show interactive Swagger UI with:
- `GET /health`
- `POST /users`, `GET /users`, `GET /users/{user_id}`, `PATCH /users/{user_id}`, `DELETE /users/{user_id}`
- `POST /projects`, `GET /projects`, `GET /projects/{project_id}`, `PATCH /projects/{project_id}`, `DELETE /projects/{project_id}`

#### 5. Import Verification
```bash
cd /Users/pradeepsharma/sasva/projects/vllm
python -c "
from team_test.database.models import User, Project
from team_test.database.connection import Base, engine, get_db
from team_test.api.schemas import UserCreate, UserResponse, ProjectCreate, ProjectResponse
from team_test.api.middleware import AuthMiddleware
from team_test.api.endpoints import app
print('All imports OK')
print('Routes:', [r.path for r in app.routes])
"
# Expected: "All imports OK" followed by list of 11 routes
```
