"""FastAPI application with CRUD endpoints for Users and Projects.

Endpoints
---------
Health
    GET  /health

Users
    POST   /users/
    GET    /users/
    GET    /users/{user_id}
    PATCH  /users/{user_id}
    DELETE /users/{user_id}

Projects
    POST   /projects/
    GET    /projects/
    GET    /projects/{project_id}
    PATCH  /projects/{project_id}
    DELETE /projects/{project_id}

Authentication
--------------
All endpoints except ``GET /health`` and the auto-generated docs paths
(``/docs``, ``/redoc``, ``/openapi.json``) require an
``Authorization: Bearer <token>`` header.  The token is validated by
:class:`team_test.api.middleware.BearerAuthMiddleware`.

Password hashing
----------------
``passlib`` with the ``bcrypt`` scheme is used to hash passwords before
storing them and to verify passwords on login-style checks.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, status
from passlib.context import CryptContext
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

# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _hash_password(plain: str) -> str:
    """Return the bcrypt hash of *plain*."""
    return _pwd_context.hash(plain)


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="team-test REST API",
    description=(
        "A simple REST API for managing **Users** and **Projects**.\n\n"
        "All endpoints (except `/health` and the docs paths) require an "
        "`Authorization: Bearer <token>` header."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Register the Bearer-token authentication middleware
app.add_middleware(BearerAuthMiddleware)


# ---------------------------------------------------------------------------
# Startup event — create tables if they don't exist yet
# ---------------------------------------------------------------------------


@app.on_event("startup")
def on_startup() -> None:
    """Initialise the database schema on application start-up."""
    init_db()


# ---------------------------------------------------------------------------
# Dependency aliases
# ---------------------------------------------------------------------------

DbSession = Annotated[Session, Depends(get_db)]


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    summary="Health check",
    response_model=MessageResponse,
    tags=["Health"],
)
def health_check() -> MessageResponse:
    """Return a simple liveness response.  No authentication required."""
    return MessageResponse(message="OK")


# ---------------------------------------------------------------------------
# User endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/users/",
    summary="Create a new user",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Users"],
)
def create_user(body: UserCreate, db: DbSession) -> User:
    """Create a new user account.

    - ``username`` and ``email`` must be unique.
    - The plain-text ``password`` is hashed with bcrypt before storage.
    """
    # Check for duplicate username
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{body.username}' is already taken.",
        )
    # Check for duplicate email
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Email '{body.email}' is already registered.",
        )

    user = User(
        username=body.username,
        email=body.email,
        full_name=body.full_name,
        is_active=body.is_active,
        hashed_password=_hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@app.get(
    "/users/",
    summary="List all users",
    response_model=list[UserResponse],
    tags=["Users"],
)
def list_users(
    db: DbSession,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip.")] = 0,
    limit: Annotated[
        int, Query(ge=1, le=200, description="Maximum number of records to return.")
    ] = 100,
) -> list[User]:
    """Return a paginated list of all users."""
    return db.query(User).offset(skip).limit(limit).all()


@app.get(
    "/users/{user_id}",
    summary="Get a user by ID",
    response_model=UserResponse,
    tags=["Users"],
)
def get_user(user_id: int, db: DbSession) -> User:
    """Return a single user by their primary key.  Raises 404 if not found."""
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id={user_id} not found.",
        )
    return user


@app.patch(
    "/users/{user_id}",
    summary="Partially update a user",
    response_model=UserResponse,
    tags=["Users"],
)
def update_user(user_id: int, body: UserUpdate, db: DbSession) -> User:
    """Partially update a user.

    Only the fields present in the request body are updated; omitted fields
    are left unchanged.  Raises 404 if the user does not exist.
    """
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id={user_id} not found.",
        )

    update_data = body.model_dump(exclude_unset=True)

    # Handle password separately — hash it before storing
    if "password" in update_data:
        plain_password = update_data.pop("password")
        if plain_password is not None:
            user.hashed_password = _hash_password(plain_password)

    for field, value in update_data.items():
        setattr(user, field, value)

    db.commit()
    db.refresh(user)
    return user


@app.delete(
    "/users/{user_id}",
    summary="Delete a user",
    response_model=MessageResponse,
    tags=["Users"],
)
def delete_user(user_id: int, db: DbSession) -> MessageResponse:
    """Delete a user and all their associated projects (cascade).

    Raises 404 if the user does not exist.
    """
    user = db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id={user_id} not found.",
        )
    db.delete(user)
    db.commit()
    return MessageResponse(message=f"User with id={user_id} deleted successfully.")


# ---------------------------------------------------------------------------
# Project endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/projects/",
    summary="Create a new project",
    response_model=ProjectResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Projects"],
)
def create_project(body: ProjectCreate, db: DbSession) -> Project:
    """Create a new project.

    - ``owner_id`` must reference an existing user; returns 404 otherwise.
    """
    owner = db.get(User, body.owner_id)
    if owner is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User with id={body.owner_id} not found.",
        )

    project = Project(
        name=body.name,
        description=body.description,
        is_active=body.is_active,
        owner_id=body.owner_id,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@app.get(
    "/projects/",
    summary="List all projects",
    response_model=list[ProjectResponse],
    tags=["Projects"],
)
def list_projects(
    db: DbSession,
    skip: Annotated[int, Query(ge=0, description="Number of records to skip.")] = 0,
    limit: Annotated[
        int, Query(ge=1, le=200, description="Maximum number of records to return.")
    ] = 100,
    owner_id: Annotated[
        int | None,
        Query(description="Filter projects by owner user ID."),
    ] = None,
) -> list[Project]:
    """Return a paginated list of projects, optionally filtered by ``owner_id``."""
    query = db.query(Project)
    if owner_id is not None:
        query = query.filter(Project.owner_id == owner_id)
    return query.offset(skip).limit(limit).all()


@app.get(
    "/projects/{project_id}",
    summary="Get a project by ID",
    response_model=ProjectResponse,
    tags=["Projects"],
)
def get_project(project_id: int, db: DbSession) -> Project:
    """Return a single project by its primary key.  Raises 404 if not found."""
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id={project_id} not found.",
        )
    return project


@app.patch(
    "/projects/{project_id}",
    summary="Partially update a project",
    response_model=ProjectResponse,
    tags=["Projects"],
)
def update_project(project_id: int, body: ProjectUpdate, db: DbSession) -> Project:
    """Partially update a project.

    Only the fields present in the request body are updated; omitted fields
    are left unchanged.  Raises 404 if the project does not exist.
    """
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id={project_id} not found.",
        )

    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(project, field, value)

    db.commit()
    db.refresh(project)
    return project


@app.delete(
    "/projects/{project_id}",
    summary="Delete a project",
    response_model=MessageResponse,
    tags=["Projects"],
)
def delete_project(project_id: int, db: DbSession) -> MessageResponse:
    """Delete a project by its primary key.  Raises 404 if not found."""
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Project with id={project_id} not found.",
        )
    db.delete(project)
    db.commit()
    return MessageResponse(message=f"Project with id={project_id} deleted successfully.")
