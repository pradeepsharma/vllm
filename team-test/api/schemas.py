"""Pydantic v2 request/response schemas for Users and Projects.

These schemas are used by the FastAPI endpoints to validate incoming request
bodies and serialise ORM model instances into JSON responses.

Design notes
------------
- ``UserCreate`` accepts a plain-text ``password``; endpoints are responsible
  for hashing it before persisting to ``User.hashed_password``.
- ``UserResponse`` deliberately omits ``hashed_password`` so it is never
  exposed over the wire.
- ``model_config = ConfigDict(from_attributes=True)`` on the *Response*
  schemas enables Pydantic to read values directly from SQLAlchemy ORM
  instances (replaces the old ``orm_mode = True`` in Pydantic v1).
- ``EmailStr`` is provided by ``pydantic[email]`` / ``email-validator``,
  which is bundled with ``fastapi[standard]`` already listed in
  ``requirements.txt``.
"""

from __future__ import annotations

import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, Field


# ---------------------------------------------------------------------------
# User schemas
# ---------------------------------------------------------------------------


class UserBase(BaseModel):
    """Fields shared by all user-facing schemas (create, update, response)."""

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="Unique username (3–50 characters).",
        examples=["alice"],
    )
    email: EmailStr = Field(
        ...,
        description="Valid e-mail address; must be unique across all users.",
        examples=["alice@example.com"],
    )
    full_name: str | None = Field(
        None,
        max_length=100,
        description="Optional display name (up to 100 characters).",
        examples=["Alice Wonderland"],
    )
    is_active: bool = Field(
        True,
        description="Whether the user account is active.",
    )


class UserCreate(UserBase):
    """Request body for ``POST /users/`` — includes a plain-text password.

    The endpoint is responsible for hashing the password before storing it
    in ``User.hashed_password``.
    """

    password: str = Field(
        ...,
        min_length=6,
        description="Plain-text password (minimum 6 characters). Never stored or returned in plain text.",
        examples=["s3cr3t!"],
    )


class UserUpdate(BaseModel):
    """Request body for ``PATCH /users/{user_id}`` — all fields are optional.

    Only the fields that are explicitly provided will be updated; omitted
    fields are left unchanged.
    """

    full_name: str | None = Field(
        None,
        max_length=100,
        description="New display name, or ``null`` to clear it.",
        examples=["Alice Smith"],
    )
    is_active: bool | None = Field(
        None,
        description="Set to ``false`` to deactivate the account.",
    )
    password: str | None = Field(
        None,
        min_length=6,
        description="New plain-text password (minimum 6 characters). Omit to keep the current password.",
        examples=["n3wP@ss!"],
    )


class UserResponse(UserBase):
    """Response schema returned by all user endpoints.

    Inherits ``username``, ``email``, ``full_name``, and ``is_active`` from
    ``UserBase``.  Adds server-assigned fields: ``id``, ``created_at``, and
    ``updated_at``.  ``hashed_password`` is intentionally excluded.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Auto-assigned primary key.", examples=[1])
    created_at: datetime.datetime = Field(
        ..., description="UTC timestamp when the user was created."
    )
    updated_at: datetime.datetime = Field(
        ..., description="UTC timestamp of the most recent update."
    )


# ---------------------------------------------------------------------------
# Project schemas
# ---------------------------------------------------------------------------


class ProjectBase(BaseModel):
    """Fields shared by all project-facing schemas (create, update, response)."""

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Project name (1–100 characters).",
        examples=["My Awesome Project"],
    )
    description: str | None = Field(
        None,
        description="Optional free-text description of the project.",
        examples=["A project that does amazing things."],
    )
    is_active: bool = Field(
        True,
        description="Whether the project is currently active.",
    )


class ProjectCreate(ProjectBase):
    """Request body for ``POST /projects/``.

    ``owner_id`` must reference an existing ``User.id``; the endpoint returns
    HTTP 404 if the referenced user does not exist.
    """

    owner_id: int = Field(
        ...,
        description="Primary key of the user who owns this project.",
        examples=[1],
    )


class ProjectUpdate(BaseModel):
    """Request body for ``PATCH /projects/{project_id}`` — all fields optional.

    Only the fields that are explicitly provided will be updated; omitted
    fields are left unchanged.
    """

    name: str | None = Field(
        None,
        min_length=1,
        max_length=100,
        description="New project name.",
        examples=["Renamed Project"],
    )
    description: str | None = Field(
        None,
        description="New description, or ``null`` to clear it.",
        examples=["Updated description."],
    )
    is_active: bool | None = Field(
        None,
        description="Set to ``false`` to deactivate the project.",
    )


class ProjectResponse(ProjectBase):
    """Response schema returned by all project endpoints.

    Inherits ``name``, ``description``, and ``is_active`` from
    ``ProjectBase``.  Adds server-assigned fields: ``id``, ``owner_id``,
    ``created_at``, and ``updated_at``.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., description="Auto-assigned primary key.", examples=[1])
    owner_id: int = Field(
        ..., description="Primary key of the owning user.", examples=[1]
    )
    created_at: datetime.datetime = Field(
        ..., description="UTC timestamp when the project was created."
    )
    updated_at: datetime.datetime = Field(
        ..., description="UTC timestamp of the most recent update."
    )


# ---------------------------------------------------------------------------
# Generic response
# ---------------------------------------------------------------------------


class MessageResponse(BaseModel):
    """Generic single-message response (e.g. for DELETE confirmations)."""

    message: str = Field(
        ...,
        description="Human-readable status message.",
        examples=["User deleted successfully."],
    )
