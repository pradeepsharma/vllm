"""Data models for the Task Management CLI Tool.

Defines the core domain objects: Status, Priority enums, and Task, Project dataclasses.
All models support serialization to/from plain dicts for JSON persistence.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string (timezone-aware)."""
    return datetime.now(timezone.utc).isoformat()


class Status(str, Enum):
    """Task lifecycle status."""

    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class Priority(str, Enum):
    """Task urgency level."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Task:
    """Represents a single work item.

    Attributes:
        title:       Short human-readable summary (required).
        id:          UUID string, auto-generated if not supplied.
        description: Optional longer description.
        status:      Current lifecycle status (default: TODO).
        priority:    Urgency level (default: MEDIUM).
        due_date:    Optional ISO-8601 date string ``"YYYY-MM-DD"``.
        project_id:  Optional reference to a :class:`Project` id.
        created_at:  ISO-8601 datetime string, set at creation time.
        updated_at:  ISO-8601 datetime string, updated on every change.
    """

    title: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    status: Status = Status.TODO
    priority: Priority = Priority.MEDIUM
    due_date: Optional[str] = None  # ISO-8601 date string "YYYY-MM-DD"
    project_id: Optional[str] = None
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "due_date": self.due_date,
            "project_id": self.project_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Reconstruct a :class:`Task` from a plain dictionary.

        Missing optional fields fall back to sensible defaults so that
        older stored records remain compatible after schema additions.
        """
        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            status=Status(data.get("status", "todo")),
            priority=Priority(data.get("priority", "medium")),
            due_date=data.get("due_date"),
            project_id=data.get("project_id"),
            created_at=data.get("created_at", _utcnow_iso()),
            updated_at=data.get("updated_at", _utcnow_iso()),
        )

    # ------------------------------------------------------------------
    # Domain logic
    # ------------------------------------------------------------------

    def is_overdue(self) -> bool:
        """Return ``True`` if the task has a past due date and is not DONE.

        A task with no due date is never considered overdue.
        A DONE task is never considered overdue regardless of its due date.
        """
        if self.due_date is None or self.status == Status.DONE:
            return False
        due = datetime.strptime(self.due_date, "%Y-%m-%d").date()
        today = datetime.now(timezone.utc).date()
        return due < today


@dataclass
class Project:
    """Groups related tasks under a named project.

    Attributes:
        name:        Human-readable project name (required).
        id:          UUID string, auto-generated if not supplied.
        description: Optional longer description.
        created_at:  ISO-8601 datetime string, set at creation time.
    """

    name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    created_at: str = field(default_factory=_utcnow_iso)

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        """Reconstruct a :class:`Project` from a plain dictionary.

        Missing optional fields fall back to sensible defaults.
        """
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=data.get("created_at", _utcnow_iso()),
        )
