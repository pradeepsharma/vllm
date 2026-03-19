"""
Data models for the Task Management CLI tool.

This module defines the core data structures used throughout the application:
  - Priority: enum for task urgency levels
  - Status: enum for task lifecycle states
  - Task: the primary entity representing a single task
  - TaskList: a collection of tasks with filtering and sorting helpers
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timezone
from enum import Enum
from typing import List, Optional, Dict, Any


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Priority(str, Enum):
    """Urgency level of a task.

    Inherits from ``str`` so that values can be compared directly to strings
    and serialised to JSON without a custom encoder.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    # Numeric weight used for sorting (higher = more urgent)
    @property
    def weight(self) -> int:
        _weights = {
            Priority.LOW: 1,
            Priority.MEDIUM: 2,
            Priority.HIGH: 3,
            Priority.CRITICAL: 4,
        }
        return _weights[self]

    @classmethod
    def from_string(cls, value: str) -> "Priority":
        """Case-insensitive lookup by value string.

        Raises
        ------
        ValueError
            If *value* does not match any known priority.
        """
        try:
            return cls(value.lower())
        except ValueError:
            valid = ", ".join(p.value for p in cls)
            raise ValueError(
                f"Invalid priority {value!r}. Valid options: {valid}"
            )


class Status(str, Enum):
    """Lifecycle state of a task.

    Inherits from ``str`` for the same reasons as :class:`Priority`.
    """

    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"

    @classmethod
    def from_string(cls, value: str) -> "Status":
        """Case-insensitive lookup by value string.

        Raises
        ------
        ValueError
            If *value* does not match any known status.
        """
        try:
            return cls(value.lower())
        except ValueError:
            valid = ", ".join(s.value for s in cls)
            raise ValueError(
                f"Invalid status {value!r}. Valid options: {valid}"
            )

    @property
    def is_terminal(self) -> bool:
        """Return ``True`` if the status represents a finished state."""
        return self in (Status.DONE, Status.CANCELLED)


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return the current UTC datetime as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


def _new_id() -> str:
    """Generate a new unique task identifier."""
    return str(uuid.uuid4())


@dataclass
class Task:
    """A single task in the task management system.

    Parameters
    ----------
    title:
        Short, human-readable name for the task (required, non-empty).
    description:
        Optional longer description or notes.
    priority:
        Urgency level; defaults to :attr:`Priority.MEDIUM`.
    status:
        Current lifecycle state; defaults to :attr:`Status.TODO`.
    due_date:
        Optional ISO-8601 date string (``YYYY-MM-DD``) by which the task
        should be completed.
    tags:
        Free-form labels for grouping / filtering tasks.
    id:
        Unique identifier; auto-generated if not supplied.
    created_at:
        ISO-8601 datetime string; set automatically on creation.
    updated_at:
        ISO-8601 datetime string; updated whenever the task is modified.
    """

    title: str
    description: str = ""
    priority: Priority = Priority.MEDIUM
    status: Status = Status.TODO
    due_date: Optional[str] = None          # "YYYY-MM-DD" or None
    tags: List[str] = field(default_factory=list)
    id: str = field(default_factory=_new_id)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        """Validate fields immediately after construction."""
        self._validate()

    def _validate(self) -> None:
        """Raise :class:`ValueError` if any field is invalid."""
        if not self.title or not self.title.strip():
            raise ValueError("Task title must not be empty.")

        if not isinstance(self.priority, Priority):
            # Accept raw strings for convenience
            self.priority = Priority.from_string(str(self.priority))

        if not isinstance(self.status, Status):
            self.status = Status.from_string(str(self.status))

        if self.due_date is not None:
            self._parse_due_date(self.due_date)  # raises if invalid

        if not isinstance(self.tags, list):
            raise ValueError("tags must be a list of strings.")

        # Normalise tags: strip whitespace, drop empty strings, deduplicate
        self.tags = list(dict.fromkeys(
            t.strip() for t in self.tags if t and t.strip()
        ))

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def update(
        self,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        priority: Optional[Priority | str] = None,
        status: Optional[Status | str] = None,
        due_date: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> None:
        """Apply one or more field updates and refresh :attr:`updated_at`.

        Only keyword arguments that are explicitly provided (not ``None``)
        are applied.  Pass ``due_date=""`` to clear the due date.

        Raises
        ------
        ValueError
            If any supplied value fails validation.
        """
        if title is not None:
            if not title.strip():
                raise ValueError("Task title must not be empty.")
            self.title = title.strip()

        if description is not None:
            self.description = description

        if priority is not None:
            self.priority = (
                priority if isinstance(priority, Priority)
                else Priority.from_string(str(priority))
            )

        if status is not None:
            self.status = (
                status if isinstance(status, Status)
                else Status.from_string(str(status))
            )

        if due_date is not None:
            if due_date == "":
                self.due_date = None
            else:
                self._parse_due_date(due_date)  # validate
                self.due_date = due_date

        if tags is not None:
            if not isinstance(tags, list):
                raise ValueError("tags must be a list of strings.")
            self.tags = list(dict.fromkeys(
                t.strip() for t in tags if t and t.strip()
            ))

        self.updated_at = _now_iso()

    def mark_done(self) -> None:
        """Convenience method to set status to :attr:`Status.DONE`."""
        self.status = Status.DONE
        self.updated_at = _now_iso()

    def mark_cancelled(self) -> None:
        """Convenience method to set status to :attr:`Status.CANCELLED`."""
        self.status = Status.CANCELLED
        self.updated_at = _now_iso()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_overdue(self) -> bool:
        """Return ``True`` if the task has a past due date and is not finished."""
        if self.due_date is None or self.status.is_terminal:
            return False
        return self._parse_due_date(self.due_date) < date.today()

    @property
    def short_id(self) -> str:
        """Return the first 8 characters of the UUID for display purposes."""
        return self.id[:8]

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the task to a plain dictionary suitable for JSON output."""
        d = asdict(self)
        # Convert enum instances to their string values
        d["priority"] = self.priority.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Task":
        """Deserialise a task from a plain dictionary (e.g. loaded from JSON).

        Raises
        ------
        KeyError
            If a required field is missing from *data*.
        ValueError
            If any field value is invalid.
        """
        required = {"title"}
        missing = required - data.keys()
        if missing:
            raise KeyError(f"Missing required field(s): {', '.join(sorted(missing))}")

        return cls(
            id=data.get("id", _new_id()),
            title=data["title"],
            description=data.get("description", ""),
            priority=Priority.from_string(data.get("priority", Priority.MEDIUM.value)),
            status=Status.from_string(data.get("status", Status.TODO.value)),
            due_date=data.get("due_date"),
            tags=list(data.get("tags", [])),
            created_at=data.get("created_at", _now_iso()),
            updated_at=data.get("updated_at", _now_iso()),
        )

    def to_json(self) -> str:
        """Return a JSON string representation of the task."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "Task":
        """Deserialise a task from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_due_date(value: str) -> date:
        """Parse a ``YYYY-MM-DD`` string into a :class:`datetime.date`.

        Raises
        ------
        ValueError
            If *value* is not a valid ISO-8601 date.
        """
        try:
            return date.fromisoformat(value)
        except (ValueError, TypeError):
            raise ValueError(
                f"due_date must be in YYYY-MM-DD format, got {value!r}."
            )

    def __repr__(self) -> str:
        return (
            f"Task(id={self.short_id!r}, title={self.title!r}, "
            f"status={self.status.value!r}, priority={self.priority.value!r})"
        )


# ---------------------------------------------------------------------------
# TaskList
# ---------------------------------------------------------------------------

class TaskList:
    """An ordered, in-memory collection of :class:`Task` objects.

    Provides CRUD operations, filtering, and sorting helpers that the CLI
    commands can delegate to.
    """

    def __init__(self, tasks: Optional[List[Task]] = None) -> None:
        self._tasks: List[Task] = list(tasks) if tasks else []

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, task: Task) -> None:
        """Append *task* to the collection.

        Raises
        ------
        ValueError
            If a task with the same ``id`` already exists.
        """
        if self.get_by_id(task.id) is not None:
            raise ValueError(f"A task with id {task.id!r} already exists.")
        self._tasks.append(task)

    def get_by_id(self, task_id: str) -> Optional[Task]:
        """Return the task with the given *task_id*, or ``None``."""
        for task in self._tasks:
            if task.id == task_id:
                return task
        return None

    def get_by_short_id(self, short_id: str) -> Optional[Task]:
        """Return the first task whose UUID starts with *short_id*, or ``None``.

        Useful for CLI commands where the user types only the first 8 chars.
        """
        for task in self._tasks:
            if task.id.startswith(short_id):
                return task
        return None

    def remove(self, task_id: str) -> Task:
        """Remove and return the task with *task_id*.

        Raises
        ------
        KeyError
            If no task with *task_id* exists.
        """
        for i, task in enumerate(self._tasks):
            if task.id == task_id:
                return self._tasks.pop(i)
        raise KeyError(f"No task found with id {task_id!r}.")

    def update(self, task_id: str, **kwargs: Any) -> Task:
        """Find the task with *task_id* and call :meth:`Task.update` on it.

        Returns the updated task.

        Raises
        ------
        KeyError
            If no task with *task_id* exists.
        """
        task = self.get_by_id(task_id)
        if task is None:
            raise KeyError(f"No task found with id {task_id!r}.")
        task.update(**kwargs)
        return task

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter_by_status(self, status: Status | str) -> "TaskList":
        """Return a new :class:`TaskList` containing only tasks with *status*."""
        if not isinstance(status, Status):
            status = Status.from_string(str(status))
        return TaskList([t for t in self._tasks if t.status == status])

    def filter_by_priority(self, priority: Priority | str) -> "TaskList":
        """Return a new :class:`TaskList` containing only tasks with *priority*."""
        if not isinstance(priority, Priority):
            priority = Priority.from_string(str(priority))
        return TaskList([t for t in self._tasks if t.priority == priority])

    def filter_by_tag(self, tag: str) -> "TaskList":
        """Return a new :class:`TaskList` containing only tasks that have *tag*."""
        tag = tag.strip()
        return TaskList([t for t in self._tasks if tag in t.tags])

    def filter_overdue(self) -> "TaskList":
        """Return a new :class:`TaskList` containing only overdue tasks."""
        return TaskList([t for t in self._tasks if t.is_overdue])

    def search(self, query: str) -> "TaskList":
        """Return tasks whose title or description contains *query* (case-insensitive)."""
        q = query.lower()
        return TaskList([
            t for t in self._tasks
            if q in t.title.lower() or q in t.description.lower()
        ])

    # ------------------------------------------------------------------
    # Sorting
    # ------------------------------------------------------------------

    def sorted_by_priority(self, descending: bool = True) -> "TaskList":
        """Return a new :class:`TaskList` sorted by priority weight."""
        return TaskList(
            sorted(self._tasks, key=lambda t: t.priority.weight, reverse=descending)
        )

    def sorted_by_due_date(self, ascending: bool = True) -> "TaskList":
        """Return a new :class:`TaskList` sorted by due date.

        Tasks without a due date are placed at the end.
        """
        def _key(t: Task):
            if t.due_date is None:
                # Sort tasks without a due date to the end
                return date.max if ascending else date.min
            return Task._parse_due_date(t.due_date)

        return TaskList(
            sorted(self._tasks, key=_key, reverse=not ascending)
        )

    def sorted_by_created_at(self, ascending: bool = True) -> "TaskList":
        """Return a new :class:`TaskList` sorted by creation time."""
        return TaskList(
            sorted(self._tasks, key=lambda t: t.created_at, reverse=not ascending)
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the entire collection to a plain dictionary."""
        return {"tasks": [t.to_dict() for t in self._tasks]}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskList":
        """Deserialise a :class:`TaskList` from a plain dictionary."""
        tasks = [Task.from_dict(td) for td in data.get("tasks", [])]
        return cls(tasks)

    def to_json(self) -> str:
        """Return a JSON string representation of the collection."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "TaskList":
        """Deserialise a :class:`TaskList` from a JSON string."""
        return cls.from_dict(json.loads(json_str))

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def all(self) -> List[Task]:
        """Return a shallow copy of all tasks as a plain list."""
        return list(self._tasks)

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self):
        return iter(self._tasks)

    def __repr__(self) -> str:
        return f"TaskList({len(self._tasks)} task(s))"

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, int]:
        """Return a count of tasks grouped by status."""
        counts: Dict[str, int] = {s.value: 0 for s in Status}
        for task in self._tasks:
            counts[task.status.value] += 1
        counts["total"] = len(self._tasks)
        return counts
