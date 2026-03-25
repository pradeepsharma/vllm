"""
task_store — Data layer for the task tracker CLI.

This module provides:
- ``TaskStatus``  — Literal type alias for valid task status values.
- ``TaskPriority`` — Literal type alias for valid task priority values.
- ``Task``        — Frozen dataclass representing a single task record.
- ``TaskStore``   — Manages persistence of tasks to a JSON flat file.

Design notes:
- Status and priority are plain ``Literal`` strings (not ``Enum``) so they
  round-trip through JSON without any custom serialization logic.
- ``TaskStore`` loads the JSON file on every read and writes atomically via
  ``os.replace()`` on every mutation, preventing data corruption on crash.
- Partial-ID lookup mirrors the git short-SHA UX: pass the first few characters
  of a task UUID and the store resolves it unambiguously, raising ``ValueError``
  if zero or more than one task matches.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Literal, Optional

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

TaskStatus = Literal["todo", "in_progress", "done"]
TaskPriority = Literal["low", "medium", "high"]

# Ordered sequences used for validation and iteration
VALID_STATUSES: tuple[str, ...] = ("todo", "in_progress", "done")
VALID_PRIORITIES: tuple[str, ...] = ("low", "medium", "high")


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """Represents a single task record.

    Attributes:
        id:         UUID4 string — globally unique identifier.
        title:      Human-readable description of the task.
        status:     Current lifecycle state: ``"todo"``, ``"in_progress"``,
                    or ``"done"``.
        priority:   Importance level: ``"low"``, ``"medium"``, or ``"high"``.
        created_at: ISO 8601 UTC timestamp set at creation time.
    """

    id: str
    title: str
    status: TaskStatus
    priority: TaskPriority
    created_at: str

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a plain ``dict`` suitable for JSON serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "priority": self.priority,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Construct a ``Task`` from a plain ``dict`` (e.g. loaded from JSON).

        Args:
            data: Dictionary with keys ``id``, ``title``, ``status``,
                  ``priority``, and ``created_at``.

        Returns:
            A new ``Task`` instance.

        Raises:
            KeyError:  If a required field is missing from *data*.
            ValueError: If ``status`` or ``priority`` hold an invalid value.
        """
        status = data["status"]
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status {status!r}. Must be one of {VALID_STATUSES}."
            )

        priority = data["priority"]
        if priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority {priority!r}. Must be one of {VALID_PRIORITIES}."
            )

        return cls(
            id=data["id"],
            title=data["title"],
            status=status,
            priority=priority,
            created_at=data["created_at"],
        )


# ---------------------------------------------------------------------------
# TaskStore
# ---------------------------------------------------------------------------


class TaskStore:
    """Manages a collection of :class:`Task` objects persisted to a JSON file.

    The store is intentionally simple: it loads the entire file on every read
    and writes the entire file on every mutation.  This keeps the
    implementation straightforward and reliable for a CLI tool that handles
    at most hundreds of tasks.

    Args:
        filepath: Path to the JSON file used for persistence.  The file is
                  created automatically on the first write if it does not
                  exist.

    Example::

        store = TaskStore("tasks.json")
        task = store.add_task("Fix the login bug", priority="high")
        print(task.id)
    """

    def __init__(self, filepath: str) -> None:
        self.filepath = filepath

    # ------------------------------------------------------------------
    # Private I/O helpers
    # ------------------------------------------------------------------

    def _load(self) -> List[Task]:
        """Load and return all tasks from the JSON file.

        Returns an empty list if the file does not exist yet.

        Raises:
            json.JSONDecodeError: If the file exists but contains invalid JSON.
            ValueError:           If any stored task has an invalid field value.
        """
        if not os.path.exists(self.filepath):
            return []

        with open(self.filepath, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        if not isinstance(raw, list):
            raise ValueError(
                f"Expected a JSON array in {self.filepath!r}, got {type(raw).__name__}."
            )

        return [Task.from_dict(item) for item in raw]

    def _save(self, tasks: List[Task]) -> None:
        """Atomically write *tasks* to the JSON file.

        Uses a temporary file in the same directory as the target so that
        ``os.replace()`` is guaranteed to be atomic on POSIX systems (same
        filesystem).  This prevents data loss if the process is interrupted
        mid-write.

        Args:
            tasks: The complete list of tasks to persist.
        """
        data = [task.to_dict() for task in tasks]
        dir_name = os.path.dirname(os.path.abspath(self.filepath))

        # Write to a temp file in the same directory, then atomically replace.
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
                fh.write("\n")  # POSIX-friendly trailing newline
            os.replace(tmp_path, self.filepath)
        except Exception:
            # Clean up the temp file if anything goes wrong.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Private lookup helper
    # ------------------------------------------------------------------

    def _resolve_partial_id(self, partial_id: str, tasks: List[Task]) -> Task:
        """Find the unique task whose ``id`` starts with *partial_id*.

        Args:
            partial_id: A prefix of a task UUID (e.g. ``"a3f2"``).
            tasks:      The current list of tasks to search.

        Returns:
            The single matching :class:`Task`.

        Raises:
            ValueError: If *partial_id* is empty, matches no tasks, or matches
                        more than one task.
        """
        if not partial_id:
            raise ValueError("partial_id must not be empty.")

        matches = [t for t in tasks if t.id.startswith(partial_id)]

        if len(matches) == 0:
            raise ValueError(
                f"No task found with ID starting with {partial_id!r}."
            )
        if len(matches) > 1:
            ids = ", ".join(t.id for t in matches)
            raise ValueError(
                f"Ambiguous ID prefix {partial_id!r} matches {len(matches)} tasks: {ids}. "
                "Please provide more characters."
            )

        return matches[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_task(
        self,
        title: str,
        priority: TaskPriority = "medium",
    ) -> Task:
        """Create a new task and persist it.

        Args:
            title:    Human-readable description of the task.  Must be a
                      non-empty string after stripping whitespace.
            priority: Importance level — ``"low"``, ``"medium"``, or
                      ``"high"``.  Defaults to ``"medium"``.

        Returns:
            The newly created :class:`Task`.

        Raises:
            ValueError: If *title* is blank or *priority* is invalid.
        """
        title = title.strip()
        if not title:
            raise ValueError("Task title must not be empty.")

        if priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority {priority!r}. Must be one of {VALID_PRIORITIES}."
            )

        task = Task(
            id=str(uuid.uuid4()),
            title=title,
            status="todo",
            priority=priority,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

        tasks = self._load()
        tasks.append(task)
        self._save(tasks)

        return task

    def list_tasks(
        self,
        status: Optional[TaskStatus] = None,
        priority: Optional[TaskPriority] = None,
    ) -> List[Task]:
        """Return tasks, optionally filtered by *status* and/or *priority*.

        Filters are applied with AND semantics: a task must satisfy every
        supplied filter to be included in the result.

        Args:
            status:   If provided, only tasks with this status are returned.
            priority: If provided, only tasks with this priority are returned.

        Returns:
            A list of matching :class:`Task` objects in insertion order.

        Raises:
            ValueError: If *status* or *priority* is not a valid value.
        """
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status {status!r}. Must be one of {VALID_STATUSES}."
            )
        if priority is not None and priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority {priority!r}. Must be one of {VALID_PRIORITIES}."
            )

        tasks = self._load()

        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        if priority is not None:
            tasks = [t for t in tasks if t.priority == priority]

        return tasks

    def update_status(self, partial_id: str, status: TaskStatus) -> Task:
        """Change the status of a task identified by a partial UUID prefix.

        Args:
            partial_id: A prefix of the target task's UUID.
            status:     The new status value — ``"todo"``, ``"in_progress"``,
                        or ``"done"``.

        Returns:
            The updated :class:`Task` (with the new status).

        Raises:
            ValueError: If *partial_id* is ambiguous or matches no task, or
                        if *status* is invalid.
        """
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status {status!r}. Must be one of {VALID_STATUSES}."
            )

        tasks = self._load()
        target = self._resolve_partial_id(partial_id, tasks)

        # Build an updated Task (dataclasses are mutable by default here).
        updated = Task(
            id=target.id,
            title=target.title,
            status=status,
            priority=target.priority,
            created_at=target.created_at,
        )

        new_tasks = [updated if t.id == target.id else t for t in tasks]
        self._save(new_tasks)

        return updated

    def delete_task(self, partial_id: str) -> Task:
        """Remove a task identified by a partial UUID prefix.

        Args:
            partial_id: A prefix of the target task's UUID.

        Returns:
            The :class:`Task` that was deleted.

        Raises:
            ValueError: If *partial_id* is ambiguous or matches no task.
        """
        tasks = self._load()
        target = self._resolve_partial_id(partial_id, tasks)

        new_tasks = [t for t in tasks if t.id != target.id]
        self._save(new_tasks)

        return target
