"""
task_store.py — Data layer for the CLI task tracker.

Responsibilities:
- Define the Task schema (TypedDict + Literal types).
- Persist tasks as a JSON array in a single file.
- Provide CRUD operations: add, list, update status, delete.
- Provide aggregate statistics.

All public functions are pure in the sense that they read the file,
perform the operation, and write the file back atomically (via a
temporary file + os.replace) so a crash mid-write never corrupts data.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Literal, Optional, TypedDict

# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

Status = Literal["todo", "in_progress", "done"]
Priority = Literal["low", "medium", "high"]

VALID_STATUSES: tuple[str, ...] = ("todo", "in_progress", "done")
VALID_PRIORITIES: tuple[str, ...] = ("low", "medium", "high")


class Task(TypedDict):
    """Schema for a single task record stored in the JSON file."""

    id: str          # UUID4 string
    title: str       # Human-readable title
    status: Status   # One of: todo | in_progress | done
    priority: Priority  # One of: low | medium | high
    created_at: str  # ISO-8601 timestamp (UTC)


# ---------------------------------------------------------------------------
# Default storage path
# ---------------------------------------------------------------------------

DEFAULT_STORE_PATH: Path = Path.home() / ".task_tracker" / "tasks.json"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_store_exists(path: Path) -> None:
    """Create the store file (and parent directories) if they don't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        _write_tasks(path, [])


def _read_tasks(path: Path) -> List[Task]:
    """
    Read and return all tasks from *path*.

    Returns an empty list if the file is missing or contains invalid JSON
    so that callers never have to handle I/O exceptions for the common case.

    Raises
    ------
    ValueError
        If the file exists but its top-level structure is not a JSON array.
    """
    if not path.exists():
        return []

    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise OSError(f"Cannot read task store at '{path}': {exc}") from exc

    if not raw:
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Task store at '{path}' contains invalid JSON: {exc}"
        ) from exc

    if not isinstance(data, list):
        raise ValueError(
            f"Task store at '{path}' must contain a JSON array, "
            f"got {type(data).__name__!r}."
        )

    return data  # type: ignore[return-value]


def _write_tasks(path: Path, tasks: List[Task]) -> None:
    """
    Atomically write *tasks* to *path* as a pretty-printed JSON array.

    Uses a sibling temporary file + ``os.replace`` so that a crash during
    the write never leaves the store in a partially-written state.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temp file in the same directory so os.replace is atomic
    # (same filesystem).
    fd, tmp_path = tempfile.mkstemp(
        dir=path.parent, prefix=".tasks_tmp_", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(tasks, fh, indent=2, ensure_ascii=False)
            fh.write("\n")  # trailing newline — friendly for git diffs
        os.replace(tmp_path, path)
    except Exception:
        # Clean up the temp file if anything went wrong before the replace.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string with 'Z' suffix."""
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_status(status: str) -> Status:
    """Raise *ValueError* if *status* is not a recognised value."""
    if status not in VALID_STATUSES:
        raise ValueError(
            f"Invalid status {status!r}. "
            f"Must be one of: {', '.join(VALID_STATUSES)}."
        )
    return status  # type: ignore[return-value]


def _validate_priority(priority: str) -> Priority:
    """Raise *ValueError* if *priority* is not a recognised value."""
    if priority not in VALID_PRIORITIES:
        raise ValueError(
            f"Invalid priority {priority!r}. "
            f"Must be one of: {', '.join(VALID_PRIORITIES)}."
        )
    return priority  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class TaskStore:
    """
    High-level interface to the JSON-backed task store.

    Parameters
    ----------
    path:
        Path to the JSON file used for persistence.  Defaults to
        ``~/.task_tracker/tasks.json``.

    Example
    -------
    >>> store = TaskStore()
    >>> task = store.add_task("Write unit tests", priority="high")
    >>> store.update_status(task["id"], "done")
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path: Path = Path(path) if path is not None else DEFAULT_STORE_PATH
        _ensure_store_exists(self.path)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add_task(
        self,
        title: str,
        *,
        priority: str = "medium",
        status: str = "todo",
    ) -> Task:
        """
        Create a new task and persist it.

        Parameters
        ----------
        title:
            Short human-readable description of the task.  Must be a
            non-empty string after stripping whitespace.
        priority:
            One of ``low``, ``medium``, ``high``.  Defaults to ``medium``.
        status:
            Initial status.  One of ``todo``, ``in_progress``, ``done``.
            Defaults to ``todo``.

        Returns
        -------
        Task
            The newly created task dict (including its generated ``id``
            and ``created_at`` fields).

        Raises
        ------
        ValueError
            If *title* is blank, or *priority* / *status* are invalid.
        """
        title = title.strip()
        if not title:
            raise ValueError("Task title must not be empty.")

        validated_priority = _validate_priority(priority)
        validated_status = _validate_status(status)

        task: Task = {
            "id": str(uuid.uuid4()),
            "title": title,
            "status": validated_status,
            "priority": validated_priority,
            "created_at": _now_iso(),
        }

        tasks = _read_tasks(self.path)
        tasks.append(task)
        _write_tasks(self.path, tasks)
        return task

    def list_tasks(
        self,
        *,
        status: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> List[Task]:
        """
        Return all tasks, optionally filtered by *status* and/or *priority*.

        Tasks are returned in insertion order (oldest first).

        Parameters
        ----------
        status:
            If provided, only tasks with this status are returned.
        priority:
            If provided, only tasks with this priority are returned.

        Raises
        ------
        ValueError
            If *status* or *priority* are provided but invalid.
        """
        if status is not None:
            _validate_status(status)
        if priority is not None:
            _validate_priority(priority)

        tasks = _read_tasks(self.path)

        if status is not None:
            tasks = [t for t in tasks if t["status"] == status]
        if priority is not None:
            tasks = [t for t in tasks if t["priority"] == priority]

        return tasks

    def get_task(self, task_id: str) -> Optional[Task]:
        """
        Return the task with the given *task_id*, or ``None`` if not found.

        Parameters
        ----------
        task_id:
            UUID string of the task to look up.
        """
        for task in _read_tasks(self.path):
            if task["id"] == task_id:
                return task
        return None

    def update_status(self, task_id: str, new_status: str) -> Task:
        """
        Change the status of an existing task.

        Parameters
        ----------
        task_id:
            UUID string of the task to update.
        new_status:
            The new status value.  Must be one of ``todo``, ``in_progress``,
            ``done``.

        Returns
        -------
        Task
            The updated task dict.

        Raises
        ------
        ValueError
            If *new_status* is invalid or no task with *task_id* exists.
        """
        validated_status = _validate_status(new_status)

        tasks = _read_tasks(self.path)
        for task in tasks:
            if task["id"] == task_id:
                task["status"] = validated_status
                _write_tasks(self.path, tasks)
                return task

        raise ValueError(f"No task found with id {task_id!r}.")

    def update_priority(self, task_id: str, new_priority: str) -> Task:
        """
        Change the priority of an existing task.

        Parameters
        ----------
        task_id:
            UUID string of the task to update.
        new_priority:
            The new priority value.  Must be one of ``low``, ``medium``,
            ``high``.

        Returns
        -------
        Task
            The updated task dict.

        Raises
        ------
        ValueError
            If *new_priority* is invalid or no task with *task_id* exists.
        """
        validated_priority = _validate_priority(new_priority)

        tasks = _read_tasks(self.path)
        for task in tasks:
            if task["id"] == task_id:
                task["priority"] = validated_priority
                _write_tasks(self.path, tasks)
                return task

        raise ValueError(f"No task found with id {task_id!r}.")

    def delete_task(self, task_id: str) -> Task:
        """
        Remove a task from the store.

        Parameters
        ----------
        task_id:
            UUID string of the task to delete.

        Returns
        -------
        Task
            The deleted task dict (so callers can display confirmation).

        Raises
        ------
        ValueError
            If no task with *task_id* exists.
        """
        tasks = _read_tasks(self.path)
        for index, task in enumerate(tasks):
            if task["id"] == task_id:
                deleted = tasks.pop(index)
                _write_tasks(self.path, tasks)
                return deleted

        raise ValueError(f"No task found with id {task_id!r}.")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict:
        """
        Return aggregate statistics about the current task store.

        Returns
        -------
        dict
            A dictionary with the following keys:

            ``total`` (int)
                Total number of tasks.
            ``by_status`` (dict[str, int])
                Count of tasks for each status value.
            ``by_priority`` (dict[str, int])
                Count of tasks for each priority value.
            ``completion_rate`` (float)
                Fraction of tasks whose status is ``done`` (0.0–1.0).
                ``0.0`` when there are no tasks.

        Example
        -------
        >>> store.get_stats()
        {
            'total': 5,
            'by_status': {'todo': 2, 'in_progress': 1, 'done': 2},
            'by_priority': {'low': 1, 'medium': 3, 'high': 1},
            'completion_rate': 0.4,
        }
        """
        tasks = _read_tasks(self.path)
        total = len(tasks)

        by_status: dict[str, int] = {s: 0 for s in VALID_STATUSES}
        by_priority: dict[str, int] = {p: 0 for p in VALID_PRIORITIES}

        for task in tasks:
            by_status[task["status"]] = by_status.get(task["status"], 0) + 1
            by_priority[task["priority"]] = by_priority.get(task["priority"], 0) + 1

        completion_rate = (by_status["done"] / total) if total > 0 else 0.0

        return {
            "total": total,
            "by_status": by_status,
            "by_priority": by_priority,
            "completion_rate": completion_rate,
        }
