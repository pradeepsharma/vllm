"""Storage layer for the Task Management CLI Tool.

Provides JSON-file-backed persistence for :class:`~models.Task` and
:class:`~models.Project` objects.  All public methods raise descriptive
exceptions on invalid input so that callers can surface meaningful error
messages to the user.

Typical usage::

    store = Storage()                   # uses default ~/.taskman/data.json
    store = Storage("/tmp/tasks.json")  # custom path (useful for tests)

    project = store.create_project("My Project")
    task    = store.create_task("Write docs", project_id=project.id)
    store.update_task(task.id, status=Status.DONE)
    store.delete_task(task.id)
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from models import Priority, Project, Status, Task

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_DATA_PATH = os.path.join(
    os.path.expanduser("~"), ".taskman", "data.json"
)

# Sentinel object used to distinguish "argument not passed" from "passed as None".
_UNSET = object()


def _utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _validate_due_date(value: str) -> None:
    """Raise :class:`ValidationError` if *value* is not a valid ``YYYY-MM-DD`` date."""
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except (ValueError, TypeError) as exc:
        raise ValidationError(
            f"Invalid due_date '{value}'. Expected format: YYYY-MM-DD."
        ) from exc


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class StorageError(Exception):
    """Base class for all storage-layer errors."""


class NotFoundError(StorageError):
    """Raised when a requested entity does not exist in the store."""


class DuplicateError(StorageError):
    """Raised when an entity with the same id already exists."""


class ValidationError(StorageError):
    """Raised when supplied data fails validation."""


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class Storage:
    """JSON-file-backed store for :class:`Task` and :class:`Project` objects.

    The on-disk format is a single JSON file with the following top-level
    structure::

        {
            "tasks":    { "<task-id>":    { … task dict … }, … },
            "projects": { "<project-id>": { … project dict … }, … }
        }

    The file is read into memory on first access and written back after every
    mutating operation.  This is intentionally simple — no locking, no
    transactions — which is appropriate for a single-user CLI tool.

    Parameters
    ----------
    data_path:
        Absolute or relative path to the JSON data file.  The parent
        directory is created automatically if it does not exist.
        Defaults to ``~/.taskman/data.json``.
    """

    def __init__(self, data_path: str = _DEFAULT_DATA_PATH) -> None:
        self._path = os.path.abspath(data_path)
        self._tasks: Dict[str, Task] = {}
        self._projects: Dict[str, Project] = {}
        self._loaded = False

    # ------------------------------------------------------------------
    # Internal I/O
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load data from disk if not already loaded."""
        if not self._loaded:
            self._load()

    def _load(self) -> None:
        """Read the JSON file from disk and populate in-memory dicts.

        If the file does not exist the store starts empty.  Corrupt JSON
        raises :class:`StorageError`.
        """
        if not os.path.exists(self._path):
            self._tasks = {}
            self._projects = {}
            self._loaded = True
            return

        try:
            with open(self._path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except json.JSONDecodeError as exc:
            raise StorageError(
                f"Data file '{self._path}' contains invalid JSON: {exc}"
            ) from exc

        tasks_raw = raw.get("tasks", {})
        projects_raw = raw.get("projects", {})

        self._tasks = {}
        for tid, tdata in tasks_raw.items():
            try:
                self._tasks[tid] = Task.from_dict(tdata)
            except (KeyError, ValueError) as exc:
                raise StorageError(
                    f"Failed to deserialise task '{tid}': {exc}"
                ) from exc

        self._projects = {}
        for pid, pdata in projects_raw.items():
            try:
                self._projects[pid] = Project.from_dict(pdata)
            except (KeyError, ValueError) as exc:
                raise StorageError(
                    f"Failed to deserialise project '{pid}': {exc}"
                ) from exc

        self._loaded = True

    def _save(self) -> None:
        """Serialise in-memory state to the JSON file.

        Creates the parent directory if necessary.
        """
        parent = os.path.dirname(self._path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        data = {
            "tasks": {tid: t.to_dict() for tid, t in self._tasks.items()},
            "projects": {pid: p.to_dict() for pid, p in self._projects.items()},
        }

        # Write to a temp file then rename for atomicity (best-effort).
        tmp_path = self._path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp_path, self._path)
        except OSError as exc:
            raise StorageError(
                f"Failed to write data file '{self._path}': {exc}"
            ) from exc
        finally:
            # Clean up temp file if rename failed.
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    # ------------------------------------------------------------------
    # Project CRUD
    # ------------------------------------------------------------------

    def create_project(
        self,
        name: str,
        description: str = "",
    ) -> Project:
        """Create and persist a new :class:`Project`.

        Parameters
        ----------
        name:
            Human-readable project name.  Must be a non-empty string.
        description:
            Optional longer description.

        Returns
        -------
        Project
            The newly created project.

        Raises
        ------
        ValidationError
            If *name* is empty or not a string.
        """
        self._ensure_loaded()
        if not isinstance(name, str) or not name.strip():
            raise ValidationError("Project name must be a non-empty string.")

        project = Project(name=name.strip(), description=description)
        self._projects[project.id] = project
        self._save()
        return project

    def get_project(self, project_id: str) -> Project:
        """Return the :class:`Project` with the given *project_id*.

        Raises
        ------
        NotFoundError
            If no project with that id exists.
        """
        self._ensure_loaded()
        project = self._projects.get(project_id)
        if project is None:
            raise NotFoundError(f"Project '{project_id}' not found.")
        return project

    def list_projects(self) -> List[Project]:
        """Return all projects, ordered by creation time (oldest first)."""
        self._ensure_loaded()
        return sorted(self._projects.values(), key=lambda p: p.created_at)

    def update_project(
        self,
        project_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Project:
        """Update mutable fields of an existing :class:`Project`.

        Only the fields that are explicitly passed (not ``None``) are changed.

        Parameters
        ----------
        project_id:
            Id of the project to update.
        name:
            New name.  Must be a non-empty string if supplied.
        description:
            New description.

        Returns
        -------
        Project
            The updated project object.

        Raises
        ------
        NotFoundError
            If no project with *project_id* exists.
        ValidationError
            If *name* is an empty string.
        """
        self._ensure_loaded()
        project = self.get_project(project_id)

        if name is not None:
            if not isinstance(name, str) or not name.strip():
                raise ValidationError("Project name must be a non-empty string.")
            project.name = name.strip()

        if description is not None:
            project.description = description

        self._save()
        return project

    def delete_project(self, project_id: str, *, cascade: bool = False) -> None:
        """Remove a :class:`Project` from the store.

        Parameters
        ----------
        project_id:
            Id of the project to delete.
        cascade:
            If ``True``, also delete all tasks that belong to this project.
            If ``False`` (default), tasks are left intact but their
            ``project_id`` field is cleared (set to ``None``).

        Raises
        ------
        NotFoundError
            If no project with *project_id* exists.
        """
        self._ensure_loaded()
        if project_id not in self._projects:
            raise NotFoundError(f"Project '{project_id}' not found.")

        del self._projects[project_id]

        # Handle orphaned tasks.
        for task in list(self._tasks.values()):
            if task.project_id == project_id:
                if cascade:
                    del self._tasks[task.id]
                else:
                    task.project_id = None

        self._save()

    # ------------------------------------------------------------------
    # Task CRUD
    # ------------------------------------------------------------------

    def create_task(
        self,
        title: str,
        description: str = "",
        status: Status = Status.TODO,
        priority: Priority = Priority.MEDIUM,
        due_date: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Task:
        """Create and persist a new :class:`Task`.

        Parameters
        ----------
        title:
            Short human-readable summary.  Must be non-empty.
        description:
            Optional longer description.
        status:
            Initial lifecycle status (default: ``Status.TODO``).
        priority:
            Urgency level (default: ``Priority.MEDIUM``).
        due_date:
            Optional ISO-8601 date string ``"YYYY-MM-DD"``.
        project_id:
            Optional id of an existing :class:`Project`.

        Returns
        -------
        Task
            The newly created task.

        Raises
        ------
        ValidationError
            If *title* is empty, *due_date* has an invalid format, or
            *project_id* refers to a non-existent project.
        """
        self._ensure_loaded()

        if not isinstance(title, str) or not title.strip():
            raise ValidationError("Task title must be a non-empty string.")

        if due_date is not None:
            _validate_due_date(due_date)

        if project_id is not None and project_id not in self._projects:
            raise ValidationError(
                f"Project '{project_id}' does not exist. "
                "Create the project first."
            )

        task = Task(
            title=title.strip(),
            description=description,
            status=status,
            priority=priority,
            due_date=due_date,
            project_id=project_id,
        )
        self._tasks[task.id] = task
        self._save()
        return task

    def get_task(self, task_id: str) -> Task:
        """Return the :class:`Task` with the given *task_id*.

        Raises
        ------
        NotFoundError
            If no task with that id exists.
        """
        self._ensure_loaded()
        task = self._tasks.get(task_id)
        if task is None:
            raise NotFoundError(f"Task '{task_id}' not found.")
        return task

    def list_tasks(
        self,
        *,
        project_id: Optional[str] = None,
        status: Optional[Status] = None,
        priority: Optional[Priority] = None,
        overdue_only: bool = False,
    ) -> List[Task]:
        """Return tasks, optionally filtered.

        Parameters
        ----------
        project_id:
            If given, only return tasks belonging to this project.
        status:
            If given, only return tasks with this status.
        priority:
            If given, only return tasks with this priority.
        overdue_only:
            If ``True``, only return tasks that are overdue.

        Returns
        -------
        list[Task]
            Matching tasks ordered by creation time (oldest first).
        """
        self._ensure_loaded()
        tasks: List[Task] = list(self._tasks.values())

        if project_id is not None:
            tasks = [t for t in tasks if t.project_id == project_id]
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        if priority is not None:
            tasks = [t for t in tasks if t.priority == priority]
        if overdue_only:
            tasks = [t for t in tasks if t.is_overdue()]

        return sorted(tasks, key=lambda t: t.created_at)

    def update_task(
        self,
        task_id: str,
        title: object = _UNSET,
        description: object = _UNSET,
        status: object = _UNSET,
        priority: object = _UNSET,
        due_date: object = _UNSET,
        project_id: object = _UNSET,
    ) -> Task:
        """Update mutable fields of an existing :class:`Task`.

        Only the fields that are explicitly passed are changed.  Pass
        ``due_date=None`` or ``project_id=None`` to *clear* those fields.

        Parameters
        ----------
        task_id:
            Id of the task to update.
        title:
            New title.  Must be non-empty if supplied.
        description:
            New description.
        status:
            New lifecycle status (:class:`~models.Status`).
        priority:
            New urgency level (:class:`~models.Priority`).
        due_date:
            New due date (``"YYYY-MM-DD"``), or ``None`` to clear it.
            Omit the argument entirely to leave the current value unchanged.
        project_id:
            New project id, or ``None`` to unlink from any project.
            Omit the argument entirely to leave the current value unchanged.

        Returns
        -------
        Task
            The updated task object.

        Raises
        ------
        NotFoundError
            If no task with *task_id* exists.
        ValidationError
            If *title* is empty, *due_date* has an invalid format, or
            *project_id* refers to a non-existent project.
        """
        self._ensure_loaded()
        task = self.get_task(task_id)

        if title is not _UNSET:
            if not isinstance(title, str) or not title.strip():
                raise ValidationError("Task title must be a non-empty string.")
            task.title = title.strip()  # type: ignore[assignment]

        if description is not _UNSET:
            task.description = description  # type: ignore[assignment]

        if status is not _UNSET:
            task.status = status  # type: ignore[assignment]

        if priority is not _UNSET:
            task.priority = priority  # type: ignore[assignment]

        if due_date is not _UNSET:
            if due_date is not None:
                _validate_due_date(due_date)  # type: ignore[arg-type]
            task.due_date = due_date  # type: ignore[assignment]

        if project_id is not _UNSET:
            if project_id is not None and project_id not in self._projects:
                raise ValidationError(
                    f"Project '{project_id}' does not exist."
                )
            task.project_id = project_id  # type: ignore[assignment]

        task.updated_at = _utcnow_iso()
        self._save()
        return task

    def delete_task(self, task_id: str) -> None:
        """Remove a :class:`Task` from the store.

        Raises
        ------
        NotFoundError
            If no task with *task_id* exists.
        """
        self._ensure_loaded()
        if task_id not in self._tasks:
            raise NotFoundError(f"Task '{task_id}' not found.")
        del self._tasks[task_id]
        self._save()

    # ------------------------------------------------------------------
    # Bulk / utility
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Delete all tasks and projects from the store and persist."""
        self._ensure_loaded()
        self._tasks.clear()
        self._projects.clear()
        self._save()

    def stats(self) -> dict:
        """Return a summary dict with counts of tasks and projects.

        The returned dict has the shape::

            {
                "total_tasks":    <int>,
                "total_projects": <int>,
                "tasks_by_status": {
                    "todo":        <int>,
                    "in_progress": <int>,
                    "done":        <int>,
                },
                "tasks_by_priority": {
                    "low":      <int>,
                    "medium":   <int>,
                    "high":     <int>,
                    "critical": <int>,
                },
                "overdue_tasks": <int>,
            }
        """
        self._ensure_loaded()
        tasks = list(self._tasks.values())

        by_status: Dict[str, int] = {s.value: 0 for s in Status}
        by_priority: Dict[str, int] = {p.value: 0 for p in Priority}
        overdue = 0

        for t in tasks:
            by_status[t.status.value] += 1
            by_priority[t.priority.value] += 1
            if t.is_overdue():
                overdue += 1

        return {
            "total_tasks": len(tasks),
            "total_projects": len(self._projects),
            "tasks_by_status": by_status,
            "tasks_by_priority": by_priority,
            "overdue_tasks": overdue,
        }
