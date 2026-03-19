"""
Business logic layer for the Task Management CLI tool.

This module provides :class:`TaskService` — the single entry-point for all
task-management operations.  It sits between the storage layer
(:mod:`storage`) and the CLI commands, orchestrating reads, writes, and
validation so that the CLI layer stays thin.

Public API
----------
- :class:`TaskNotFoundError`   – raised when a task cannot be located
- :class:`DuplicateTaskError`  – raised when a duplicate task ID is detected
- :class:`ServiceError`        – base class for all service-level errors
- :class:`TaskService`         – the main service class

Typical usage::

    from storage import TaskStorage
    from services import TaskService

    service = TaskService(TaskStorage("~/.tasks/tasks.json"))

    task = service.create_task("Write unit tests", priority="high", tags=["dev"])
    service.update_task(task.id, status="in_progress")
    service.complete_task(task.id)
    service.delete_task(task.id)

    results = service.list_tasks(status="todo", sort_by="priority")
    stats   = service.get_statistics()
"""

from __future__ import annotations

import pathlib
import sys
import importlib.util
from typing import Any, Dict, List, Optional, Union


# ---------------------------------------------------------------------------
# Lazy model / storage imports (mirrors the pattern used in storage.py so
# that this file works regardless of how sys.path is configured).
# ---------------------------------------------------------------------------

def _load_module(name: str, filepath: pathlib.Path):
    """Load *filepath* as a module named *name* and register it in sys.modules."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name!r} from {filepath}")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _get_models():
    """Return the task_models module, importing it if necessary."""
    if "task_models" in sys.modules:
        return sys.modules["task_models"]
    _here = pathlib.Path(__file__).parent
    return _load_module("task_models", _here / "models.py")


def _get_storage_module():
    """Return the task_storage module, importing it if necessary."""
    if "task_storage" in sys.modules:
        return sys.modules["task_storage"]
    _here = pathlib.Path(__file__).parent
    return _load_module("task_storage", _here / "storage.py")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ServiceError(Exception):
    """Base class for all service-level errors."""


class TaskNotFoundError(ServiceError):
    """Raised when a task cannot be found by its ID or short ID.

    Attributes
    ----------
    task_id : str
        The identifier that was searched for.
    """

    def __init__(self, task_id: str) -> None:
        super().__init__(f"No task found with id or short-id {task_id!r}.")
        self.task_id = task_id


class DuplicateTaskError(ServiceError):
    """Raised when attempting to add a task whose ID already exists.

    Attributes
    ----------
    task_id : str
        The duplicate identifier.
    """

    def __init__(self, task_id: str) -> None:
        super().__init__(f"A task with id {task_id!r} already exists.")
        self.task_id = task_id


# ---------------------------------------------------------------------------
# TaskService
# ---------------------------------------------------------------------------

#: Valid sort-by keys accepted by :meth:`TaskService.list_tasks`.
_VALID_SORT_KEYS = frozenset({"priority", "due_date", "created_at", "title"})


class TaskService:
    """Orchestrates all task-management operations.

    Parameters
    ----------
    storage:
        A :class:`~storage.TaskStorage` instance used for persistence.
        The service loads the task list from storage on first access and
        writes it back after every mutating operation (auto-save).

    Notes
    -----
    The service uses *lazy loading*: the task list is not read from disk
    until the first operation that needs it.  This avoids unnecessary I/O
    when the service object is constructed but never used.

    All mutating methods (:meth:`create_task`, :meth:`update_task`, etc.)
    automatically persist the updated task list to disk via
    :meth:`~storage.TaskStorage.save`.
    """

    def __init__(self, storage) -> None:
        self._storage = storage
        self._task_list = None  # loaded lazily

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load(self):
        """Return the in-memory task list, loading from disk if needed."""
        if self._task_list is None:
            self._task_list = self._storage.load()
        return self._task_list

    def _save(self) -> None:
        """Persist the current in-memory task list to disk."""
        self._storage.save(self._task_list)

    def _resolve_task(self, task_id: str):
        """Return the task matching *task_id* (full or short ID).

        Raises
        ------
        TaskNotFoundError
            If no task matches *task_id*.
        """
        task_list = self._load()
        # Try exact match first
        task = task_list.get_by_id(task_id)
        if task is not None:
            return task
        # Fall back to short-id prefix match
        task = task_list.get_by_short_id(task_id)
        if task is not None:
            return task
        raise TaskNotFoundError(task_id)

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def create_task(
        self,
        title: str,
        *,
        description: str = "",
        priority: Union[str, Any] = "medium",
        status: Union[str, Any] = "todo",
        due_date: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """Create a new task, persist it, and return it.

        Parameters
        ----------
        title:
            Short, human-readable name for the task (required, non-empty).
        description:
            Optional longer description or notes.
        priority:
            Urgency level string (``"low"``, ``"medium"``, ``"high"``,
            ``"critical"``) or a :class:`~models.Priority` enum value.
            Defaults to ``"medium"``.
        status:
            Initial lifecycle state string (``"todo"``, ``"in_progress"``,
            ``"done"``, ``"cancelled"``) or a :class:`~models.Status` enum
            value.  Defaults to ``"todo"``.
        due_date:
            Optional ISO-8601 date string (``YYYY-MM-DD``).
        tags:
            Optional list of string labels.

        Returns
        -------
        Task
            The newly created and persisted task.

        Raises
        ------
        ValueError
            If *title* is empty, *priority* / *status* are invalid strings,
            or *due_date* is not in ``YYYY-MM-DD`` format.
        """
        models = _get_models()
        task = models.Task(
            title=title,
            description=description,
            priority=priority,
            status=status,
            due_date=due_date,
            tags=tags if tags is not None else [],
        )
        task_list = self._load()
        try:
            task_list.add(task)
        except ValueError as exc:
            # Re-raise as DuplicateTaskError if it's an ID collision
            if "already exists" in str(exc):
                raise DuplicateTaskError(task.id) from exc
            raise
        self._save()
        return task

    def get_task(self, task_id: str):
        """Return the task identified by *task_id* (full or short UUID).

        Parameters
        ----------
        task_id:
            Full UUID or the first 8 characters of the UUID.

        Returns
        -------
        Task
            The matching task.

        Raises
        ------
        TaskNotFoundError
            If no task matches *task_id*.
        """
        return self._resolve_task(task_id)

    def update_task(
        self,
        task_id: str,
        *,
        title: Optional[str] = None,
        description: Optional[str] = None,
        priority: Optional[Union[str, Any]] = None,
        status: Optional[Union[str, Any]] = None,
        due_date: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ):
        """Update one or more fields of an existing task and persist.

        Only keyword arguments that are explicitly provided (not ``None``)
        are applied.  Pass ``due_date=""`` to clear the due date.

        Parameters
        ----------
        task_id:
            Full UUID or short ID of the task to update.
        title:
            New title (non-empty string).
        description:
            New description.
        priority:
            New priority string or enum value.
        status:
            New status string or enum value.
        due_date:
            New due date in ``YYYY-MM-DD`` format, or ``""`` to clear.
        tags:
            New list of tags (replaces existing tags).

        Returns
        -------
        Task
            The updated task.

        Raises
        ------
        TaskNotFoundError
            If no task matches *task_id*.
        ValueError
            If any supplied value fails validation.
        """
        task = self._resolve_task(task_id)
        kwargs: Dict[str, Any] = {}
        if title is not None:
            kwargs["title"] = title
        if description is not None:
            kwargs["description"] = description
        if priority is not None:
            kwargs["priority"] = priority
        if status is not None:
            kwargs["status"] = status
        if due_date is not None:
            kwargs["due_date"] = due_date
        if tags is not None:
            kwargs["tags"] = tags
        task.update(**kwargs)
        self._save()
        return task

    def delete_task(self, task_id: str):
        """Remove a task from the list and persist.

        Parameters
        ----------
        task_id:
            Full UUID or short ID of the task to delete.

        Returns
        -------
        Task
            The removed task.

        Raises
        ------
        TaskNotFoundError
            If no task matches *task_id*.
        """
        # Resolve to full ID first (handles short IDs)
        task = self._resolve_task(task_id)
        task_list = self._load()
        task_list.remove(task.id)
        self._save()
        return task

    # ------------------------------------------------------------------
    # Status transitions
    # ------------------------------------------------------------------

    def complete_task(self, task_id: str):
        """Mark a task as done and persist.

        Parameters
        ----------
        task_id:
            Full UUID or short ID of the task to complete.

        Returns
        -------
        Task
            The updated task with status ``done``.

        Raises
        ------
        TaskNotFoundError
            If no task matches *task_id*.
        """
        task = self._resolve_task(task_id)
        task.mark_done()
        self._save()
        return task

    def cancel_task(self, task_id: str):
        """Mark a task as cancelled and persist.

        Parameters
        ----------
        task_id:
            Full UUID or short ID of the task to cancel.

        Returns
        -------
        Task
            The updated task with status ``cancelled``.

        Raises
        ------
        TaskNotFoundError
            If no task matches *task_id*.
        """
        task = self._resolve_task(task_id)
        task.mark_cancelled()
        self._save()
        return task

    def start_task(self, task_id: str):
        """Mark a task as in-progress and persist.

        Parameters
        ----------
        task_id:
            Full UUID or short ID of the task to start.

        Returns
        -------
        Task
            The updated task with status ``in_progress``.

        Raises
        ------
        TaskNotFoundError
            If no task matches *task_id*.
        """
        task = self._resolve_task(task_id)
        task.update(status="in_progress")
        self._save()
        return task

    # ------------------------------------------------------------------
    # Querying / listing
    # ------------------------------------------------------------------

    def list_tasks(
        self,
        *,
        status: Optional[str] = None,
        priority: Optional[str] = None,
        tag: Optional[str] = None,
        overdue_only: bool = False,
        search_query: Optional[str] = None,
        sort_by: str = "created_at",
        ascending: bool = True,
    ):
        """Return a filtered and sorted list of tasks.

        All filter parameters are optional and combinable.  Filters are
        applied in the following order:
        ``status`` → ``priority`` → ``tag`` → ``overdue_only`` → ``search_query``.

        Parameters
        ----------
        status:
            If given, only tasks with this status are returned.
        priority:
            If given, only tasks with this priority are returned.
        tag:
            If given, only tasks that carry this tag are returned.
        overdue_only:
            If ``True``, only overdue tasks are returned.
        search_query:
            If given, only tasks whose title or description contains this
            string (case-insensitive) are returned.
        sort_by:
            Field to sort by.  One of ``"priority"``, ``"due_date"``,
            ``"created_at"``, ``"title"``.  Defaults to ``"created_at"``.
        ascending:
            Sort direction.  Defaults to ``True`` (ascending).
            For ``"priority"``, ``ascending=True`` (the default) means
            most-urgent first (critical → low); pass ``ascending=False``
            to get least-urgent first (low → critical).

        Returns
        -------
        list[Task]
            A plain list of :class:`~models.Task` objects matching all
            supplied filters, in the requested order.

        Raises
        ------
        ValueError
            If *sort_by* is not one of the accepted values.
        """
        if sort_by not in _VALID_SORT_KEYS:
            raise ValueError(
                f"Invalid sort_by {sort_by!r}. "
                f"Valid options: {', '.join(sorted(_VALID_SORT_KEYS))}"
            )

        task_list = self._load()

        # --- Apply filters ---
        if status is not None:
            task_list = task_list.filter_by_status(status)

        if priority is not None:
            task_list = task_list.filter_by_priority(priority)

        if tag is not None:
            task_list = task_list.filter_by_tag(tag)

        if overdue_only:
            task_list = task_list.filter_overdue()

        if search_query is not None:
            task_list = task_list.search(search_query)

        # --- Apply sort ---
        if sort_by == "priority":
            # ascending=True (default) → most-urgent first (descending by weight)
            # ascending=False          → least-urgent first (ascending by weight)
            task_list = task_list.sorted_by_priority(descending=ascending)
        elif sort_by == "due_date":
            task_list = task_list.sorted_by_due_date(ascending=ascending)
        elif sort_by == "created_at":
            task_list = task_list.sorted_by_created_at(ascending=ascending)
        elif sort_by == "title":
            tasks = sorted(task_list.all(), key=lambda t: t.title.lower(), reverse=not ascending)
            return tasks

        return task_list.all()

    def get_overdue_tasks(self):
        """Return all overdue tasks sorted by due date (earliest first).

        Returns
        -------
        list[Task]
            Tasks that are past their due date and not in a terminal state.
        """
        return self.list_tasks(overdue_only=True, sort_by="due_date", ascending=True)

    def search_tasks(self, query: str):
        """Search tasks by title or description (case-insensitive).

        Parameters
        ----------
        query:
            The search string.

        Returns
        -------
        list[Task]
            Tasks whose title or description contains *query*.
        """
        return self.list_tasks(search_query=query, sort_by="created_at")

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> Dict[str, Any]:
        """Return a summary of task counts grouped by status and priority.

        Returns
        -------
        dict
            A dictionary with the following keys:

            ``total``
                Total number of tasks.
            ``by_status``
                Dict mapping each status value to its task count.
            ``by_priority``
                Dict mapping each priority value to its task count.
            ``overdue``
                Number of overdue tasks.
        """
        models = _get_models()
        task_list = self._load()

        by_status = task_list.summary()
        # Remove 'total' from by_status (we report it separately)
        total = by_status.pop("total", len(task_list))

        by_priority: Dict[str, int] = {p.value: 0 for p in models.Priority}
        for task in task_list:
            by_priority[task.priority.value] += 1

        overdue_count = len(task_list.filter_overdue())

        return {
            "total": total,
            "by_status": by_status,
            "by_priority": by_priority,
            "overdue": overdue_count,
        }

    # ------------------------------------------------------------------
    # Bulk operations
    # ------------------------------------------------------------------

    def bulk_complete(self, task_ids: List[str]) -> List:
        """Mark multiple tasks as done in a single operation.

        Tasks that cannot be found are silently skipped; the method returns
        only the tasks that were successfully updated.

        Parameters
        ----------
        task_ids:
            List of full UUIDs or short IDs.

        Returns
        -------
        list[Task]
            Tasks that were successfully marked as done.
        """
        updated = []
        for tid in task_ids:
            try:
                task = self._resolve_task(tid)
                task.mark_done()
                updated.append(task)
            except TaskNotFoundError:
                continue
        if updated:
            self._save()
        return updated

    def bulk_delete(self, task_ids: List[str]) -> List:
        """Delete multiple tasks in a single operation.

        Tasks that cannot be found are silently skipped.

        Parameters
        ----------
        task_ids:
            List of full UUIDs or short IDs.

        Returns
        -------
        list[Task]
            Tasks that were successfully deleted.
        """
        task_list = self._load()
        deleted = []
        for tid in task_ids:
            try:
                task = self._resolve_task(tid)
                task_list.remove(task.id)
                deleted.append(task)
            except (TaskNotFoundError, KeyError):
                continue
        if deleted:
            self._save()
        return deleted

    # ------------------------------------------------------------------
    # Storage helpers
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Force a reload of the task list from disk.

        Useful when the underlying storage file may have been modified
        externally (e.g. by another process).
        """
        self._task_list = None

    def backup(self, suffix: str = ".bak") -> pathlib.Path:
        """Create a backup of the storage file.

        Parameters
        ----------
        suffix:
            Suffix appended to the original filename.  Defaults to ``".bak"``.

        Returns
        -------
        pathlib.Path
            Path to the newly created backup file.

        Raises
        ------
        ~storage.StorageReadError
            If the storage file does not exist.
        ~storage.StorageWriteError
            If the backup cannot be written.
        """
        return self._storage.backup(suffix=suffix)

    @property
    def storage_path(self) -> pathlib.Path:
        """The path to the underlying storage file."""
        return self._storage.path

    @property
    def task_count(self) -> int:
        """The total number of tasks currently in the list."""
        return len(self._load())

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        count = len(self._task_list) if self._task_list is not None else "?"
        return f"TaskService(storage={self._storage!r}, tasks={count})"
