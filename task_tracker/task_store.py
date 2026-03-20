"""Core data layer for the Task Tracker CLI application.

This module provides the :class:`TaskStore` class, which manages persistent
storage of tasks using a JSON file on disk.  It also defines the
:class:`Task` dataclass and the :class:`TaskStatus` enumeration that model
a single task record throughout the application.

Typical usage::

    store = TaskStore()                          # uses default ~/.task_tracker/tasks.json
    task  = store.add("Write unit tests", "high")
    store.update_status(task.id, TaskStatus.IN_PROGRESS)
    tasks = store.list_tasks(status=TaskStatus.IN_PROGRESS)
    store.delete(task.id)
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default directory that stores the tasks JSON file.
DEFAULT_STORE_DIR: Path = Path.home() / ".task_tracker"

#: Default filename for the tasks JSON file.
DEFAULT_STORE_FILE: str = "tasks.json"

#: ISO-8601 datetime format used when serialising / deserialising timestamps.
_DT_FORMAT: str = "%Y-%m-%dT%H:%M:%S.%f%z"

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class TaskStatus(str, Enum):
    """Lifecycle status of a task.

    Inheriting from :class:`str` means that a :class:`TaskStatus` value can
    be compared directly to plain strings and serialised to JSON without a
    custom encoder.
    """

    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_string(cls, value: str) -> "TaskStatus":
        """Return the matching :class:`TaskStatus` for *value* (case-insensitive).

        Args:
            value: A string representation of a status, e.g. ``"todo"``,
                ``"IN_PROGRESS"``, or ``"Done"``.

        Returns:
            The corresponding :class:`TaskStatus` member.

        Raises:
            ValueError: If *value* does not match any known status.
        """
        normalised = value.strip().lower()
        for member in cls:
            if member.value == normalised:
                return member
        valid = ", ".join(m.value for m in cls)
        raise ValueError(
            f"Unknown task status {value!r}. Valid values are: {valid}"
        )

    def label(self) -> str:
        """Return a human-readable label for display purposes."""
        return self.value.replace("_", " ").title()


class TaskPriority(str, Enum):
    """Priority level of a task.

    Inheriting from :class:`str` allows direct string comparison and
    JSON serialisation without a custom encoder.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_string(cls, value: str) -> "TaskPriority":
        """Return the matching :class:`TaskPriority` for *value* (case-insensitive).

        Args:
            value: A string representation of a priority, e.g. ``"high"``.

        Returns:
            The corresponding :class:`TaskPriority` member.

        Raises:
            ValueError: If *value* does not match any known priority.
        """
        normalised = value.strip().lower()
        for member in cls:
            if member.value == normalised:
                return member
        valid = ", ".join(m.value for m in cls)
        raise ValueError(
            f"Unknown task priority {value!r}. Valid values are: {valid}"
        )

    def label(self) -> str:
        """Return a human-readable label for display purposes."""
        return self.value.title()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """Represents a single task record.

    Attributes:
        id:          Unique identifier (UUID4 string).
        title:       Short description of the task.
        status:      Current lifecycle status.
        priority:    Importance level of the task.
        created_at:  UTC timestamp when the task was first created.
        updated_at:  UTC timestamp of the most recent modification.
        tags:        Optional list of free-form tag strings.
        notes:       Optional free-form notes / description.
    """

    id: str
    title: str
    status: TaskStatus
    priority: TaskPriority
    created_at: datetime
    updated_at: datetime
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise the task to a plain :class:`dict` suitable for JSON.

        Enum values are stored as their string ``value``; datetimes are
        stored as ISO-8601 strings with UTC timezone information.

        Returns:
            A JSON-serialisable dictionary representation of this task.
        """
        raw = asdict(self)
        raw["status"] = self.status.value
        raw["priority"] = self.priority.value
        raw["created_at"] = _format_dt(self.created_at)
        raw["updated_at"] = _format_dt(self.updated_at)
        return raw

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        """Deserialise a task from a plain :class:`dict` (e.g. loaded from JSON).

        Args:
            data: Dictionary with the same keys produced by :meth:`to_dict`.

        Returns:
            A fully populated :class:`Task` instance.

        Raises:
            KeyError:   If a required field is missing from *data*.
            ValueError: If a field value cannot be parsed (e.g. bad status).
        """
        return cls(
            id=data["id"],
            title=data["title"],
            status=TaskStatus.from_string(data["status"]),
            priority=TaskPriority.from_string(data["priority"]),
            created_at=_parse_dt(data["created_at"]),
            updated_at=_parse_dt(data["updated_at"]),
            tags=list(data.get("tags", [])),
            notes=data.get("notes", ""),
        )

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Task):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return (
            f"Task(id={self.id!r}, title={self.title!r}, "
            f"status={self.status.value!r}, priority={self.priority.value!r})"
        )


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class TaskNotFoundError(KeyError):
    """Raised when a task with the requested ID does not exist in the store."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        super().__init__(f"No task found with id {task_id!r}")

    def __str__(self) -> str:  # pragma: no cover
        return f"No task found with id {self.task_id!r}"


class TaskStoreError(OSError):
    """Raised when the backing JSON file cannot be read or written."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    """Return the current UTC time as a timezone-aware :class:`datetime`."""
    return datetime.now(tz=timezone.utc)


def _format_dt(dt: datetime) -> str:
    """Serialise *dt* to an ISO-8601 string.

    If *dt* is naive (no tzinfo), it is assumed to be UTC and the ``+00:00``
    suffix is appended before formatting.

    Args:
        dt: The datetime to format.

    Returns:
        An ISO-8601 string, e.g. ``"2026-03-20T12:34:56.789012+00:00"``.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _parse_dt(value: str) -> datetime:
    """Parse an ISO-8601 datetime string produced by :func:`_format_dt`.

    Args:
        value: An ISO-8601 string, optionally with timezone info.

    Returns:
        A timezone-aware :class:`datetime` in UTC.

    Raises:
        ValueError: If *value* cannot be parsed.
    """
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _new_id() -> str:
    """Generate a new unique task identifier (UUID4)."""
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# TaskStore
# ---------------------------------------------------------------------------


class TaskStore:
    """Persistent, file-backed store for :class:`Task` objects.

    Tasks are persisted as a JSON array in a single file on disk.  All
    mutating operations write the full file atomically (write to a temporary
    file then rename) so that a crash mid-write cannot corrupt existing data.

    Args:
        store_path: Path to the JSON file used for persistence.  Defaults to
            ``~/.task_tracker/tasks.json``.  Parent directories are created
            automatically on first write.

    Example::

        store = TaskStore()
        task  = store.add("Buy groceries")
        store.update_status(task.id, TaskStatus.DONE)
        for t in store.list_tasks():
            print(t.title)
    """

    def __init__(
        self,
        store_path: Optional[Path | str] = None,
    ) -> None:
        if store_path is None:
            self._path: Path = DEFAULT_STORE_DIR / DEFAULT_STORE_FILE
        else:
            self._path = Path(store_path)

        # In-memory cache: ordered dict keyed by task id.
        self._tasks: dict[str, Task] = {}
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        """The absolute path to the backing JSON file."""
        return self._path.resolve()

    # ------------------------------------------------------------------
    # CRUD operations
    # ------------------------------------------------------------------

    def add(
        self,
        title: str,
        priority: str | TaskPriority = TaskPriority.MEDIUM,
        tags: Optional[list[str]] = None,
        notes: str = "",
    ) -> Task:
        """Create a new task and persist it.

        Args:
            title:    Short description of the task.  Must be non-empty after
                      stripping whitespace.
            priority: Priority level — either a :class:`TaskPriority` member
                      or a plain string (``"low"``, ``"medium"``, ``"high"``).
                      Defaults to ``"medium"``.
            tags:     Optional list of tag strings.
            notes:    Optional free-form notes.

        Returns:
            The newly created :class:`Task`.

        Raises:
            ValueError:      If *title* is empty or *priority* is invalid.
            TaskStoreError:  If the backing file cannot be written.
        """
        title = title.strip()
        if not title:
            raise ValueError("Task title must not be empty.")

        if isinstance(priority, str):
            priority = TaskPriority.from_string(priority)

        now = _now_utc()
        task = Task(
            id=_new_id(),
            title=title,
            status=TaskStatus.TODO,
            priority=priority,
            created_at=now,
            updated_at=now,
            tags=list(tags) if tags else [],
            notes=notes,
        )

        self._ensure_loaded()
        self._tasks[task.id] = task
        self._save()
        return task

    def get(self, task_id: str) -> Task:
        """Retrieve a single task by its ID.

        Args:
            task_id: The UUID string of the task to retrieve.

        Returns:
            The matching :class:`Task`.

        Raises:
            TaskNotFoundError: If no task with *task_id* exists.
            TaskStoreError:    If the backing file cannot be read.
        """
        self._ensure_loaded()
        try:
            return self._tasks[task_id]
        except KeyError:
            raise TaskNotFoundError(task_id)

    def update(
        self,
        task_id: str,
        *,
        title: Optional[str] = None,
        priority: Optional[str | TaskPriority] = None,
        tags: Optional[list[str]] = None,
        notes: Optional[str] = None,
    ) -> Task:
        """Update one or more fields of an existing task.

        Only the keyword arguments that are explicitly provided (i.e. not
        ``None``) are applied.  ``updated_at`` is always refreshed.

        Args:
            task_id:  The UUID string of the task to update.
            title:    New title (must be non-empty if provided).
            priority: New priority level.
            tags:     New list of tags (replaces existing tags).
            notes:    New notes string.

        Returns:
            The updated :class:`Task`.

        Raises:
            TaskNotFoundError: If no task with *task_id* exists.
            ValueError:        If *title* is empty or *priority* is invalid.
            TaskStoreError:    If the backing file cannot be written.
        """
        self._ensure_loaded()
        task = self.get(task_id)

        if title is not None:
            title = title.strip()
            if not title:
                raise ValueError("Task title must not be empty.")
            task.title = title

        if priority is not None:
            if isinstance(priority, str):
                priority = TaskPriority.from_string(priority)
            task.priority = priority

        if tags is not None:
            task.tags = list(tags)

        if notes is not None:
            task.notes = notes

        task.updated_at = _now_utc()
        self._save()
        return task

    def update_status(self, task_id: str, status: str | TaskStatus) -> Task:
        """Change the status of an existing task.

        Args:
            task_id: The UUID string of the task to update.
            status:  New status — either a :class:`TaskStatus` member or a
                     plain string (``"todo"``, ``"in_progress"``, ``"done"``).

        Returns:
            The updated :class:`Task`.

        Raises:
            TaskNotFoundError: If no task with *task_id* exists.
            ValueError:        If *status* is invalid.
            TaskStoreError:    If the backing file cannot be written.
        """
        if isinstance(status, str):
            status = TaskStatus.from_string(status)

        self._ensure_loaded()
        task = self.get(task_id)
        task.status = status
        task.updated_at = _now_utc()
        self._save()
        return task

    def delete(self, task_id: str) -> Task:
        """Remove a task from the store permanently.

        Args:
            task_id: The UUID string of the task to delete.

        Returns:
            The :class:`Task` that was deleted.

        Raises:
            TaskNotFoundError: If no task with *task_id* exists.
            TaskStoreError:    If the backing file cannot be written.
        """
        self._ensure_loaded()
        if task_id not in self._tasks:
            raise TaskNotFoundError(task_id)
        task = self._tasks.pop(task_id)
        self._save()
        return task

    # ------------------------------------------------------------------
    # Query / listing
    # ------------------------------------------------------------------

    def list_tasks(
        self,
        *,
        status: Optional[str | TaskStatus] = None,
        priority: Optional[str | TaskPriority] = None,
        tags: Optional[list[str]] = None,
        search: Optional[str] = None,
    ) -> list[Task]:
        """Return a filtered, sorted list of tasks.

        All filter arguments are optional and combinable.  Tasks are returned
        sorted by priority (high → medium → low) and then by creation time
        (oldest first).

        Args:
            status:   Only include tasks with this status.
            priority: Only include tasks with this priority.
            tags:     Only include tasks that have *all* of the given tags.
            search:   Case-insensitive substring match against the task title
                      and notes fields.

        Returns:
            A list of :class:`Task` objects matching all supplied filters.

        Raises:
            ValueError:     If *status* or *priority* is an invalid string.
            TaskStoreError: If the backing file cannot be read.
        """
        self._ensure_loaded()

        # Normalise filter arguments.
        status_filter: Optional[TaskStatus] = None
        if status is not None:
            status_filter = (
                status if isinstance(status, TaskStatus)
                else TaskStatus.from_string(status)
            )

        priority_filter: Optional[TaskPriority] = None
        if priority is not None:
            priority_filter = (
                priority if isinstance(priority, TaskPriority)
                else TaskPriority.from_string(priority)
            )

        tag_set: Optional[set[str]] = (
            {t.lower() for t in tags} if tags else None
        )

        search_lower: Optional[str] = (
            search.lower() if search else None
        )

        results: list[Task] = []
        for task in self._tasks.values():
            if status_filter is not None and task.status != status_filter:
                continue
            if priority_filter is not None and task.priority != priority_filter:
                continue
            if tag_set is not None:
                task_tags = {t.lower() for t in task.tags}
                if not tag_set.issubset(task_tags):
                    continue
            if search_lower is not None:
                haystack = (task.title + " " + task.notes).lower()
                if search_lower not in haystack:
                    continue
            results.append(task)

        # Sort: priority descending (high=0, medium=1, low=2), then created_at ascending.
        _priority_order = {
            TaskPriority.HIGH: 0,
            TaskPriority.MEDIUM: 1,
            TaskPriority.LOW: 2,
        }
        results.sort(
            key=lambda t: (_priority_order[t.priority], t.created_at)
        )
        return results

    def count(
        self,
        *,
        status: Optional[str | TaskStatus] = None,
    ) -> int:
        """Return the number of tasks, optionally filtered by *status*.

        Args:
            status: If provided, count only tasks with this status.

        Returns:
            Integer count of matching tasks.

        Raises:
            ValueError:     If *status* is an invalid string.
            TaskStoreError: If the backing file cannot be read.
        """
        return len(self.list_tasks(status=status))

    def all_tags(self) -> list[str]:
        """Return a sorted list of all unique tags across all tasks.

        Returns:
            Alphabetically sorted list of tag strings.

        Raises:
            TaskStoreError: If the backing file cannot be read.
        """
        self._ensure_loaded()
        tag_set: set[str] = set()
        for task in self._tasks.values():
            tag_set.update(task.tags)
        return sorted(tag_set)

    def __iter__(self) -> Iterator[Task]:
        """Iterate over all tasks in insertion order."""
        self._ensure_loaded()
        yield from self._tasks.values()

    def __len__(self) -> int:
        """Return the total number of tasks in the store."""
        self._ensure_loaded()
        return len(self._tasks)

    def __contains__(self, task_id: str) -> bool:
        """Return ``True`` if a task with *task_id* exists in the store."""
        self._ensure_loaded()
        return task_id in self._tasks

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """Force a reload of tasks from disk, discarding any in-memory state.

        This is useful when multiple processes share the same backing file and
        you need to pick up changes made by another process.

        Raises:
            TaskStoreError: If the backing file exists but cannot be read or
                parsed.
        """
        self._loaded = False
        self._tasks = {}
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        """Load tasks from disk if they have not been loaded yet."""
        if not self._loaded:
            self._load()

    def _load(self) -> None:
        """Read and parse the backing JSON file into ``self._tasks``.

        If the file does not exist the store starts empty.  If the file exists
        but is malformed a :class:`TaskStoreError` is raised so the caller can
        surface a meaningful error message rather than a raw JSON exception.

        Raises:
            TaskStoreError: If the file exists but cannot be read or parsed.
        """
        self._tasks = {}
        if not self._path.exists():
            self._loaded = True
            return

        try:
            raw_text = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise TaskStoreError(
                f"Cannot read task store at {self._path}: {exc}"
            ) from exc

        try:
            raw_list: list[dict] = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise TaskStoreError(
                f"Task store at {self._path} contains invalid JSON: {exc}"
            ) from exc

        if not isinstance(raw_list, list):
            raise TaskStoreError(
                f"Task store at {self._path} is malformed: "
                "expected a JSON array at the top level."
            )

        for item in raw_list:
            try:
                task = Task.from_dict(item)
            except (KeyError, ValueError) as exc:
                raise TaskStoreError(
                    f"Task store at {self._path} contains an invalid task "
                    f"record: {exc}"
                ) from exc
            self._tasks[task.id] = task

        self._loaded = True

    def _save(self) -> None:
        """Serialise ``self._tasks`` and write them to the backing JSON file.

        The write is performed atomically: the data is first written to a
        temporary file in the same directory, then renamed over the target
        file.  This prevents partial writes from corrupting the store.

        Raises:
            TaskStoreError: If the directory cannot be created or the file
                cannot be written.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise TaskStoreError(
                f"Cannot create task store directory {self._path.parent}: {exc}"
            ) from exc

        payload = json.dumps(
            [task.to_dict() for task in self._tasks.values()],
            indent=2,
            ensure_ascii=False,
        )

        # Write atomically via a sibling temp file.
        tmp_path = self._path.with_suffix(".tmp")
        try:
            tmp_path.write_text(payload, encoding="utf-8")
            os.replace(tmp_path, self._path)
        except OSError as exc:
            # Best-effort cleanup of the temp file.
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise TaskStoreError(
                f"Cannot write task store to {self._path}: {exc}"
            ) from exc
