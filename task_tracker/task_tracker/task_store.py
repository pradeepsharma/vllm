"""
task_store.py — Data layer for the task tracker.

Provides the TaskStore class, which owns all JSON persistence and in-memory
task management.  Every public method that mutates state immediately flushes
the updated task list back to disk so the file is always consistent.

Custom exceptions
-----------------
TaskNotFoundError   — raised when no task matches the supplied partial ID.
AmbiguousIDError    — raised when more than one task matches the partial ID.

Task schema (plain dict)
------------------------
{
    "id":         str   # UUID4 hex string
    "title":      str   # free-form task description
    "status":     str   # "todo" | "in_progress" | "done"
    "priority":   str   # "low" | "medium" | "high"
    "created_at": str   # ISO-8601 UTC timestamp, e.g. "2026-03-24T12:00:00.000000"
}

Environment variable
--------------------
TASK_TRACKER_FILE — override the default storage path.
Default: ~/.task_tracker/tasks.json
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Default storage path
# ---------------------------------------------------------------------------

_DEFAULT_PATH = Path.home() / ".task_tracker" / "tasks.json"

# ---------------------------------------------------------------------------
# Valid field values (mirrors Click Choice constraints in cli.py)
# ---------------------------------------------------------------------------

VALID_STATUSES: frozenset[str] = frozenset({"todo", "in_progress", "done"})
VALID_PRIORITIES: frozenset[str] = frozenset({"low", "medium", "high"})


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class TaskNotFoundError(Exception):
    """Raised when no task matches the supplied partial ID."""

    def __init__(self, partial_id: str) -> None:
        self.partial_id = partial_id
        super().__init__(f"No task found matching ID fragment '{partial_id}'.")


class AmbiguousIDError(Exception):
    """Raised when more than one task matches the supplied partial ID."""

    def __init__(self, partial_id: str, candidates: list[dict]) -> None:
        self.partial_id = partial_id
        self.candidates = candidates
        ids = ", ".join(t["id"] for t in candidates)
        super().__init__(
            f"Partial ID '{partial_id}' is ambiguous — matches: {ids}"
        )


# ---------------------------------------------------------------------------
# TaskStore
# ---------------------------------------------------------------------------


class TaskStore:
    """Manages a flat JSON file of task dicts.

    Parameters
    ----------
    path:
        Path to the JSON storage file.  Defaults to the value of the
        ``TASK_TRACKER_FILE`` environment variable, or
        ``~/.task_tracker/tasks.json`` if the variable is not set.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        if path is None:
            env_path = os.environ.get("TASK_TRACKER_FILE")
            path = Path(env_path) if env_path else _DEFAULT_PATH
        self._path = Path(path)
        self._tasks: list[dict] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_task(self, title: str, priority: str = "medium") -> dict:
        """Create a new task and persist it.

        Parameters
        ----------
        title:
            Human-readable task description.  Must be a non-empty string.
        priority:
            One of ``"low"``, ``"medium"``, or ``"high"``.

        Returns
        -------
        dict
            The newly created task dict.

        Raises
        ------
        ValueError
            If *title* is blank or *priority* is not a recognised value.
        """
        title = title.strip()
        if not title:
            raise ValueError("Task title must not be empty.")
        if priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority '{priority}'. "
                f"Choose from: {', '.join(sorted(VALID_PRIORITIES))}."
            )

        task: dict = {
            "id": uuid.uuid4().hex,
            "title": title,
            "status": "todo",
            "priority": priority,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._tasks.append(task)
        self._save()
        return task

    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> list[dict]:
        """Return tasks, optionally filtered by status and/or priority.

        Parameters
        ----------
        status:
            If provided, only tasks whose ``status`` field equals this value
            are returned.  Must be one of ``"todo"``, ``"in_progress"``,
            ``"done"``, or ``None`` (no filter).
        priority:
            If provided, only tasks whose ``priority`` field equals this value
            are returned.  Must be one of ``"low"``, ``"medium"``, ``"high"``,
            or ``None`` (no filter).

        Returns
        -------
        list[dict]
            Filtered (or full) list of task dicts, in insertion order.

        Raises
        ------
        ValueError
            If *status* or *priority* is not a recognised value.
        """
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. "
                f"Choose from: {', '.join(sorted(VALID_STATUSES))}."
            )
        if priority is not None and priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority '{priority}'. "
                f"Choose from: {', '.join(sorted(VALID_PRIORITIES))}."
            )

        result = self._tasks
        if status is not None:
            result = [t for t in result if t["status"] == status]
        if priority is not None:
            result = [t for t in result if t["priority"] == priority]
        return list(result)

    def update_status(self, partial_id: str, status: str) -> dict:
        """Change the status of a task identified by a partial ID.

        Parameters
        ----------
        partial_id:
            A prefix or substring of the task's UUID hex string.
        status:
            The new status value.  Must be one of ``"todo"``,
            ``"in_progress"``, or ``"done"``.

        Returns
        -------
        dict
            The updated task dict.

        Raises
        ------
        ValueError
            If *status* is not a recognised value.
        TaskNotFoundError
            If no task matches *partial_id*.
        AmbiguousIDError
            If more than one task matches *partial_id*.
        """
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. "
                f"Choose from: {', '.join(sorted(VALID_STATUSES))}."
            )
        task = self._match(partial_id)
        task["status"] = status
        self._save()
        return task

    def delete_task(self, partial_id: str) -> dict:
        """Remove a task identified by a partial ID.

        Parameters
        ----------
        partial_id:
            A prefix or substring of the task's UUID hex string.

        Returns
        -------
        dict
            The deleted task dict (snapshot before removal).

        Raises
        ------
        TaskNotFoundError
            If no task matches *partial_id*.
        AmbiguousIDError
            If more than one task matches *partial_id*.
        """
        task = self._match(partial_id)
        self._tasks = [t for t in self._tasks if t["id"] != task["id"]]
        self._save()
        return task

    def get_stats(self) -> dict:
        """Return aggregate statistics about the current task list.

        Returns
        -------
        dict
            A dict with the following keys:

            * ``"total"``       — total number of tasks (int)
            * ``"by_status"``   — mapping of status → count (dict[str, int])
            * ``"by_priority"`` — mapping of priority → count (dict[str, int])
        """
        by_status: dict[str, int] = {s: 0 for s in sorted(VALID_STATUSES)}
        by_priority: dict[str, int] = {p: 0 for p in sorted(VALID_PRIORITIES)}

        for task in self._tasks:
            by_status[task["status"]] = by_status.get(task["status"], 0) + 1
            by_priority[task["priority"]] = (
                by_priority.get(task["priority"], 0) + 1
            )

        return {
            "total": len(self._tasks),
            "by_status": by_status,
            "by_priority": by_priority,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[dict]:
        """Read tasks from the JSON file.

        If the file does not exist an empty list is returned.  If the file
        exists but is malformed, a ``json.JSONDecodeError`` is propagated to
        the caller.

        Returns
        -------
        list[dict]
            The list of task dicts stored on disk.
        """
        if not self._path.exists():
            return []
        with self._path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array in '{self._path}', "
                f"got {type(data).__name__}."
            )
        return data

    def _save(self) -> None:
        """Flush the in-memory task list to the JSON file.

        The parent directory is created automatically if it does not exist.
        The file is written atomically via a temporary name on the same
        filesystem, then renamed, to avoid partial writes corrupting the store.
        """
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self._path.with_suffix(".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                json.dump(self._tasks, fh, indent=2, ensure_ascii=False)
                fh.write("\n")  # POSIX-friendly trailing newline
            tmp_path.replace(self._path)
        except Exception:
            # Clean up the temp file if anything goes wrong.
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _match(self, partial_id: str) -> dict:
        """Resolve a partial ID to exactly one task.

        Matching strategy (in order):
        1. Exact match on ``task["id"]``.
        2. Prefix match — ``task["id"].startswith(partial_id)``.
        3. Substring match — ``partial_id in task["id"]``.

        If exactly one task is found at any stage, it is returned immediately.
        If multiple tasks match, ``AmbiguousIDError`` is raised.
        If no task matches after all three stages, ``TaskNotFoundError`` is
        raised.

        Parameters
        ----------
        partial_id:
            A prefix, substring, or full UUID hex string.

        Returns
        -------
        dict
            The single matching task dict.

        Raises
        ------
        TaskNotFoundError
            If no task matches *partial_id*.
        AmbiguousIDError
            If more than one task matches *partial_id*.
        """
        partial_id = partial_id.strip()

        # Stage 1 — exact match
        exact = [t for t in self._tasks if t["id"] == partial_id]
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            # Theoretically impossible with UUID4, but be defensive.
            raise AmbiguousIDError(partial_id, exact)

        # Stage 2 — prefix match
        prefix = [t for t in self._tasks if t["id"].startswith(partial_id)]
        if len(prefix) == 1:
            return prefix[0]
        if len(prefix) > 1:
            raise AmbiguousIDError(partial_id, prefix)

        # Stage 3 — substring match
        substr = [t for t in self._tasks if partial_id in t["id"]]
        if len(substr) == 1:
            return substr[0]
        if len(substr) > 1:
            raise AmbiguousIDError(partial_id, substr)

        raise TaskNotFoundError(partial_id)
