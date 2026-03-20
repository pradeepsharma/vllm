"""TaskStore: thin wrapper around Storage providing the task_store.py interface.

Reads/writes tasks to a local tasks.json file.
Each task has:
  - id         (uuid4 string)
  - title      (str)
  - status     ("todo" | "in_progress" | "done")
  - priority   ("low" | "medium" | "high")
  - created_at (ISO-8601 UTC timestamp)
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from models import Priority, Status, Task
from storage import NotFoundError, Storage, ValidationError

__all__ = ["TaskStore", "NotFoundError", "ValidationError"]

_DEFAULT_TASKS_JSON = os.path.join(
    os.path.expanduser("~"), ".taskman", "tasks.json"
)


class TaskStore:
    """Reads/writes tasks to a local tasks.json file.

    This class is a thin, ergonomic wrapper around :class:`~storage.Storage`
    that exposes a simplified interface for the Click+Rich CLI.  All JSON
    persistence, atomic writes, and parent-directory creation are delegated
    to ``Storage`` — no duplicate logic lives here.

    Parameters
    ----------
    path:
        Path to the JSON file.  Defaults to ``~/.taskman/tasks.json``.
        Pass a custom path (e.g. a tmp file) for testing so that tests
        remain isolated from the user's real data.
    """

    def __init__(self, path: str = _DEFAULT_TASKS_JSON) -> None:
        self._storage = Storage(data_path=path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_task(
        self,
        title: str,
        priority: str = "medium",
    ) -> Task:
        """Create a new task and persist it.

        Parameters
        ----------
        title:
            Human-readable task title (required, non-empty).
        priority:
            One of ``"low"``, ``"medium"``, ``"high"`` (default ``"medium"``).
            The value is case-insensitive.

        Returns
        -------
        Task
            The newly created :class:`~models.Task` instance.

        Raises
        ------
        ValidationError
            If *title* is empty or *priority* is not a valid value.
        ValueError
            If *priority* is not a recognised :class:`~models.Priority` member.
        """
        # Coerce to enum — raises ValueError for unrecognised values, which
        # callers (e.g. the CLI) can catch and surface as a user-friendly error.
        pri = Priority(priority.lower())
        return self._storage.create_task(title=title, priority=pri)

    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> List[Task]:
        """Return tasks, optionally filtered by status and/or priority.

        Parameters
        ----------
        status:
            Optional filter: ``"todo"``, ``"in_progress"``, or ``"done"``.
            The value is case-insensitive.
        priority:
            Optional filter: ``"low"``, ``"medium"``, ``"high"``, or
            ``"critical"``.  The value is case-insensitive.

        Returns
        -------
        List[Task]
            Matching tasks ordered by ``created_at`` ascending (oldest first).

        Raises
        ------
        ValueError
            If *status* or *priority* is not a recognised enum member.
        """
        status_enum = Status(status.lower()) if status else None
        priority_enum = Priority(priority.lower()) if priority else None
        return self._storage.list_tasks(
            status=status_enum,
            priority=priority_enum,
        )

    def update_status(self, task_id: str, new_status: str) -> Task:
        """Update the status of a task identified by *task_id*.

        Supports partial id matching: if *task_id* is a prefix of exactly
        one stored task id, that task is updated.

        Parameters
        ----------
        task_id:
            Full or partial UUID string.  At least one character is required.
        new_status:
            One of ``"todo"``, ``"in_progress"``, ``"done"``.
            The value is case-insensitive.

        Returns
        -------
        Task
            The updated :class:`~models.Task` instance.

        Raises
        ------
        NotFoundError
            If no task matches *task_id* or more than one task matches the
            given prefix.
        ValueError
            If *new_status* is not a recognised :class:`~models.Status` member.
        """
        resolved_id = self._resolve_id(task_id)
        status_enum = Status(new_status.lower())
        return self._storage.update_task(resolved_id, status=status_enum)

    def delete_task(self, task_id: str) -> None:
        """Delete a task by full or partial id.

        Parameters
        ----------
        task_id:
            Full or partial UUID string.  At least one character is required.

        Raises
        ------
        NotFoundError
            If no task matches *task_id* or more than one task matches the
            given prefix.
        """
        resolved_id = self._resolve_id(task_id)
        self._storage.delete_task(resolved_id)

    def stats(self) -> dict:
        """Return a summary dictionary with task counts.

        Delegates directly to :meth:`~storage.Storage.stats`.

        Returns
        -------
        dict
            A dict with the shape::

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
        return self._storage.stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_id(self, partial_id: str) -> str:
        """Return the full task id for a full or partial *partial_id*.

        Scans all stored tasks and returns the id of the unique task whose
        id starts with *partial_id*.

        Parameters
        ----------
        partial_id:
            A prefix (or full value) of a task UUID string.

        Returns
        -------
        str
            The full UUID of the uniquely matched task.

        Raises
        ------
        NotFoundError
            If zero tasks match the prefix (task does not exist) or if more
            than one task matches (prefix is ambiguous — user must supply
            more characters).
        """
        all_tasks = self._storage.list_tasks()
        matches = [t for t in all_tasks if t.id.startswith(partial_id)]
        if len(matches) == 1:
            return matches[0].id
        if len(matches) == 0:
            raise NotFoundError(
                f"No task found matching id prefix '{partial_id}'."
            )
        raise NotFoundError(
            f"Ambiguous id prefix '{partial_id}' matches {len(matches)} tasks. "
            "Please provide more characters."
        )
