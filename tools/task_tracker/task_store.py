"""
task_store.py — Core data layer for the CLI Task Tracker.

Provides the TaskStore class that persists tasks as JSON on disk.
Each task is a plain dict with five fields:
    id, title, status, priority, created_at
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

VALID_STATUSES: tuple[str, ...] = ("todo", "in_progress", "done")
VALID_PRIORITIES: tuple[str, ...] = ("low", "medium", "high")


# ---------------------------------------------------------------------------
# TaskStore
# ---------------------------------------------------------------------------


class TaskStore:
    """Manages a JSON-backed list of task dictionaries.

    Parameters
    ----------
    store_path:
        Path to the JSON file used for persistence.  Defaults to
        ``"tasks.json"`` in the current working directory.  Pass a
        different path (e.g. a ``tmp_path`` fixture) to isolate tests.
    """

    def __init__(self, store_path: str = "tasks.json") -> None:
        self.store_path = Path(store_path)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[dict]:
        """Load tasks from disk.

        Returns an empty list when the file does not exist, cannot be
        parsed as JSON, or its root value is not a JSON array.
        """
        if not self.store_path.exists():
            return []

        try:
            with self.store_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError:
            # Corrupted file — start fresh rather than crashing.
            return []

        if not isinstance(data, list):
            return []

        return data

    def _save(self, tasks: list[dict]) -> None:
        """Persist *tasks* to disk as pretty-printed JSON.

        The parent directory is created automatically if it does not
        already exist.
        """
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        with self.store_path.open("w", encoding="utf-8") as fh:
            json.dump(tasks, fh, indent=2, ensure_ascii=False)

    def _find_by_partial_id(self, partial_id: str, tasks: list[dict]) -> dict:
        """Return the unique task whose ``id`` contains *partial_id*.

        Parameters
        ----------
        partial_id:
            A substring of the full UUID (e.g. the first 8 characters).
        tasks:
            The current in-memory task list.

        Raises
        ------
        ValueError
            If no task matches, or if more than one task matches
            (ambiguous partial ID).
        """
        partial_id = partial_id.strip()
        matches = [t for t in tasks if partial_id in t["id"]]

        if len(matches) == 0:
            raise ValueError(
                f"No task found matching partial ID '{partial_id}'."
            )

        if len(matches) > 1:
            ids = ", ".join(t["id"] for t in matches)
            raise ValueError(
                f"Ambiguous partial ID '{partial_id}' matches "
                f"{len(matches)} tasks: {ids}"
            )

        return matches[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_task(self, title: str, priority: str = "medium") -> dict:
        """Create a new task and persist it.

        Parameters
        ----------
        title:
            Human-readable description of the task.  Leading/trailing
            whitespace is stripped.  Must not be empty after stripping.
        priority:
            One of ``"low"``, ``"medium"``, or ``"high"``.

        Returns
        -------
        dict
            The newly created task dictionary.

        Raises
        ------
        ValueError
            If *title* is blank or *priority* is not a valid value.
        """
        title = title.strip()
        if not title:
            raise ValueError("Task title must not be empty.")

        if priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority '{priority}'. "
                f"Must be one of: {', '.join(VALID_PRIORITIES)}."
            )

        task: dict = {
            "id": str(uuid.uuid4()),
            "title": title,
            "status": "todo",
            "priority": priority,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        tasks = self._load()
        tasks.append(task)
        self._save(tasks)

        return task

    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> list[dict]:
        """Return tasks, optionally filtered by *status* and/or *priority*.

        Results are sorted by ``created_at`` in ascending order (oldest
        first).

        Parameters
        ----------
        status:
            When provided, only tasks with this status are returned.
            Must be one of ``VALID_STATUSES``.
        priority:
            When provided, only tasks with this priority are returned.
            Must be one of ``VALID_PRIORITIES``.

        Returns
        -------
        list[dict]
            Filtered and sorted task list.

        Raises
        ------
        ValueError
            If *status* or *priority* is not a recognised value.
        """
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. "
                f"Must be one of: {', '.join(VALID_STATUSES)}."
            )

        if priority is not None and priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority '{priority}'. "
                f"Must be one of: {', '.join(VALID_PRIORITIES)}."
            )

        tasks = self._load()

        if status is not None:
            tasks = [t for t in tasks if t.get("status") == status]

        if priority is not None:
            tasks = [t for t in tasks if t.get("priority") == priority]

        tasks.sort(key=lambda t: t.get("created_at", ""))

        return tasks

    def update_status(self, partial_id: str, new_status: str) -> dict:
        """Change the status of a task identified by *partial_id*.

        Parameters
        ----------
        partial_id:
            A substring of the target task's UUID.
        new_status:
            The desired new status.  Must be one of ``VALID_STATUSES``.

        Returns
        -------
        dict
            The updated task dictionary (same object reference as stored).

        Raises
        ------
        ValueError
            If *new_status* is invalid, or if *partial_id* matches zero
            or more than one task.
        """
        if new_status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{new_status}'. "
                f"Must be one of: {', '.join(VALID_STATUSES)}."
            )

        tasks = self._load()
        task = self._find_by_partial_id(partial_id, tasks)
        task["status"] = new_status
        self._save(tasks)

        return task

    def delete_task(self, partial_id: str) -> dict:
        """Remove a task identified by *partial_id* from the store.

        Parameters
        ----------
        partial_id:
            A substring of the target task's UUID.

        Returns
        -------
        dict
            A snapshot of the deleted task dictionary.

        Raises
        ------
        ValueError
            If *partial_id* matches zero or more than one task.
        """
        tasks = self._load()
        task = self._find_by_partial_id(partial_id, tasks)

        remaining = [t for t in tasks if t["id"] != task["id"]]
        self._save(remaining)

        return task
