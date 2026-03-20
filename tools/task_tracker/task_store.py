"""
task_store.py — Data layer for the Task Tracker CLI.

Owns all persistence logic: reads and writes a JSON file (tasks.json) and
exposes a clean TaskStore class with methods for CRUD operations and filtering.

Each task is a plain dict with five fields:
    id         : UUID4 string
    title      : str
    status     : "todo" | "in_progress" | "done"
    priority   : "low" | "medium" | "high"
    created_at : ISO-8601 timestamp string
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_STATUSES = ("todo", "in_progress", "done")
VALID_PRIORITIES = ("low", "medium", "high")


# ---------------------------------------------------------------------------
# TaskStore
# ---------------------------------------------------------------------------


class TaskStore:
    """Manages task persistence in a JSON flat file.

    Parameters
    ----------
    store_path:
        Path to the JSON file used for persistence.  Defaults to
        ``tasks.json`` in the current working directory.  Passing a
        different path enables test isolation via temporary files.
    """

    def __init__(self, store_path: str = "tasks.json") -> None:
        self.store_path = store_path

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[dict]:
        """Load and return the task list from the JSON file.

        Returns an empty list if the file does not exist yet.
        """
        if not os.path.exists(self.store_path):
            return []
        with open(self.store_path, "r", encoding="utf-8") as fh:
            try:
                data = json.load(fh)
            except json.JSONDecodeError:
                # Corrupted file — start fresh rather than crashing.
                return []
        if not isinstance(data, list):
            return []
        return data

    def _save(self, tasks: list[dict]) -> None:
        """Persist *tasks* to the JSON file (pretty-printed, UTF-8)."""
        with open(self.store_path, "w", encoding="utf-8") as fh:
            json.dump(tasks, fh, indent=2, ensure_ascii=False)

    def _find_by_partial_id(self, partial_id: str, tasks: list[dict]) -> dict:
        """Return the single task whose ``id`` contains *partial_id*.

        Raises
        ------
        ValueError
            If zero or more than one task matches the partial ID.
        """
        partial_id = partial_id.strip()
        matches = [t for t in tasks if partial_id in t["id"]]
        if not matches:
            raise ValueError(
                f"No task found matching partial ID '{partial_id}'."
            )
        if len(matches) > 1:
            ids = ", ".join(t["id"] for t in matches)
            raise ValueError(
                f"Ambiguous partial ID '{partial_id}' matches {len(matches)} tasks: {ids}"
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
            Human-readable task title.  Must be non-empty.
        priority:
            One of ``"low"``, ``"medium"``, or ``"high"``.

        Returns
        -------
        dict
            The newly created task dict.

        Raises
        ------
        ValueError
            If *title* is blank or *priority* is invalid.
        """
        title = title.strip()
        if not title:
            raise ValueError("Task title must not be empty.")
        if priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority '{priority}'. Choose from: {', '.join(VALID_PRIORITIES)}"
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

        Parameters
        ----------
        status:
            If provided, only tasks with this status are returned.
        priority:
            If provided, only tasks with this priority are returned.

        Returns
        -------
        list[dict]
            Filtered (or full) task list, ordered by ``created_at``.

        Raises
        ------
        ValueError
            If *status* or *priority* is not a recognised value.
        """
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. Choose from: {', '.join(VALID_STATUSES)}"
            )
        if priority is not None and priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority '{priority}'. Choose from: {', '.join(VALID_PRIORITIES)}"
            )

        tasks = self._load()

        if status is not None:
            tasks = [t for t in tasks if t.get("status") == status]
        if priority is not None:
            tasks = [t for t in tasks if t.get("priority") == priority]

        # Sort by creation time (oldest first) for stable, predictable output.
        tasks.sort(key=lambda t: t.get("created_at", ""))
        return tasks

    def update_status(self, partial_id: str, new_status: str) -> dict:
        """Update the status of the task identified by *partial_id*.

        Parameters
        ----------
        partial_id:
            A prefix or substring of the target task's UUID.
        new_status:
            The desired new status.  Must be one of the valid statuses.

        Returns
        -------
        dict
            The updated task dict.

        Raises
        ------
        ValueError
            If *new_status* is invalid, or if *partial_id* matches zero or
            more than one task.
        """
        if new_status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{new_status}'. Choose from: {', '.join(VALID_STATUSES)}"
            )

        tasks = self._load()
        task = self._find_by_partial_id(partial_id, tasks)
        task["status"] = new_status
        self._save(tasks)
        return task

    def delete_task(self, partial_id: str) -> dict:
        """Delete the task identified by *partial_id* and return it.

        Parameters
        ----------
        partial_id:
            A prefix or substring of the target task's UUID.

        Returns
        -------
        dict
            The deleted task dict (snapshot before deletion).

        Raises
        ------
        ValueError
            If *partial_id* matches zero or more than one task.
        """
        tasks = self._load()
        task = self._find_by_partial_id(partial_id, tasks)
        tasks = [t for t in tasks if t["id"] != task["id"]]
        self._save(tasks)
        return task
