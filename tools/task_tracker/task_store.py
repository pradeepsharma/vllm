"""
task_store.py — Data layer for the CLI task tracker.

Provides the TaskStore class which persists tasks as a flat JSON array
in a file on disk. All writes are atomic (write to .tmp, then rename)
to prevent data corruption on crash.

Task schema (dict):
    id         : str  — UUID4 string
    title      : str  — human-readable description
    status     : str  — one of VALID_STATUSES
    priority   : str  — one of VALID_PRIORITIES
    created_at : str  — ISO 8601 UTC timestamp
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

VALID_STATUSES: set[str] = {"todo", "in_progress", "done"}
VALID_PRIORITIES: set[str] = {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# TaskStore
# ---------------------------------------------------------------------------


class TaskStore:
    """Manages a collection of tasks persisted to a JSON file.

    Parameters
    ----------
    filepath:
        Path to the JSON file used for persistence.  Defaults to
        ``"tasks.json"`` in the current working directory.  The file
        (and any missing parent directories) are created automatically
        on the first write.
    """

    def __init__(self, filepath: str | Path = "tasks.json") -> None:
        self.filepath: Path = Path(filepath)
        self._tasks: list[dict] = []
        self._load()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Read tasks from *self.filepath*.

        If the file does not exist, or its contents are not valid JSON,
        ``self._tasks`` is initialised to an empty list (a fresh store).
        """
        if not self.filepath.exists():
            self._tasks = []
            return

        try:
            raw = self.filepath.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, list):
                self._tasks = data
            else:
                # Unexpected format — start fresh rather than crash.
                self._tasks = []
        except json.JSONDecodeError:
            self._tasks = []

    def _save(self) -> None:
        """Atomically write *self._tasks* to *self.filepath*.

        The data is first written to a sibling ``.tmp`` file, then
        renamed over the target path.  This guarantees that a crash
        mid-write cannot leave a partially-written (corrupt) JSON file.
        """
        # Ensure parent directories exist.
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = self.filepath.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(self._tasks, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Atomic rename — on POSIX this is guaranteed atomic; on Windows
        # Path.replace() is used which is as close as the stdlib gets.
        tmp_path.replace(self.filepath)

    def _find_by_partial_id(self, partial_id: str) -> Optional[dict]:
        """Return the unique task whose ``id`` starts with *partial_id*.

        Parameters
        ----------
        partial_id:
            A prefix of a task UUID (e.g. ``"a3f2504e"``).

        Returns
        -------
        dict | None
            The matching task dict, or ``None`` if no task matches.

        Raises
        ------
        ValueError
            If more than one task matches the given prefix (ambiguous).
        """
        matches = [t for t in self._tasks if t["id"].startswith(partial_id)]

        if len(matches) > 1:
            ids = ", ".join(t["id"] for t in matches)
            raise ValueError(
                f"Ambiguous task prefix '{partial_id}' matches multiple tasks: {ids}"
            )

        return matches[0] if matches else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_task(self, title: str, priority: str = "medium") -> dict:
        """Create a new task and persist it.

        Parameters
        ----------
        title:
            Human-readable description of the task.
        priority:
            One of ``"low"``, ``"medium"``, or ``"high"``.
            Defaults to ``"medium"``.

        Returns
        -------
        dict
            The newly created task dict.

        Raises
        ------
        ValueError
            If *priority* is not a member of :data:`VALID_PRIORITIES`.
        """
        if priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority '{priority}'. "
                f"Must be one of: {', '.join(sorted(VALID_PRIORITIES))}"
            )

        task: dict = {
            "id": str(uuid.uuid4()),
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
        """Return tasks, optionally filtered by *status* and/or *priority*.

        Parameters
        ----------
        status:
            If provided, only tasks with this status are returned.
            Must be one of :data:`VALID_STATUSES`.
        priority:
            If provided, only tasks with this priority are returned.
            Must be one of :data:`VALID_PRIORITIES`.

        Returns
        -------
        list[dict]
            A (possibly empty) list of matching task dicts.  The list is
            a shallow copy — mutating it does not affect the store.

        Raises
        ------
        ValueError
            If *status* or *priority* is provided but invalid.
        """
        if status is not None and status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. "
                f"Must be one of: {', '.join(sorted(VALID_STATUSES))}"
            )
        if priority is not None and priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority '{priority}'. "
                f"Must be one of: {', '.join(sorted(VALID_PRIORITIES))}"
            )

        result = list(self._tasks)  # shallow copy

        if status is not None:
            result = [t for t in result if t["status"] == status]
        if priority is not None:
            result = [t for t in result if t["priority"] == priority]

        return result

    def update_status(self, partial_id: str, new_status: str) -> dict:
        """Change the status of a task identified by a partial ID prefix.

        Parameters
        ----------
        partial_id:
            A prefix of the target task's UUID.
        new_status:
            The desired new status.  Must be one of :data:`VALID_STATUSES`.

        Returns
        -------
        dict
            The updated task dict (reflects the new status).

        Raises
        ------
        ValueError
            If *new_status* is invalid, or if *partial_id* is ambiguous.
        KeyError
            If no task matches *partial_id*.
        """
        if new_status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{new_status}'. "
                f"Must be one of: {', '.join(sorted(VALID_STATUSES))}"
            )

        task = self._find_by_partial_id(partial_id)
        if task is None:
            raise KeyError(f"No task found with id prefix '{partial_id}'")

        task["status"] = new_status
        self._save()
        return task

    def delete_task(self, partial_id: str) -> dict:
        """Remove a task identified by a partial ID prefix.

        Parameters
        ----------
        partial_id:
            A prefix of the target task's UUID.

        Returns
        -------
        dict
            The deleted task dict (as it existed before deletion).

        Raises
        ------
        ValueError
            If *partial_id* is ambiguous (matches more than one task).
        KeyError
            If no task matches *partial_id*.
        """
        task = self._find_by_partial_id(partial_id)
        if task is None:
            raise KeyError(f"No task found with id prefix '{partial_id}'")

        self._tasks.remove(task)
        self._save()
        return task
