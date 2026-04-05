"""
task_store.py — Data layer for the task tracker.

Provides the TaskStore class which handles all JSON persistence for tasks.
Each task is stored as a dictionary with the following fields:
  - id:         UUID4 string (unique identifier)
  - title:      str (human-readable task description)
  - status:     Literal["todo", "in_progress", "done"]
  - priority:   Literal["low", "medium", "high"]
  - created_at: ISO 8601 UTC timestamp string

The store reads the full JSON file on every operation and writes it back
atomically, keeping the implementation simple and reliable for a local CLI tool.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Literal, Optional, TypedDict


# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------

Status = Literal["todo", "in_progress", "done"]
Priority = Literal["low", "medium", "high"]

VALID_STATUSES: tuple[str, ...] = ("todo", "in_progress", "done")
VALID_PRIORITIES: tuple[str, ...] = ("low", "medium", "high")


class Task(TypedDict):
    """Typed dictionary representing a single task."""

    id: str
    title: str
    status: str
    priority: str
    created_at: str


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class TaskNotFoundError(Exception):
    """Raised when no task matches the given partial ID."""

    def __init__(self, partial_id: str) -> None:
        super().__init__(f"No task found matching ID prefix: '{partial_id}'")
        self.partial_id = partial_id


class AmbiguousTaskIDError(Exception):
    """Raised when multiple tasks match the given partial ID."""

    def __init__(self, partial_id: str, matches: list[str]) -> None:
        ids = ", ".join(matches)
        super().__init__(
            f"Ambiguous ID prefix '{partial_id}' matches multiple tasks: {ids}"
        )
        self.partial_id = partial_id
        self.matches = matches


class InvalidStatusError(Exception):
    """Raised when an invalid status value is provided."""

    def __init__(self, status: str) -> None:
        super().__init__(
            f"Invalid status '{status}'. Must be one of: {', '.join(VALID_STATUSES)}"
        )
        self.status = status


class InvalidPriorityError(Exception):
    """Raised when an invalid priority value is provided."""

    def __init__(self, priority: str) -> None:
        super().__init__(
            f"Invalid priority '{priority}'. Must be one of: {', '.join(VALID_PRIORITIES)}"
        )
        self.priority = priority


# ---------------------------------------------------------------------------
# TaskStore
# ---------------------------------------------------------------------------


class TaskStore:
    """
    Manages task persistence via a local JSON file.

    All mutating operations follow a read-modify-write pattern:
      1. Load the current task list from disk.
      2. Apply the requested change in memory.
      3. Write the updated list back to disk atomically.

    This keeps the implementation simple and correct for a single-user
    local CLI tool where concurrent access is not a concern.

    Args:
        filepath: Path to the JSON file used for storage.
                  Defaults to ``tasks.json`` in the current working directory.
    """

    DEFAULT_FILEPATH = "tasks.json"

    def __init__(self, filepath: Optional[str] = None) -> None:
        self.filepath: str = filepath if filepath is not None else self.DEFAULT_FILEPATH

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[Task]:
        """
        Read and return all tasks from the JSON file.

        Returns an empty list if the file does not exist yet.

        Raises:
            json.JSONDecodeError: If the file exists but contains invalid JSON.
            OSError: If the file cannot be read for reasons other than absence.
        """
        if not os.path.exists(self.filepath):
            return []

        with open(self.filepath, "r", encoding="utf-8") as fh:
            raw = fh.read().strip()

        # Treat an empty file the same as a missing file.
        if not raw:
            return []

        data = json.loads(raw)

        if not isinstance(data, list):
            raise ValueError(
                f"Expected a JSON array in '{self.filepath}', got {type(data).__name__}"
            )

        return data  # type: ignore[return-value]

    def _save(self, tasks: list[Task]) -> None:
        """
        Write the task list to the JSON file atomically.

        Uses a temporary file in the same directory as the target file so
        that the ``os.replace`` rename is atomic on POSIX systems.

        Args:
            tasks: The complete list of tasks to persist.

        Raises:
            OSError: If the file cannot be written.
        """
        target_dir = os.path.dirname(os.path.abspath(self.filepath))
        os.makedirs(target_dir, exist_ok=True)

        # Write to a temp file first, then atomically replace the target.
        fd, tmp_path = tempfile.mkstemp(dir=target_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(tasks, fh, indent=2, ensure_ascii=False)
                fh.write("\n")  # POSIX-friendly trailing newline
            os.replace(tmp_path, self.filepath)
        except Exception:
            # Clean up the temp file if anything goes wrong.
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def _find_task(self, partial_id: str, tasks: list[Task]) -> Task:
        """
        Locate a single task whose ``id`` starts with *partial_id*.

        Args:
            partial_id: A prefix of the task UUID to search for.
            tasks:      The list of tasks to search within.

        Returns:
            The matching Task dict.

        Raises:
            ValueError:          If *partial_id* is empty.
            TaskNotFoundError:   If no task matches.
            AmbiguousTaskIDError: If more than one task matches.
        """
        if not partial_id:
            raise ValueError("partial_id must not be empty")

        matches = [t for t in tasks if t["id"].startswith(partial_id)]

        if len(matches) == 0:
            raise TaskNotFoundError(partial_id)
        if len(matches) > 1:
            raise AmbiguousTaskIDError(partial_id, [t["id"] for t in matches])

        return matches[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_task(self, title: str, priority: str = "medium") -> Task:
        """
        Create a new task and persist it.

        Args:
            title:    Human-readable description of the task. Must be non-empty.
            priority: One of ``"low"``, ``"medium"``, or ``"high"``.
                      Defaults to ``"medium"``.

        Returns:
            The newly created Task dict.

        Raises:
            ValueError:        If *title* is empty or whitespace-only.
            InvalidPriorityError: If *priority* is not a valid value.
        """
        title = title.strip()
        if not title:
            raise ValueError("Task title must not be empty")

        if priority not in VALID_PRIORITIES:
            raise InvalidPriorityError(priority)

        task: Task = {
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
    ) -> list[Task]:
        """
        Return all tasks, optionally filtered by status and/or priority.

        Args:
            status:   If provided, only return tasks with this status.
                      Must be one of ``"todo"``, ``"in_progress"``, ``"done"``.
            priority: If provided, only return tasks with this priority.
                      Must be one of ``"low"``, ``"medium"``, ``"high"``.

        Returns:
            A list of Task dicts matching the given filters, ordered by
            ``created_at`` ascending (oldest first).

        Raises:
            InvalidStatusError:   If *status* is not a valid value.
            InvalidPriorityError: If *priority* is not a valid value.
        """
        if status is not None and status not in VALID_STATUSES:
            raise InvalidStatusError(status)
        if priority is not None and priority not in VALID_PRIORITIES:
            raise InvalidPriorityError(priority)

        tasks = self._load()

        if status is not None:
            tasks = [t for t in tasks if t["status"] == status]
        if priority is not None:
            tasks = [t for t in tasks if t["priority"] == priority]

        # Sort by creation time (ascending) for a stable, predictable order.
        tasks.sort(key=lambda t: t.get("created_at", ""))

        return tasks

    def update_status(self, partial_id: str, new_status: str) -> Task:
        """
        Change the status of a task identified by a partial ID prefix.

        Any status transition is permitted (see state machine in the design doc).

        Args:
            partial_id: A prefix of the target task's UUID.
            new_status: The desired new status. Must be one of
                        ``"todo"``, ``"in_progress"``, ``"done"``.

        Returns:
            The updated Task dict.

        Raises:
            InvalidStatusError:   If *new_status* is not a valid value.
            TaskNotFoundError:    If no task matches *partial_id*.
            AmbiguousTaskIDError: If multiple tasks match *partial_id*.
        """
        if new_status not in VALID_STATUSES:
            raise InvalidStatusError(new_status)

        tasks = self._load()
        task = self._find_task(partial_id, tasks)
        task["status"] = new_status
        self._save(tasks)

        return task

    def delete_task(self, partial_id: str) -> Task:
        """
        Remove a task identified by a partial ID prefix.

        Args:
            partial_id: A prefix of the target task's UUID.

        Returns:
            The deleted Task dict (as it existed before deletion).

        Raises:
            TaskNotFoundError:    If no task matches *partial_id*.
            AmbiguousTaskIDError: If multiple tasks match *partial_id*.
        """
        tasks = self._load()
        task = self._find_task(partial_id, tasks)
        tasks.remove(task)
        self._save(tasks)

        return task

    def get_stats(self) -> dict[str, int]:
        """
        Return a summary of task counts grouped by status.

        Returns:
            A dict with keys ``"total"``, ``"todo"``, ``"in_progress"``,
            ``"done"``, and one key per priority level (``"low"``,
            ``"medium"``, ``"high"``).
        """
        tasks = self._load()

        stats: dict[str, int] = {
            "total": len(tasks),
            "todo": 0,
            "in_progress": 0,
            "done": 0,
            "low": 0,
            "medium": 0,
            "high": 0,
        }

        for task in tasks:
            status = task.get("status", "")
            priority = task.get("priority", "")
            if status in stats:
                stats[status] += 1
            if priority in stats:
                stats[priority] += 1

        return stats
