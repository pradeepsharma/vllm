"""
task_store.py — Data layer for the task tracker CLI.

Provides the TaskStore class which handles all JSON I/O and business logic:
  - Reading and writing tasks.json atomically
  - Generating UUIDs and ISO timestamps for new tasks
  - Filtering tasks by status and/or priority
  - Partial-UUID matching (git-style short IDs)
  - CRUD operations: add, list, update_status, delete
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Valid domain values
# ---------------------------------------------------------------------------

VALID_STATUSES = {"todo", "in_progress", "done"}
VALID_PRIORITIES = {"low", "medium", "high"}


class TaskStore:
    """Manages task persistence in a local JSON flat file.

    Each task is stored as a plain Python dict with the following schema::

        {
            "id":         str,   # UUID4 hex string
            "title":      str,   # human-readable task title
            "status":     str,   # one of: "todo", "in_progress", "done"
            "priority":   str,   # one of: "low", "medium", "high"
            "created_at": str,   # ISO 8601 UTC timestamp
        }

    The entire file is read on every operation and written back atomically
    (write to a temp file, then rename) to avoid partial-write corruption.
    This is intentionally simple and correct for a single-user CLI tool.
    """

    def __init__(self, filepath: str = "tasks.json") -> None:
        """Initialise the store with the path to the JSON data file.

        Args:
            filepath: Path to the JSON file used for persistence.
                      Defaults to ``"tasks.json"`` in the current directory.
        """
        self.filepath = str(filepath)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> list[dict]:
        """Read and return all tasks from the JSON file.

        Returns an empty list if the file does not exist yet.

        Returns:
            A list of task dicts.

        Raises:
            ValueError: If the file exists but contains invalid JSON or its
                        top-level value is not a list.
        """
        path = Path(self.filepath)
        if not path.exists():
            return []

        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise OSError(f"Cannot read tasks file '{self.filepath}': {exc}") from exc

        if not raw.strip():
            return []

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"tasks file '{self.filepath}' contains invalid JSON: {exc}"
            ) from exc

        if not isinstance(data, list):
            raise ValueError(
                f"tasks file '{self.filepath}' must contain a JSON array at the top level"
            )

        return data

    def _save(self, tasks: list[dict]) -> None:
        """Write the task list back to the JSON file atomically.

        The data is first written to a sibling temp file, then renamed into
        place so that a crash mid-write never leaves a corrupt file.

        Args:
            tasks: The full list of task dicts to persist.

        Raises:
            OSError: If the file cannot be written.
        """
        path = Path(self.filepath)
        # Ensure the parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = path.with_suffix(".tmp")
        try:
            tmp_path.write_text(
                json.dumps(tasks, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            # Atomic rename (POSIX) / best-effort on Windows
            os.replace(tmp_path, path)
        except OSError as exc:
            # Clean up the temp file if something went wrong
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
            raise OSError(f"Cannot write tasks file '{self.filepath}': {exc}") from exc

    def _find_task(self, tasks: list[dict], partial_id: str) -> dict:
        """Locate a single task by a partial (prefix) UUID match.

        Matching is case-insensitive and uses ``str.startswith()`` on the
        stored UUID string, mirroring the familiar git short-SHA UX.

        Args:
            tasks:      The full list of task dicts to search.
            partial_id: A prefix of the task UUID to match against.

        Returns:
            The single matching task dict.

        Raises:
            ValueError: If ``partial_id`` is empty, matches no tasks, or
                        matches more than one task (ambiguous prefix).
        """
        if not partial_id or not partial_id.strip():
            raise ValueError("partial_id must not be empty")

        needle = partial_id.strip().lower()
        matches = [t for t in tasks if t["id"].lower().startswith(needle)]

        if len(matches) == 0:
            raise ValueError(f"No task found matching id prefix '{partial_id}'")
        if len(matches) > 1:
            ids = ", ".join(t["id"][:8] for t in matches)
            raise ValueError(
                f"Ambiguous id prefix '{partial_id}' matches {len(matches)} tasks: {ids}"
            )

        return matches[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_task(self, title: str, priority: str = "medium") -> dict:
        """Create a new task and persist it.

        Args:
            title:    Human-readable task title. Must be a non-empty string.
            priority: One of ``"low"``, ``"medium"``, ``"high"``.
                      Defaults to ``"medium"``.

        Returns:
            The newly created task dict (including its generated ``id`` and
            ``created_at`` fields).

        Raises:
            ValueError: If ``title`` is empty or ``priority`` is invalid.
        """
        title = title.strip() if title else ""
        if not title:
            raise ValueError("Task title must not be empty")

        priority = priority.strip().lower() if priority else ""
        if priority not in VALID_PRIORITIES:
            raise ValueError(
                f"Invalid priority '{priority}'. Must be one of: "
                + ", ".join(sorted(VALID_PRIORITIES))
            )

        task: dict = {
            "id": str(uuid.uuid4()),
            "title": title,
            "status": "todo",
            "priority": priority,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
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
        """Return tasks, optionally filtered by status and/or priority.

        Filters are applied with AND semantics: a task must satisfy *all*
        supplied filters to be included in the result.

        Args:
            status:   If provided, only return tasks with this status.
                      Must be one of ``"todo"``, ``"in_progress"``, ``"done"``.
            priority: If provided, only return tasks with this priority.
                      Must be one of ``"low"``, ``"medium"``, ``"high"``.

        Returns:
            A (possibly empty) list of matching task dicts, in insertion order.

        Raises:
            ValueError: If ``status`` or ``priority`` is provided but invalid.
        """
        if status is not None:
            status = status.strip().lower()
            if status not in VALID_STATUSES:
                raise ValueError(
                    f"Invalid status '{status}'. Must be one of: "
                    + ", ".join(sorted(VALID_STATUSES))
                )

        if priority is not None:
            priority = priority.strip().lower()
            if priority not in VALID_PRIORITIES:
                raise ValueError(
                    f"Invalid priority '{priority}'. Must be one of: "
                    + ", ".join(sorted(VALID_PRIORITIES))
                )

        tasks = self._load()

        if status is not None:
            tasks = [t for t in tasks if t.get("status") == status]
        if priority is not None:
            tasks = [t for t in tasks if t.get("priority") == priority]

        return tasks

    def update_status(self, partial_id: str, new_status: str) -> dict:
        """Update the status of a task identified by a partial UUID prefix.

        Args:
            partial_id: A prefix of the task UUID to match (case-insensitive).
            new_status: The new status value. Must be one of ``"todo"``,
                        ``"in_progress"``, ``"done"``.

        Returns:
            The updated task dict (with the new status applied).

        Raises:
            ValueError: If ``new_status`` is invalid, or if ``partial_id``
                        matches zero or more than one task.
        """
        new_status = new_status.strip().lower() if new_status else ""
        if new_status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{new_status}'. Must be one of: "
                + ", ".join(sorted(VALID_STATUSES))
            )

        tasks = self._load()
        task = self._find_task(tasks, partial_id)
        task["status"] = new_status
        self._save(tasks)

        return task

    def delete_task(self, partial_id: str) -> dict:
        """Delete a task identified by a partial UUID prefix.

        Args:
            partial_id: A prefix of the task UUID to match (case-insensitive).

        Returns:
            The deleted task dict (as it existed before deletion).

        Raises:
            ValueError: If ``partial_id`` matches zero or more than one task.
        """
        tasks = self._load()
        task = self._find_task(tasks, partial_id)
        tasks.remove(task)
        self._save(tasks)

        return task
