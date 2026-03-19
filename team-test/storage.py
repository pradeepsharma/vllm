"""
Storage layer for the Task Management CLI tool.

This module provides a file-backed persistence mechanism for
:class:`~models.TaskList` objects.  All data is stored as a single
JSON file on disk.

Public API
----------
- :class:`StorageError`      – base exception for all storage failures
- :class:`StorageReadError`  – raised when the file cannot be read / parsed
- :class:`StorageWriteError` – raised when the file cannot be written
- :class:`TaskStorage`       – the main storage class

Typical usage::

    from storage import TaskStorage

    store = TaskStorage("~/.tasks/tasks.json")
    task_list = store.load()          # returns TaskList (empty if file absent)
    task_list.add(some_task)
    store.save(task_list)             # persists to disk atomically
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import tempfile
from typing import Optional

# ---------------------------------------------------------------------------
# Lazy import of models so that storage.py can be loaded independently of
# how the caller has set up sys.path (the test suite loads models.py via
# importlib; we mirror that pattern here).
# ---------------------------------------------------------------------------
import importlib.util
import sys

def _load_models():
    """Return the models module, importing it if necessary."""
    if "task_models" in sys.modules:
        return sys.modules["task_models"]
    # Try a sibling-file import (same directory as this file)
    _here = pathlib.Path(__file__).parent
    _models_path = _here / "models.py"
    if not _models_path.exists():
        raise ImportError(
            f"Cannot find models.py next to storage.py (looked in {_here})"
        )
    spec = importlib.util.spec_from_file_location("task_models", _models_path)
    mod = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
    sys.modules["task_models"] = mod
    spec.loader.exec_module(mod)                          # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StorageError(Exception):
    """Base class for all storage-related errors."""


class StorageReadError(StorageError):
    """Raised when the task file cannot be read or parsed.

    Attributes
    ----------
    path : pathlib.Path
        The file path that could not be read.
    """

    def __init__(self, message: str, path: pathlib.Path) -> None:
        super().__init__(message)
        self.path = path


class StorageWriteError(StorageError):
    """Raised when the task file cannot be written.

    Attributes
    ----------
    path : pathlib.Path
        The file path that could not be written.
    """

    def __init__(self, message: str, path: pathlib.Path) -> None:
        super().__init__(message)
        self.path = path


# ---------------------------------------------------------------------------
# TaskStorage
# ---------------------------------------------------------------------------

class TaskStorage:
    """Persist a :class:`~models.TaskList` to a JSON file on disk.

    Parameters
    ----------
    path:
        Path to the JSON file.  ``~`` and environment variables are expanded
        automatically.  Parent directories are created on first :meth:`save`.
    indent:
        JSON indentation level used when writing the file.  Defaults to 2.
    encoding:
        File encoding.  Defaults to ``"utf-8"``.

    Notes
    -----
    Writes are performed atomically: the new content is first written to a
    temporary file in the same directory, then renamed over the target file.
    This prevents data loss if the process is interrupted mid-write.
    """

    def __init__(
        self,
        path: str | os.PathLike,
        *,
        indent: int = 2,
        encoding: str = "utf-8",
    ) -> None:
        self._path = pathlib.Path(os.path.expandvars(os.path.expanduser(path)))
        self._indent = indent
        self._encoding = encoding

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> pathlib.Path:
        """The resolved, absolute path to the storage file."""
        return self._path

    @property
    def exists(self) -> bool:
        """Return ``True`` if the storage file currently exists on disk."""
        return self._path.exists()

    # ------------------------------------------------------------------
    # Core I/O
    # ------------------------------------------------------------------

    def load(self) -> "task_models.TaskList":  # type: ignore[name-defined]
        """Load and return the :class:`~models.TaskList` from disk.

        If the file does not exist an empty :class:`~models.TaskList` is
        returned — this is the expected behaviour on first run.

        Raises
        ------
        StorageReadError
            If the file exists but cannot be opened or contains invalid JSON /
            an unexpected data structure.
        """
        models = _load_models()
        TaskList = models.TaskList

        if not self._path.exists():
            return TaskList()

        try:
            raw = self._path.read_text(encoding=self._encoding)
        except OSError as exc:
            raise StorageReadError(
                f"Cannot read task file {self._path}: {exc}",
                self._path,
            ) from exc

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StorageReadError(
                f"Task file {self._path} contains invalid JSON: {exc}",
                self._path,
            ) from exc

        if not isinstance(data, dict):
            raise StorageReadError(
                f"Task file {self._path} has unexpected format "
                f"(expected a JSON object, got {type(data).__name__!r}).",
                self._path,
            )

        try:
            return TaskList.from_dict(data)
        except (KeyError, ValueError) as exc:
            raise StorageReadError(
                f"Task file {self._path} contains invalid task data: {exc}",
                self._path,
            ) from exc

    def save(self, task_list: "task_models.TaskList") -> None:  # type: ignore[name-defined]
        """Persist *task_list* to disk atomically.

        Parent directories are created automatically if they do not exist.

        Parameters
        ----------
        task_list:
            The :class:`~models.TaskList` to persist.

        Raises
        ------
        StorageWriteError
            If the file cannot be written (e.g. permission denied, disk full).
        TypeError
            If *task_list* is not a :class:`~models.TaskList` instance.
        """
        models = _load_models()
        if not isinstance(task_list, models.TaskList):
            raise TypeError(
                f"task_list must be a TaskList instance, got {type(task_list).__name__!r}"
            )

        # Ensure parent directory exists
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageWriteError(
                f"Cannot create directory {self._path.parent}: {exc}",
                self._path,
            ) from exc

        payload = json.dumps(task_list.to_dict(), indent=self._indent, ensure_ascii=False)

        # Atomic write: write to a temp file, then rename
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=self._path.parent,
                prefix=".tmp_tasks_",
                suffix=".json",
            )
            try:
                with os.fdopen(fd, "w", encoding=self._encoding) as fh:
                    fh.write(payload)
            except Exception:
                # Clean up the temp file if writing fails
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
            # Atomic rename
            shutil.move(tmp_path, self._path)
        except StorageWriteError:
            raise
        except OSError as exc:
            raise StorageWriteError(
                f"Cannot write task file {self._path}: {exc}",
                self._path,
            ) from exc

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def delete(self) -> bool:
        """Delete the storage file from disk.

        Returns
        -------
        bool
            ``True`` if the file was deleted, ``False`` if it did not exist.

        Raises
        ------
        StorageWriteError
            If the file exists but cannot be deleted.
        """
        if not self._path.exists():
            return False
        try:
            self._path.unlink()
            return True
        except OSError as exc:
            raise StorageWriteError(
                f"Cannot delete task file {self._path}: {exc}",
                self._path,
            ) from exc

    def backup(self, suffix: str = ".bak") -> pathlib.Path:
        """Create a backup copy of the storage file.

        Parameters
        ----------
        suffix:
            Suffix appended to the original filename for the backup.
            Defaults to ``".bak"``.

        Returns
        -------
        pathlib.Path
            Path to the newly created backup file.

        Raises
        ------
        StorageReadError
            If the source file does not exist.
        StorageWriteError
            If the backup file cannot be written.
        """
        if not self._path.exists():
            raise StorageReadError(
                f"Cannot back up {self._path}: file does not exist.",
                self._path,
            )
        backup_path = self._path.with_suffix(self._path.suffix + suffix)
        try:
            shutil.copy2(self._path, backup_path)
        except OSError as exc:
            raise StorageWriteError(
                f"Cannot create backup at {backup_path}: {exc}",
                backup_path,
            ) from exc
        return backup_path

    def file_size(self) -> Optional[int]:
        """Return the size of the storage file in bytes, or ``None`` if absent."""
        try:
            return self._path.stat().st_size
        except OSError:
            return None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        exists = "exists" if self.exists else "absent"
        return f"TaskStorage({str(self._path)!r}, {exists})"
