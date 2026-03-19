"""
task-manager — A Python task management CLI tool.

This package provides a lightweight, file-backed task manager that can be
used entirely from the command line.  All data is persisted as a single
JSON file on disk (default: ``~/.tasks/tasks.json``).

Modules
-------
models
    Core data structures: :class:`~models.Task`, :class:`~models.TaskList`,
    :class:`~models.Priority`, and :class:`~models.Status`.
storage
    File-backed persistence layer (:class:`~storage.TaskStorage`).
services
    Business-logic orchestration (:class:`~services.TaskService`).
cli
    ``argparse``-based command-line interface and ``main()`` entry point.

Quick start
-----------
Run directly::

    python cli.py --help

Or, after installing the package::

    tasks --help
"""

__version__ = "1.0.0"
__author__ = "Task Manager Contributors"
__license__ = "MIT"

# ---------------------------------------------------------------------------
# Public re-exports — import the most commonly used symbols so that callers
# can do ``from task_manager import Task`` instead of
# ``from task_manager.models import Task``.
# ---------------------------------------------------------------------------

from importlib import import_module as _import_module
import pathlib as _pathlib
import importlib.util as _util
import sys as _sys


def _load_sibling(name: str, filename: str):
    """Load a sibling .py file as a module and register it in sys.modules."""
    if name in _sys.modules:
        return _sys.modules[name]
    _here = _pathlib.Path(__file__).parent
    _path = _here / filename
    if not _path.exists():
        raise ImportError(f"Cannot find {filename!r} in {_here}")
    spec = _util.spec_from_file_location(name, _path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create module spec for {name!r}")
    mod = _util.module_from_spec(spec)  # type: ignore[arg-type]
    _sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# Load sibling modules so that ``from task_manager import …`` works even
# when the package is used without a proper install (e.g. ``python cli.py``).
_models_mod  = _load_sibling("task_models",   "models.py")
_storage_mod = _load_sibling("task_storage",  "storage.py")
_service_mod = _load_sibling("task_services", "services.py")

# Core model types
Priority          = _models_mod.Priority
Status            = _models_mod.Status
Task              = _models_mod.Task
TaskList          = _models_mod.TaskList

# Storage
TaskStorage       = _storage_mod.TaskStorage
StorageError      = _storage_mod.StorageError
StorageReadError  = _storage_mod.StorageReadError
StorageWriteError = _storage_mod.StorageWriteError

# Service
TaskService       = _service_mod.TaskService
ServiceError      = _service_mod.ServiceError
TaskNotFoundError = _service_mod.TaskNotFoundError
DuplicateTaskError = _service_mod.DuplicateTaskError

__all__ = [
    # Version
    "__version__",
    # Models
    "Priority",
    "Status",
    "Task",
    "TaskList",
    # Storage
    "TaskStorage",
    "StorageError",
    "StorageReadError",
    "StorageWriteError",
    # Service
    "TaskService",
    "ServiceError",
    "TaskNotFoundError",
    "DuplicateTaskError",
]
