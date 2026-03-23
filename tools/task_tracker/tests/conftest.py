"""
conftest.py — Shared pytest fixtures for the task_tracker test suite.

Fixtures
--------
tmp_store(tmp_path)
    A fresh TaskStore instance backed by a temp file.  Each test function
    gets its own isolated store (scope="function").

runner()
    A Click CliRunner instance for invoking CLI commands in tests.

tmp_store_path(tmp_path)
    The string path to the temp tasks.json file, suitable for passing to
    CliRunner via env={"TASK_STORE_PATH": tmp_store_path}.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

# Ensure the task_tracker package directory is importable regardless of how
# pytest is invoked (e.g. from the repo root or from tools/task_tracker/).
_TRACKER_DIR = Path(__file__).parent.parent
if str(_TRACKER_DIR) not in sys.path:
    sys.path.insert(0, str(_TRACKER_DIR))

from task_store import TaskStore  # noqa: E402


@pytest.fixture(scope="function")
def tmp_store(tmp_path: Path) -> TaskStore:
    """Return a fresh TaskStore backed by a temporary tasks.json file.

    Each test function receives its own isolated store so tests cannot
    interfere with one another.
    """
    return TaskStore(tmp_path / "tasks.json")


@pytest.fixture(scope="function")
def runner() -> CliRunner:
    """Return a Click CliRunner instance for invoking CLI commands in tests."""
    return CliRunner()


@pytest.fixture(scope="function")
def tmp_store_path(tmp_path: Path) -> str:
    """Return the string path to a temporary tasks.json file.

    Use this with CliRunner's env parameter::

        result = runner.invoke(cli, ["list"],
                               env={"TASK_STORE_PATH": tmp_store_path})
    """
    return str(tmp_path / "tasks.json")
