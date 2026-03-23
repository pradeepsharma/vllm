"""
tests/test_store.py — Unit tests for task_store.TaskStore.

Covers every public method of TaskStore:
  - add_task: happy path, default priority, invalid priority, persistence
  - list_tasks: no filter, status filter, priority filter, combined filter,
                invalid status
  - update_status: success, persistence, not found, ambiguous prefix,
                   invalid status
  - delete_task: success, returns deleted dict, persistence, not found
  - persistence: full save/load round-trip
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the task_tracker directory is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_store import TaskStore  # noqa: E402


# ---------------------------------------------------------------------------
# test_add_task_returns_dict
# ---------------------------------------------------------------------------


def test_add_task_returns_dict(tmp_store: TaskStore) -> None:
    """add_task returns a dict with all required keys and correct values."""
    task = tmp_store.add_task("Buy milk", "high")

    assert isinstance(task, dict)
    # All five required keys must be present
    for key in ("id", "title", "status", "priority", "created_at"):
        assert key in task, f"Missing key: {key}"

    assert task["title"] == "Buy milk"
    assert task["status"] == "todo"
    assert task["priority"] == "high"


# ---------------------------------------------------------------------------
# test_add_task_default_priority
# ---------------------------------------------------------------------------


def test_add_task_default_priority(tmp_store: TaskStore) -> None:
    """add_task uses 'medium' as the default priority."""
    task = tmp_store.add_task("Task")
    assert task["priority"] == "medium"


# ---------------------------------------------------------------------------
# test_add_task_invalid_priority
# ---------------------------------------------------------------------------


def test_add_task_invalid_priority(tmp_store: TaskStore) -> None:
    """add_task raises ValueError for an unrecognised priority."""
    with pytest.raises(ValueError):
        tmp_store.add_task("Task", "urgent")


# ---------------------------------------------------------------------------
# test_add_task_persists
# ---------------------------------------------------------------------------


def test_add_task_persists(tmp_path: Path) -> None:
    """A task added via one TaskStore instance is visible in a second instance."""
    store1 = TaskStore(tmp_path / "tasks.json")
    task = store1.add_task("Persisted task", "low")

    # Create a brand-new instance pointing at the same file
    store2 = TaskStore(tmp_path / "tasks.json")
    tasks = store2.list_tasks()

    assert len(tasks) == 1
    assert tasks[0]["id"] == task["id"]
    assert tasks[0]["title"] == "Persisted task"


# ---------------------------------------------------------------------------
# test_list_tasks_no_filter
# ---------------------------------------------------------------------------


def test_list_tasks_no_filter(tmp_store: TaskStore) -> None:
    """list_tasks() with no arguments returns all tasks."""
    tmp_store.add_task("Task 1", "low")
    tmp_store.add_task("Task 2", "medium")
    tmp_store.add_task("Task 3", "high")

    tasks = tmp_store.list_tasks()
    assert len(tasks) == 3


# ---------------------------------------------------------------------------
# test_list_tasks_filter_status
# ---------------------------------------------------------------------------


def test_list_tasks_filter_status(tmp_store: TaskStore) -> None:
    """list_tasks(status='done') returns only tasks with that status."""
    t1 = tmp_store.add_task("Task A", "medium")
    tmp_store.add_task("Task B", "medium")

    # Mark the first task as done
    tmp_store.update_status(t1["id"][:8], "done")

    done_tasks = tmp_store.list_tasks(status="done")
    assert len(done_tasks) == 1
    assert done_tasks[0]["status"] == "done"
    assert done_tasks[0]["title"] == "Task A"


# ---------------------------------------------------------------------------
# test_list_tasks_filter_priority
# ---------------------------------------------------------------------------


def test_list_tasks_filter_priority(tmp_store: TaskStore) -> None:
    """list_tasks(priority='high') returns only high-priority tasks."""
    tmp_store.add_task("High task", "high")
    tmp_store.add_task("Low task", "low")
    tmp_store.add_task("Medium task", "medium")

    high_tasks = tmp_store.list_tasks(priority="high")
    assert len(high_tasks) == 1
    assert high_tasks[0]["priority"] == "high"
    assert high_tasks[0]["title"] == "High task"


# ---------------------------------------------------------------------------
# test_list_tasks_combined_filter
# ---------------------------------------------------------------------------


def test_list_tasks_combined_filter(tmp_store: TaskStore) -> None:
    """list_tasks with both status and priority filters works correctly."""
    t1 = tmp_store.add_task("High todo", "high")
    tmp_store.add_task("Low todo", "low")
    tmp_store.add_task("High done", "high")

    # Mark the third task done
    all_tasks = tmp_store.list_tasks()
    high_tasks = [t for t in all_tasks if t["priority"] == "high"]
    # Mark the second high-priority task done (not t1)
    other_high = next(t for t in high_tasks if t["id"] != t1["id"])
    tmp_store.update_status(other_high["id"][:8], "done")

    # Combined filter: todo + high → only t1
    result = tmp_store.list_tasks(status="todo", priority="high")
    assert len(result) == 1
    assert result[0]["id"] == t1["id"]
    assert result[0]["title"] == "High todo"


# ---------------------------------------------------------------------------
# test_list_tasks_invalid_status
# ---------------------------------------------------------------------------


def test_list_tasks_invalid_status(tmp_store: TaskStore) -> None:
    """list_tasks raises ValueError for an unrecognised status."""
    with pytest.raises(ValueError):
        tmp_store.list_tasks(status="invalid")


# ---------------------------------------------------------------------------
# test_update_status_success
# ---------------------------------------------------------------------------


def test_update_status_success(tmp_store: TaskStore) -> None:
    """update_status with a valid 8-char prefix returns the updated task."""
    task = tmp_store.add_task("Mark me done", "medium")
    updated = tmp_store.update_status(task["id"][:8], "done")

    assert updated["status"] == "done"
    assert updated["id"] == task["id"]


# ---------------------------------------------------------------------------
# test_update_status_persists
# ---------------------------------------------------------------------------


def test_update_status_persists(tmp_path: Path) -> None:
    """A status change made via one store is visible in a freshly loaded store."""
    store1 = TaskStore(tmp_path / "tasks.json")
    task = store1.add_task("Persist status", "medium")
    store1.update_status(task["id"][:8], "done")

    store2 = TaskStore(tmp_path / "tasks.json")
    tasks = store2.list_tasks()
    assert len(tasks) == 1
    assert tasks[0]["status"] == "done"


# ---------------------------------------------------------------------------
# test_update_status_not_found
# ---------------------------------------------------------------------------


def test_update_status_not_found(tmp_store: TaskStore) -> None:
    """update_status raises KeyError when no task matches the prefix."""
    with pytest.raises(KeyError):
        tmp_store.update_status("nonexistent", "done")


# ---------------------------------------------------------------------------
# test_update_status_ambiguous
# ---------------------------------------------------------------------------


def test_update_status_ambiguous(tmp_store: TaskStore) -> None:
    """update_status raises ValueError when the prefix matches multiple tasks."""
    # Directly patch _tasks to force a shared prefix
    tmp_store._tasks = [
        {
            "id": "aaaa1111-0000-0000-0000-000000000001",
            "title": "Task A",
            "status": "todo",
            "priority": "medium",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
        {
            "id": "aaaa2222-0000-0000-0000-000000000002",
            "title": "Task B",
            "status": "todo",
            "priority": "medium",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    ]
    with pytest.raises(ValueError, match="Ambiguous"):
        tmp_store.update_status("aaaa", "done")


# ---------------------------------------------------------------------------
# test_update_status_invalid_status
# ---------------------------------------------------------------------------


def test_update_status_invalid_status(tmp_store: TaskStore) -> None:
    """update_status raises ValueError for an unrecognised status string."""
    task = tmp_store.add_task("Bad status task", "medium")
    with pytest.raises(ValueError):
        tmp_store.update_status(task["id"], "invalid")


# ---------------------------------------------------------------------------
# test_delete_task_success
# ---------------------------------------------------------------------------


def test_delete_task_success(tmp_store: TaskStore) -> None:
    """delete_task removes the task so list_tasks() returns an empty list."""
    task = tmp_store.add_task("Delete me", "low")
    tmp_store.delete_task(task["id"][:8])

    assert tmp_store.list_tasks() == []


# ---------------------------------------------------------------------------
# test_delete_task_returns_deleted
# ---------------------------------------------------------------------------


def test_delete_task_returns_deleted(tmp_store: TaskStore) -> None:
    """delete_task returns the dict of the task that was deleted."""
    task = tmp_store.add_task("Return me on delete", "high")
    deleted = tmp_store.delete_task(task["id"][:8])

    assert deleted["id"] == task["id"]
    assert deleted["title"] == "Return me on delete"
    assert deleted["priority"] == "high"


# ---------------------------------------------------------------------------
# test_delete_task_persists
# ---------------------------------------------------------------------------


def test_delete_task_persists(tmp_path: Path) -> None:
    """A deletion made via one store is reflected in a freshly loaded store."""
    store1 = TaskStore(tmp_path / "tasks.json")
    task = store1.add_task("Persist deletion", "medium")
    store1.delete_task(task["id"][:8])

    store2 = TaskStore(tmp_path / "tasks.json")
    assert store2.list_tasks() == []


# ---------------------------------------------------------------------------
# test_delete_task_not_found
# ---------------------------------------------------------------------------


def test_delete_task_not_found(tmp_store: TaskStore) -> None:
    """delete_task raises KeyError when no task matches the given prefix."""
    with pytest.raises(KeyError):
        tmp_store.delete_task("nonexistent-id")


# ---------------------------------------------------------------------------
# test_persistence_save_load_roundtrip
# ---------------------------------------------------------------------------


def test_persistence_save_load_roundtrip(tmp_path: Path) -> None:
    """All tasks added to one store are present with correct fields after reload."""
    store1 = TaskStore(tmp_path / "tasks.json")
    t1 = store1.add_task("Alpha", "high")
    t2 = store1.add_task("Beta", "medium")
    t3 = store1.add_task("Gamma", "low")

    # Load a fresh store from the same file
    store2 = TaskStore(tmp_path / "tasks.json")
    tasks = store2.list_tasks()

    assert len(tasks) == 3

    ids = {t["id"] for t in tasks}
    assert t1["id"] in ids
    assert t2["id"] in ids
    assert t3["id"] in ids

    titles = {t["title"] for t in tasks}
    assert titles == {"Alpha", "Beta", "Gamma"}

    priorities = {t["priority"] for t in tasks}
    assert priorities == {"high", "medium", "low"}

    # All tasks should start as "todo"
    assert all(t["status"] == "todo" for t in tasks)

    # All tasks should have a created_at timestamp
    assert all("created_at" in t for t in tasks)
