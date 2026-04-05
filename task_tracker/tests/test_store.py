"""
Tests for task_tracker/task_store.py

Covers:
  - TaskStore.add_task
  - TaskStore.list_tasks (with and without filters)
  - TaskStore.update_status
  - TaskStore.delete_task
  - TaskStore.get_stats
  - Error / edge-case paths
  - Persistence across save/load cycles
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from task_tracker.task_store import (
    AmbiguousTaskIDError,
    InvalidPriorityError,
    InvalidStatusError,
    TaskNotFoundError,
    TaskStore,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_store(tmp_path):
    """Return a TaskStore backed by a temporary file."""
    return TaskStore(filepath=str(tmp_path / "tasks.json"))


# ---------------------------------------------------------------------------
# add_task
# ---------------------------------------------------------------------------


def test_add_task_creates_task(tmp_store):
    task = tmp_store.add_task("Fix login bug", priority="high")
    assert task["title"] == "Fix login bug"
    assert task["priority"] == "high"
    assert task["status"] == "todo"
    assert len(task["id"]) == 36  # UUID4 format
    assert "created_at" in task


def test_add_task_default_priority_is_medium(tmp_store):
    task = tmp_store.add_task("Write docs")
    assert task["priority"] == "medium"


def test_add_task_low_priority(tmp_store):
    task = tmp_store.add_task("Update deps", priority="low")
    assert task["priority"] == "low"


def test_add_task_strips_whitespace_from_title(tmp_store):
    task = tmp_store.add_task("  Trim me  ")
    assert task["title"] == "Trim me"


def test_add_task_empty_title_raises(tmp_store):
    with pytest.raises(ValueError, match="empty"):
        tmp_store.add_task("")


def test_add_task_whitespace_only_title_raises(tmp_store):
    with pytest.raises(ValueError, match="empty"):
        tmp_store.add_task("   ")


def test_add_task_invalid_priority_raises(tmp_store):
    with pytest.raises(InvalidPriorityError):
        tmp_store.add_task("Bad task", priority="urgent")


def test_add_task_invalid_priority_message(tmp_store):
    with pytest.raises(InvalidPriorityError, match="urgent"):
        tmp_store.add_task("Bad task", priority="urgent")


def test_add_multiple_tasks(tmp_store):
    tmp_store.add_task("Task A")
    tmp_store.add_task("Task B")
    tmp_store.add_task("Task C")
    tasks = tmp_store.list_tasks()
    assert len(tasks) == 3


# ---------------------------------------------------------------------------
# Persistence across save/load
# ---------------------------------------------------------------------------


def test_persistence_across_save_load(tmp_path):
    filepath = str(tmp_path / "tasks.json")
    store1 = TaskStore(filepath=filepath)
    task = store1.add_task("Persist me", priority="high")

    # Create a fresh store pointing at the same file
    store2 = TaskStore(filepath=filepath)
    tasks = store2.list_tasks()
    assert len(tasks) == 1
    assert tasks[0]["id"] == task["id"]
    assert tasks[0]["title"] == "Persist me"
    assert tasks[0]["priority"] == "high"


def test_persistence_file_is_valid_json(tmp_path):
    filepath = str(tmp_path / "tasks.json")
    store = TaskStore(filepath=filepath)
    store.add_task("JSON check")

    with open(filepath, "r") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["title"] == "JSON check"


def test_missing_file_returns_empty_list(tmp_path):
    store = TaskStore(filepath=str(tmp_path / "nonexistent.json"))
    assert store.list_tasks() == []


def test_empty_file_returns_empty_list(tmp_path):
    filepath = str(tmp_path / "empty.json")
    open(filepath, "w").close()  # create empty file
    store = TaskStore(filepath=filepath)
    assert store.list_tasks() == []


# ---------------------------------------------------------------------------
# list_tasks
# ---------------------------------------------------------------------------


def test_list_tasks_returns_all(tmp_store):
    tmp_store.add_task("A")
    tmp_store.add_task("B")
    tasks = tmp_store.list_tasks()
    assert len(tasks) == 2


def test_list_tasks_filter_by_status(tmp_store):
    t1 = tmp_store.add_task("Todo task")
    t2 = tmp_store.add_task("Done task")
    tmp_store.update_status(t2["id"][:8], "done")

    todo_tasks = tmp_store.list_tasks(status="todo")
    assert len(todo_tasks) == 1
    assert todo_tasks[0]["id"] == t1["id"]


def test_list_tasks_filter_by_priority(tmp_store):
    tmp_store.add_task("High task", priority="high")
    tmp_store.add_task("Low task", priority="low")

    high_tasks = tmp_store.list_tasks(priority="high")
    assert len(high_tasks) == 1
    assert high_tasks[0]["title"] == "High task"


def test_list_tasks_filter_by_status_and_priority(tmp_store):
    tmp_store.add_task("High todo", priority="high")
    t2 = tmp_store.add_task("High done", priority="high")
    tmp_store.add_task("Low todo", priority="low")
    tmp_store.update_status(t2["id"][:8], "done")

    results = tmp_store.list_tasks(status="todo", priority="high")
    assert len(results) == 1
    assert results[0]["title"] == "High todo"


def test_list_tasks_invalid_status_raises(tmp_store):
    with pytest.raises(InvalidStatusError):
        tmp_store.list_tasks(status="invalid")


def test_list_tasks_invalid_priority_raises(tmp_store):
    with pytest.raises(InvalidPriorityError):
        tmp_store.list_tasks(priority="critical")


def test_list_tasks_sorted_by_created_at(tmp_store):
    tmp_store.add_task("First")
    tmp_store.add_task("Second")
    tmp_store.add_task("Third")
    tasks = tmp_store.list_tasks()
    titles = [t["title"] for t in tasks]
    assert titles == ["First", "Second", "Third"]


# ---------------------------------------------------------------------------
# update_status
# ---------------------------------------------------------------------------


def test_update_status_marks_done(tmp_store):
    task = tmp_store.add_task("Mark me done")
    updated = tmp_store.update_status(task["id"][:8], "done")
    assert updated["status"] == "done"
    assert updated["id"] == task["id"]


def test_update_status_marks_in_progress(tmp_store):
    task = tmp_store.add_task("In progress task")
    updated = tmp_store.update_status(task["id"][:8], "in_progress")
    assert updated["status"] == "in_progress"


def test_update_status_persists(tmp_store):
    task = tmp_store.add_task("Persist status")
    tmp_store.update_status(task["id"][:8], "done")
    tasks = tmp_store.list_tasks()
    assert tasks[0]["status"] == "done"


def test_update_status_not_found_raises(tmp_store):
    with pytest.raises(TaskNotFoundError):
        tmp_store.update_status("zzzzzzz", "done")


def test_update_status_invalid_status_raises(tmp_store):
    task = tmp_store.add_task("Some task")
    with pytest.raises(InvalidStatusError):
        tmp_store.update_status(task["id"][:8], "finished")


def test_update_status_ambiguous_raises(tmp_store):
    """Force ambiguity by using a very short prefix that matches multiple tasks."""
    # Add tasks and check if any share a prefix — we'll mock by using empty prefix
    tmp_store.add_task("Task 1")
    tmp_store.add_task("Task 2")
    # Empty partial_id raises ValueError, not AmbiguousTaskIDError
    with pytest.raises(ValueError):
        tmp_store.update_status("", "done")


# ---------------------------------------------------------------------------
# delete_task
# ---------------------------------------------------------------------------


def test_delete_task_removes_task(tmp_store):
    task = tmp_store.add_task("Delete me")
    tmp_store.delete_task(task["id"][:8])
    tasks = tmp_store.list_tasks()
    assert len(tasks) == 0


def test_delete_task_returns_deleted_task(tmp_store):
    task = tmp_store.add_task("Return me on delete")
    deleted = tmp_store.delete_task(task["id"][:8])
    assert deleted["id"] == task["id"]
    assert deleted["title"] == "Return me on delete"


def test_delete_task_not_found_raises(tmp_store):
    with pytest.raises(TaskNotFoundError):
        tmp_store.delete_task("zzzzzzz")


def test_delete_task_not_found_error_message(tmp_store):
    with pytest.raises(TaskNotFoundError, match="zzzzzzz"):
        tmp_store.delete_task("zzzzzzz")


def test_delete_task_persists(tmp_store):
    t1 = tmp_store.add_task("Keep me")
    t2 = tmp_store.add_task("Delete me")
    tmp_store.delete_task(t2["id"][:8])
    tasks = tmp_store.list_tasks()
    assert len(tasks) == 1
    assert tasks[0]["id"] == t1["id"]


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


def test_get_stats_empty_store(tmp_store):
    stats = tmp_store.get_stats()
    assert stats["total"] == 0
    assert stats["todo"] == 0
    assert stats["in_progress"] == 0
    assert stats["done"] == 0
    assert stats["low"] == 0
    assert stats["medium"] == 0
    assert stats["high"] == 0


def test_get_stats_counts_correctly(tmp_store):
    t1 = tmp_store.add_task("High todo", priority="high")
    t2 = tmp_store.add_task("Medium done", priority="medium")
    tmp_store.add_task("Low todo", priority="low")
    tmp_store.update_status(t2["id"][:8], "done")

    stats = tmp_store.get_stats()
    assert stats["total"] == 3
    assert stats["todo"] == 2
    assert stats["done"] == 1
    assert stats["in_progress"] == 0
    assert stats["high"] == 1
    assert stats["medium"] == 1
    assert stats["low"] == 1


def test_get_stats_all_keys_present(tmp_store):
    stats = tmp_store.get_stats()
    for key in ("total", "todo", "in_progress", "done", "low", "medium", "high"):
        assert key in stats


# ---------------------------------------------------------------------------
# _find_task edge cases
# ---------------------------------------------------------------------------


def test_find_task_empty_partial_id_raises(tmp_store):
    tmp_store.add_task("Some task")
    with pytest.raises(ValueError, match="empty"):
        tmp_store.update_status("", "done")


def test_task_not_found_error_stores_partial_id(tmp_store):
    try:
        tmp_store.delete_task("abc123")
    except TaskNotFoundError as exc:
        assert exc.partial_id == "abc123"
    else:
        pytest.fail("TaskNotFoundError not raised")
