"""
tests/test_store.py — Unit tests for TaskStore (task_tracker.task_store).

16 tests covering:
  - add_task: happy path, default priority, empty title, invalid priority
  - list_tasks: all, filter by status, filter by priority, combined filter,
                invalid status, invalid priority
  - update_status: happy path, invalid status, no match, ambiguous match
  - delete_task: happy path, no match
  - _load/_save: empty file, corrupt JSON
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from task_tracker.task_store import TaskStore, VALID_PRIORITIES, VALID_STATUSES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_store(tmp_path: Path) -> TaskStore:
    """Return a TaskStore backed by a fresh temp file."""
    return TaskStore(filepath=str(tmp_path / "tasks.json"))


# ---------------------------------------------------------------------------
# add_task
# ---------------------------------------------------------------------------


class TestAddTask:
    def test_add_task_returns_correct_fields(self, tmp_store: TaskStore) -> None:
        """add_task returns a dict with all required fields."""
        task = tmp_store.add_task(title="Buy milk", priority="high")

        assert task["title"] == "Buy milk"
        assert task["priority"] == "high"
        assert task["status"] == "todo"
        assert len(task["id"]) == 36  # UUID4 with dashes
        assert task["created_at"].endswith("+00:00")  # UTC ISO timestamp

    def test_add_task_default_priority_is_medium(self, tmp_store: TaskStore) -> None:
        """add_task uses 'medium' as the default priority."""
        task = tmp_store.add_task(title="Default priority task")
        assert task["priority"] == "medium"

    def test_add_task_persists_to_file(self, tmp_store: TaskStore) -> None:
        """add_task writes the task to the JSON file."""
        tmp_store.add_task(title="Persisted task", priority="low")
        raw = Path(tmp_store.filepath).read_text(encoding="utf-8")
        data = json.loads(raw)
        assert len(data) == 1
        assert data[0]["title"] == "Persisted task"

    def test_add_task_empty_title_raises(self, tmp_store: TaskStore) -> None:
        """add_task raises ValueError for an empty title."""
        with pytest.raises(ValueError, match="title must not be empty"):
            tmp_store.add_task(title="   ")

    def test_add_task_invalid_priority_raises(self, tmp_store: TaskStore) -> None:
        """add_task raises ValueError for an invalid priority."""
        with pytest.raises(ValueError, match="Invalid priority"):
            tmp_store.add_task(title="Bad priority", priority="critical")


# ---------------------------------------------------------------------------
# list_tasks
# ---------------------------------------------------------------------------


class TestListTasks:
    def test_list_tasks_returns_all_when_no_filter(self, tmp_store: TaskStore) -> None:
        """list_tasks with no filters returns all tasks."""
        tmp_store.add_task("Task A", priority="high")
        tmp_store.add_task("Task B", priority="low")
        tasks = tmp_store.list_tasks()
        assert len(tasks) == 2
        titles = {t["title"] for t in tasks}
        assert titles == {"Task A", "Task B"}

    def test_list_tasks_filter_by_status(self, tmp_store: TaskStore) -> None:
        """list_tasks filters correctly by status."""
        t1 = tmp_store.add_task("Todo task", priority="medium")
        t2 = tmp_store.add_task("Done task", priority="medium")
        tmp_store.update_status(t2["id"], "done")

        todo_tasks = tmp_store.list_tasks(status="todo")
        done_tasks = tmp_store.list_tasks(status="done")

        assert len(todo_tasks) == 1
        assert todo_tasks[0]["title"] == "Todo task"
        assert len(done_tasks) == 1
        assert done_tasks[0]["title"] == "Done task"

    def test_list_tasks_filter_by_priority(self, tmp_store: TaskStore) -> None:
        """list_tasks filters correctly by priority."""
        tmp_store.add_task("High task", priority="high")
        tmp_store.add_task("Low task", priority="low")

        high_tasks = tmp_store.list_tasks(priority="high")
        assert len(high_tasks) == 1
        assert high_tasks[0]["title"] == "High task"

    def test_list_tasks_combined_filter(self, tmp_store: TaskStore) -> None:
        """list_tasks applies status AND priority filters together."""
        t1 = tmp_store.add_task("High todo", priority="high")
        t2 = tmp_store.add_task("High done", priority="high")
        tmp_store.update_status(t2["id"], "done")
        tmp_store.add_task("Low todo", priority="low")

        results = tmp_store.list_tasks(status="todo", priority="high")
        assert len(results) == 1
        assert results[0]["title"] == "High todo"

    def test_list_tasks_invalid_status_raises(self, tmp_store: TaskStore) -> None:
        """list_tasks raises ValueError for an invalid status filter."""
        with pytest.raises(ValueError, match="Invalid status"):
            tmp_store.list_tasks(status="pending")

    def test_list_tasks_invalid_priority_raises(self, tmp_store: TaskStore) -> None:
        """list_tasks raises ValueError for an invalid priority filter."""
        with pytest.raises(ValueError, match="Invalid priority"):
            tmp_store.list_tasks(priority="urgent")


# ---------------------------------------------------------------------------
# update_status
# ---------------------------------------------------------------------------


class TestUpdateStatus:
    def test_update_status_happy_path(self, tmp_store: TaskStore) -> None:
        """update_status changes the task status and returns the updated task."""
        task = tmp_store.add_task("My task", priority="medium")
        updated = tmp_store.update_status(task["id"], "done")

        assert updated["status"] == "done"
        assert updated["id"] == task["id"]
        # Verify it's persisted
        tasks = tmp_store.list_tasks(status="done")
        assert len(tasks) == 1

    def test_update_status_invalid_status_raises(self, tmp_store: TaskStore) -> None:
        """update_status raises ValueError for an invalid new_status."""
        task = tmp_store.add_task("My task", priority="medium")
        with pytest.raises(ValueError, match="Invalid status"):
            tmp_store.update_status(task["id"], "finished")

    def test_update_status_no_match_raises(self, tmp_store: TaskStore) -> None:
        """update_status raises ValueError when partial_id matches nothing."""
        tmp_store.add_task("My task", priority="medium")
        with pytest.raises(ValueError, match="No task found"):
            tmp_store.update_status("zzzzzzz", "done")

    def test_update_status_ambiguous_raises(self, tmp_store: TaskStore) -> None:
        """update_status raises ValueError when partial_id is ambiguous."""
        # Force two tasks with the same UUID prefix by patching IDs directly
        store = tmp_store
        store.add_task("Task 1", priority="low")
        store.add_task("Task 2", priority="low")
        # Load and manually set IDs to share a prefix
        tasks = store._load()
        tasks[0]["id"] = "aaaa1111-0000-0000-0000-000000000001"
        tasks[1]["id"] = "aaaa2222-0000-0000-0000-000000000002"
        store._save(tasks)

        with pytest.raises(ValueError, match="Ambiguous"):
            store.update_status("aaaa", "done")


# ---------------------------------------------------------------------------
# delete_task
# ---------------------------------------------------------------------------


class TestDeleteTask:
    def test_delete_task_removes_task(self, tmp_store: TaskStore) -> None:
        """delete_task removes the task from the store and returns it."""
        task = tmp_store.add_task("To be deleted", priority="low")
        deleted = tmp_store.delete_task(task["id"])

        assert deleted["id"] == task["id"]
        assert deleted["title"] == "To be deleted"
        remaining = tmp_store.list_tasks()
        assert len(remaining) == 0

    def test_delete_task_no_match_raises(self, tmp_store: TaskStore) -> None:
        """delete_task raises ValueError when partial_id matches nothing."""
        tmp_store.add_task("Some task", priority="medium")
        with pytest.raises(ValueError, match="No task found"):
            tmp_store.delete_task("zzzzzzz")


# ---------------------------------------------------------------------------
# Edge cases: _load
# ---------------------------------------------------------------------------


class TestLoadEdgeCases:
    def test_load_returns_empty_list_when_file_missing(self, tmp_path: Path) -> None:
        """_load returns [] when the JSON file does not exist."""
        store = TaskStore(filepath=str(tmp_path / "nonexistent.json"))
        assert store._load() == []

    def test_load_raises_on_corrupt_json(self, tmp_path: Path) -> None:
        """_load raises ValueError when the file contains invalid JSON."""
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json", encoding="utf-8")
        store = TaskStore(filepath=str(bad_file))
        with pytest.raises(ValueError, match="invalid JSON"):
            store._load()
