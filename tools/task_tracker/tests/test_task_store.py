"""
tests/test_task_store.py — Unit tests for task_store.TaskStore.

Covers:
  - add_task: happy path, invalid priority
  - list_tasks: no filter, status filter, priority filter, combined filter, invalid args
  - update_status: happy path, invalid status, not found, ambiguous prefix
  - delete_task: happy path, not found, ambiguous prefix
  - persistence: data survives a reload from disk
  - atomic write: .tmp file is cleaned up
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import pytest

# Make sure the parent directory is on sys.path so we can import task_store
sys.path.insert(0, str(Path(__file__).parent.parent))

from task_store import VALID_PRIORITIES, VALID_STATUSES, TaskStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> TaskStore:
    """Return a fresh TaskStore backed by a temp file."""
    return TaskStore(tmp_path / "tasks.json")


@pytest.fixture()
def populated_store(store: TaskStore) -> TaskStore:
    """Return a TaskStore pre-loaded with three tasks of varying attributes."""
    store.add_task("Fix login bug", priority="high")
    store.add_task("Write docs", priority="medium")
    store.add_task("Refactor DB layer", priority="low")
    return store


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_valid_statuses_contains_expected_values():
    assert VALID_STATUSES == {"todo", "in_progress", "done"}


def test_valid_priorities_contains_expected_values():
    assert VALID_PRIORITIES == {"low", "medium", "high"}


# ---------------------------------------------------------------------------
# add_task
# ---------------------------------------------------------------------------


class TestAddTask:
    def test_returns_task_dict_with_correct_fields(self, store: TaskStore):
        task = store.add_task("My task", priority="high")
        assert task["title"] == "My task"
        assert task["priority"] == "high"
        assert task["status"] == "todo"
        assert "id" in task
        assert len(task["id"]) == 36  # UUID4 format
        assert "created_at" in task

    def test_default_priority_is_medium(self, store: TaskStore):
        task = store.add_task("No priority given")
        assert task["priority"] == "medium"

    def test_task_is_persisted_to_disk(self, store: TaskStore):
        task = store.add_task("Persisted task", priority="low")
        # Re-load from the same file
        reloaded = TaskStore(store.filepath)
        tasks = reloaded.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["id"] == task["id"]
        assert tasks[0]["title"] == "Persisted task"

    def test_multiple_tasks_accumulate(self, store: TaskStore):
        store.add_task("Task A")
        store.add_task("Task B")
        store.add_task("Task C")
        assert len(store.list_tasks()) == 3

    def test_invalid_priority_raises_value_error(self, store: TaskStore):
        with pytest.raises(ValueError, match="Invalid priority"):
            store.add_task("Bad task", priority="urgent")

    def test_invalid_priority_message_contains_valid_options(self, store: TaskStore):
        with pytest.raises(ValueError) as exc_info:
            store.add_task("Bad task", priority="critical")
        msg = str(exc_info.value)
        assert "high" in msg
        assert "low" in msg
        assert "medium" in msg

    def test_created_at_is_iso8601_utc(self, store: TaskStore):
        task = store.add_task("Timestamp check")
        # ISO 8601 UTC strings end with +00:00
        assert "+00:00" in task["created_at"] or task["created_at"].endswith("Z")

    def test_each_task_gets_unique_id(self, store: TaskStore):
        t1 = store.add_task("Task 1")
        t2 = store.add_task("Task 2")
        assert t1["id"] != t2["id"]


# ---------------------------------------------------------------------------
# list_tasks
# ---------------------------------------------------------------------------


class TestListTasks:
    def test_empty_store_returns_empty_list(self, store: TaskStore):
        assert store.list_tasks() == []

    def test_no_filter_returns_all_tasks(self, populated_store: TaskStore):
        tasks = populated_store.list_tasks()
        assert len(tasks) == 3

    def test_filter_by_status_todo(self, populated_store: TaskStore):
        tasks = populated_store.list_tasks(status="todo")
        assert len(tasks) == 3  # all start as todo
        assert all(t["status"] == "todo" for t in tasks)

    def test_filter_by_status_done_returns_only_done(self, populated_store: TaskStore):
        # Mark one task done
        all_tasks = populated_store.list_tasks()
        populated_store.update_status(all_tasks[0]["id"], "done")

        done_tasks = populated_store.list_tasks(status="done")
        assert len(done_tasks) == 1
        assert done_tasks[0]["status"] == "done"

    def test_filter_by_priority_high(self, populated_store: TaskStore):
        tasks = populated_store.list_tasks(priority="high")
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Fix login bug"
        assert tasks[0]["priority"] == "high"

    def test_filter_by_priority_low(self, populated_store: TaskStore):
        tasks = populated_store.list_tasks(priority="low")
        assert len(tasks) == 1
        assert tasks[0]["priority"] == "low"

    def test_combined_status_and_priority_filter(self, populated_store: TaskStore):
        tasks = populated_store.list_tasks(status="todo", priority="medium")
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Write docs"

    def test_combined_filter_no_match_returns_empty(self, populated_store: TaskStore):
        # Mark the high-priority task done, then filter for todo+high
        all_tasks = populated_store.list_tasks(priority="high")
        populated_store.update_status(all_tasks[0]["id"], "done")
        tasks = populated_store.list_tasks(status="todo", priority="high")
        assert tasks == []

    def test_invalid_status_raises_value_error(self, store: TaskStore):
        with pytest.raises(ValueError, match="Invalid status"):
            store.list_tasks(status="pending")

    def test_invalid_priority_raises_value_error(self, store: TaskStore):
        with pytest.raises(ValueError, match="Invalid priority"):
            store.list_tasks(priority="urgent")

    def test_returned_list_is_a_copy(self, populated_store: TaskStore):
        """Mutating the returned list must not affect the store."""
        tasks = populated_store.list_tasks()
        tasks.clear()
        assert len(populated_store.list_tasks()) == 3


# ---------------------------------------------------------------------------
# update_status
# ---------------------------------------------------------------------------


class TestUpdateStatus:
    def test_marks_task_done_by_full_id(self, store: TaskStore):
        task = store.add_task("Complete me")
        updated = store.update_status(task["id"], "done")
        assert updated["status"] == "done"
        assert updated["id"] == task["id"]

    def test_marks_task_done_by_prefix(self, store: TaskStore):
        task = store.add_task("Complete me by prefix")
        prefix = task["id"][:8]
        updated = store.update_status(prefix, "done")
        assert updated["status"] == "done"

    def test_update_to_in_progress(self, store: TaskStore):
        task = store.add_task("In progress task")
        updated = store.update_status(task["id"], "in_progress")
        assert updated["status"] == "in_progress"

    def test_status_change_is_persisted(self, store: TaskStore):
        task = store.add_task("Persist status change")
        store.update_status(task["id"], "done")
        reloaded = TaskStore(store.filepath)
        tasks = reloaded.list_tasks()
        assert tasks[0]["status"] == "done"

    def test_invalid_status_raises_value_error(self, store: TaskStore):
        task = store.add_task("Bad status")
        with pytest.raises(ValueError, match="Invalid status"):
            store.update_status(task["id"], "completed")

    def test_not_found_raises_key_error(self, store: TaskStore):
        with pytest.raises(KeyError, match="No task found"):
            store.update_status("nonexistent-prefix", "done")

    def test_ambiguous_prefix_raises_value_error(self, store: TaskStore):
        """Two tasks sharing the same prefix should raise ValueError."""
        # We can't easily force UUID collision, so we patch _tasks directly.
        store._tasks = [
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
            store.update_status("aaaa", "done")


# ---------------------------------------------------------------------------
# delete_task
# ---------------------------------------------------------------------------


class TestDeleteTask:
    def test_delete_by_full_id(self, store: TaskStore):
        task = store.add_task("Delete me")
        deleted = store.delete_task(task["id"])
        assert deleted["id"] == task["id"]
        assert store.list_tasks() == []

    def test_delete_by_prefix(self, store: TaskStore):
        task = store.add_task("Delete by prefix")
        prefix = task["id"][:8]
        deleted = store.delete_task(prefix)
        assert deleted["id"] == task["id"]

    def test_deletion_is_persisted(self, store: TaskStore):
        task = store.add_task("Persist deletion")
        store.delete_task(task["id"])
        reloaded = TaskStore(store.filepath)
        assert reloaded.list_tasks() == []

    def test_delete_one_of_many(self, populated_store: TaskStore):
        all_tasks = populated_store.list_tasks()
        target = all_tasks[1]
        populated_store.delete_task(target["id"])
        remaining = populated_store.list_tasks()
        assert len(remaining) == 2
        assert all(t["id"] != target["id"] for t in remaining)

    def test_not_found_raises_key_error(self, store: TaskStore):
        with pytest.raises(KeyError, match="No task found"):
            store.delete_task("nonexistent-prefix")

    def test_ambiguous_prefix_raises_value_error(self, store: TaskStore):
        store._tasks = [
            {
                "id": "bbbb1111-0000-0000-0000-000000000001",
                "title": "Task A",
                "status": "todo",
                "priority": "medium",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "id": "bbbb2222-0000-0000-0000-000000000002",
                "title": "Task B",
                "status": "todo",
                "priority": "medium",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
        ]
        with pytest.raises(ValueError, match="Ambiguous"):
            store.delete_task("bbbb")


# ---------------------------------------------------------------------------
# Persistence & atomic write
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_fresh_store_on_missing_file(self, tmp_path: Path):
        store = TaskStore(tmp_path / "nonexistent.json")
        assert store.list_tasks() == []

    def test_corrupt_json_starts_fresh(self, tmp_path: Path):
        bad_file = tmp_path / "corrupt.json"
        bad_file.write_text("NOT VALID JSON", encoding="utf-8")
        store = TaskStore(bad_file)
        assert store.list_tasks() == []

    def test_unexpected_json_type_starts_fresh(self, tmp_path: Path):
        bad_file = tmp_path / "bad_type.json"
        bad_file.write_text('{"key": "value"}', encoding="utf-8")
        store = TaskStore(bad_file)
        assert store.list_tasks() == []

    def test_tmp_file_is_cleaned_up_after_save(self, tmp_path: Path):
        store = TaskStore(tmp_path / "tasks.json")
        store.add_task("Atomic write test")
        tmp_file = store.filepath.with_suffix(".tmp")
        assert not tmp_file.exists(), ".tmp file should be removed after atomic rename"

    def test_json_file_is_valid_after_save(self, tmp_path: Path):
        store = TaskStore(tmp_path / "tasks.json")
        store.add_task("Valid JSON check")
        raw = store.filepath.read_text(encoding="utf-8")
        data = json.loads(raw)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["title"] == "Valid JSON check"
