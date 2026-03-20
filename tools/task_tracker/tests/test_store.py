"""
test_store.py — Unit tests for TaskStore (task_store.py).

Tests cover:
  - add_task: happy path, empty title, invalid priority, persistence
  - list_tasks: no filter, status filter, priority filter, combined filter,
                invalid filter values, ordering by created_at
  - update_status: happy path, invalid status, no match, ambiguous match
  - delete_task: happy path, no match, ambiguous match, persistence
  - _load: missing file, corrupted JSON, non-list JSON
  - _find_by_partial_id: exact match, prefix match, substring match,
                          no match, ambiguous match
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

# Ensure the task_tracker package root is on sys.path so that
# `from task_store import TaskStore` works regardless of how pytest is invoked.
_PACKAGE_ROOT = str(Path(__file__).parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from task_store import TaskStore, VALID_STATUSES, VALID_PRIORITIES  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_store(tmp_path: Path) -> TaskStore:
    """Return a TaskStore backed by a temporary file (auto-cleaned up)."""
    store_file = tmp_path / "tasks.json"
    return TaskStore(store_path=str(store_file))


@pytest.fixture()
def populated_store(tmp_path: Path) -> TaskStore:
    """Return a TaskStore pre-populated with three tasks of varying attributes."""
    store_file = tmp_path / "tasks.json"
    store = TaskStore(store_path=str(store_file))
    store.add_task(title="Task Alpha", priority="low")
    store.add_task(title="Task Beta", priority="medium")
    store.add_task(title="Task Gamma", priority="high")
    return store


# ===========================================================================
# _load — private helper
# ===========================================================================


class TestLoad:
    def test_returns_empty_list_when_file_missing(self, tmp_path: Path) -> None:
        store = TaskStore(store_path=str(tmp_path / "nonexistent.json"))
        assert store._load() == []

    def test_returns_empty_list_for_corrupted_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{ this is not valid JSON !!!", encoding="utf-8")
        store = TaskStore(store_path=str(bad_file))
        assert store._load() == []

    def test_returns_empty_list_when_json_is_not_a_list(self, tmp_path: Path) -> None:
        obj_file = tmp_path / "obj.json"
        obj_file.write_text(json.dumps({"key": "value"}), encoding="utf-8")
        store = TaskStore(store_path=str(obj_file))
        assert store._load() == []

    def test_returns_tasks_from_valid_file(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Hello", priority="high")
        loaded = tmp_store._load()
        assert len(loaded) == 1
        assert loaded[0]["id"] == task["id"]


# ===========================================================================
# add_task
# ===========================================================================


class TestAddTask:
    def test_returns_task_dict_with_all_fields(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Fix login bug", priority="high")
        assert "id" in task
        assert "title" in task
        assert "status" in task
        assert "priority" in task
        assert "created_at" in task

    def test_id_is_valid_uuid4_string(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Check UUID", priority="medium")
        # Should not raise
        parsed = uuid.UUID(task["id"], version=4)
        assert str(parsed) == task["id"]

    def test_title_is_stored_correctly(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("My important task", priority="low")
        assert task["title"] == "My important task"

    def test_title_is_stripped_of_whitespace(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("  Padded title  ", priority="low")
        assert task["title"] == "Padded title"

    def test_initial_status_is_todo(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("New task", priority="medium")
        assert task["status"] == "todo"

    def test_priority_is_stored_correctly(self, tmp_store: TaskStore) -> None:
        for prio in VALID_PRIORITIES:
            task = tmp_store.add_task(f"Task with {prio} priority", priority=prio)
            assert task["priority"] == prio

    def test_created_at_is_iso8601_string(self, tmp_store: TaskStore) -> None:
        from datetime import datetime
        task = tmp_store.add_task("Timestamp check", priority="medium")
        # Should parse without error
        dt = datetime.fromisoformat(task["created_at"])
        assert dt is not None

    def test_task_is_persisted_to_file(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Persist me", priority="high")
        # Create a fresh store pointing at the same file
        fresh_store = TaskStore(store_path=tmp_store.store_path)
        tasks = fresh_store._load()
        assert len(tasks) == 1
        assert tasks[0]["id"] == task["id"]

    def test_multiple_tasks_accumulate(self, tmp_store: TaskStore) -> None:
        tmp_store.add_task("First", priority="low")
        tmp_store.add_task("Second", priority="medium")
        tmp_store.add_task("Third", priority="high")
        tasks = tmp_store._load()
        assert len(tasks) == 3

    def test_raises_value_error_for_empty_title(self, tmp_store: TaskStore) -> None:
        with pytest.raises(ValueError, match="empty"):
            tmp_store.add_task("", priority="medium")

    def test_raises_value_error_for_whitespace_only_title(self, tmp_store: TaskStore) -> None:
        with pytest.raises(ValueError, match="empty"):
            tmp_store.add_task("   ", priority="medium")

    def test_raises_value_error_for_invalid_priority(self, tmp_store: TaskStore) -> None:
        with pytest.raises(ValueError, match="priority"):
            tmp_store.add_task("Valid title", priority="urgent")

    def test_default_priority_is_medium(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Default priority task")
        assert task["priority"] == "medium"

    def test_each_task_gets_unique_id(self, tmp_store: TaskStore) -> None:
        t1 = tmp_store.add_task("Task 1")
        t2 = tmp_store.add_task("Task 2")
        assert t1["id"] != t2["id"]


# ===========================================================================
# list_tasks
# ===========================================================================


class TestListTasks:
    def test_returns_empty_list_when_no_tasks(self, tmp_store: TaskStore) -> None:
        assert tmp_store.list_tasks() == []

    def test_returns_all_tasks_without_filters(self, populated_store: TaskStore) -> None:
        tasks = populated_store.list_tasks()
        assert len(tasks) == 3

    def test_filter_by_status_todo(self, populated_store: TaskStore) -> None:
        tasks = populated_store.list_tasks(status="todo")
        assert all(t["status"] == "todo" for t in tasks)
        assert len(tasks) == 3  # all start as todo

    def test_filter_by_status_done_returns_only_done_tasks(
        self, populated_store: TaskStore
    ) -> None:
        all_tasks = populated_store.list_tasks()
        # Mark first task as done
        populated_store.update_status(all_tasks[0]["id"], "done")
        done_tasks = populated_store.list_tasks(status="done")
        assert len(done_tasks) == 1
        assert done_tasks[0]["status"] == "done"

    def test_filter_by_status_in_progress(self, populated_store: TaskStore) -> None:
        all_tasks = populated_store.list_tasks()
        populated_store.update_status(all_tasks[1]["id"], "in_progress")
        in_progress = populated_store.list_tasks(status="in_progress")
        assert len(in_progress) == 1
        assert in_progress[0]["status"] == "in_progress"

    def test_filter_by_priority_low(self, populated_store: TaskStore) -> None:
        tasks = populated_store.list_tasks(priority="low")
        assert len(tasks) == 1
        assert tasks[0]["priority"] == "low"
        assert tasks[0]["title"] == "Task Alpha"

    def test_filter_by_priority_high(self, populated_store: TaskStore) -> None:
        tasks = populated_store.list_tasks(priority="high")
        assert len(tasks) == 1
        assert tasks[0]["priority"] == "high"
        assert tasks[0]["title"] == "Task Gamma"

    def test_combined_status_and_priority_filter(self, populated_store: TaskStore) -> None:
        # All tasks are "todo" initially; filter for todo + medium
        tasks = populated_store.list_tasks(status="todo", priority="medium")
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Task Beta"
        assert tasks[0]["status"] == "todo"
        assert tasks[0]["priority"] == "medium"

    def test_combined_filter_returns_empty_when_no_match(
        self, populated_store: TaskStore
    ) -> None:
        tasks = populated_store.list_tasks(status="done", priority="high")
        assert tasks == []

    def test_tasks_sorted_by_created_at(self, tmp_store: TaskStore) -> None:
        t1 = tmp_store.add_task("First")
        t2 = tmp_store.add_task("Second")
        t3 = tmp_store.add_task("Third")
        tasks = tmp_store.list_tasks()
        ids = [t["id"] for t in tasks]
        assert ids == [t1["id"], t2["id"], t3["id"]]

    def test_raises_value_error_for_invalid_status(self, tmp_store: TaskStore) -> None:
        with pytest.raises(ValueError, match="status"):
            tmp_store.list_tasks(status="pending")

    def test_raises_value_error_for_invalid_priority(self, tmp_store: TaskStore) -> None:
        with pytest.raises(ValueError, match="priority"):
            tmp_store.list_tasks(priority="critical")

    def test_none_filters_return_all_tasks(self, populated_store: TaskStore) -> None:
        tasks = populated_store.list_tasks(status=None, priority=None)
        assert len(tasks) == 3


# ===========================================================================
# update_status
# ===========================================================================


class TestUpdateStatus:
    def test_updates_status_to_done(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Mark me done")
        updated = tmp_store.update_status(task["id"], "done")
        assert updated["status"] == "done"
        assert updated["id"] == task["id"]

    def test_updates_status_to_in_progress(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Start me")
        updated = tmp_store.update_status(task["id"], "in_progress")
        assert updated["status"] == "in_progress"

    def test_updates_status_back_to_todo(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Reopen me")
        tmp_store.update_status(task["id"], "done")
        updated = tmp_store.update_status(task["id"], "todo")
        assert updated["status"] == "todo"

    def test_update_is_persisted(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Persist status")
        tmp_store.update_status(task["id"], "done")
        fresh = TaskStore(store_path=tmp_store.store_path)
        tasks = fresh.list_tasks()
        assert tasks[0]["status"] == "done"

    def test_partial_id_prefix_match(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Prefix match")
        prefix = task["id"][:6]
        updated = tmp_store.update_status(prefix, "done")
        assert updated["id"] == task["id"]
        assert updated["status"] == "done"

    def test_partial_id_substring_match(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Substring match")
        # Use a middle portion of the UUID
        substring = task["id"][4:12]
        updated = tmp_store.update_status(substring, "in_progress")
        assert updated["id"] == task["id"]

    def test_raises_value_error_for_invalid_status(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Invalid status")
        with pytest.raises(ValueError, match="status"):
            tmp_store.update_status(task["id"], "archived")

    def test_raises_value_error_when_no_task_matches(self, tmp_store: TaskStore) -> None:
        tmp_store.add_task("Some task")
        with pytest.raises(ValueError, match="No task found"):
            tmp_store.update_status("zzz-nonexistent-zzz", "done")

    def test_raises_value_error_when_ambiguous_match(self, tmp_path: Path) -> None:
        """Force two tasks to share a common substring in their IDs."""
        store_file = tmp_path / "tasks.json"
        # Craft two tasks with known IDs that share a common substring
        shared = "aaaa"
        tasks = [
            {
                "id": f"{shared}0000-0000-0000-0000-000000000001",
                "title": "Task One",
                "status": "todo",
                "priority": "medium",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "id": f"{shared}0000-0000-0000-0000-000000000002",
                "title": "Task Two",
                "status": "todo",
                "priority": "medium",
                "created_at": "2026-01-01T00:00:01+00:00",
            },
        ]
        store_file.write_text(json.dumps(tasks), encoding="utf-8")
        store = TaskStore(store_path=str(store_file))
        with pytest.raises(ValueError, match="Ambiguous"):
            store.update_status(shared, "done")

    def test_other_tasks_unaffected_by_update(self, populated_store: TaskStore) -> None:
        all_tasks = populated_store.list_tasks()
        target = all_tasks[0]
        populated_store.update_status(target["id"], "done")
        remaining = populated_store.list_tasks(status="todo")
        assert len(remaining) == 2
        remaining_ids = {t["id"] for t in remaining}
        assert target["id"] not in remaining_ids


# ===========================================================================
# delete_task
# ===========================================================================


class TestDeleteTask:
    def test_returns_deleted_task_dict(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Delete me")
        deleted = tmp_store.delete_task(task["id"])
        assert deleted["id"] == task["id"]
        assert deleted["title"] == "Delete me"

    def test_task_no_longer_in_store_after_deletion(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Gone")
        tmp_store.delete_task(task["id"])
        tasks = tmp_store.list_tasks()
        assert all(t["id"] != task["id"] for t in tasks)

    def test_deletion_is_persisted(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Persist deletion")
        tmp_store.delete_task(task["id"])
        fresh = TaskStore(store_path=tmp_store.store_path)
        assert fresh.list_tasks() == []

    def test_only_target_task_is_deleted(self, populated_store: TaskStore) -> None:
        all_tasks = populated_store.list_tasks()
        target = all_tasks[1]  # delete the middle task
        populated_store.delete_task(target["id"])
        remaining = populated_store.list_tasks()
        assert len(remaining) == 2
        remaining_ids = {t["id"] for t in remaining}
        assert target["id"] not in remaining_ids

    def test_partial_id_prefix_match(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Prefix delete")
        prefix = task["id"][:8]
        deleted = tmp_store.delete_task(prefix)
        assert deleted["id"] == task["id"]
        assert tmp_store.list_tasks() == []

    def test_partial_id_substring_match(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Substring delete")
        substring = task["id"][5:13]
        deleted = tmp_store.delete_task(substring)
        assert deleted["id"] == task["id"]

    def test_raises_value_error_when_no_task_matches(self, tmp_store: TaskStore) -> None:
        tmp_store.add_task("Existing task")
        with pytest.raises(ValueError, match="No task found"):
            tmp_store.delete_task("zzz-nonexistent-zzz")

    def test_raises_value_error_when_ambiguous_match(self, tmp_path: Path) -> None:
        shared = "bbbb"
        tasks = [
            {
                "id": f"{shared}1111-1111-1111-1111-111111111111",
                "title": "Task A",
                "status": "todo",
                "priority": "low",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "id": f"{shared}2222-2222-2222-2222-222222222222",
                "title": "Task B",
                "status": "todo",
                "priority": "high",
                "created_at": "2026-01-01T00:00:01+00:00",
            },
        ]
        store_file = tmp_path / "tasks.json"
        store_file.write_text(json.dumps(tasks), encoding="utf-8")
        store = TaskStore(store_path=str(store_file))
        with pytest.raises(ValueError, match="Ambiguous"):
            store.delete_task(shared)

    def test_store_is_empty_after_deleting_all_tasks(self, populated_store: TaskStore) -> None:
        all_tasks = populated_store.list_tasks()
        for task in all_tasks:
            populated_store.delete_task(task["id"])
        assert populated_store.list_tasks() == []


# ===========================================================================
# _find_by_partial_id — private helper (tested via public API above,
# but also directly for completeness)
# ===========================================================================


class TestFindByPartialId:
    def test_exact_full_id_match(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Exact match")
        tasks = tmp_store._load()
        found = tmp_store._find_by_partial_id(task["id"], tasks)
        assert found["id"] == task["id"]

    def test_prefix_match(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Prefix")
        tasks = tmp_store._load()
        found = tmp_store._find_by_partial_id(task["id"][:10], tasks)
        assert found["id"] == task["id"]

    def test_strips_whitespace_from_partial_id(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Whitespace")
        tasks = tmp_store._load()
        padded = f"  {task['id'][:8]}  "
        found = tmp_store._find_by_partial_id(padded, tasks)
        assert found["id"] == task["id"]

    def test_raises_when_no_match(self, tmp_store: TaskStore) -> None:
        tmp_store.add_task("Some task")
        tasks = tmp_store._load()
        with pytest.raises(ValueError, match="No task found"):
            tmp_store._find_by_partial_id("zzz-no-match-zzz", tasks)

    def test_raises_when_multiple_matches(self, tmp_path: Path) -> None:
        shared = "cccc"
        tasks = [
            {
                "id": f"{shared}aaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "title": "T1",
                "status": "todo",
                "priority": "low",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "id": f"{shared}bbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "title": "T2",
                "status": "todo",
                "priority": "low",
                "created_at": "2026-01-01T00:00:01+00:00",
            },
        ]
        store_file = tmp_path / "tasks.json"
        store_file.write_text(json.dumps(tasks), encoding="utf-8")
        store = TaskStore(store_path=str(store_file))
        with pytest.raises(ValueError, match="Ambiguous"):
            store._find_by_partial_id(shared, tasks)
