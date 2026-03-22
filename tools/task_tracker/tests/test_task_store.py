"""
test_task_store.py — Unit tests for the TaskStore data layer.

Covers:
  - add_task: happy path, blank title, invalid priority
  - list_tasks: no filter, status filter, priority filter, combined filter,
                invalid filter values, sort order
  - update_status: happy path, invalid status, no match, ambiguous match
  - delete_task: happy path, no match, ambiguous match
  - _load: missing file, corrupted JSON, non-list JSON
  - _find_by_partial_id: exact match, prefix match, ambiguous, no match
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from task_store import TaskStore, VALID_STATUSES, VALID_PRIORITIES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> TaskStore:
    """Return a TaskStore backed by a temporary file."""
    return TaskStore(store_path=str(tmp_path / "tasks.json"))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _add_n(store: TaskStore, n: int, priority: str = "medium") -> list[dict]:
    """Add *n* tasks to *store* and return them."""
    return [store.add_task(f"Task {i}", priority=priority) for i in range(n)]


# ===========================================================================
# add_task
# ===========================================================================


class TestAddTask:
    def test_returns_dict_with_required_fields(self, store: TaskStore) -> None:
        task = store.add_task("Write docs")
        assert isinstance(task, dict)
        for field in ("id", "title", "status", "priority", "created_at"):
            assert field in task, f"Missing field: {field}"

    def test_default_status_is_todo(self, store: TaskStore) -> None:
        task = store.add_task("Default status task")
        assert task["status"] == "todo"

    def test_default_priority_is_medium(self, store: TaskStore) -> None:
        task = store.add_task("Default priority task")
        assert task["priority"] == "medium"

    def test_title_stored_correctly(self, store: TaskStore) -> None:
        task = store.add_task("  My Task  ")
        assert task["title"] == "My Task"  # stripped

    def test_id_is_valid_uuid(self, store: TaskStore) -> None:
        task = store.add_task("UUID check")
        # Should not raise
        uuid.UUID(task["id"])

    def test_created_at_is_iso_format(self, store: TaskStore) -> None:
        from datetime import datetime
        task = store.add_task("Timestamp check")
        # Should parse without error
        dt = datetime.fromisoformat(task["created_at"])
        assert dt is not None

    def test_high_priority(self, store: TaskStore) -> None:
        task = store.add_task("Urgent", priority="high")
        assert task["priority"] == "high"

    def test_low_priority(self, store: TaskStore) -> None:
        task = store.add_task("Chill", priority="low")
        assert task["priority"] == "low"

    def test_task_persisted_to_disk(self, store: TaskStore, tmp_path: Path) -> None:
        task = store.add_task("Persist me")
        store_file = tmp_path / "tasks.json"
        assert store_file.exists()
        data = json.loads(store_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["id"] == task["id"]

    def test_multiple_tasks_accumulate(self, store: TaskStore) -> None:
        store.add_task("Task A")
        store.add_task("Task B")
        store.add_task("Task C")
        tasks = store.list_tasks()
        assert len(tasks) == 3

    def test_blank_title_raises_value_error(self, store: TaskStore) -> None:
        with pytest.raises(ValueError, match="empty"):
            store.add_task("")

    def test_whitespace_only_title_raises_value_error(self, store: TaskStore) -> None:
        with pytest.raises(ValueError, match="empty"):
            store.add_task("   ")

    def test_invalid_priority_raises_value_error(self, store: TaskStore) -> None:
        with pytest.raises(ValueError, match="priority"):
            store.add_task("Bad priority", priority="critical")

    def test_all_valid_priorities_accepted(self, store: TaskStore) -> None:
        for p in VALID_PRIORITIES:
            task = store.add_task(f"Task with {p}", priority=p)
            assert task["priority"] == p


# ===========================================================================
# list_tasks
# ===========================================================================


class TestListTasks:
    def test_empty_store_returns_empty_list(self, store: TaskStore) -> None:
        result = store.list_tasks()
        assert result == []

    def test_returns_all_tasks_without_filter(self, store: TaskStore) -> None:
        _add_n(store, 3)
        result = store.list_tasks()
        assert len(result) == 3

    def test_filter_by_status_todo(self, store: TaskStore) -> None:
        t1 = store.add_task("Todo task")
        t2 = store.add_task("Another task")
        # Update one to in_progress
        store.update_status(t2["id"], "in_progress")
        result = store.list_tasks(status="todo")
        ids = [t["id"] for t in result]
        assert t1["id"] in ids
        assert t2["id"] not in ids

    def test_filter_by_status_done(self, store: TaskStore) -> None:
        t1 = store.add_task("Done task")
        store.update_status(t1["id"], "done")
        store.add_task("Still todo")
        result = store.list_tasks(status="done")
        assert len(result) == 1
        assert result[0]["id"] == t1["id"]

    def test_filter_by_priority_high(self, store: TaskStore) -> None:
        t_high = store.add_task("High priority", priority="high")
        store.add_task("Low priority", priority="low")
        result = store.list_tasks(priority="high")
        assert len(result) == 1
        assert result[0]["id"] == t_high["id"]

    def test_filter_by_priority_low(self, store: TaskStore) -> None:
        store.add_task("High priority", priority="high")
        t_low = store.add_task("Low priority", priority="low")
        result = store.list_tasks(priority="low")
        assert len(result) == 1
        assert result[0]["id"] == t_low["id"]

    def test_combined_status_and_priority_filter(self, store: TaskStore) -> None:
        t1 = store.add_task("High todo", priority="high")
        t2 = store.add_task("High done", priority="high")
        store.add_task("Low todo", priority="low")
        store.update_status(t2["id"], "done")
        result = store.list_tasks(status="todo", priority="high")
        assert len(result) == 1
        assert result[0]["id"] == t1["id"]

    def test_sorted_by_created_at_ascending(self, store: TaskStore) -> None:
        tasks = _add_n(store, 5)
        result = store.list_tasks()
        result_ids = [t["id"] for t in result]
        expected_ids = [t["id"] for t in tasks]
        assert result_ids == expected_ids

    def test_invalid_status_raises_value_error(self, store: TaskStore) -> None:
        with pytest.raises(ValueError, match="status"):
            store.list_tasks(status="pending")

    def test_invalid_priority_raises_value_error(self, store: TaskStore) -> None:
        with pytest.raises(ValueError, match="priority"):
            store.list_tasks(priority="urgent")

    def test_filter_returns_empty_when_no_match(self, store: TaskStore) -> None:
        store.add_task("Low task", priority="low")
        result = store.list_tasks(priority="high")
        assert result == []

    def test_all_valid_statuses_accepted(self, store: TaskStore) -> None:
        for s in VALID_STATUSES:
            # Should not raise
            store.list_tasks(status=s)

    def test_all_valid_priorities_accepted(self, store: TaskStore) -> None:
        for p in VALID_PRIORITIES:
            # Should not raise
            store.list_tasks(priority=p)


# ===========================================================================
# update_status
# ===========================================================================


class TestUpdateStatus:
    def test_updates_status_and_returns_task(self, store: TaskStore) -> None:
        task = store.add_task("Update me")
        updated = store.update_status(task["id"], "in_progress")
        assert updated["status"] == "in_progress"
        assert updated["id"] == task["id"]

    def test_update_persisted_to_disk(self, store: TaskStore) -> None:
        task = store.add_task("Persist update")
        store.update_status(task["id"], "done")
        # Reload via a fresh store instance
        reloaded = store.list_tasks()
        assert reloaded[0]["status"] == "done"

    def test_update_with_partial_id(self, store: TaskStore) -> None:
        task = store.add_task("Partial ID update")
        partial = task["id"][:8]
        updated = store.update_status(partial, "done")
        assert updated["status"] == "done"

    def test_all_valid_statuses_accepted(self, store: TaskStore) -> None:
        task = store.add_task("Status cycle")
        for s in VALID_STATUSES:
            updated = store.update_status(task["id"], s)
            assert updated["status"] == s

    def test_invalid_status_raises_value_error(self, store: TaskStore) -> None:
        task = store.add_task("Invalid status")
        with pytest.raises(ValueError, match="status"):
            store.update_status(task["id"], "pending")

    def test_no_match_raises_value_error(self, store: TaskStore) -> None:
        store.add_task("Some task")
        with pytest.raises(ValueError, match="No task found"):
            store.update_status("nonexistent-id-xyz", "done")

    def test_ambiguous_partial_id_raises_value_error(self, store: TaskStore) -> None:
        # Force two tasks with a shared prefix by manipulating the store file
        store_path = store.store_path
        tasks = [
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
                "created_at": "2026-01-01T00:00:01+00:00",
            },
        ]
        store_path.write_text(json.dumps(tasks))
        with pytest.raises(ValueError, match="Ambiguous"):
            store.update_status("aaaa", "done")


# ===========================================================================
# delete_task
# ===========================================================================


class TestDeleteTask:
    def test_deletes_task_and_returns_snapshot(self, store: TaskStore) -> None:
        task = store.add_task("Delete me")
        deleted = store.delete_task(task["id"])
        assert deleted["id"] == task["id"]
        assert deleted["title"] == "Delete me"

    def test_task_removed_from_store(self, store: TaskStore) -> None:
        task = store.add_task("Remove this")
        store.delete_task(task["id"])
        remaining = store.list_tasks()
        assert all(t["id"] != task["id"] for t in remaining)

    def test_other_tasks_unaffected(self, store: TaskStore) -> None:
        t1 = store.add_task("Keep me")
        t2 = store.add_task("Delete me")
        store.delete_task(t2["id"])
        remaining = store.list_tasks()
        assert len(remaining) == 1
        assert remaining[0]["id"] == t1["id"]

    def test_delete_with_partial_id(self, store: TaskStore) -> None:
        task = store.add_task("Partial delete")
        partial = task["id"][:8]
        deleted = store.delete_task(partial)
        assert deleted["id"] == task["id"]
        assert store.list_tasks() == []

    def test_no_match_raises_value_error(self, store: TaskStore) -> None:
        store.add_task("Some task")
        with pytest.raises(ValueError, match="No task found"):
            store.delete_task("nonexistent-xyz-999")

    def test_ambiguous_partial_id_raises_value_error(self, store: TaskStore) -> None:
        store_path = store.store_path
        tasks = [
            {
                "id": "bbbb1111-0000-0000-0000-000000000001",
                "title": "Task X",
                "status": "todo",
                "priority": "medium",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "id": "bbbb2222-0000-0000-0000-000000000002",
                "title": "Task Y",
                "status": "todo",
                "priority": "medium",
                "created_at": "2026-01-01T00:00:01+00:00",
            },
        ]
        store_path.write_text(json.dumps(tasks))
        with pytest.raises(ValueError, match="Ambiguous"):
            store.delete_task("bbbb")

    def test_delete_all_tasks_leaves_empty_store(self, store: TaskStore) -> None:
        tasks = _add_n(store, 3)
        for t in tasks:
            store.delete_task(t["id"])
        assert store.list_tasks() == []


# ===========================================================================
# _load — resilience tests
# ===========================================================================


class TestLoad:
    def test_missing_file_returns_empty_list(self, tmp_path: Path) -> None:
        s = TaskStore(store_path=str(tmp_path / "nonexistent.json"))
        assert s._load() == []

    def test_corrupted_json_returns_empty_list(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.json"
        p.write_text("{ this is not valid json !!!")
        s = TaskStore(store_path=str(p))
        assert s._load() == []

    def test_non_list_json_returns_empty_list(self, tmp_path: Path) -> None:
        p = tmp_path / "obj.json"
        p.write_text(json.dumps({"key": "value"}))
        s = TaskStore(store_path=str(p))
        assert s._load() == []

    def test_valid_json_array_loaded_correctly(self, tmp_path: Path) -> None:
        p = tmp_path / "tasks.json"
        data = [{"id": "abc", "title": "T", "status": "todo",
                 "priority": "medium", "created_at": "2026-01-01T00:00:00+00:00"}]
        p.write_text(json.dumps(data))
        s = TaskStore(store_path=str(p))
        loaded = s._load()
        assert len(loaded) == 1
        assert loaded[0]["id"] == "abc"


# ===========================================================================
# _save — directory creation
# ===========================================================================


class TestSave:
    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "tasks.json"
        s = TaskStore(store_path=str(nested))
        s.add_task("Nested save")
        assert nested.exists()
        data = json.loads(nested.read_text())
        assert len(data) == 1
