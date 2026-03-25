"""
test_store.py — Unit tests for task_tracker.task_store.TaskStore.

Covers:
  - add_task: happy path, blank title, invalid priority
  - list_tasks: no filter, status filter, priority filter, combined filter, invalid args
  - update_status: happy path, invalid status, not found, ambiguous
  - delete_task: happy path, not found, ambiguous
  - get_stats: empty store, populated store
  - _match: exact, prefix, substring, not found, ambiguous
  - Persistence: data survives a new TaskStore instance pointing at the same file
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from task_tracker.task_store import (
    AmbiguousIDError,
    TaskNotFoundError,
    TaskStore,
    VALID_PRIORITIES,
    VALID_STATUSES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_store(tmp_path: Path) -> TaskStore:
    """Return a fresh TaskStore backed by a temp file."""
    return TaskStore(path=tmp_path / "tasks.json")


@pytest.fixture()
def populated_store(tmp_path: Path) -> TaskStore:
    """Return a TaskStore with three pre-loaded tasks."""
    store = TaskStore(path=tmp_path / "tasks.json")
    store.add_task("Alpha task", priority="high")
    store.add_task("Beta task", priority="medium")
    store.add_task("Gamma task", priority="low")
    return store


# ---------------------------------------------------------------------------
# add_task
# ---------------------------------------------------------------------------


class TestAddTask:
    def test_add_returns_task_dict(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Do something", priority="medium")
        assert isinstance(task, dict)
        assert task["title"] == "Do something"
        assert task["status"] == "todo"
        assert task["priority"] == "medium"
        assert "id" in task
        assert "created_at" in task

    def test_add_increments_list(self, tmp_store: TaskStore) -> None:
        tmp_store.add_task("Task 1")
        tmp_store.add_task("Task 2")
        assert len(tmp_store.list_tasks()) == 2

    def test_add_strips_whitespace_from_title(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("  Padded title  ")
        assert task["title"] == "Padded title"

    def test_add_blank_title_raises(self, tmp_store: TaskStore) -> None:
        with pytest.raises(ValueError, match="empty"):
            tmp_store.add_task("   ")

    def test_add_invalid_priority_raises(self, tmp_store: TaskStore) -> None:
        with pytest.raises(ValueError, match="priority"):
            tmp_store.add_task("Task", priority="urgent")

    def test_add_all_valid_priorities(self, tmp_store: TaskStore) -> None:
        for p in ("low", "medium", "high"):
            task = tmp_store.add_task(f"Task {p}", priority=p)
            assert task["priority"] == p

    def test_add_default_priority_is_medium(self, tmp_store: TaskStore) -> None:
        task = tmp_store.add_task("Default priority task")
        assert task["priority"] == "medium"

    def test_add_persists_to_disk(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.json"
        store = TaskStore(path=path)
        store.add_task("Persisted task")
        # Re-open the file and verify
        data = json.loads(path.read_text())
        assert len(data) == 1
        assert data[0]["title"] == "Persisted task"


# ---------------------------------------------------------------------------
# list_tasks
# ---------------------------------------------------------------------------


class TestListTasks:
    def test_list_empty_store(self, tmp_store: TaskStore) -> None:
        assert tmp_store.list_tasks() == []

    def test_list_all_tasks(self, populated_store: TaskStore) -> None:
        tasks = populated_store.list_tasks()
        assert len(tasks) == 3

    def test_list_filter_by_status_todo(self, populated_store: TaskStore) -> None:
        tasks = populated_store.list_tasks(status="todo")
        assert len(tasks) == 3
        assert all(t["status"] == "todo" for t in tasks)

    def test_list_filter_by_status_done(self, populated_store: TaskStore) -> None:
        # Mark one done first
        first_id = populated_store.list_tasks()[0]["id"]
        populated_store.update_status(first_id, "done")
        done_tasks = populated_store.list_tasks(status="done")
        assert len(done_tasks) == 1
        assert done_tasks[0]["status"] == "done"

    def test_list_filter_by_priority(self, populated_store: TaskStore) -> None:
        high_tasks = populated_store.list_tasks(priority="high")
        assert len(high_tasks) == 1
        assert high_tasks[0]["priority"] == "high"

    def test_list_combined_filter(self, populated_store: TaskStore) -> None:
        # All tasks are todo; only one is high priority
        tasks = populated_store.list_tasks(status="todo", priority="high")
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Alpha task"

    def test_list_invalid_status_raises(self, tmp_store: TaskStore) -> None:
        with pytest.raises(ValueError, match="status"):
            tmp_store.list_tasks(status="pending")

    def test_list_invalid_priority_raises(self, tmp_store: TaskStore) -> None:
        with pytest.raises(ValueError, match="priority"):
            tmp_store.list_tasks(priority="critical")


# ---------------------------------------------------------------------------
# update_status
# ---------------------------------------------------------------------------


class TestUpdateStatus:
    def test_update_status_done(self, populated_store: TaskStore) -> None:
        task_id = populated_store.list_tasks()[0]["id"]
        updated = populated_store.update_status(task_id, "done")
        assert updated["status"] == "done"
        assert updated["id"] == task_id

    def test_update_status_in_progress(self, populated_store: TaskStore) -> None:
        task_id = populated_store.list_tasks()[1]["id"]
        updated = populated_store.update_status(task_id, "in_progress")
        assert updated["status"] == "in_progress"

    def test_update_status_persists(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.json"
        store = TaskStore(path=path)
        task = store.add_task("Persist status")
        store.update_status(task["id"], "done")
        # Re-load from disk
        store2 = TaskStore(path=path)
        tasks = store2.list_tasks()
        assert tasks[0]["status"] == "done"

    def test_update_invalid_status_raises(self, populated_store: TaskStore) -> None:
        task_id = populated_store.list_tasks()[0]["id"]
        with pytest.raises(ValueError, match="status"):
            populated_store.update_status(task_id, "finished")

    def test_update_not_found_raises(self, populated_store: TaskStore) -> None:
        with pytest.raises(TaskNotFoundError) as exc_info:
            populated_store.update_status("zzzzzzzzz", "done")
        assert "zzzzzzzzz" in str(exc_info.value)

    def test_update_partial_id(self, populated_store: TaskStore) -> None:
        full_id = populated_store.list_tasks()[0]["id"]
        partial = full_id[:8]
        updated = populated_store.update_status(partial, "done")
        assert updated["id"] == full_id
        assert updated["status"] == "done"


# ---------------------------------------------------------------------------
# delete_task
# ---------------------------------------------------------------------------


class TestDeleteTask:
    def test_delete_removes_task(self, populated_store: TaskStore) -> None:
        before = populated_store.list_tasks()
        task_id = before[0]["id"]
        deleted = populated_store.delete_task(task_id)
        assert deleted["id"] == task_id
        after = populated_store.list_tasks()
        assert len(after) == len(before) - 1
        assert all(t["id"] != task_id for t in after)

    def test_delete_returns_snapshot(self, populated_store: TaskStore) -> None:
        task = populated_store.list_tasks()[0]
        deleted = populated_store.delete_task(task["id"])
        assert deleted["title"] == task["title"]
        assert deleted["priority"] == task["priority"]

    def test_delete_not_found_raises(self, populated_store: TaskStore) -> None:
        with pytest.raises(TaskNotFoundError):
            populated_store.delete_task("nonexistent000")

    def test_delete_persists(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.json"
        store = TaskStore(path=path)
        task = store.add_task("To be deleted")
        store.delete_task(task["id"])
        store2 = TaskStore(path=path)
        assert store2.list_tasks() == []


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------


class TestGetStats:
    def test_stats_empty_store(self, tmp_store: TaskStore) -> None:
        stats = tmp_store.get_stats()
        assert stats["total"] == 0
        assert stats["by_status"]["todo"] == 0
        assert stats["by_status"]["done"] == 0
        assert stats["by_priority"]["high"] == 0

    def test_stats_counts_correctly(self, populated_store: TaskStore) -> None:
        stats = populated_store.get_stats()
        assert stats["total"] == 3
        assert stats["by_status"]["todo"] == 3
        assert stats["by_priority"]["high"] == 1
        assert stats["by_priority"]["medium"] == 1
        assert stats["by_priority"]["low"] == 1

    def test_stats_after_done(self, populated_store: TaskStore) -> None:
        task_id = populated_store.list_tasks()[0]["id"]
        populated_store.update_status(task_id, "done")
        stats = populated_store.get_stats()
        assert stats["by_status"]["done"] == 1
        assert stats["by_status"]["todo"] == 2

    def test_stats_has_all_status_keys(self, tmp_store: TaskStore) -> None:
        stats = tmp_store.get_stats()
        for s in VALID_STATUSES:
            assert s in stats["by_status"]

    def test_stats_has_all_priority_keys(self, tmp_store: TaskStore) -> None:
        stats = tmp_store.get_stats()
        for p in VALID_PRIORITIES:
            assert p in stats["by_priority"]


# ---------------------------------------------------------------------------
# _match (partial ID resolution)
# ---------------------------------------------------------------------------


class TestMatch:
    def test_match_exact_id(self, populated_store: TaskStore) -> None:
        task = populated_store.list_tasks()[0]
        matched = populated_store._match(task["id"])
        assert matched["id"] == task["id"]

    def test_match_prefix(self, populated_store: TaskStore) -> None:
        task = populated_store.list_tasks()[0]
        matched = populated_store._match(task["id"][:8])
        assert matched["id"] == task["id"]

    def test_match_not_found_raises(self, populated_store: TaskStore) -> None:
        with pytest.raises(TaskNotFoundError) as exc_info:
            populated_store._match("00000000")
        assert "00000000" in str(exc_info.value)

    def test_match_ambiguous_raises(self, tmp_path: Path) -> None:
        """Force two tasks to share a common prefix to trigger AmbiguousIDError."""
        path = tmp_path / "tasks.json"
        store = TaskStore(path=path)
        # Inject two tasks with IDs sharing the same 4-char prefix
        store._tasks = [
            {
                "id": "abcd1111111111111111111111111111",
                "title": "Task A",
                "status": "todo",
                "priority": "medium",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
            {
                "id": "abcd2222222222222222222222222222",
                "title": "Task B",
                "status": "todo",
                "priority": "medium",
                "created_at": "2026-01-01T00:00:00+00:00",
            },
        ]
        with pytest.raises(AmbiguousIDError) as exc_info:
            store._match("abcd")
        assert "abcd" in str(exc_info.value)
        assert len(exc_info.value.candidates) == 2


# ---------------------------------------------------------------------------
# Persistence / env var
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_data_survives_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "tasks.json"
        store1 = TaskStore(path=path)
        store1.add_task("Survive reload", priority="high")
        store2 = TaskStore(path=path)
        tasks = store2.list_tasks()
        assert len(tasks) == 1
        assert tasks[0]["title"] == "Survive reload"
        assert tasks[0]["priority"] == "high"

    def test_env_var_overrides_default_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        custom_path = str(tmp_path / "custom_tasks.json")
        monkeypatch.setenv("TASK_TRACKER_FILE", custom_path)
        store = TaskStore()
        store.add_task("Env var task")
        assert Path(custom_path).exists()
        data = json.loads(Path(custom_path).read_text())
        assert data[0]["title"] == "Env var task"

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "nonexistent.json"
        store = TaskStore(path=path)
        assert store.list_tasks() == []
