"""Tests for team-test/task_store.py.

Covers:
- TaskStore initialisation (default path, custom path)
- add_task(): happy path, all priorities, empty title, invalid priority
- list_tasks(): no filters, status filter, priority filter, combined filters
- update_status(): happy path, partial id matching, not-found, ambiguous prefix,
  invalid status
- delete_task(): happy path, partial id matching, not-found, ambiguous prefix
- stats(): shape, correct counts, empty store
- _resolve_id(): exact match, prefix match, not-found, ambiguous prefix
- Integration: full end-to-end workflow (add → list → update → delete → stats)
- Isolation: each test uses a fresh temp file, never touching the user's real data
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure team-test directory is on sys.path
# ---------------------------------------------------------------------------
_TEAM_TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TEAM_TEST_DIR not in sys.path:
    sys.path.insert(0, _TEAM_TEST_DIR)

from models import Priority, Status, Task  # noqa: E402
from task_store import NotFoundError, TaskStore, ValidationError  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path: Path) -> TaskStore:
    """Return a fresh TaskStore backed by a temp file."""
    return TaskStore(path=str(tmp_path / "tasks.json"))


@pytest.fixture()
def store_with_task(store: TaskStore):
    """Return (store, task) with one task already added."""
    task = store.add_task("Initial task", priority="medium")
    return store, task


# ===========================================================================
# Initialisation
# ===========================================================================


class TestTaskStoreInit:
    def test_custom_path_accepted(self, tmp_path: Path):
        """TaskStore accepts a custom path without raising."""
        path = str(tmp_path / "custom_tasks.json")
        ts = TaskStore(path=path)
        # Should be usable immediately
        assert ts.list_tasks() == []

    def test_default_path_does_not_raise(self):
        """Constructing with no arguments uses the default path without error."""
        # We don't actually call any methods to avoid touching the real file.
        ts = TaskStore()
        assert ts._storage is not None

    def test_file_created_on_first_write(self, tmp_path: Path):
        """The JSON file is created when the first task is added."""
        path = str(tmp_path / "new_tasks.json")
        ts = TaskStore(path=path)
        ts.add_task("First task")
        assert os.path.exists(path)

    def test_parent_directory_created_automatically(self, tmp_path: Path):
        """Nested directories are created automatically."""
        path = str(tmp_path / "a" / "b" / "c" / "tasks.json")
        ts = TaskStore(path=path)
        ts.add_task("Nested dir task")
        assert os.path.exists(path)


# ===========================================================================
# add_task()
# ===========================================================================


class TestAddTask:
    def test_returns_task_instance(self, store: TaskStore):
        task = store.add_task("Write docs")
        assert isinstance(task, Task)

    def test_title_stored(self, store: TaskStore):
        task = store.add_task("Write docs")
        assert task.title == "Write docs"

    def test_default_priority_is_medium(self, store: TaskStore):
        task = store.add_task("Default priority")
        assert task.priority is Priority.MEDIUM

    def test_default_status_is_todo(self, store: TaskStore):
        task = store.add_task("Default status")
        assert task.status is Status.TODO

    def test_id_is_uuid(self, store: TaskStore):
        task = store.add_task("UUID check")
        parsed = uuid.UUID(task.id)
        assert str(parsed) == task.id

    def test_two_tasks_have_different_ids(self, store: TaskStore):
        t1 = store.add_task("Task A")
        t2 = store.add_task("Task B")
        assert t1.id != t2.id

    def test_explicit_priority_low(self, store: TaskStore):
        task = store.add_task("Low prio", priority="low")
        assert task.priority is Priority.LOW

    def test_explicit_priority_high(self, store: TaskStore):
        task = store.add_task("High prio", priority="high")
        assert task.priority is Priority.HIGH

    def test_explicit_priority_critical(self, store: TaskStore):
        task = store.add_task("Critical prio", priority="critical")
        assert task.priority is Priority.CRITICAL

    def test_priority_is_case_insensitive(self, store: TaskStore):
        task = store.add_task("Case test", priority="HIGH")
        assert task.priority is Priority.HIGH

    def test_task_persisted_to_disk(self, tmp_path: Path):
        path = str(tmp_path / "tasks.json")
        ts1 = TaskStore(path=path)
        task = ts1.add_task("Persisted task")

        ts2 = TaskStore(path=path)
        tasks = ts2.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].id == task.id
        assert tasks[0].title == "Persisted task"

    def test_empty_title_raises_validation_error(self, store: TaskStore):
        with pytest.raises(ValidationError):
            store.add_task("")

    def test_invalid_priority_raises_value_error(self, store: TaskStore):
        with pytest.raises(ValueError):
            store.add_task("Bad priority", priority="urgent")

    def test_multiple_tasks_added_sequentially(self, store: TaskStore):
        store.add_task("Task 1")
        store.add_task("Task 2")
        store.add_task("Task 3")
        assert len(store.list_tasks()) == 3


# ===========================================================================
# list_tasks()
# ===========================================================================


class TestListTasks:
    def test_empty_store_returns_empty_list(self, store: TaskStore):
        assert store.list_tasks() == []

    def test_returns_all_tasks_when_no_filter(self, store: TaskStore):
        store.add_task("Task A")
        store.add_task("Task B")
        store.add_task("Task C")
        tasks = store.list_tasks()
        assert len(tasks) == 3

    def test_returns_task_instances(self, store: TaskStore):
        store.add_task("Task A")
        tasks = store.list_tasks()
        assert all(isinstance(t, Task) for t in tasks)

    def test_ordered_by_created_at_ascending(self, store: TaskStore):
        t1 = store.add_task("First")
        t2 = store.add_task("Second")
        t3 = store.add_task("Third")
        listed = store.list_tasks()
        assert [t.id for t in listed] == [t1.id, t2.id, t3.id]

    def test_filter_by_status_todo(self, store: TaskStore):
        t1 = store.add_task("Todo task")
        t2 = store.add_task("Done task")
        store.update_status(t2.id, "done")
        result = store.list_tasks(status="todo")
        assert len(result) == 1
        assert result[0].id == t1.id

    def test_filter_by_status_done(self, store: TaskStore):
        t1 = store.add_task("Todo task")
        t2 = store.add_task("Done task")
        store.update_status(t2.id, "done")
        result = store.list_tasks(status="done")
        assert len(result) == 1
        assert result[0].id == t2.id

    def test_filter_by_status_in_progress(self, store: TaskStore):
        t1 = store.add_task("In progress task")
        t2 = store.add_task("Todo task")
        store.update_status(t1.id, "in_progress")
        result = store.list_tasks(status="in_progress")
        assert len(result) == 1
        assert result[0].id == t1.id

    def test_filter_by_priority_low(self, store: TaskStore):
        t1 = store.add_task("Low task", priority="low")
        t2 = store.add_task("High task", priority="high")
        result = store.list_tasks(priority="low")
        assert len(result) == 1
        assert result[0].id == t1.id

    def test_filter_by_priority_high(self, store: TaskStore):
        t1 = store.add_task("Low task", priority="low")
        t2 = store.add_task("High task", priority="high")
        result = store.list_tasks(priority="high")
        assert len(result) == 1
        assert result[0].id == t2.id

    def test_filter_by_priority_critical(self, store: TaskStore):
        t1 = store.add_task("Critical task", priority="critical")
        t2 = store.add_task("Medium task", priority="medium")
        result = store.list_tasks(priority="critical")
        assert len(result) == 1
        assert result[0].id == t1.id

    def test_combined_status_and_priority_filter(self, store: TaskStore):
        t1 = store.add_task("High todo", priority="high")
        t2 = store.add_task("High done", priority="high")
        t3 = store.add_task("Low todo", priority="low")
        store.update_status(t2.id, "done")
        result = store.list_tasks(status="todo", priority="high")
        assert len(result) == 1
        assert result[0].id == t1.id

    def test_filter_returns_empty_when_no_match(self, store: TaskStore):
        store.add_task("Todo task")
        result = store.list_tasks(status="done")
        assert result == []

    def test_invalid_status_raises_value_error(self, store: TaskStore):
        with pytest.raises(ValueError):
            store.list_tasks(status="invalid_status")

    def test_invalid_priority_raises_value_error(self, store: TaskStore):
        with pytest.raises(ValueError):
            store.list_tasks(priority="urgent")

    def test_status_filter_is_case_insensitive(self, store: TaskStore):
        t1 = store.add_task("Todo task")
        result = store.list_tasks(status="TODO")
        assert len(result) == 1
        assert result[0].id == t1.id

    def test_priority_filter_is_case_insensitive(self, store: TaskStore):
        t1 = store.add_task("High task", priority="high")
        result = store.list_tasks(priority="HIGH")
        assert len(result) == 1
        assert result[0].id == t1.id


# ===========================================================================
# update_status()
# ===========================================================================


class TestUpdateStatus:
    def test_updates_status_to_done(self, store_with_task):
        store, task = store_with_task
        updated = store.update_status(task.id, "done")
        assert updated.status is Status.DONE

    def test_updates_status_to_in_progress(self, store_with_task):
        store, task = store_with_task
        updated = store.update_status(task.id, "in_progress")
        assert updated.status is Status.IN_PROGRESS

    def test_updates_status_to_todo(self, store_with_task):
        store, task = store_with_task
        # First mark done, then reopen
        store.update_status(task.id, "done")
        updated = store.update_status(task.id, "todo")
        assert updated.status is Status.TODO

    def test_returns_task_instance(self, store_with_task):
        store, task = store_with_task
        result = store.update_status(task.id, "done")
        assert isinstance(result, Task)

    def test_update_persisted_to_disk(self, tmp_path: Path):
        path = str(tmp_path / "tasks.json")
        ts1 = TaskStore(path=path)
        task = ts1.add_task("Persist status")
        ts1.update_status(task.id, "done")

        ts2 = TaskStore(path=path)
        tasks = ts2.list_tasks()
        assert tasks[0].status is Status.DONE

    def test_partial_id_prefix_works(self, store: TaskStore):
        task = store.add_task("Partial id task")
        # Use first 8 characters as prefix
        updated = store.update_status(task.id[:8], "done")
        assert updated.status is Status.DONE

    def test_full_id_works(self, store_with_task):
        store, task = store_with_task
        updated = store.update_status(task.id, "done")
        assert updated.id == task.id

    def test_not_found_raises_not_found_error(self, store: TaskStore):
        with pytest.raises(NotFoundError, match="No task found"):
            store.update_status("nonexistent-id-xyz", "done")

    def test_ambiguous_prefix_raises_not_found_error(self, store: TaskStore):
        """Two tasks with the same prefix should raise NotFoundError."""
        # Create two tasks and use a prefix that matches both
        t1 = store.add_task("Task one")
        t2 = store.add_task("Task two")
        # Use a prefix that is common to both (empty string matches all)
        with pytest.raises(NotFoundError, match="Ambiguous"):
            store.update_status("", "done")

    def test_invalid_status_raises_value_error(self, store_with_task):
        store, task = store_with_task
        with pytest.raises(ValueError):
            store.update_status(task.id, "invalid_status")

    def test_status_is_case_insensitive(self, store_with_task):
        store, task = store_with_task
        updated = store.update_status(task.id, "DONE")
        assert updated.status is Status.DONE


# ===========================================================================
# delete_task()
# ===========================================================================


class TestDeleteTask:
    def test_task_removed_from_store(self, store_with_task):
        store, task = store_with_task
        store.delete_task(task.id)
        assert store.list_tasks() == []

    def test_delete_returns_none(self, store_with_task):
        store, task = store_with_task
        result = store.delete_task(task.id)
        assert result is None

    def test_delete_persisted_to_disk(self, tmp_path: Path):
        path = str(tmp_path / "tasks.json")
        ts1 = TaskStore(path=path)
        task = ts1.add_task("Delete me")
        ts1.delete_task(task.id)

        ts2 = TaskStore(path=path)
        assert ts2.list_tasks() == []

    def test_partial_id_prefix_works(self, store: TaskStore):
        task = store.add_task("Partial delete task")
        store.delete_task(task.id[:8])
        assert store.list_tasks() == []

    def test_full_id_works(self, store_with_task):
        store, task = store_with_task
        store.delete_task(task.id)
        assert store.list_tasks() == []

    def test_not_found_raises_not_found_error(self, store: TaskStore):
        with pytest.raises(NotFoundError, match="No task found"):
            store.delete_task("nonexistent-id-xyz")

    def test_ambiguous_prefix_raises_not_found_error(self, store: TaskStore):
        """Two tasks with the same prefix should raise NotFoundError."""
        store.add_task("Task one")
        store.add_task("Task two")
        with pytest.raises(NotFoundError, match="Ambiguous"):
            store.delete_task("")

    def test_other_tasks_unaffected(self, store: TaskStore):
        t1 = store.add_task("Keep me")
        t2 = store.add_task("Delete me")
        store.delete_task(t2.id)
        remaining = store.list_tasks()
        assert len(remaining) == 1
        assert remaining[0].id == t1.id


# ===========================================================================
# stats()
# ===========================================================================


class TestStats:
    def test_empty_store_returns_zero_counts(self, store: TaskStore):
        s = store.stats()
        assert s["total_tasks"] == 0
        assert s["total_projects"] == 0
        assert s["overdue_tasks"] == 0

    def test_stats_keys_present(self, store: TaskStore):
        s = store.stats()
        expected_keys = {
            "total_tasks",
            "total_projects",
            "tasks_by_status",
            "tasks_by_priority",
            "overdue_tasks",
        }
        assert set(s.keys()) == expected_keys

    def test_tasks_by_status_keys(self, store: TaskStore):
        s = store.stats()
        assert set(s["tasks_by_status"].keys()) == {"todo", "in_progress", "done"}

    def test_tasks_by_priority_keys(self, store: TaskStore):
        s = store.stats()
        assert set(s["tasks_by_priority"].keys()) == {"low", "medium", "high", "critical"}

    def test_total_tasks_count(self, store: TaskStore):
        store.add_task("Task A")
        store.add_task("Task B")
        store.add_task("Task C")
        s = store.stats()
        assert s["total_tasks"] == 3

    def test_tasks_by_status_counts(self, store: TaskStore):
        t1 = store.add_task("Todo task")
        t2 = store.add_task("In progress task")
        t3 = store.add_task("Done task")
        store.update_status(t2.id, "in_progress")
        store.update_status(t3.id, "done")
        s = store.stats()
        assert s["tasks_by_status"]["todo"] == 1
        assert s["tasks_by_status"]["in_progress"] == 1
        assert s["tasks_by_status"]["done"] == 1

    def test_tasks_by_priority_counts(self, store: TaskStore):
        store.add_task("Low task", priority="low")
        store.add_task("Medium task", priority="medium")
        store.add_task("High task", priority="high")
        store.add_task("Critical task", priority="critical")
        s = store.stats()
        assert s["tasks_by_priority"]["low"] == 1
        assert s["tasks_by_priority"]["medium"] == 1
        assert s["tasks_by_priority"]["high"] == 1
        assert s["tasks_by_priority"]["critical"] == 1

    def test_stats_after_delete(self, store: TaskStore):
        t1 = store.add_task("Task A")
        t2 = store.add_task("Task B")
        store.delete_task(t1.id)
        s = store.stats()
        assert s["total_tasks"] == 1

    def test_stats_after_status_update(self, store: TaskStore):
        t1 = store.add_task("Task A")
        store.update_status(t1.id, "done")
        s = store.stats()
        assert s["tasks_by_status"]["done"] == 1
        assert s["tasks_by_status"]["todo"] == 0


# ===========================================================================
# _resolve_id() (internal helper — tested indirectly via public methods)
# ===========================================================================


class TestResolveId:
    def test_exact_id_resolves(self, store: TaskStore):
        task = store.add_task("Exact match")
        # update_status uses _resolve_id internally
        updated = store.update_status(task.id, "done")
        assert updated.id == task.id

    def test_prefix_resolves_to_unique_task(self, store: TaskStore):
        task = store.add_task("Prefix match")
        # Use first 12 characters as prefix
        updated = store.update_status(task.id[:12], "done")
        assert updated.id == task.id

    def test_no_match_raises_not_found_error(self, store: TaskStore):
        with pytest.raises(NotFoundError, match="No task found"):
            store.update_status("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx", "done")

    def test_ambiguous_prefix_raises_not_found_error(self, store: TaskStore):
        store.add_task("Task one")
        store.add_task("Task two")
        # Empty string matches all tasks → ambiguous
        with pytest.raises(NotFoundError, match="Ambiguous"):
            store.delete_task("")

    def test_single_char_prefix_resolves_when_unique(self, store: TaskStore):
        """A single-character prefix works if it uniquely identifies one task."""
        # Create a task and use the first character of its UUID as prefix.
        # This is only reliable if there's exactly one task in the store.
        task = store.add_task("Single char prefix")
        first_char = task.id[0]
        # Clear and re-add to ensure uniqueness
        store2 = TaskStore(path=store._storage._path)
        # Since we only have one task, the first char should be unique
        updated = store.update_status(first_char, "done")
        assert updated.status is Status.DONE


# ===========================================================================
# Integration: full end-to-end workflow
# ===========================================================================


class TestIntegration:
    def test_full_workflow(self, tmp_path: Path):
        """Add → list → update status → delete → stats."""
        path = str(tmp_path / "tasks.json")
        store = TaskStore(path=path)

        # 1. Add tasks
        t1 = store.add_task("Task Alpha", priority="high")
        t2 = store.add_task("Task Beta", priority="low")
        t3 = store.add_task("Task Gamma", priority="medium")

        # 2. List all tasks
        tasks = store.list_tasks()
        assert len(tasks) == 3
        titles = {t.title for t in tasks}
        assert "Task Alpha" in titles
        assert "Task Beta" in titles
        assert "Task Gamma" in titles

        # 3. Update status
        store.update_status(t1.id, "in_progress")
        store.update_status(t2.id, "done")

        # 4. Filter by status
        todo_tasks = store.list_tasks(status="todo")
        assert len(todo_tasks) == 1
        assert todo_tasks[0].id == t3.id

        done_tasks = store.list_tasks(status="done")
        assert len(done_tasks) == 1
        assert done_tasks[0].id == t2.id

        # 5. Stats
        s = store.stats()
        assert s["total_tasks"] == 3
        assert s["tasks_by_status"]["todo"] == 1
        assert s["tasks_by_status"]["in_progress"] == 1
        assert s["tasks_by_status"]["done"] == 1
        assert s["tasks_by_priority"]["high"] == 1
        assert s["tasks_by_priority"]["low"] == 1
        assert s["tasks_by_priority"]["medium"] == 1

        # 6. Delete a task
        store.delete_task(t2.id)
        assert len(store.list_tasks()) == 2

        # 7. Reload from disk and verify state is preserved
        store2 = TaskStore(path=path)
        tasks2 = store2.list_tasks()
        assert len(tasks2) == 2
        ids = {t.id for t in tasks2}
        assert t1.id in ids
        assert t3.id in ids
        assert t2.id not in ids

    def test_priority_filter_workflow(self, store: TaskStore):
        """Add tasks with different priorities and filter them."""
        store.add_task("Low task", priority="low")
        store.add_task("Medium task", priority="medium")
        store.add_task("High task", priority="high")
        store.add_task("Critical task", priority="critical")

        assert len(store.list_tasks(priority="low")) == 1
        assert len(store.list_tasks(priority="medium")) == 1
        assert len(store.list_tasks(priority="high")) == 1
        assert len(store.list_tasks(priority="critical")) == 1

        # Verify titles
        assert store.list_tasks(priority="high")[0].title == "High task"
        assert store.list_tasks(priority="critical")[0].title == "Critical task"

    def test_status_lifecycle(self, store: TaskStore):
        """Task goes through full status lifecycle: todo → in_progress → done → todo."""
        task = store.add_task("Lifecycle task")
        assert task.status is Status.TODO

        updated = store.update_status(task.id, "in_progress")
        assert updated.status is Status.IN_PROGRESS

        updated = store.update_status(task.id, "done")
        assert updated.status is Status.DONE

        updated = store.update_status(task.id, "todo")
        assert updated.status is Status.TODO

    def test_partial_id_workflow(self, store: TaskStore):
        """Partial id matching works for both update and delete."""
        task = store.add_task("Partial id task")
        prefix = task.id[:10]

        # Update via prefix
        updated = store.update_status(prefix, "in_progress")
        assert updated.status is Status.IN_PROGRESS

        # Delete via prefix
        store.delete_task(prefix)
        assert store.list_tasks() == []

    def test_data_isolation_between_stores(self, tmp_path: Path):
        """Two TaskStore instances pointing to different files are independent."""
        path1 = str(tmp_path / "store1.json")
        path2 = str(tmp_path / "store2.json")

        ts1 = TaskStore(path=path1)
        ts2 = TaskStore(path=path2)

        ts1.add_task("Store 1 task")
        ts2.add_task("Store 2 task")

        assert len(ts1.list_tasks()) == 1
        assert len(ts2.list_tasks()) == 1
        assert ts1.list_tasks()[0].title == "Store 1 task"
        assert ts2.list_tasks()[0].title == "Store 2 task"

    def test_stats_reflects_all_changes(self, store: TaskStore):
        """Stats accurately reflect the current state after multiple operations."""
        t1 = store.add_task("Task 1", priority="high")
        t2 = store.add_task("Task 2", priority="low")
        t3 = store.add_task("Task 3", priority="medium")

        store.update_status(t1.id, "done")
        store.update_status(t2.id, "in_progress")
        store.delete_task(t3.id)

        s = store.stats()
        assert s["total_tasks"] == 2
        assert s["tasks_by_status"]["done"] == 1
        assert s["tasks_by_status"]["in_progress"] == 1
        assert s["tasks_by_status"]["todo"] == 0
        assert s["tasks_by_priority"]["high"] == 1
        assert s["tasks_by_priority"]["low"] == 1
        assert s["tasks_by_priority"]["medium"] == 0
