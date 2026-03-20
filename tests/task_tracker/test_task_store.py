"""Comprehensive tests for task_tracker/task_store.py.

Covers:
- TaskStatus and TaskPriority enumerations (from_string, label, str-comparison)
- Task dataclass (to_dict / from_dict round-trip, __eq__, __hash__, __repr__)
- TaskNotFoundError / TaskStoreError custom exceptions
- Internal helpers (_format_dt, _parse_dt, _now_utc, _new_id)
- TaskStore CRUD: add, get, update, update_status, delete
- TaskStore queries: list_tasks (all filter combos), count, all_tags
- TaskStore dunder helpers: __iter__, __len__, __contains__
- TaskStore persistence: reload, atomic save, corrupt-file error paths
- Full end-to-end integration flow
"""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from task_tracker.task_store import (
    DEFAULT_STORE_DIR,
    DEFAULT_STORE_FILE,
    Task,
    TaskNotFoundError,
    TaskPriority,
    TaskStatus,
    TaskStore,
    TaskStoreError,
    _format_dt,
    _new_id,
    _now_utc,
    _parse_dt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_store(tmp_path: Path) -> TaskStore:
    """Return a TaskStore backed by a temporary file."""
    return TaskStore(store_path=tmp_path / "tasks.json")


# ---------------------------------------------------------------------------
# TaskStatus
# ---------------------------------------------------------------------------


class TestTaskStatus:
    def test_values(self):
        assert TaskStatus.TODO.value == "todo"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.DONE.value == "done"

    def test_str_comparison(self):
        # TaskStatus inherits from str
        assert TaskStatus.TODO == "todo"
        assert TaskStatus.IN_PROGRESS == "in_progress"
        assert TaskStatus.DONE == "done"

    def test_from_string_exact(self):
        assert TaskStatus.from_string("todo") is TaskStatus.TODO
        assert TaskStatus.from_string("in_progress") is TaskStatus.IN_PROGRESS
        assert TaskStatus.from_string("done") is TaskStatus.DONE

    def test_from_string_case_insensitive(self):
        assert TaskStatus.from_string("TODO") is TaskStatus.TODO
        assert TaskStatus.from_string("In_Progress") is TaskStatus.IN_PROGRESS
        assert TaskStatus.from_string("DONE") is TaskStatus.DONE

    def test_from_string_strips_whitespace(self):
        assert TaskStatus.from_string("  todo  ") is TaskStatus.TODO

    def test_from_string_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown task status"):
            TaskStatus.from_string("pending")

    def test_from_string_error_message_contains_valid_values(self):
        with pytest.raises(ValueError) as exc_info:
            TaskStatus.from_string("bad")
        msg = str(exc_info.value)
        assert "todo" in msg
        assert "in_progress" in msg
        assert "done" in msg

    def test_label(self):
        assert TaskStatus.TODO.label() == "Todo"
        assert TaskStatus.IN_PROGRESS.label() == "In Progress"
        assert TaskStatus.DONE.label() == "Done"


# ---------------------------------------------------------------------------
# TaskPriority
# ---------------------------------------------------------------------------


class TestTaskPriority:
    def test_values(self):
        assert TaskPriority.LOW.value == "low"
        assert TaskPriority.MEDIUM.value == "medium"
        assert TaskPriority.HIGH.value == "high"

    def test_str_comparison(self):
        assert TaskPriority.LOW == "low"
        assert TaskPriority.MEDIUM == "medium"
        assert TaskPriority.HIGH == "high"

    def test_from_string_exact(self):
        assert TaskPriority.from_string("low") is TaskPriority.LOW
        assert TaskPriority.from_string("medium") is TaskPriority.MEDIUM
        assert TaskPriority.from_string("high") is TaskPriority.HIGH

    def test_from_string_case_insensitive(self):
        assert TaskPriority.from_string("LOW") is TaskPriority.LOW
        assert TaskPriority.from_string("Medium") is TaskPriority.MEDIUM
        assert TaskPriority.from_string("HIGH") is TaskPriority.HIGH

    def test_from_string_strips_whitespace(self):
        assert TaskPriority.from_string("  high  ") is TaskPriority.HIGH

    def test_from_string_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="Unknown task priority"):
            TaskPriority.from_string("urgent")

    def test_from_string_error_message_contains_valid_values(self):
        with pytest.raises(ValueError) as exc_info:
            TaskPriority.from_string("bad")
        msg = str(exc_info.value)
        assert "low" in msg
        assert "medium" in msg
        assert "high" in msg

    def test_label(self):
        assert TaskPriority.LOW.label() == "Low"
        assert TaskPriority.MEDIUM.label() == "Medium"
        assert TaskPriority.HIGH.label() == "High"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_now_utc_is_aware(self):
        dt = _now_utc()
        assert dt.tzinfo is not None
        assert dt.tzinfo == timezone.utc

    def test_format_dt_aware(self):
        dt = datetime(2026, 3, 20, 12, 0, 0, 123456, tzinfo=timezone.utc)
        result = _format_dt(dt)
        assert "2026-03-20" in result
        assert "12:00:00" in result

    def test_format_dt_naive_gets_utc_suffix(self):
        dt = datetime(2026, 3, 20, 12, 0, 0)
        result = _format_dt(dt)
        # Should include UTC offset info
        assert "+00:00" in result or "Z" in result or "UTC" in result

    def test_parse_dt_roundtrip(self):
        original = datetime(2026, 3, 20, 12, 34, 56, 789012, tzinfo=timezone.utc)
        serialised = _format_dt(original)
        parsed = _parse_dt(serialised)
        assert parsed == original
        assert parsed.tzinfo is not None

    def test_parse_dt_naive_string_gets_utc(self):
        result = _parse_dt("2026-03-20T12:00:00")
        assert result.tzinfo is not None

    def test_new_id_is_valid_uuid4(self):
        id1 = _new_id()
        id2 = _new_id()
        # Should be parseable as UUID
        uuid.UUID(id1, version=4)
        uuid.UUID(id2, version=4)
        # Should be unique
        assert id1 != id2


# ---------------------------------------------------------------------------
# Task dataclass
# ---------------------------------------------------------------------------


class TestTask:
    def _make_task(self, **kwargs) -> Task:
        now = datetime(2026, 3, 20, 10, 0, 0, tzinfo=timezone.utc)
        defaults = dict(
            id=str(uuid.uuid4()),
            title="Test task",
            status=TaskStatus.TODO,
            priority=TaskPriority.MEDIUM,
            created_at=now,
            updated_at=now,
            tags=["alpha", "beta"],
            notes="Some notes",
        )
        defaults.update(kwargs)
        return Task(**defaults)

    def test_to_dict_contains_expected_keys(self):
        task = self._make_task()
        d = task.to_dict()
        for key in ("id", "title", "status", "priority", "created_at", "updated_at", "tags", "notes"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_status_is_string(self):
        task = self._make_task(status=TaskStatus.IN_PROGRESS)
        d = task.to_dict()
        assert d["status"] == "in_progress"
        assert isinstance(d["status"], str)

    def test_to_dict_priority_is_string(self):
        task = self._make_task(priority=TaskPriority.HIGH)
        d = task.to_dict()
        assert d["priority"] == "high"
        assert isinstance(d["priority"], str)

    def test_to_dict_datetimes_are_strings(self):
        task = self._make_task()
        d = task.to_dict()
        assert isinstance(d["created_at"], str)
        assert isinstance(d["updated_at"], str)

    def test_from_dict_roundtrip(self):
        task = self._make_task()
        d = task.to_dict()
        restored = Task.from_dict(d)
        assert restored.id == task.id
        assert restored.title == task.title
        assert restored.status == task.status
        assert restored.priority == task.priority
        assert restored.created_at == task.created_at
        assert restored.updated_at == task.updated_at
        assert restored.tags == task.tags
        assert restored.notes == task.notes

    def test_from_dict_missing_required_key_raises(self):
        task = self._make_task()
        d = task.to_dict()
        del d["title"]
        with pytest.raises(KeyError):
            Task.from_dict(d)

    def test_from_dict_invalid_status_raises(self):
        task = self._make_task()
        d = task.to_dict()
        d["status"] = "flying"
        with pytest.raises(ValueError):
            Task.from_dict(d)

    def test_from_dict_invalid_priority_raises(self):
        task = self._make_task()
        d = task.to_dict()
        d["priority"] = "critical"
        with pytest.raises(ValueError):
            Task.from_dict(d)

    def test_from_dict_optional_fields_default(self):
        task = self._make_task()
        d = task.to_dict()
        d.pop("tags", None)
        d.pop("notes", None)
        restored = Task.from_dict(d)
        assert restored.tags == []
        assert restored.notes == ""

    def test_eq_same_id(self):
        task_a = self._make_task()
        task_b = self._make_task(id=task_a.id, title="Different title")
        assert task_a == task_b

    def test_eq_different_id(self):
        task_a = self._make_task()
        task_b = self._make_task()
        assert task_a != task_b

    def test_eq_non_task_returns_not_implemented(self):
        task = self._make_task()
        result = task.__eq__("not a task")
        assert result is NotImplemented

    def test_hash_same_id(self):
        task_a = self._make_task()
        task_b = self._make_task(id=task_a.id)
        assert hash(task_a) == hash(task_b)

    def test_hash_usable_in_set(self):
        task_a = self._make_task()
        task_b = self._make_task(id=task_a.id)
        task_c = self._make_task()
        s = {task_a, task_b, task_c}
        assert len(s) == 2

    def test_repr_contains_key_fields(self):
        task = self._make_task(title="My Task", status=TaskStatus.DONE, priority=TaskPriority.HIGH)
        r = repr(task)
        assert "My Task" in r
        assert "done" in r
        assert "high" in r
        assert task.id in r


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class TestExceptions:
    def test_task_not_found_error_is_key_error(self):
        err = TaskNotFoundError("abc-123")
        assert isinstance(err, KeyError)

    def test_task_not_found_error_stores_task_id(self):
        err = TaskNotFoundError("abc-123")
        assert err.task_id == "abc-123"

    def test_task_not_found_error_message(self):
        err = TaskNotFoundError("abc-123")
        assert "abc-123" in str(err)

    def test_task_store_error_is_os_error(self):
        err = TaskStoreError("disk full")
        assert isinstance(err, OSError)


# ---------------------------------------------------------------------------
# TaskStore — basic construction
# ---------------------------------------------------------------------------


class TestTaskStoreConstruction:
    def test_default_path_uses_home(self):
        store = TaskStore()
        assert store.path == (DEFAULT_STORE_DIR / DEFAULT_STORE_FILE).resolve()

    def test_custom_path_string(self, tmp_path):
        p = str(tmp_path / "custom.json")
        store = TaskStore(store_path=p)
        assert store.path == Path(p).resolve()

    def test_custom_path_path_object(self, tmp_path):
        p = tmp_path / "custom.json"
        store = TaskStore(store_path=p)
        assert store.path == p.resolve()

    def test_starts_empty(self, tmp_store):
        assert len(tmp_store) == 0


# ---------------------------------------------------------------------------
# TaskStore — add
# ---------------------------------------------------------------------------


class TestTaskStoreAdd:
    def test_add_returns_task(self, tmp_store):
        task = tmp_store.add("Write tests")
        assert isinstance(task, Task)
        assert task.title == "Write tests"

    def test_add_default_status_is_todo(self, tmp_store):
        task = tmp_store.add("Write tests")
        assert task.status == TaskStatus.TODO

    def test_add_default_priority_is_medium(self, tmp_store):
        task = tmp_store.add("Write tests")
        assert task.priority == TaskPriority.MEDIUM

    def test_add_custom_priority_string(self, tmp_store):
        task = tmp_store.add("Write tests", priority="high")
        assert task.priority == TaskPriority.HIGH

    def test_add_custom_priority_enum(self, tmp_store):
        task = tmp_store.add("Write tests", priority=TaskPriority.LOW)
        assert task.priority == TaskPriority.LOW

    def test_add_with_tags(self, tmp_store):
        task = tmp_store.add("Write tests", tags=["ci", "backend"])
        assert task.tags == ["ci", "backend"]

    def test_add_with_notes(self, tmp_store):
        task = tmp_store.add("Write tests", notes="Important notes here")
        assert task.notes == "Important notes here"

    def test_add_strips_title_whitespace(self, tmp_store):
        task = tmp_store.add("  Write tests  ")
        assert task.title == "Write tests"

    def test_add_empty_title_raises(self, tmp_store):
        with pytest.raises(ValueError, match="empty"):
            tmp_store.add("")

    def test_add_whitespace_only_title_raises(self, tmp_store):
        with pytest.raises(ValueError, match="empty"):
            tmp_store.add("   ")

    def test_add_invalid_priority_raises(self, tmp_store):
        with pytest.raises(ValueError):
            tmp_store.add("Task", priority="critical")

    def test_add_assigns_unique_ids(self, tmp_store):
        t1 = tmp_store.add("Task 1")
        t2 = tmp_store.add("Task 2")
        assert t1.id != t2.id

    def test_add_persists_to_file(self, tmp_store):
        task = tmp_store.add("Persisted task")
        # Read the file directly and verify
        data = json.loads(tmp_store.path.read_text())
        assert any(item["id"] == task.id for item in data)

    def test_add_increments_len(self, tmp_store):
        assert len(tmp_store) == 0
        tmp_store.add("Task 1")
        assert len(tmp_store) == 1
        tmp_store.add("Task 2")
        assert len(tmp_store) == 2

    def test_add_timestamps_are_utc_aware(self, tmp_store):
        task = tmp_store.add("Task")
        assert task.created_at.tzinfo is not None
        assert task.updated_at.tzinfo is not None
        assert task.created_at == task.updated_at


# ---------------------------------------------------------------------------
# TaskStore — get
# ---------------------------------------------------------------------------


class TestTaskStoreGet:
    def test_get_returns_correct_task(self, tmp_store):
        task = tmp_store.add("Find me")
        retrieved = tmp_store.get(task.id)
        assert retrieved.id == task.id
        assert retrieved.title == "Find me"

    def test_get_nonexistent_raises_task_not_found(self, tmp_store):
        with pytest.raises(TaskNotFoundError) as exc_info:
            tmp_store.get("nonexistent-id")
        assert "nonexistent-id" in str(exc_info.value)

    def test_get_after_reload(self, tmp_path):
        store1 = TaskStore(store_path=tmp_path / "tasks.json")
        task = store1.add("Persistent task")

        store2 = TaskStore(store_path=tmp_path / "tasks.json")
        retrieved = store2.get(task.id)
        assert retrieved.title == "Persistent task"


# ---------------------------------------------------------------------------
# TaskStore — update
# ---------------------------------------------------------------------------


class TestTaskStoreUpdate:
    def test_update_title(self, tmp_store):
        task = tmp_store.add("Old title")
        updated = tmp_store.update(task.id, title="New title")
        assert updated.title == "New title"

    def test_update_priority_string(self, tmp_store):
        task = tmp_store.add("Task")
        updated = tmp_store.update(task.id, priority="high")
        assert updated.priority == TaskPriority.HIGH

    def test_update_priority_enum(self, tmp_store):
        task = tmp_store.add("Task")
        updated = tmp_store.update(task.id, priority=TaskPriority.LOW)
        assert updated.priority == TaskPriority.LOW

    def test_update_tags(self, tmp_store):
        task = tmp_store.add("Task", tags=["old"])
        updated = tmp_store.update(task.id, tags=["new1", "new2"])
        assert updated.tags == ["new1", "new2"]

    def test_update_notes(self, tmp_store):
        task = tmp_store.add("Task", notes="old notes")
        updated = tmp_store.update(task.id, notes="new notes")
        assert updated.notes == "new notes"

    def test_update_refreshes_updated_at(self, tmp_store):
        task = tmp_store.add("Task")
        original_updated_at = task.updated_at
        time.sleep(0.01)
        updated = tmp_store.update(task.id, title="New title")
        assert updated.updated_at >= original_updated_at

    def test_update_does_not_change_unspecified_fields(self, tmp_store):
        task = tmp_store.add("Task", priority="high", tags=["x"], notes="note")
        updated = tmp_store.update(task.id, title="New title")
        assert updated.priority == TaskPriority.HIGH
        assert updated.tags == ["x"]
        assert updated.notes == "note"

    def test_update_empty_title_raises(self, tmp_store):
        task = tmp_store.add("Task")
        with pytest.raises(ValueError, match="empty"):
            tmp_store.update(task.id, title="")

    def test_update_invalid_priority_raises(self, tmp_store):
        task = tmp_store.add("Task")
        with pytest.raises(ValueError):
            tmp_store.update(task.id, priority="critical")

    def test_update_nonexistent_raises_task_not_found(self, tmp_store):
        with pytest.raises(TaskNotFoundError):
            tmp_store.update("bad-id", title="New")

    def test_update_persists_to_file(self, tmp_store):
        task = tmp_store.add("Task")
        tmp_store.update(task.id, title="Updated title")
        data = json.loads(tmp_store.path.read_text())
        record = next(item for item in data if item["id"] == task.id)
        assert record["title"] == "Updated title"


# ---------------------------------------------------------------------------
# TaskStore — update_status
# ---------------------------------------------------------------------------


class TestTaskStoreUpdateStatus:
    def test_update_status_with_enum(self, tmp_store):
        task = tmp_store.add("Task")
        updated = tmp_store.update_status(task.id, TaskStatus.IN_PROGRESS)
        assert updated.status == TaskStatus.IN_PROGRESS

    def test_update_status_with_string(self, tmp_store):
        task = tmp_store.add("Task")
        updated = tmp_store.update_status(task.id, "done")
        assert updated.status == TaskStatus.DONE

    def test_update_status_refreshes_updated_at(self, tmp_store):
        task = tmp_store.add("Task")
        original_updated_at = task.updated_at
        time.sleep(0.01)
        updated = tmp_store.update_status(task.id, TaskStatus.DONE)
        assert updated.updated_at >= original_updated_at

    def test_update_status_invalid_string_raises(self, tmp_store):
        task = tmp_store.add("Task")
        with pytest.raises(ValueError):
            tmp_store.update_status(task.id, "flying")

    def test_update_status_nonexistent_raises_task_not_found(self, tmp_store):
        with pytest.raises(TaskNotFoundError):
            tmp_store.update_status("bad-id", TaskStatus.DONE)

    def test_update_status_persists_to_file(self, tmp_store):
        task = tmp_store.add("Task")
        tmp_store.update_status(task.id, TaskStatus.DONE)
        data = json.loads(tmp_store.path.read_text())
        record = next(item for item in data if item["id"] == task.id)
        assert record["status"] == "done"


# ---------------------------------------------------------------------------
# TaskStore — delete
# ---------------------------------------------------------------------------


class TestTaskStoreDelete:
    def test_delete_returns_deleted_task(self, tmp_store):
        task = tmp_store.add("Delete me")
        deleted = tmp_store.delete(task.id)
        assert deleted.id == task.id
        assert deleted.title == "Delete me"

    def test_delete_removes_from_store(self, tmp_store):
        task = tmp_store.add("Delete me")
        tmp_store.delete(task.id)
        assert task.id not in tmp_store

    def test_delete_decrements_len(self, tmp_store):
        task = tmp_store.add("Task")
        assert len(tmp_store) == 1
        tmp_store.delete(task.id)
        assert len(tmp_store) == 0

    def test_delete_nonexistent_raises_task_not_found(self, tmp_store):
        with pytest.raises(TaskNotFoundError) as exc_info:
            tmp_store.delete("nonexistent-id")
        assert "nonexistent-id" in str(exc_info.value)

    def test_delete_persists_to_file(self, tmp_store):
        task = tmp_store.add("Task")
        tmp_store.delete(task.id)
        data = json.loads(tmp_store.path.read_text())
        assert not any(item["id"] == task.id for item in data)

    def test_delete_get_after_delete_raises(self, tmp_store):
        task = tmp_store.add("Task")
        tmp_store.delete(task.id)
        with pytest.raises(TaskNotFoundError):
            tmp_store.get(task.id)


# ---------------------------------------------------------------------------
# TaskStore — list_tasks
# ---------------------------------------------------------------------------


class TestTaskStoreListTasks:
    def test_list_all_tasks(self, tmp_store):
        tmp_store.add("Task 1")
        tmp_store.add("Task 2")
        tmp_store.add("Task 3")
        tasks = tmp_store.list_tasks()
        assert len(tasks) == 3

    def test_list_empty_store(self, tmp_store):
        assert tmp_store.list_tasks() == []

    def test_list_filter_by_status_enum(self, tmp_store):
        t1 = tmp_store.add("Task 1")
        t2 = tmp_store.add("Task 2")
        tmp_store.update_status(t1.id, TaskStatus.DONE)
        results = tmp_store.list_tasks(status=TaskStatus.DONE)
        assert len(results) == 1
        assert results[0].id == t1.id

    def test_list_filter_by_status_string(self, tmp_store):
        t1 = tmp_store.add("Task 1")
        tmp_store.add("Task 2")
        tmp_store.update_status(t1.id, TaskStatus.IN_PROGRESS)
        results = tmp_store.list_tasks(status="in_progress")
        assert len(results) == 1
        assert results[0].id == t1.id

    def test_list_filter_by_priority_enum(self, tmp_store):
        tmp_store.add("High task", priority="high")
        tmp_store.add("Low task", priority="low")
        results = tmp_store.list_tasks(priority=TaskPriority.HIGH)
        assert len(results) == 1
        assert results[0].title == "High task"

    def test_list_filter_by_priority_string(self, tmp_store):
        tmp_store.add("High task", priority="high")
        tmp_store.add("Low task", priority="low")
        results = tmp_store.list_tasks(priority="low")
        assert len(results) == 1
        assert results[0].title == "Low task"

    def test_list_filter_by_tags_single(self, tmp_store):
        tmp_store.add("Tagged", tags=["backend"])
        tmp_store.add("Untagged")
        results = tmp_store.list_tasks(tags=["backend"])
        assert len(results) == 1
        assert results[0].title == "Tagged"

    def test_list_filter_by_tags_multiple_all_required(self, tmp_store):
        tmp_store.add("Both tags", tags=["backend", "ci"])
        tmp_store.add("One tag", tags=["backend"])
        results = tmp_store.list_tasks(tags=["backend", "ci"])
        assert len(results) == 1
        assert results[0].title == "Both tags"

    def test_list_filter_by_tags_case_insensitive(self, tmp_store):
        tmp_store.add("Tagged", tags=["Backend"])
        results = tmp_store.list_tasks(tags=["backend"])
        assert len(results) == 1

    def test_list_filter_by_search_title(self, tmp_store):
        tmp_store.add("Write unit tests")
        tmp_store.add("Deploy to production")
        results = tmp_store.list_tasks(search="unit")
        assert len(results) == 1
        assert results[0].title == "Write unit tests"

    def test_list_filter_by_search_notes(self, tmp_store):
        tmp_store.add("Task A", notes="important deadline")
        tmp_store.add("Task B", notes="routine work")
        results = tmp_store.list_tasks(search="deadline")
        assert len(results) == 1
        assert results[0].title == "Task A"

    def test_list_filter_by_search_case_insensitive(self, tmp_store):
        tmp_store.add("Write Unit Tests")
        results = tmp_store.list_tasks(search="unit tests")
        assert len(results) == 1

    def test_list_combined_filters(self, tmp_store):
        t1 = tmp_store.add("Backend tests", priority="high", tags=["backend"])
        tmp_store.add("Frontend tests", priority="high", tags=["frontend"])
        tmp_store.add("Backend docs", priority="low", tags=["backend"])
        results = tmp_store.list_tasks(priority="high", tags=["backend"])
        assert len(results) == 1
        assert results[0].id == t1.id

    def test_list_sorted_by_priority_then_created_at(self, tmp_store):
        tmp_store.add("Low task", priority="low")
        tmp_store.add("High task", priority="high")
        tmp_store.add("Medium task", priority="medium")
        results = tmp_store.list_tasks()
        assert results[0].priority == TaskPriority.HIGH
        assert results[1].priority == TaskPriority.MEDIUM
        assert results[2].priority == TaskPriority.LOW

    def test_list_invalid_status_raises(self, tmp_store):
        with pytest.raises(ValueError):
            tmp_store.list_tasks(status="flying")

    def test_list_invalid_priority_raises(self, tmp_store):
        with pytest.raises(ValueError):
            tmp_store.list_tasks(priority="critical")


# ---------------------------------------------------------------------------
# TaskStore — count
# ---------------------------------------------------------------------------


class TestTaskStoreCount:
    def test_count_all(self, tmp_store):
        tmp_store.add("Task 1")
        tmp_store.add("Task 2")
        assert tmp_store.count() == 2

    def test_count_empty(self, tmp_store):
        assert tmp_store.count() == 0

    def test_count_by_status(self, tmp_store):
        t1 = tmp_store.add("Task 1")
        tmp_store.add("Task 2")
        tmp_store.update_status(t1.id, TaskStatus.DONE)
        assert tmp_store.count(status=TaskStatus.DONE) == 1
        assert tmp_store.count(status=TaskStatus.TODO) == 1

    def test_count_by_status_string(self, tmp_store):
        t1 = tmp_store.add("Task 1")
        tmp_store.add("Task 2")
        tmp_store.update_status(t1.id, TaskStatus.IN_PROGRESS)
        assert tmp_store.count(status="in_progress") == 1


# ---------------------------------------------------------------------------
# TaskStore — all_tags
# ---------------------------------------------------------------------------


class TestTaskStoreAllTags:
    def test_all_tags_empty_store(self, tmp_store):
        assert tmp_store.all_tags() == []

    def test_all_tags_returns_sorted_unique(self, tmp_store):
        tmp_store.add("Task 1", tags=["zebra", "alpha"])
        tmp_store.add("Task 2", tags=["beta", "alpha"])
        tags = tmp_store.all_tags()
        assert tags == sorted(set(tags))
        assert "alpha" in tags
        assert "beta" in tags
        assert "zebra" in tags
        # alpha appears in both tasks but only once in result
        assert tags.count("alpha") == 1

    def test_all_tags_no_tags(self, tmp_store):
        tmp_store.add("Task 1")
        tmp_store.add("Task 2")
        assert tmp_store.all_tags() == []


# ---------------------------------------------------------------------------
# TaskStore — dunder helpers
# ---------------------------------------------------------------------------


class TestTaskStoreDunders:
    def test_len_empty(self, tmp_store):
        assert len(tmp_store) == 0

    def test_len_after_adds(self, tmp_store):
        tmp_store.add("T1")
        tmp_store.add("T2")
        assert len(tmp_store) == 2

    def test_contains_existing(self, tmp_store):
        task = tmp_store.add("Task")
        assert task.id in tmp_store

    def test_contains_nonexistent(self, tmp_store):
        assert "nonexistent-id" not in tmp_store

    def test_iter_yields_all_tasks(self, tmp_store):
        t1 = tmp_store.add("Task 1")
        t2 = tmp_store.add("Task 2")
        ids = {t.id for t in tmp_store}
        assert t1.id in ids
        assert t2.id in ids
        assert len(ids) == 2

    def test_iter_empty_store(self, tmp_store):
        assert list(tmp_store) == []


# ---------------------------------------------------------------------------
# TaskStore — persistence / reload
# ---------------------------------------------------------------------------


class TestTaskStorePersistence:
    def test_reload_picks_up_external_changes(self, tmp_path):
        store1 = TaskStore(store_path=tmp_path / "tasks.json")
        task = store1.add("Original task")

        # Simulate external modification
        store2 = TaskStore(store_path=tmp_path / "tasks.json")
        store2.update(task.id, title="Modified externally")

        # store1 should see the change after reload
        store1.reload()
        assert store1.get(task.id).title == "Modified externally"

    def test_reload_discards_in_memory_state(self, tmp_path):
        store = TaskStore(store_path=tmp_path / "tasks.json")
        task = store.add("Task")
        # Manually corrupt in-memory state
        store._tasks = {}
        store._loaded = True
        # reload should restore from disk
        store.reload()
        assert task.id in store

    def test_new_store_loads_existing_file(self, tmp_path):
        store1 = TaskStore(store_path=tmp_path / "tasks.json")
        t1 = store1.add("Task 1")
        t2 = store1.add("Task 2")

        store2 = TaskStore(store_path=tmp_path / "tasks.json")
        assert len(store2) == 2
        assert store2.get(t1.id).title == "Task 1"
        assert store2.get(t2.id).title == "Task 2"

    def test_atomic_write_creates_no_tmp_file_on_success(self, tmp_store):
        tmp_store.add("Task")
        tmp_file = tmp_store.path.with_suffix(".tmp")
        assert not tmp_file.exists()

    def test_corrupt_json_raises_task_store_error(self, tmp_path):
        store_path = tmp_path / "tasks.json"
        store_path.write_text("this is not valid json", encoding="utf-8")
        store = TaskStore(store_path=store_path)
        with pytest.raises(TaskStoreError, match="invalid JSON"):
            store.reload()

    def test_non_array_json_raises_task_store_error(self, tmp_path):
        store_path = tmp_path / "tasks.json"
        store_path.write_text('{"key": "value"}', encoding="utf-8")
        store = TaskStore(store_path=store_path)
        with pytest.raises(TaskStoreError, match="JSON array"):
            store.reload()

    def test_invalid_task_record_raises_task_store_error(self, tmp_path):
        store_path = tmp_path / "tasks.json"
        # Missing required 'title' field
        bad_record = [{"id": "abc", "status": "todo", "priority": "low",
                       "created_at": "2026-01-01T00:00:00+00:00",
                       "updated_at": "2026-01-01T00:00:00+00:00"}]
        store_path.write_text(json.dumps(bad_record), encoding="utf-8")
        store = TaskStore(store_path=store_path)
        with pytest.raises(TaskStoreError, match="invalid task record"):
            store.reload()

    def test_nonexistent_file_starts_empty(self, tmp_path):
        store = TaskStore(store_path=tmp_path / "nonexistent.json")
        assert len(store) == 0

    def test_parent_dirs_created_on_save(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "tasks.json"
        store = TaskStore(store_path=nested)
        store.add("Task")
        assert nested.exists()


# ---------------------------------------------------------------------------
# End-to-end integration flow
# ---------------------------------------------------------------------------


class TestEndToEndFlow:
    def test_full_lifecycle(self, tmp_path):
        """Create → read → update → change status → delete, with persistence."""
        store_path = tmp_path / "tasks.json"

        # --- Phase 1: create tasks ---
        store = TaskStore(store_path=store_path)
        t1 = store.add("Write unit tests", priority="high", tags=["ci", "backend"])
        t2 = store.add("Deploy to staging", priority="medium", tags=["ops"])
        t3 = store.add("Update docs", priority="low", notes="Needs review")

        assert len(store) == 3

        # --- Phase 2: verify persistence across store instances ---
        store2 = TaskStore(store_path=store_path)
        assert len(store2) == 3
        assert store2.get(t1.id).title == "Write unit tests"
        assert store2.get(t1.id).priority == TaskPriority.HIGH

        # --- Phase 3: update fields ---
        store2.update(t1.id, title="Write comprehensive unit tests", tags=["ci", "backend", "pytest"])
        assert store2.get(t1.id).title == "Write comprehensive unit tests"
        assert "pytest" in store2.get(t1.id).tags

        # --- Phase 4: status transitions ---
        store2.update_status(t1.id, TaskStatus.IN_PROGRESS)
        store2.update_status(t2.id, TaskStatus.DONE)

        in_progress = store2.list_tasks(status=TaskStatus.IN_PROGRESS)
        done = store2.list_tasks(status=TaskStatus.DONE)
        todo = store2.list_tasks(status=TaskStatus.TODO)

        assert len(in_progress) == 1 and in_progress[0].id == t1.id
        assert len(done) == 1 and done[0].id == t2.id
        assert len(todo) == 1 and todo[0].id == t3.id

        # --- Phase 5: filtering ---
        backend_tasks = store2.list_tasks(tags=["backend"])
        assert len(backend_tasks) == 1

        search_results = store2.list_tasks(search="docs")
        assert len(search_results) == 1
        assert search_results[0].id == t3.id

        # --- Phase 6: counts ---
        assert store2.count() == 3
        assert store2.count(status=TaskStatus.DONE) == 1

        # --- Phase 7: all_tags ---
        tags = store2.all_tags()
        assert "ci" in tags
        assert "backend" in tags
        assert "ops" in tags
        assert "pytest" in tags

        # --- Phase 8: delete ---
        store2.delete(t2.id)
        assert len(store2) == 2
        with pytest.raises(TaskNotFoundError):
            store2.get(t2.id)

        # --- Phase 9: verify final state persists ---
        store3 = TaskStore(store_path=store_path)
        assert len(store3) == 2
        assert t2.id not in store3

    def test_priority_sort_order_in_list(self, tmp_path):
        """High-priority tasks appear before medium, medium before low."""
        store = TaskStore(store_path=tmp_path / "tasks.json")
        store.add("Low task", priority="low")
        store.add("High task", priority="high")
        store.add("Medium task", priority="medium")
        store.add("Another high task", priority="high")

        results = store.list_tasks()
        priorities = [t.priority for t in results]
        # All HIGH tasks come first
        high_indices = [i for i, p in enumerate(priorities) if p == TaskPriority.HIGH]
        medium_indices = [i for i, p in enumerate(priorities) if p == TaskPriority.MEDIUM]
        low_indices = [i for i, p in enumerate(priorities) if p == TaskPriority.LOW]
        assert max(high_indices) < min(medium_indices)
        assert max(medium_indices) < min(low_indices)

    def test_unicode_content_persists_correctly(self, tmp_path):
        """Non-ASCII characters in title/notes/tags survive serialisation."""
        store = TaskStore(store_path=tmp_path / "tasks.json")
        task = store.add("Tâche: écrire des tests 🚀", notes="Résumé: ñoño", tags=["café"])

        store2 = TaskStore(store_path=tmp_path / "tasks.json")
        loaded = store2.get(task.id)
        assert loaded.title == "Tâche: écrire des tests 🚀"
        assert loaded.notes == "Résumé: ñoño"
        assert "café" in loaded.tags
