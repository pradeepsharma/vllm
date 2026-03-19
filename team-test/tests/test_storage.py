"""Tests for team-test/storage.py.

Covers:
- Storage initialisation (default path, custom path, lazy loading)
- JSON persistence: save/load round-trip, corrupt file handling
- Project CRUD: create, get, list, update, delete (cascade & non-cascade)
- Task CRUD: create, get, list (with filters), update (sentinel logic), delete
- Validation errors: empty title/name, bad due_date format, non-existent project_id
- NotFoundError for missing task/project ids
- stats() correctness
- clear() wipes everything
- Full end-to-end workflow: project → tasks → update → delete
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup — same pattern as test_models.py
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEAM_TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

for _p in (WORKSPACE_ROOT, TEAM_TEST_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import Priority, Project, Status, Task  # noqa: E402
from storage import (  # noqa: E402
    DuplicateError,
    NotFoundError,
    Storage,
    StorageError,
    ValidationError,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_store(tmp_path: Path) -> Storage:
    """Return a fresh Storage backed by a temp file."""
    return Storage(str(tmp_path / "data.json"))


@pytest.fixture()
def store_with_project(tmp_store: Storage):
    """Return (store, project) with one project already created."""
    project = tmp_store.create_project("Alpha", description="First project")
    return tmp_store, project


@pytest.fixture()
def store_with_task(store_with_project):
    """Return (store, project, task) with one task already created."""
    store, project = store_with_project
    task = store.create_task(
        "Initial task",
        description="A description",
        project_id=project.id,
    )
    return store, project, task


# ===========================================================================
# Storage initialisation
# ===========================================================================


class TestStorageInit:
    def test_custom_path_stored(self, tmp_path: Path):
        path = str(tmp_path / "custom.json")
        store = Storage(path)
        assert store._path == os.path.abspath(path)

    def test_not_loaded_until_first_access(self, tmp_path: Path):
        store = Storage(str(tmp_path / "data.json"))
        assert store._loaded is False

    def test_lazy_load_on_list_tasks(self, tmp_store: Storage):
        tasks = tmp_store.list_tasks()
        assert tmp_store._loaded is True
        assert tasks == []

    def test_lazy_load_on_list_projects(self, tmp_store: Storage):
        projects = tmp_store.list_projects()
        assert tmp_store._loaded is True
        assert projects == []

    def test_missing_file_gives_empty_store(self, tmp_store: Storage):
        assert tmp_store.list_tasks() == []
        assert tmp_store.list_projects() == []


# ===========================================================================
# Persistence (save / load round-trip)
# ===========================================================================


class TestPersistence:
    def test_data_survives_reload(self, tmp_path: Path):
        path = str(tmp_path / "data.json")
        store1 = Storage(path)
        project = store1.create_project("Persist me")
        task = store1.create_task("Task A", project_id=project.id)

        # Open a fresh Storage instance pointing at the same file.
        store2 = Storage(path)
        assert len(store2.list_projects()) == 1
        assert store2.list_projects()[0].name == "Persist me"
        assert len(store2.list_tasks()) == 1
        assert store2.list_tasks()[0].title == "Task A"

    def test_json_file_structure(self, tmp_path: Path):
        path = str(tmp_path / "data.json")
        store = Storage(path)
        store.create_project("Struct test")
        store.create_task("T1")

        with open(path, "r", encoding="utf-8") as fh:
            raw = json.load(fh)

        assert "tasks" in raw
        assert "projects" in raw
        assert isinstance(raw["tasks"], dict)
        assert isinstance(raw["projects"], dict)

    def test_corrupt_json_raises_storage_error(self, tmp_path: Path):
        path = str(tmp_path / "data.json")
        with open(path, "w") as fh:
            fh.write("{ not valid json !!!")

        store = Storage(path)
        with pytest.raises(StorageError, match="invalid JSON"):
            store.list_tasks()

    def test_empty_json_object_is_valid(self, tmp_path: Path):
        path = str(tmp_path / "data.json")
        with open(path, "w") as fh:
            json.dump({}, fh)

        store = Storage(path)
        assert store.list_tasks() == []
        assert store.list_projects() == []

    def test_parent_directory_created_automatically(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c" / "data.json"
        store = Storage(str(nested))
        store.create_task("Nested dir task")
        assert nested.exists()

    def test_atomic_write_no_tmp_file_left_behind(self, tmp_path: Path):
        path = str(tmp_path / "data.json")
        store = Storage(path)
        store.create_task("Atomic write")
        tmp = path + ".tmp"
        assert not os.path.exists(tmp)


# ===========================================================================
# Project CRUD
# ===========================================================================


class TestCreateProject:
    def test_returns_project_instance(self, tmp_store: Storage):
        project = tmp_store.create_project("My Project")
        assert isinstance(project, Project)

    def test_name_is_stored(self, tmp_store: Storage):
        project = tmp_store.create_project("My Project")
        assert project.name == "My Project"

    def test_description_is_stored(self, tmp_store: Storage):
        project = tmp_store.create_project("P", description="Desc")
        assert project.description == "Desc"

    def test_id_is_uuid(self, tmp_store: Storage):
        project = tmp_store.create_project("UUID check")
        parsed = uuid.UUID(project.id)
        assert str(parsed) == project.id

    def test_name_is_stripped(self, tmp_store: Storage):
        project = tmp_store.create_project("  Padded  ")
        assert project.name == "Padded"

    def test_empty_name_raises_validation_error(self, tmp_store: Storage):
        with pytest.raises(ValidationError, match="non-empty"):
            tmp_store.create_project("")

    def test_whitespace_only_name_raises_validation_error(self, tmp_store: Storage):
        with pytest.raises(ValidationError, match="non-empty"):
            tmp_store.create_project("   ")

    def test_project_persisted_to_disk(self, tmp_path: Path):
        path = str(tmp_path / "data.json")
        store = Storage(path)
        project = store.create_project("Persisted")
        store2 = Storage(path)
        fetched = store2.get_project(project.id)
        assert fetched.name == "Persisted"


class TestGetProject:
    def test_returns_correct_project(self, store_with_project):
        store, project = store_with_project
        fetched = store.get_project(project.id)
        assert fetched.id == project.id
        assert fetched.name == project.name

    def test_missing_id_raises_not_found(self, tmp_store: Storage):
        with pytest.raises(NotFoundError, match="not found"):
            tmp_store.get_project("nonexistent-id")


class TestListProjects:
    def test_empty_store_returns_empty_list(self, tmp_store: Storage):
        assert tmp_store.list_projects() == []

    def test_returns_all_projects(self, tmp_store: Storage):
        tmp_store.create_project("A")
        tmp_store.create_project("B")
        tmp_store.create_project("C")
        assert len(tmp_store.list_projects()) == 3

    def test_ordered_by_created_at(self, tmp_store: Storage):
        p1 = tmp_store.create_project("First")
        p2 = tmp_store.create_project("Second")
        p3 = tmp_store.create_project("Third")
        listed = tmp_store.list_projects()
        assert [p.id for p in listed] == [p1.id, p2.id, p3.id]


class TestUpdateProject:
    def test_update_name(self, store_with_project):
        store, project = store_with_project
        updated = store.update_project(project.id, name="New Name")
        assert updated.name == "New Name"

    def test_update_description(self, store_with_project):
        store, project = store_with_project
        updated = store.update_project(project.id, description="New desc")
        assert updated.description == "New desc"

    def test_update_persisted(self, tmp_path: Path):
        path = str(tmp_path / "data.json")
        store = Storage(path)
        project = store.create_project("Old Name")
        store.update_project(project.id, name="New Name")

        store2 = Storage(path)
        fetched = store2.get_project(project.id)
        assert fetched.name == "New Name"

    def test_no_args_leaves_project_unchanged(self, store_with_project):
        store, project = store_with_project
        original_name = project.name
        updated = store.update_project(project.id)
        assert updated.name == original_name

    def test_empty_name_raises_validation_error(self, store_with_project):
        store, project = store_with_project
        with pytest.raises(ValidationError, match="non-empty"):
            store.update_project(project.id, name="")

    def test_missing_id_raises_not_found(self, tmp_store: Storage):
        with pytest.raises(NotFoundError):
            tmp_store.update_project("bad-id", name="X")


class TestDeleteProject:
    def test_project_removed(self, store_with_project):
        store, project = store_with_project
        store.delete_project(project.id)
        with pytest.raises(NotFoundError):
            store.get_project(project.id)

    def test_missing_id_raises_not_found(self, tmp_store: Storage):
        with pytest.raises(NotFoundError):
            tmp_store.delete_project("bad-id")

    def test_cascade_false_clears_task_project_id(self, store_with_task):
        store, project, task = store_with_task
        store.delete_project(project.id, cascade=False)
        fetched_task = store.get_task(task.id)
        assert fetched_task.project_id is None

    def test_cascade_true_deletes_tasks(self, store_with_task):
        store, project, task = store_with_task
        store.delete_project(project.id, cascade=True)
        with pytest.raises(NotFoundError):
            store.get_task(task.id)

    def test_cascade_false_does_not_delete_tasks(self, store_with_task):
        store, project, task = store_with_task
        store.delete_project(project.id, cascade=False)
        # Task still exists
        fetched = store.get_task(task.id)
        assert fetched.id == task.id

    def test_cascade_only_affects_project_tasks(self, tmp_store: Storage):
        p1 = tmp_store.create_project("P1")
        p2 = tmp_store.create_project("P2")
        t1 = tmp_store.create_task("T1", project_id=p1.id)
        t2 = tmp_store.create_task("T2", project_id=p2.id)

        tmp_store.delete_project(p1.id, cascade=True)

        with pytest.raises(NotFoundError):
            tmp_store.get_task(t1.id)
        # t2 should still exist
        assert tmp_store.get_task(t2.id).id == t2.id


# ===========================================================================
# Task CRUD
# ===========================================================================


class TestCreateTask:
    def test_returns_task_instance(self, tmp_store: Storage):
        task = tmp_store.create_task("My task")
        assert isinstance(task, Task)

    def test_title_stored(self, tmp_store: Storage):
        task = tmp_store.create_task("My task")
        assert task.title == "My task"

    def test_title_stripped(self, tmp_store: Storage):
        task = tmp_store.create_task("  Padded  ")
        assert task.title == "Padded"

    def test_defaults(self, tmp_store: Storage):
        task = tmp_store.create_task("Defaults")
        assert task.status is Status.TODO
        assert task.priority is Priority.MEDIUM
        assert task.due_date is None
        assert task.project_id is None
        assert task.description == ""

    def test_explicit_fields(self, store_with_project):
        store, project = store_with_project
        task = store.create_task(
            "Explicit",
            description="Desc",
            status=Status.IN_PROGRESS,
            priority=Priority.HIGH,
            due_date="2027-01-01",
            project_id=project.id,
        )
        assert task.description == "Desc"
        assert task.status is Status.IN_PROGRESS
        assert task.priority is Priority.HIGH
        assert task.due_date == "2027-01-01"
        assert task.project_id == project.id

    def test_empty_title_raises_validation_error(self, tmp_store: Storage):
        with pytest.raises(ValidationError, match="non-empty"):
            tmp_store.create_task("")

    def test_whitespace_title_raises_validation_error(self, tmp_store: Storage):
        with pytest.raises(ValidationError, match="non-empty"):
            tmp_store.create_task("   ")

    def test_invalid_due_date_raises_validation_error(self, tmp_store: Storage):
        with pytest.raises(ValidationError, match="YYYY-MM-DD"):
            tmp_store.create_task("Bad date", due_date="31-12-2027")

    def test_nonexistent_project_id_raises_validation_error(self, tmp_store: Storage):
        with pytest.raises(ValidationError, match="does not exist"):
            tmp_store.create_task("Task", project_id="no-such-project")

    def test_task_persisted(self, tmp_path: Path):
        path = str(tmp_path / "data.json")
        store = Storage(path)
        task = store.create_task("Persisted task")
        store2 = Storage(path)
        fetched = store2.get_task(task.id)
        assert fetched.title == "Persisted task"

    def test_id_is_uuid(self, tmp_store: Storage):
        task = tmp_store.create_task("UUID task")
        parsed = uuid.UUID(task.id)
        assert str(parsed) == task.id

    def test_two_tasks_have_different_ids(self, tmp_store: Storage):
        t1 = tmp_store.create_task("T1")
        t2 = tmp_store.create_task("T2")
        assert t1.id != t2.id


class TestGetTask:
    def test_returns_correct_task(self, store_with_task):
        store, project, task = store_with_task
        fetched = store.get_task(task.id)
        assert fetched.id == task.id
        assert fetched.title == task.title

    def test_missing_id_raises_not_found(self, tmp_store: Storage):
        with pytest.raises(NotFoundError, match="not found"):
            tmp_store.get_task("nonexistent-id")


class TestListTasks:
    def test_empty_store_returns_empty_list(self, tmp_store: Storage):
        assert tmp_store.list_tasks() == []

    def test_returns_all_tasks(self, tmp_store: Storage):
        tmp_store.create_task("T1")
        tmp_store.create_task("T2")
        tmp_store.create_task("T3")
        assert len(tmp_store.list_tasks()) == 3

    def test_ordered_by_created_at(self, tmp_store: Storage):
        t1 = tmp_store.create_task("First")
        t2 = tmp_store.create_task("Second")
        t3 = tmp_store.create_task("Third")
        listed = tmp_store.list_tasks()
        assert [t.id for t in listed] == [t1.id, t2.id, t3.id]

    def test_filter_by_project_id(self, tmp_store: Storage):
        p1 = tmp_store.create_project("P1")
        p2 = tmp_store.create_project("P2")
        t1 = tmp_store.create_task("T1", project_id=p1.id)
        t2 = tmp_store.create_task("T2", project_id=p2.id)
        t3 = tmp_store.create_task("T3", project_id=p1.id)

        result = tmp_store.list_tasks(project_id=p1.id)
        ids = {t.id for t in result}
        assert ids == {t1.id, t3.id}
        assert t2.id not in ids

    def test_filter_by_status(self, tmp_store: Storage):
        t1 = tmp_store.create_task("T1", status=Status.TODO)
        t2 = tmp_store.create_task("T2", status=Status.DONE)
        t3 = tmp_store.create_task("T3", status=Status.TODO)

        result = tmp_store.list_tasks(status=Status.TODO)
        ids = {t.id for t in result}
        assert ids == {t1.id, t3.id}
        assert t2.id not in ids

    def test_filter_by_priority(self, tmp_store: Storage):
        t1 = tmp_store.create_task("T1", priority=Priority.HIGH)
        t2 = tmp_store.create_task("T2", priority=Priority.LOW)
        t3 = tmp_store.create_task("T3", priority=Priority.HIGH)

        result = tmp_store.list_tasks(priority=Priority.HIGH)
        ids = {t.id for t in result}
        assert ids == {t1.id, t3.id}

    def test_filter_overdue_only(self, tmp_store: Storage):
        past = (datetime.now(timezone.utc) - timedelta(days=2)).strftime("%Y-%m-%d")
        future = (datetime.now(timezone.utc) + timedelta(days=10)).strftime("%Y-%m-%d")

        t_overdue = tmp_store.create_task("Overdue", due_date=past)
        t_future = tmp_store.create_task("Future", due_date=future)
        t_no_date = tmp_store.create_task("No date")

        result = tmp_store.list_tasks(overdue_only=True)
        ids = {t.id for t in result}
        assert t_overdue.id in ids
        assert t_future.id not in ids
        assert t_no_date.id not in ids

    def test_combined_filters(self, tmp_store: Storage):
        p = tmp_store.create_project("P")
        past = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")

        t1 = tmp_store.create_task(
            "T1", project_id=p.id, status=Status.IN_PROGRESS, due_date=past
        )
        t2 = tmp_store.create_task(
            "T2", project_id=p.id, status=Status.DONE, due_date=past
        )
        t3 = tmp_store.create_task("T3", status=Status.IN_PROGRESS, due_date=past)

        result = tmp_store.list_tasks(
            project_id=p.id, status=Status.IN_PROGRESS, overdue_only=True
        )
        assert len(result) == 1
        assert result[0].id == t1.id


class TestUpdateTask:
    def test_update_title(self, store_with_task):
        store, project, task = store_with_task
        updated = store.update_task(task.id, title="New Title")
        assert updated.title == "New Title"

    def test_update_description(self, store_with_task):
        store, project, task = store_with_task
        updated = store.update_task(task.id, description="New desc")
        assert updated.description == "New desc"

    def test_update_status(self, store_with_task):
        store, project, task = store_with_task
        updated = store.update_task(task.id, status=Status.DONE)
        assert updated.status is Status.DONE

    def test_update_priority(self, store_with_task):
        store, project, task = store_with_task
        updated = store.update_task(task.id, priority=Priority.CRITICAL)
        assert updated.priority is Priority.CRITICAL

    def test_update_due_date(self, store_with_task):
        store, project, task = store_with_task
        updated = store.update_task(task.id, due_date="2028-06-15")
        assert updated.due_date == "2028-06-15"

    def test_clear_due_date_with_none(self, store_with_task):
        store, project, task = store_with_task
        store.update_task(task.id, due_date="2028-01-01")
        updated = store.update_task(task.id, due_date=None)
        assert updated.due_date is None

    def test_update_project_id(self, tmp_store: Storage):
        p1 = tmp_store.create_project("P1")
        p2 = tmp_store.create_project("P2")
        task = tmp_store.create_task("T", project_id=p1.id)
        updated = tmp_store.update_task(task.id, project_id=p2.id)
        assert updated.project_id == p2.id

    def test_clear_project_id_with_none(self, store_with_task):
        store, project, task = store_with_task
        updated = store.update_task(task.id, project_id=None)
        assert updated.project_id is None

    def test_updated_at_changes(self, store_with_task):
        store, project, task = store_with_task
        original_updated_at = task.updated_at
        updated = store.update_task(task.id, title="Changed")
        assert updated.updated_at >= original_updated_at

    def test_no_args_still_updates_updated_at(self, store_with_task):
        store, project, task = store_with_task
        original_updated_at = task.updated_at
        updated = store.update_task(task.id)
        assert updated.updated_at >= original_updated_at

    def test_update_persisted(self, tmp_path: Path):
        path = str(tmp_path / "data.json")
        store = Storage(path)
        task = store.create_task("Original")
        store.update_task(task.id, title="Updated", status=Status.DONE)

        store2 = Storage(path)
        fetched = store2.get_task(task.id)
        assert fetched.title == "Updated"
        assert fetched.status is Status.DONE

    def test_missing_id_raises_not_found(self, tmp_store: Storage):
        with pytest.raises(NotFoundError):
            tmp_store.update_task("bad-id", title="X")

    def test_empty_title_raises_validation_error(self, store_with_task):
        store, project, task = store_with_task
        with pytest.raises(ValidationError, match="non-empty"):
            store.update_task(task.id, title="")

    def test_invalid_due_date_raises_validation_error(self, store_with_task):
        store, project, task = store_with_task
        with pytest.raises(ValidationError, match="YYYY-MM-DD"):
            store.update_task(task.id, due_date="not-a-date")

    def test_nonexistent_project_id_raises_validation_error(self, store_with_task):
        store, project, task = store_with_task
        with pytest.raises(ValidationError, match="does not exist"):
            store.update_task(task.id, project_id="no-such-project")

    def test_unset_fields_not_changed(self, store_with_task):
        """Fields not passed to update_task must remain unchanged."""
        store, project, task = store_with_task
        original_title = task.title
        original_status = task.status
        original_priority = task.priority
        original_due_date = task.due_date
        original_project_id = task.project_id

        # Only update description
        updated = store.update_task(task.id, description="Only this changed")

        assert updated.title == original_title
        assert updated.status is original_status
        assert updated.priority is original_priority
        assert updated.due_date == original_due_date
        assert updated.project_id == original_project_id


class TestDeleteTask:
    def test_task_removed(self, store_with_task):
        store, project, task = store_with_task
        store.delete_task(task.id)
        with pytest.raises(NotFoundError):
            store.get_task(task.id)

    def test_missing_id_raises_not_found(self, tmp_store: Storage):
        with pytest.raises(NotFoundError):
            tmp_store.delete_task("bad-id")

    def test_delete_persisted(self, tmp_path: Path):
        path = str(tmp_path / "data.json")
        store = Storage(path)
        task = store.create_task("To delete")
        store.delete_task(task.id)

        store2 = Storage(path)
        assert store2.list_tasks() == []

    def test_other_tasks_unaffected(self, tmp_store: Storage):
        t1 = tmp_store.create_task("Keep me")
        t2 = tmp_store.create_task("Delete me")
        tmp_store.delete_task(t2.id)
        assert len(tmp_store.list_tasks()) == 1
        assert tmp_store.list_tasks()[0].id == t1.id


# ===========================================================================
# stats()
# ===========================================================================


class TestStats:
    def test_empty_store(self, tmp_store: Storage):
        s = tmp_store.stats()
        assert s["total_tasks"] == 0
        assert s["total_projects"] == 0
        assert s["overdue_tasks"] == 0
        assert s["tasks_by_status"] == {"todo": 0, "in_progress": 0, "done": 0}
        assert s["tasks_by_priority"] == {
            "low": 0, "medium": 0, "high": 0, "critical": 0
        }

    def test_counts_tasks_and_projects(self, tmp_store: Storage):
        tmp_store.create_project("P1")
        tmp_store.create_project("P2")
        tmp_store.create_task("T1")
        tmp_store.create_task("T2")
        tmp_store.create_task("T3")
        s = tmp_store.stats()
        assert s["total_tasks"] == 3
        assert s["total_projects"] == 2

    def test_tasks_by_status(self, tmp_store: Storage):
        tmp_store.create_task("T1", status=Status.TODO)
        tmp_store.create_task("T2", status=Status.IN_PROGRESS)
        tmp_store.create_task("T3", status=Status.DONE)
        tmp_store.create_task("T4", status=Status.TODO)
        s = tmp_store.stats()
        assert s["tasks_by_status"]["todo"] == 2
        assert s["tasks_by_status"]["in_progress"] == 1
        assert s["tasks_by_status"]["done"] == 1

    def test_tasks_by_priority(self, tmp_store: Storage):
        tmp_store.create_task("T1", priority=Priority.LOW)
        tmp_store.create_task("T2", priority=Priority.MEDIUM)
        tmp_store.create_task("T3", priority=Priority.HIGH)
        tmp_store.create_task("T4", priority=Priority.CRITICAL)
        tmp_store.create_task("T5", priority=Priority.HIGH)
        s = tmp_store.stats()
        assert s["tasks_by_priority"]["low"] == 1
        assert s["tasks_by_priority"]["medium"] == 1
        assert s["tasks_by_priority"]["high"] == 2
        assert s["tasks_by_priority"]["critical"] == 1

    def test_overdue_tasks_count(self, tmp_store: Storage):
        past = (datetime.now(timezone.utc) - timedelta(days=3)).strftime("%Y-%m-%d")
        future = (datetime.now(timezone.utc) + timedelta(days=3)).strftime("%Y-%m-%d")
        tmp_store.create_task("Overdue1", due_date=past)
        tmp_store.create_task("Overdue2", due_date=past)
        tmp_store.create_task("Future", due_date=future)
        tmp_store.create_task("Done overdue", due_date=past, status=Status.DONE)
        s = tmp_store.stats()
        assert s["overdue_tasks"] == 2

    def test_stats_keys_present(self, tmp_store: Storage):
        s = tmp_store.stats()
        assert "total_tasks" in s
        assert "total_projects" in s
        assert "tasks_by_status" in s
        assert "tasks_by_priority" in s
        assert "overdue_tasks" in s


# ===========================================================================
# clear()
# ===========================================================================


class TestClear:
    def test_clears_all_tasks_and_projects(self, tmp_store: Storage):
        tmp_store.create_project("P")
        tmp_store.create_task("T")
        tmp_store.clear()
        assert tmp_store.list_tasks() == []
        assert tmp_store.list_projects() == []

    def test_clear_persisted(self, tmp_path: Path):
        path = str(tmp_path / "data.json")
        store = Storage(path)
        store.create_project("P")
        store.create_task("T")
        store.clear()

        store2 = Storage(path)
        assert store2.list_tasks() == []
        assert store2.list_projects() == []


# ===========================================================================
# Exception hierarchy
# ===========================================================================


class TestExceptionHierarchy:
    def test_not_found_is_storage_error(self):
        assert issubclass(NotFoundError, StorageError)

    def test_duplicate_is_storage_error(self):
        assert issubclass(DuplicateError, StorageError)

    def test_validation_is_storage_error(self):
        assert issubclass(ValidationError, StorageError)


# ===========================================================================
# End-to-end workflow
# ===========================================================================


class TestEndToEndWorkflow:
    def test_full_project_task_lifecycle(self, tmp_path: Path):
        """Create a project, add tasks, update them, delete one, verify state."""
        path = str(tmp_path / "data.json")
        store = Storage(path)

        # 1. Create project
        project = store.create_project("E2E Project", description="End-to-end test")
        assert store.get_project(project.id).name == "E2E Project"

        # 2. Add tasks
        t1 = store.create_task(
            "Design API",
            status=Status.DONE,
            priority=Priority.HIGH,
            project_id=project.id,
        )
        t2 = store.create_task(
            "Implement storage",
            status=Status.IN_PROGRESS,
            priority=Priority.HIGH,
            project_id=project.id,
        )
        t3 = store.create_task(
            "Write tests",
            status=Status.TODO,
            priority=Priority.MEDIUM,
            project_id=project.id,
        )

        assert len(store.list_tasks(project_id=project.id)) == 3

        # 3. Update a task
        store.update_task(t2.id, status=Status.DONE)
        assert store.get_task(t2.id).status is Status.DONE

        # 4. Filter by status
        todo_tasks = store.list_tasks(status=Status.TODO, project_id=project.id)
        assert len(todo_tasks) == 1
        assert todo_tasks[0].id == t3.id

        # 5. Delete a task
        store.delete_task(t1.id)
        assert len(store.list_tasks(project_id=project.id)) == 2

        # 6. Stats
        s = store.stats()
        assert s["total_tasks"] == 2
        assert s["total_projects"] == 1
        assert s["tasks_by_status"]["done"] == 1
        assert s["tasks_by_status"]["todo"] == 1

        # 7. Reload from disk and verify
        store2 = Storage(path)
        assert len(store2.list_tasks()) == 2
        assert store2.get_project(project.id).name == "E2E Project"

        # 8. Delete project with cascade
        store2.delete_project(project.id, cascade=True)
        assert store2.list_tasks() == []
        assert store2.list_projects() == []

    def test_task_without_project(self, tmp_path: Path):
        """Tasks can exist independently of any project."""
        path = str(tmp_path / "data.json")
        store = Storage(path)

        task = store.create_task("Standalone task", priority=Priority.CRITICAL)
        assert task.project_id is None

        store2 = Storage(path)
        fetched = store2.get_task(task.id)
        assert fetched.title == "Standalone task"
        assert fetched.priority is Priority.CRITICAL
        assert fetched.project_id is None

    def test_overdue_workflow(self, tmp_store: Storage):
        """Overdue detection works correctly across the full lifecycle."""
        past = (datetime.now(timezone.utc) - timedelta(days=5)).strftime("%Y-%m-%d")
        future = (datetime.now(timezone.utc) + timedelta(days=5)).strftime("%Y-%m-%d")

        t_overdue = tmp_store.create_task("Overdue task", due_date=past)
        t_future = tmp_store.create_task("Future task", due_date=future)

        overdue_list = tmp_store.list_tasks(overdue_only=True)
        assert len(overdue_list) == 1
        assert overdue_list[0].id == t_overdue.id

        # Marking overdue task as DONE removes it from overdue list
        tmp_store.update_task(t_overdue.id, status=Status.DONE)
        overdue_list_after = tmp_store.list_tasks(overdue_only=True)
        assert len(overdue_list_after) == 0
