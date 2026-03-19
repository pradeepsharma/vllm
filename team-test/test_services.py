"""
Comprehensive tests for team-test/services.py

Covers:
  - ServiceError exception hierarchy
  - TaskNotFoundError / DuplicateTaskError attributes
  - TaskService construction and properties (storage_path, task_count, repr)
  - create_task(): happy path, all fields, invalid inputs
  - get_task(): by full ID, by short ID, not found
  - update_task(): each field, clear due_date, not found, invalid values
  - delete_task(): happy path, by short ID, not found
  - complete_task(): marks done, not found
  - cancel_task(): marks cancelled, not found
  - start_task(): marks in_progress, not found
  - list_tasks(): no filters, filter by status/priority/tag/overdue/search,
                  all sort_by options, ascending/descending, invalid sort_by
  - get_overdue_tasks(): returns only overdue, sorted by due_date
  - search_tasks(): case-insensitive title and description search
  - get_statistics(): counts by status, priority, overdue
  - bulk_complete(): all found, some not found, empty list
  - bulk_delete(): all found, some not found, empty list
  - reload(): forces re-read from disk
  - backup(): delegates to storage
  - Integration: full CRUD workflow persisted across multiple service instances
"""

import importlib.util
import pathlib
import sys
import tempfile
import os
from datetime import date, timedelta

import pytest

# ---------------------------------------------------------------------------
# Load modules dynamically (no package setup required)
# ---------------------------------------------------------------------------

def _load_module(name: str, filepath: pathlib.Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)                          # type: ignore[union-attr]
    return mod


_here = pathlib.Path(__file__).parent
_models   = _load_module("task_models",   _here / "models.py")
_storage  = _load_module("task_storage",  _here / "storage.py")
_services = _load_module("task_services", _here / "services.py")

Task         = _models.Task
TaskList     = _models.TaskList
Priority     = _models.Priority
Status       = _models.Status

TaskStorage      = _storage.TaskStorage
StorageReadError = _storage.StorageReadError

TaskService       = _services.TaskService
ServiceError      = _services.ServiceError
TaskNotFoundError = _services.TaskNotFoundError
DuplicateTaskError = _services.DuplicateTaskError


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture()
def storage(tmp_dir):
    return TaskStorage(tmp_dir / "tasks.json")


@pytest.fixture()
def service(storage):
    return TaskService(storage)


@pytest.fixture()
def populated_service(service):
    """Service pre-loaded with three tasks."""
    service.create_task("Alpha", priority="low",      tags=["a"])
    service.create_task("Beta",  priority="high",     tags=["b", "shared"])
    service.create_task("Gamma", priority="critical", tags=["c", "shared"])
    return service


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _future_date(days: int = 30) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def _past_date(days: int = 5) -> str:
    return (date.today() - timedelta(days=days)).isoformat()


# ===========================================================================
# Exception hierarchy
# ===========================================================================

class TestExceptionHierarchy:
    def test_service_error_is_exception(self):
        err = ServiceError("oops")
        assert isinstance(err, Exception)
        assert "oops" in str(err)

    def test_task_not_found_is_service_error(self):
        err = TaskNotFoundError("abc123")
        assert isinstance(err, ServiceError)
        assert err.task_id == "abc123"
        assert "abc123" in str(err)

    def test_duplicate_task_is_service_error(self):
        err = DuplicateTaskError("xyz")
        assert isinstance(err, ServiceError)
        assert err.task_id == "xyz"
        assert "xyz" in str(err)


# ===========================================================================
# TaskService — construction & properties
# ===========================================================================

class TestTaskServiceProperties:
    def test_storage_path(self, service, storage):
        assert service.storage_path == storage.path

    def test_task_count_empty(self, service):
        assert service.task_count == 0

    def test_task_count_after_creates(self, service):
        service.create_task("T1")
        service.create_task("T2")
        assert service.task_count == 2

    def test_repr_contains_storage(self, service):
        r = repr(service)
        assert "TaskService" in r

    def test_repr_shows_unknown_count_before_load(self, storage):
        # A brand-new service that has never been used should show "?"
        svc = TaskService(storage)
        assert "?" in repr(svc)

    def test_repr_shows_count_after_load(self, service):
        service.create_task("T")
        r = repr(service)
        assert "1" in r


# ===========================================================================
# create_task()
# ===========================================================================

class TestCreateTask:
    def test_create_minimal(self, service):
        task = service.create_task("Buy milk")
        assert task.title == "Buy milk"
        assert task.priority is Priority.MEDIUM
        assert task.status is Status.TODO
        assert task.description == ""
        assert task.due_date is None
        assert task.tags == []
        assert len(task.id) == 36

    def test_create_with_all_fields(self, service):
        task = service.create_task(
            "Deploy service",
            description="Push to prod",
            priority="high",
            status="in_progress",
            due_date=_future_date(),
            tags=["devops", "prod"],
        )
        assert task.title == "Deploy service"
        assert task.description == "Push to prod"
        assert task.priority is Priority.HIGH
        assert task.status is Status.IN_PROGRESS
        assert task.due_date == _future_date()
        assert "devops" in task.tags
        assert "prod" in task.tags

    def test_create_persists_to_disk(self, service, storage):
        task = service.create_task("Persisted task")
        # Load fresh from disk
        loaded = storage.load()
        assert len(loaded) == 1
        assert loaded.all()[0].id == task.id

    def test_create_empty_title_raises(self, service):
        with pytest.raises(ValueError, match="title must not be empty"):
            service.create_task("")

    def test_create_whitespace_title_raises(self, service):
        with pytest.raises(ValueError, match="title must not be empty"):
            service.create_task("   ")

    def test_create_invalid_priority_raises(self, service):
        with pytest.raises(ValueError):
            service.create_task("T", priority="urgent")

    def test_create_invalid_status_raises(self, service):
        with pytest.raises(ValueError):
            service.create_task("T", status="pending")

    def test_create_invalid_due_date_raises(self, service):
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            service.create_task("T", due_date="31-12-2030")

    def test_create_increments_task_count(self, service):
        assert service.task_count == 0
        service.create_task("T1")
        assert service.task_count == 1
        service.create_task("T2")
        assert service.task_count == 2

    def test_create_returns_task_object(self, service):
        task = service.create_task("T")
        assert isinstance(task, Task)

    def test_create_with_priority_enum(self, service):
        task = service.create_task("T", priority=Priority.CRITICAL)
        assert task.priority is Priority.CRITICAL

    def test_create_with_status_enum(self, service):
        task = service.create_task("T", status=Status.DONE)
        assert task.status is Status.DONE


# ===========================================================================
# get_task()
# ===========================================================================

class TestGetTask:
    def test_get_by_full_id(self, service):
        created = service.create_task("Find me")
        found = service.get_task(created.id)
        assert found.id == created.id
        assert found.title == "Find me"

    def test_get_by_short_id(self, service):
        created = service.create_task("Short ID test")
        found = service.get_task(created.short_id)
        assert found.id == created.id

    def test_get_not_found_raises(self, service):
        with pytest.raises(TaskNotFoundError) as exc_info:
            service.get_task("nonexistent-id")
        assert exc_info.value.task_id == "nonexistent-id"

    def test_get_not_found_error_message(self, service):
        with pytest.raises(TaskNotFoundError, match="nonexistent-id"):
            service.get_task("nonexistent-id")


# ===========================================================================
# update_task()
# ===========================================================================

class TestUpdateTask:
    def test_update_title(self, service):
        task = service.create_task("Old title")
        updated = service.update_task(task.id, title="New title")
        assert updated.title == "New title"

    def test_update_description(self, service):
        task = service.create_task("T")
        updated = service.update_task(task.id, description="Some notes")
        assert updated.description == "Some notes"

    def test_update_priority(self, service):
        task = service.create_task("T")
        updated = service.update_task(task.id, priority="critical")
        assert updated.priority is Priority.CRITICAL

    def test_update_status(self, service):
        task = service.create_task("T")
        updated = service.update_task(task.id, status="in_progress")
        assert updated.status is Status.IN_PROGRESS

    def test_update_due_date(self, service):
        task = service.create_task("T")
        updated = service.update_task(task.id, due_date=_future_date())
        assert updated.due_date == _future_date()

    def test_update_clear_due_date(self, service):
        task = service.create_task("T", due_date=_future_date())
        updated = service.update_task(task.id, due_date="")
        assert updated.due_date is None

    def test_update_tags(self, service):
        task = service.create_task("T", tags=["old"])
        updated = service.update_task(task.id, tags=["new1", "new2"])
        assert updated.tags == ["new1", "new2"]

    def test_update_persists_to_disk(self, service, storage):
        task = service.create_task("T")
        service.update_task(task.id, title="Updated")
        loaded = storage.load()
        assert loaded.get_by_id(task.id).title == "Updated"

    def test_update_not_found_raises(self, service):
        with pytest.raises(TaskNotFoundError):
            service.update_task("no-such-id", title="X")

    def test_update_by_short_id(self, service):
        task = service.create_task("T")
        updated = service.update_task(task.short_id, title="Via short ID")
        assert updated.title == "Via short ID"

    def test_update_empty_title_raises(self, service):
        task = service.create_task("T")
        with pytest.raises(ValueError, match="title must not be empty"):
            service.update_task(task.id, title="")

    def test_update_invalid_priority_raises(self, service):
        task = service.create_task("T")
        with pytest.raises(ValueError):
            service.update_task(task.id, priority="super-urgent")

    def test_update_invalid_due_date_raises(self, service):
        task = service.create_task("T")
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            service.update_task(task.id, due_date="not-a-date")

    def test_update_no_kwargs_still_returns_task(self, service):
        task = service.create_task("T")
        result = service.update_task(task.id)
        assert result.id == task.id


# ===========================================================================
# delete_task()
# ===========================================================================

class TestDeleteTask:
    def test_delete_removes_task(self, service):
        task = service.create_task("Delete me")
        service.delete_task(task.id)
        assert service.task_count == 0

    def test_delete_returns_removed_task(self, service):
        task = service.create_task("Delete me")
        removed = service.delete_task(task.id)
        assert removed.id == task.id
        assert removed.title == "Delete me"

    def test_delete_by_short_id(self, service):
        task = service.create_task("Delete by short ID")
        service.delete_task(task.short_id)
        assert service.task_count == 0

    def test_delete_persists_to_disk(self, service, storage):
        task = service.create_task("T")
        service.delete_task(task.id)
        loaded = storage.load()
        assert len(loaded) == 0

    def test_delete_not_found_raises(self, service):
        with pytest.raises(TaskNotFoundError):
            service.delete_task("no-such-id")

    def test_delete_only_removes_target(self, service):
        t1 = service.create_task("Keep me")
        t2 = service.create_task("Delete me")
        service.delete_task(t2.id)
        assert service.task_count == 1
        assert service.get_task(t1.id).title == "Keep me"


# ===========================================================================
# complete_task() / cancel_task() / start_task()
# ===========================================================================

class TestStatusTransitions:
    def test_complete_task(self, service):
        task = service.create_task("T")
        result = service.complete_task(task.id)
        assert result.status is Status.DONE

    def test_complete_task_persists(self, service, storage):
        task = service.create_task("T")
        service.complete_task(task.id)
        loaded = storage.load()
        assert loaded.get_by_id(task.id).status is Status.DONE

    def test_complete_task_not_found_raises(self, service):
        with pytest.raises(TaskNotFoundError):
            service.complete_task("no-such-id")

    def test_cancel_task(self, service):
        task = service.create_task("T")
        result = service.cancel_task(task.id)
        assert result.status is Status.CANCELLED

    def test_cancel_task_persists(self, service, storage):
        task = service.create_task("T")
        service.cancel_task(task.id)
        loaded = storage.load()
        assert loaded.get_by_id(task.id).status is Status.CANCELLED

    def test_cancel_task_not_found_raises(self, service):
        with pytest.raises(TaskNotFoundError):
            service.cancel_task("no-such-id")

    def test_start_task(self, service):
        task = service.create_task("T")
        result = service.start_task(task.id)
        assert result.status is Status.IN_PROGRESS

    def test_start_task_persists(self, service, storage):
        task = service.create_task("T")
        service.start_task(task.id)
        loaded = storage.load()
        assert loaded.get_by_id(task.id).status is Status.IN_PROGRESS

    def test_start_task_not_found_raises(self, service):
        with pytest.raises(TaskNotFoundError):
            service.start_task("no-such-id")

    def test_complete_by_short_id(self, service):
        task = service.create_task("T")
        result = service.complete_task(task.short_id)
        assert result.status is Status.DONE

    def test_cancel_by_short_id(self, service):
        task = service.create_task("T")
        result = service.cancel_task(task.short_id)
        assert result.status is Status.CANCELLED


# ===========================================================================
# list_tasks()
# ===========================================================================

class TestListTasks:
    def test_list_all_no_filters(self, populated_service):
        tasks = populated_service.list_tasks()
        assert len(tasks) == 3

    def test_list_returns_plain_list(self, populated_service):
        tasks = populated_service.list_tasks()
        assert isinstance(tasks, list)
        assert all(isinstance(t, Task) for t in tasks)

    def test_list_empty_service(self, service):
        tasks = service.list_tasks()
        assert tasks == []

    # --- Filter by status ---
    def test_filter_by_status_todo(self, service):
        service.create_task("T1")
        t2 = service.create_task("T2")
        service.complete_task(t2.id)
        tasks = service.list_tasks(status="todo")
        assert len(tasks) == 1
        assert tasks[0].title == "T1"

    def test_filter_by_status_done(self, service):
        t1 = service.create_task("T1")
        service.create_task("T2")
        service.complete_task(t1.id)
        tasks = service.list_tasks(status="done")
        assert len(tasks) == 1
        assert tasks[0].title == "T1"

    def test_filter_by_status_no_match(self, service):
        service.create_task("T1")
        tasks = service.list_tasks(status="cancelled")
        assert tasks == []

    # --- Filter by priority ---
    def test_filter_by_priority(self, populated_service):
        tasks = populated_service.list_tasks(priority="critical")
        assert len(tasks) == 1
        assert tasks[0].title == "Gamma"

    def test_filter_by_priority_no_match(self, service):
        service.create_task("T", priority="low")
        tasks = service.list_tasks(priority="critical")
        assert tasks == []

    # --- Filter by tag ---
    def test_filter_by_tag(self, populated_service):
        tasks = populated_service.list_tasks(tag="shared")
        assert len(tasks) == 2
        titles = {t.title for t in tasks}
        assert titles == {"Beta", "Gamma"}

    def test_filter_by_tag_no_match(self, populated_service):
        tasks = populated_service.list_tasks(tag="nonexistent")
        assert tasks == []

    # --- Filter overdue ---
    def test_filter_overdue_only(self, service):
        service.create_task("Past due", due_date=_past_date())
        service.create_task("Future due", due_date=_future_date())
        service.create_task("No due date")
        tasks = service.list_tasks(overdue_only=True)
        assert len(tasks) == 1
        assert tasks[0].title == "Past due"

    def test_filter_overdue_excludes_done(self, service):
        t = service.create_task("Past due done", due_date=_past_date())
        service.complete_task(t.id)
        tasks = service.list_tasks(overdue_only=True)
        assert tasks == []

    # --- Search query ---
    def test_search_by_title(self, service):
        service.create_task("Fix the bug")
        service.create_task("Write docs")
        tasks = service.list_tasks(search_query="bug")
        assert len(tasks) == 1
        assert tasks[0].title == "Fix the bug"

    def test_search_by_description(self, service):
        service.create_task("T1", description="important backend work")
        service.create_task("T2", description="frontend styling")
        tasks = service.list_tasks(search_query="backend")
        assert len(tasks) == 1
        assert tasks[0].title == "T1"

    def test_search_case_insensitive(self, service):
        service.create_task("Deploy to PRODUCTION")
        tasks = service.list_tasks(search_query="production")
        assert len(tasks) == 1

    def test_search_no_match(self, service):
        service.create_task("T")
        tasks = service.list_tasks(search_query="zzz")
        assert tasks == []

    # --- Combined filters ---
    def test_combined_status_and_priority(self, service):
        service.create_task("T1", priority="high")
        t2 = service.create_task("T2", priority="high")
        service.complete_task(t2.id)
        tasks = service.list_tasks(status="todo", priority="high")
        assert len(tasks) == 1
        assert tasks[0].title == "T1"

    def test_combined_tag_and_overdue(self, service):
        service.create_task("Overdue tagged", due_date=_past_date(), tags=["urgent"])
        service.create_task("Future tagged", due_date=_future_date(), tags=["urgent"])
        service.create_task("Overdue untagged", due_date=_past_date())
        tasks = service.list_tasks(tag="urgent", overdue_only=True)
        assert len(tasks) == 1
        assert tasks[0].title == "Overdue tagged"

    # --- Sort by priority ---
    def test_sort_by_priority_default_most_urgent_first(self, service):
        """Default sort_by='priority' with ascending=True puts Critical first."""
        service.create_task("Low",      priority="low")
        service.create_task("High",     priority="high")
        service.create_task("Critical", priority="critical")
        # ascending=True (default) → most-urgent first (critical → low)
        tasks = service.list_tasks(sort_by="priority")
        assert tasks[0].title == "Critical"
        assert tasks[-1].title == "Low"

    def test_sort_by_priority_least_urgent_first(self, service):
        """ascending=False puts Low first (least-urgent first)."""
        service.create_task("Low",      priority="low")
        service.create_task("High",     priority="high")
        service.create_task("Critical", priority="critical")
        tasks = service.list_tasks(sort_by="priority", ascending=False)
        assert tasks[0].title == "Low"
        assert tasks[-1].title == "Critical"

    def test_sort_by_due_date_ascending(self, service):
        service.create_task("Far",  due_date=_future_date(60))
        service.create_task("Near", due_date=_future_date(10))
        tasks = service.list_tasks(sort_by="due_date", ascending=True)
        assert tasks[0].title == "Near"
        assert tasks[1].title == "Far"

    def test_sort_by_title_ascending(self, service):
        service.create_task("Zebra")
        service.create_task("Apple")
        service.create_task("Mango")
        tasks = service.list_tasks(sort_by="title", ascending=True)
        assert tasks[0].title == "Apple"
        assert tasks[-1].title == "Zebra"

    def test_sort_by_title_descending(self, service):
        service.create_task("Zebra")
        service.create_task("Apple")
        tasks = service.list_tasks(sort_by="title", ascending=False)
        assert tasks[0].title == "Zebra"
        assert tasks[-1].title == "Apple"

    def test_sort_by_created_at(self, service):
        t1 = service.create_task("First")
        t2 = service.create_task("Second")
        tasks = service.list_tasks(sort_by="created_at", ascending=True)
        assert tasks[0].id == t1.id
        assert tasks[1].id == t2.id

    def test_invalid_sort_by_raises(self, service):
        with pytest.raises(ValueError, match="Invalid sort_by"):
            service.list_tasks(sort_by="nonexistent_field")


# ===========================================================================
# get_overdue_tasks()
# ===========================================================================

class TestGetOverdueTasks:
    def test_returns_only_overdue(self, service):
        service.create_task("Overdue",  due_date=_past_date(3))
        service.create_task("Future",   due_date=_future_date(3))
        service.create_task("No date")
        tasks = service.get_overdue_tasks()
        assert len(tasks) == 1
        assert tasks[0].title == "Overdue"

    def test_sorted_by_due_date_ascending(self, service):
        service.create_task("Later overdue",  due_date=_past_date(1))
        service.create_task("Earlier overdue", due_date=_past_date(10))
        tasks = service.get_overdue_tasks()
        assert tasks[0].title == "Earlier overdue"
        assert tasks[1].title == "Later overdue"

    def test_excludes_done_tasks(self, service):
        t = service.create_task("Done overdue", due_date=_past_date())
        service.complete_task(t.id)
        tasks = service.get_overdue_tasks()
        assert tasks == []

    def test_empty_when_no_overdue(self, service):
        service.create_task("Future", due_date=_future_date())
        tasks = service.get_overdue_tasks()
        assert tasks == []


# ===========================================================================
# search_tasks()
# ===========================================================================

class TestSearchTasks:
    def test_search_by_title(self, service):
        service.create_task("Fix login bug")
        service.create_task("Update README")
        tasks = service.search_tasks("login")
        assert len(tasks) == 1
        assert tasks[0].title == "Fix login bug"

    def test_search_by_description(self, service):
        service.create_task("T1", description="needs database migration")
        service.create_task("T2", description="frontend only")
        tasks = service.search_tasks("database")
        assert len(tasks) == 1
        assert tasks[0].title == "T1"

    def test_search_case_insensitive(self, service):
        service.create_task("Deploy to STAGING")
        tasks = service.search_tasks("staging")
        assert len(tasks) == 1

    def test_search_no_results(self, service):
        service.create_task("T")
        tasks = service.search_tasks("zzz_no_match")
        assert tasks == []

    def test_search_returns_list(self, service):
        service.create_task("T")
        result = service.search_tasks("T")
        assert isinstance(result, list)


# ===========================================================================
# get_statistics()
# ===========================================================================

class TestGetStatistics:
    def test_empty_statistics(self, service):
        stats = service.get_statistics()
        assert stats["total"] == 0
        assert stats["by_status"]["todo"] == 0
        assert stats["by_status"]["done"] == 0
        assert stats["by_priority"]["low"] == 0
        assert stats["overdue"] == 0

    def test_statistics_counts_by_status(self, service):
        service.create_task("T1")
        t2 = service.create_task("T2")
        service.complete_task(t2.id)
        stats = service.get_statistics()
        assert stats["total"] == 2
        assert stats["by_status"]["todo"] == 1
        assert stats["by_status"]["done"] == 1

    def test_statistics_counts_by_priority(self, service):
        service.create_task("T1", priority="low")
        service.create_task("T2", priority="high")
        service.create_task("T3", priority="high")
        stats = service.get_statistics()
        assert stats["by_priority"]["low"] == 1
        assert stats["by_priority"]["high"] == 2
        assert stats["by_priority"]["medium"] == 0

    def test_statistics_overdue_count(self, service):
        service.create_task("Overdue", due_date=_past_date())
        service.create_task("Future",  due_date=_future_date())
        stats = service.get_statistics()
        assert stats["overdue"] == 1

    def test_statistics_has_all_status_keys(self, service):
        stats = service.get_statistics()
        assert "todo" in stats["by_status"]
        assert "in_progress" in stats["by_status"]
        assert "done" in stats["by_status"]
        assert "cancelled" in stats["by_status"]

    def test_statistics_has_all_priority_keys(self, service):
        stats = service.get_statistics()
        assert "low" in stats["by_priority"]
        assert "medium" in stats["by_priority"]
        assert "high" in stats["by_priority"]
        assert "critical" in stats["by_priority"]

    def test_statistics_total_matches_task_count(self, service):
        service.create_task("T1")
        service.create_task("T2")
        service.create_task("T3")
        stats = service.get_statistics()
        assert stats["total"] == service.task_count


# ===========================================================================
# bulk_complete()
# ===========================================================================

class TestBulkComplete:
    def test_bulk_complete_all_found(self, service):
        t1 = service.create_task("T1")
        t2 = service.create_task("T2")
        updated = service.bulk_complete([t1.id, t2.id])
        assert len(updated) == 2
        assert all(t.status is Status.DONE for t in updated)

    def test_bulk_complete_persists(self, service, storage):
        t1 = service.create_task("T1")
        t2 = service.create_task("T2")
        service.bulk_complete([t1.id, t2.id])
        loaded = storage.load()
        for t in loaded:
            assert t.status is Status.DONE

    def test_bulk_complete_skips_not_found(self, service):
        t1 = service.create_task("T1")
        updated = service.bulk_complete([t1.id, "nonexistent-id"])
        assert len(updated) == 1
        assert updated[0].id == t1.id

    def test_bulk_complete_all_not_found_returns_empty(self, service):
        updated = service.bulk_complete(["bad1", "bad2"])
        assert updated == []

    def test_bulk_complete_empty_list(self, service):
        updated = service.bulk_complete([])
        assert updated == []

    def test_bulk_complete_by_short_id(self, service):
        t1 = service.create_task("T1")
        updated = service.bulk_complete([t1.short_id])
        assert len(updated) == 1
        assert updated[0].status is Status.DONE


# ===========================================================================
# bulk_delete()
# ===========================================================================

class TestBulkDelete:
    def test_bulk_delete_all_found(self, service):
        t1 = service.create_task("T1")
        t2 = service.create_task("T2")
        deleted = service.bulk_delete([t1.id, t2.id])
        assert len(deleted) == 2
        assert service.task_count == 0

    def test_bulk_delete_persists(self, service, storage):
        t1 = service.create_task("T1")
        t2 = service.create_task("T2")
        service.bulk_delete([t1.id, t2.id])
        loaded = storage.load()
        assert len(loaded) == 0

    def test_bulk_delete_skips_not_found(self, service):
        t1 = service.create_task("T1")
        deleted = service.bulk_delete([t1.id, "nonexistent-id"])
        assert len(deleted) == 1
        assert deleted[0].id == t1.id

    def test_bulk_delete_all_not_found_returns_empty(self, service):
        deleted = service.bulk_delete(["bad1", "bad2"])
        assert deleted == []

    def test_bulk_delete_empty_list(self, service):
        deleted = service.bulk_delete([])
        assert deleted == []

    def test_bulk_delete_only_removes_targets(self, service):
        t1 = service.create_task("Keep")
        t2 = service.create_task("Delete")
        service.bulk_delete([t2.id])
        assert service.task_count == 1
        assert service.get_task(t1.id).title == "Keep"


# ===========================================================================
# reload()
# ===========================================================================

class TestReload:
    def test_reload_picks_up_external_changes(self, service, storage):
        """Simulate an external write and verify reload picks it up."""
        # Write a task directly to storage (bypassing the service)
        tl = TaskList()
        tl.add(Task(title="External task"))
        storage.save(tl)

        # Service hasn't loaded yet — first access should see the task
        assert service.task_count == 1

    def test_reload_after_external_change(self, service, storage):
        """After reload(), the service should see externally added tasks."""
        # Service loads once
        _ = service.task_count  # triggers initial load

        # External write
        tl = TaskList()
        tl.add(Task(title="Added externally"))
        storage.save(tl)

        # Without reload, service still sees old state
        assert service.task_count == 0

        # After reload, service sees new state
        service.reload()
        assert service.task_count == 1
        assert service.get_task(service.list_tasks()[0].id).title == "Added externally"


# ===========================================================================
# backup()
# ===========================================================================

class TestBackup:
    def test_backup_creates_file(self, service, storage):
        service.create_task("T")
        bak_path = service.backup()
        assert bak_path.exists()

    def test_backup_contains_correct_data(self, service, storage):
        import json
        service.create_task("Backup me")
        bak_path = service.backup()
        data = json.loads(bak_path.read_text(encoding="utf-8"))
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["title"] == "Backup me"

    def test_backup_no_file_raises(self, service):
        with pytest.raises(StorageReadError):
            service.backup()

    def test_backup_custom_suffix(self, service):
        service.create_task("T")
        bak_path = service.backup(suffix=".backup2")
        assert bak_path.name.endswith(".backup2")


# ===========================================================================
# Integration — full CRUD workflow across multiple service instances
# ===========================================================================

class TestIntegrationWorkflow:
    def test_full_crud_workflow(self, tmp_dir):
        """Simulate a realistic CLI session across multiple service instances."""
        path = tmp_dir / "tasks.json"

        # --- Session 1: create tasks ---
        svc1 = TaskService(TaskStorage(path))
        t1 = svc1.create_task(
            "Design schema",
            priority="high",
            tags=["backend"],
            due_date=_future_date(30),
        )
        t2 = svc1.create_task("Write unit tests", priority="medium", tags=["testing"])
        t3 = svc1.create_task(
            "Deploy to staging",
            priority="critical",
            tags=["devops", "backend"],
            due_date=_past_date(2),
        )
        assert svc1.task_count == 3

        # --- Session 2: update and transition ---
        svc2 = TaskService(TaskStorage(path))
        svc2.start_task(t1.id)
        svc2.complete_task(t2.id)

        # Verify in-memory state
        assert svc2.get_task(t1.id).status is Status.IN_PROGRESS
        assert svc2.get_task(t2.id).status is Status.DONE

        # --- Session 3: query ---
        svc3 = TaskService(TaskStorage(path))
        todo_tasks = svc3.list_tasks(status="todo")
        assert len(todo_tasks) == 1
        assert todo_tasks[0].title == "Deploy to staging"

        overdue = svc3.get_overdue_tasks()
        assert len(overdue) == 1
        assert overdue[0].title == "Deploy to staging"

        backend_tasks = svc3.list_tasks(tag="backend")
        assert len(backend_tasks) == 2

        stats = svc3.get_statistics()
        assert stats["total"] == 3
        assert stats["by_status"]["in_progress"] == 1
        assert stats["by_status"]["done"] == 1
        assert stats["by_status"]["todo"] == 1
        assert stats["overdue"] == 1

        # --- Session 4: bulk operations ---
        svc4 = TaskService(TaskStorage(path))
        svc4.bulk_complete([t3.id])
        assert svc4.get_task(t3.id).status is Status.DONE

        # Overdue should now be 0 (t3 is done)
        stats4 = svc4.get_statistics()
        assert stats4["overdue"] == 0

        # --- Session 5: delete and verify ---
        svc5 = TaskService(TaskStorage(path))
        svc5.delete_task(t3.id)
        assert svc5.task_count == 2
        with pytest.raises(TaskNotFoundError):
            svc5.get_task(t3.id)

        # --- Session 6: search ---
        svc6 = TaskService(TaskStorage(path))
        results = svc6.search_tasks("schema")
        assert len(results) == 1
        assert results[0].title == "Design schema"

    def test_service_survives_empty_storage(self, tmp_dir):
        """A service on a non-existent file should start with zero tasks."""
        svc = TaskService(TaskStorage(tmp_dir / "new.json"))
        assert svc.task_count == 0
        stats = svc.get_statistics()
        assert stats["total"] == 0

    def test_two_independent_services(self, tmp_dir):
        """Two services pointing at different files should not interfere."""
        svc_a = TaskService(TaskStorage(tmp_dir / "a.json"))
        svc_b = TaskService(TaskStorage(tmp_dir / "b.json"))

        svc_a.create_task("Task A1")
        svc_a.create_task("Task A2")
        svc_b.create_task("Task B1")

        assert svc_a.task_count == 2
        assert svc_b.task_count == 1

        # Reload from disk to confirm isolation
        svc_a2 = TaskService(TaskStorage(tmp_dir / "a.json"))
        svc_b2 = TaskService(TaskStorage(tmp_dir / "b.json"))
        assert svc_a2.task_count == 2
        assert svc_b2.task_count == 1

    def test_sort_and_filter_combined_workflow(self, tmp_dir):
        """Verify filtering + sorting produces correct ordered results."""
        svc = TaskService(TaskStorage(tmp_dir / "tasks.json"))
        svc.create_task("Low todo",      priority="low",      due_date=_future_date(20))
        svc.create_task("High todo",     priority="high",     due_date=_future_date(5))
        svc.create_task("Critical todo", priority="critical", due_date=_future_date(10))
        t_done = svc.create_task("High done", priority="high")
        svc.complete_task(t_done.id)

        # Filter todo, sort by priority (ascending=True → most-urgent first)
        tasks = svc.list_tasks(status="todo", sort_by="priority")
        assert tasks[0].title == "Critical todo"
        assert tasks[-1].title == "Low todo"

        # Filter todo, sort by due_date ascending
        tasks_by_date = svc.list_tasks(status="todo", sort_by="due_date", ascending=True)
        assert tasks_by_date[0].title == "High todo"
        assert tasks_by_date[-1].title == "Low todo"
