"""Tests for team-test/services.py.

Covers:
- TaskService.add(): happy path, string enum coercion, validation errors
- TaskService.get(): happy path, not-found error
- TaskService.list_all(): no filters, project filter, status filter, priority filter,
  overdue_only filter, string enum coercion in filters
- TaskService.search(): keyword match in title, keyword match in description,
  case-insensitive, empty keyword, project_id scoping, no matches
- TaskService.complete() / start() / reopen(): status transitions
- TaskService.update(): field updates, string enum coercion
- TaskService.complete_all(): all tasks, project-scoped, already-done tasks skipped
- TaskService.purge_completed(): deletes DONE tasks, returns count, project-scoped
- TaskService.move_to_project(): move to another project, unlink (None)
- TaskService.remove(): happy path, not-found error
- TaskService.overdue_report(): shape, ordering, only overdue tasks
- TaskService.format_task(): format strings for various states
- ProjectService.create(): happy path, validation error
- ProjectService.get(): happy path, not-found error
- ProjectService.list_all(): ordering
- ProjectService.find_by_name(): case-insensitive substring match
- ProjectService.rename(): happy path, validation error
- ProjectService.update_description(): happy path
- ProjectService.remove(): cascade=False (unlinks tasks), cascade=True (deletes tasks)
- ProjectService.summary(): shape, completion_pct calculation, zero-task project
- ProjectService.all_summaries(): returns one entry per project
- ProjectService.format_project(): format string
- Integration: full workflow end-to-end
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup — same pattern as other test files in this package
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEAM_TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

for _p in (WORKSPACE_ROOT, TEAM_TEST_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import Priority, Status, Task, Project  # noqa: E402
from storage import Storage, NotFoundError, ValidationError  # noqa: E402
from services import TaskService, ProjectService, ServiceError  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _past(days: int = 5) -> str:
    """Return a date string *days* in the past."""
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")


def _future(days: int = 5) -> str:
    """Return a date string *days* in the future."""
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_store(tmp_path: Path) -> Storage:
    """Return a fresh Storage backed by a temp file."""
    return Storage(str(tmp_path / "data.json"))


@pytest.fixture()
def task_svc(tmp_store: Storage) -> TaskService:
    """Return a TaskService backed by a fresh temp store."""
    return TaskService(tmp_store)


@pytest.fixture()
def proj_svc(tmp_store: Storage) -> ProjectService:
    """Return a ProjectService backed by the same fresh temp store."""
    return ProjectService(tmp_store)


@pytest.fixture()
def services(tmp_store: Storage):
    """Return (task_svc, proj_svc) sharing the same store."""
    return TaskService(tmp_store), ProjectService(tmp_store)


# ===========================================================================
# TaskService — add()
# ===========================================================================


class TestTaskServiceAdd:
    def test_returns_task_instance(self, task_svc: TaskService):
        task = task_svc.add("Write docs")
        assert isinstance(task, Task)

    def test_title_stored(self, task_svc: TaskService):
        task = task_svc.add("Write docs")
        assert task.title == "Write docs"

    def test_defaults(self, task_svc: TaskService):
        task = task_svc.add("Default task")
        assert task.status is Status.TODO
        assert task.priority is Priority.MEDIUM
        assert task.description == ""
        assert task.due_date is None
        assert task.project_id is None

    def test_string_status_coercion(self, task_svc: TaskService):
        task = task_svc.add("In progress task", status="in_progress")
        assert task.status is Status.IN_PROGRESS

    def test_string_priority_coercion(self, task_svc: TaskService):
        task = task_svc.add("High priority task", priority="high")
        assert task.priority is Priority.HIGH

    def test_enum_status_accepted(self, task_svc: TaskService):
        task = task_svc.add("Done task", status=Status.DONE)
        assert task.status is Status.DONE

    def test_enum_priority_accepted(self, task_svc: TaskService):
        task = task_svc.add("Critical task", priority=Priority.CRITICAL)
        assert task.priority is Priority.CRITICAL

    def test_due_date_stored(self, task_svc: TaskService):
        task = task_svc.add("Due task", due_date="2030-01-15")
        assert task.due_date == "2030-01-15"

    def test_description_stored(self, task_svc: TaskService):
        task = task_svc.add("Described task", description="A longer description")
        assert task.description == "A longer description"

    def test_project_id_stored(self, tmp_store: Storage):
        svc = TaskService(tmp_store)
        project = tmp_store.create_project("My Project")
        task = svc.add("Project task", project_id=project.id)
        assert task.project_id == project.id

    def test_empty_title_raises_validation_error(self, task_svc: TaskService):
        with pytest.raises(ValidationError, match="non-empty"):
            task_svc.add("")

    def test_invalid_status_string_raises_validation_error(self, task_svc: TaskService):
        with pytest.raises(ValidationError, match="Invalid status"):
            task_svc.add("Bad status", status="flying")

    def test_invalid_priority_string_raises_validation_error(self, task_svc: TaskService):
        with pytest.raises(ValidationError, match="Invalid priority"):
            task_svc.add("Bad priority", priority="urgent")

    def test_invalid_due_date_raises_validation_error(self, task_svc: TaskService):
        with pytest.raises(ValidationError, match="due_date"):
            task_svc.add("Bad date", due_date="not-a-date")

    def test_nonexistent_project_id_raises_validation_error(self, task_svc: TaskService):
        with pytest.raises(ValidationError):
            task_svc.add("Orphan task", project_id="nonexistent-project-id")


# ===========================================================================
# TaskService — get()
# ===========================================================================


class TestTaskServiceGet:
    def test_returns_correct_task(self, task_svc: TaskService):
        created = task_svc.add("Fetch me")
        fetched = task_svc.get(created.id)
        assert fetched.id == created.id
        assert fetched.title == "Fetch me"

    def test_missing_id_raises_not_found(self, task_svc: TaskService):
        with pytest.raises(NotFoundError, match="not found"):
            task_svc.get("nonexistent-id")


# ===========================================================================
# TaskService — list_all()
# ===========================================================================


class TestTaskServiceListAll:
    def test_empty_store_returns_empty_list(self, task_svc: TaskService):
        assert task_svc.list_all() == []

    def test_returns_all_tasks(self, task_svc: TaskService):
        task_svc.add("Task A")
        task_svc.add("Task B")
        task_svc.add("Task C")
        assert len(task_svc.list_all()) == 3

    def test_ordered_by_created_at(self, task_svc: TaskService):
        t1 = task_svc.add("First")
        t2 = task_svc.add("Second")
        t3 = task_svc.add("Third")
        ids = [t.id for t in task_svc.list_all()]
        assert ids == [t1.id, t2.id, t3.id]

    def test_filter_by_project_id(self, tmp_store: Storage):
        svc = TaskService(tmp_store)
        p1 = tmp_store.create_project("P1")
        p2 = tmp_store.create_project("P2")
        svc.add("Task in P1", project_id=p1.id)
        svc.add("Task in P2", project_id=p2.id)
        svc.add("Standalone task")

        p1_tasks = svc.list_all(project_id=p1.id)
        assert len(p1_tasks) == 1
        assert p1_tasks[0].title == "Task in P1"

    def test_filter_by_status_enum(self, task_svc: TaskService):
        task_svc.add("Todo task", status=Status.TODO)
        task_svc.add("Done task", status=Status.DONE)
        todo = task_svc.list_all(status=Status.TODO)
        assert len(todo) == 1
        assert todo[0].title == "Todo task"

    def test_filter_by_status_string(self, task_svc: TaskService):
        task_svc.add("Todo task", status=Status.TODO)
        task_svc.add("Done task", status=Status.DONE)
        done = task_svc.list_all(status="done")
        assert len(done) == 1
        assert done[0].title == "Done task"

    def test_filter_by_priority_enum(self, task_svc: TaskService):
        task_svc.add("Low task", priority=Priority.LOW)
        task_svc.add("High task", priority=Priority.HIGH)
        high = task_svc.list_all(priority=Priority.HIGH)
        assert len(high) == 1
        assert high[0].title == "High task"

    def test_filter_by_priority_string(self, task_svc: TaskService):
        task_svc.add("Low task", priority=Priority.LOW)
        task_svc.add("Critical task", priority=Priority.CRITICAL)
        critical = task_svc.list_all(priority="critical")
        assert len(critical) == 1
        assert critical[0].title == "Critical task"

    def test_filter_overdue_only(self, task_svc: TaskService):
        task_svc.add("Overdue task", due_date=_past())
        task_svc.add("Future task", due_date=_future())
        task_svc.add("No due date task")
        overdue = task_svc.list_all(overdue_only=True)
        assert len(overdue) == 1
        assert overdue[0].title == "Overdue task"

    def test_invalid_status_string_raises_validation_error(self, task_svc: TaskService):
        with pytest.raises(ValidationError, match="Invalid status"):
            task_svc.list_all(status="invalid_status")

    def test_invalid_priority_string_raises_validation_error(self, task_svc: TaskService):
        with pytest.raises(ValidationError, match="Invalid priority"):
            task_svc.list_all(priority="super_urgent")


# ===========================================================================
# TaskService — search()
# ===========================================================================


class TestTaskServiceSearch:
    def test_match_in_title(self, task_svc: TaskService):
        task_svc.add("Write unit tests")
        task_svc.add("Deploy to production")
        results = task_svc.search("unit")
        assert len(results) == 1
        assert results[0].title == "Write unit tests"

    def test_match_in_description(self, task_svc: TaskService):
        task_svc.add("Task A", description="Contains the keyword needle here")
        task_svc.add("Task B", description="Nothing special")
        results = task_svc.search("needle")
        assert len(results) == 1
        assert results[0].title == "Task A"

    def test_case_insensitive(self, task_svc: TaskService):
        task_svc.add("Write UNIT Tests")
        results = task_svc.search("unit")
        assert len(results) == 1

    def test_empty_keyword_returns_all(self, task_svc: TaskService):
        task_svc.add("Task A")
        task_svc.add("Task B")
        results = task_svc.search("")
        assert len(results) == 2

    def test_no_match_returns_empty(self, task_svc: TaskService):
        task_svc.add("Task A")
        results = task_svc.search("xyzzy_no_match")
        assert results == []

    def test_project_id_scoping(self, tmp_store: Storage):
        svc = TaskService(tmp_store)
        p1 = tmp_store.create_project("P1")
        p2 = tmp_store.create_project("P2")
        svc.add("Alpha task", project_id=p1.id)
        svc.add("Alpha task in P2", project_id=p2.id)
        results = svc.search("alpha", project_id=p1.id)
        assert len(results) == 1
        assert results[0].project_id == p1.id

    def test_match_in_both_title_and_description(self, task_svc: TaskService):
        """A task matching in both title and description appears only once."""
        task_svc.add("Alpha task", description="Alpha description")
        results = task_svc.search("alpha")
        assert len(results) == 1


# ===========================================================================
# TaskService — complete() / start() / reopen()
# ===========================================================================


class TestTaskServiceStatusTransitions:
    def test_complete_sets_done(self, task_svc: TaskService):
        task = task_svc.add("Finish me")
        updated = task_svc.complete(task.id)
        assert updated.status is Status.DONE

    def test_complete_persists(self, tmp_store: Storage):
        svc = TaskService(tmp_store)
        task = svc.add("Finish me")
        svc.complete(task.id)
        fetched = svc.get(task.id)
        assert fetched.status is Status.DONE

    def test_complete_missing_task_raises_not_found(self, task_svc: TaskService):
        with pytest.raises(NotFoundError):
            task_svc.complete("nonexistent-id")

    def test_start_sets_in_progress(self, task_svc: TaskService):
        task = task_svc.add("Start me")
        updated = task_svc.start(task.id)
        assert updated.status is Status.IN_PROGRESS

    def test_start_persists(self, tmp_store: Storage):
        svc = TaskService(tmp_store)
        task = svc.add("Start me")
        svc.start(task.id)
        fetched = svc.get(task.id)
        assert fetched.status is Status.IN_PROGRESS

    def test_start_missing_task_raises_not_found(self, task_svc: TaskService):
        with pytest.raises(NotFoundError):
            task_svc.start("nonexistent-id")

    def test_reopen_sets_todo(self, task_svc: TaskService):
        task = task_svc.add("Reopen me", status=Status.DONE)
        updated = task_svc.reopen(task.id)
        assert updated.status is Status.TODO

    def test_reopen_persists(self, tmp_store: Storage):
        svc = TaskService(tmp_store)
        task = svc.add("Reopen me", status=Status.DONE)
        svc.reopen(task.id)
        fetched = svc.get(task.id)
        assert fetched.status is Status.TODO

    def test_reopen_missing_task_raises_not_found(self, task_svc: TaskService):
        with pytest.raises(NotFoundError):
            task_svc.reopen("nonexistent-id")

    def test_full_lifecycle_transition(self, task_svc: TaskService):
        """todo → in_progress → done → todo."""
        task = task_svc.add("Lifecycle task")
        assert task.status is Status.TODO

        task = task_svc.start(task.id)
        assert task.status is Status.IN_PROGRESS

        task = task_svc.complete(task.id)
        assert task.status is Status.DONE

        task = task_svc.reopen(task.id)
        assert task.status is Status.TODO


# ===========================================================================
# TaskService — update()
# ===========================================================================


class TestTaskServiceUpdate:
    def test_update_title(self, task_svc: TaskService):
        task = task_svc.add("Old title")
        updated = task_svc.update(task.id, title="New title")
        assert updated.title == "New title"

    def test_update_description(self, task_svc: TaskService):
        task = task_svc.add("Task")
        updated = task_svc.update(task.id, description="New description")
        assert updated.description == "New description"

    def test_update_status_with_enum(self, task_svc: TaskService):
        task = task_svc.add("Task")
        updated = task_svc.update(task.id, status=Status.IN_PROGRESS)
        assert updated.status is Status.IN_PROGRESS

    def test_update_status_with_string(self, task_svc: TaskService):
        task = task_svc.add("Task")
        updated = task_svc.update(task.id, status="done")
        assert updated.status is Status.DONE

    def test_update_priority_with_enum(self, task_svc: TaskService):
        task = task_svc.add("Task")
        updated = task_svc.update(task.id, priority=Priority.HIGH)
        assert updated.priority is Priority.HIGH

    def test_update_priority_with_string(self, task_svc: TaskService):
        task = task_svc.add("Task")
        updated = task_svc.update(task.id, priority="critical")
        assert updated.priority is Priority.CRITICAL

    def test_update_due_date(self, task_svc: TaskService):
        task = task_svc.add("Task")
        updated = task_svc.update(task.id, due_date="2030-06-15")
        assert updated.due_date == "2030-06-15"

    def test_update_missing_task_raises_not_found(self, task_svc: TaskService):
        with pytest.raises(NotFoundError):
            task_svc.update("nonexistent-id", title="New title")

    def test_update_invalid_status_raises_validation_error(self, task_svc: TaskService):
        task = task_svc.add("Task")
        with pytest.raises(ValidationError, match="Invalid status"):
            task_svc.update(task.id, status="flying")

    def test_update_invalid_priority_raises_validation_error(self, task_svc: TaskService):
        task = task_svc.add("Task")
        with pytest.raises(ValidationError, match="Invalid priority"):
            task_svc.update(task.id, priority="super_urgent")

    def test_no_args_leaves_task_unchanged(self, task_svc: TaskService):
        task = task_svc.add("Unchanged task")
        original_title = task.title
        updated = task_svc.update(task.id)
        assert updated.title == original_title


# ===========================================================================
# TaskService — complete_all()
# ===========================================================================


class TestTaskServiceCompleteAll:
    def test_completes_all_non_done_tasks(self, task_svc: TaskService):
        task_svc.add("Task A")
        task_svc.add("Task B")
        task_svc.add("Task C")
        completed = task_svc.complete_all()
        assert len(completed) == 3
        for t in completed:
            assert t.status is Status.DONE

    def test_already_done_tasks_not_returned(self, task_svc: TaskService):
        task_svc.add("Already done", status=Status.DONE)
        task_svc.add("Pending task")
        completed = task_svc.complete_all()
        assert len(completed) == 1
        assert completed[0].title == "Pending task"

    def test_all_already_done_returns_empty_list(self, task_svc: TaskService):
        task_svc.add("Done A", status=Status.DONE)
        task_svc.add("Done B", status=Status.DONE)
        completed = task_svc.complete_all()
        assert completed == []

    def test_project_scoped_complete_all(self, tmp_store: Storage):
        svc = TaskService(tmp_store)
        p1 = tmp_store.create_project("P1")
        p2 = tmp_store.create_project("P2")
        svc.add("P1 task A", project_id=p1.id)
        svc.add("P1 task B", project_id=p1.id)
        svc.add("P2 task", project_id=p2.id)

        completed = svc.complete_all(project_id=p1.id)
        assert len(completed) == 2
        # P2 task should remain untouched
        p2_tasks = svc.list_all(project_id=p2.id)
        assert p2_tasks[0].status is Status.TODO

    def test_project_scoped_nonexistent_project_raises_not_found(
        self, task_svc: TaskService
    ):
        with pytest.raises(NotFoundError):
            task_svc.complete_all(project_id="nonexistent-project-id")

    def test_complete_all_persists(self, tmp_store: Storage):
        svc = TaskService(tmp_store)
        svc.add("Task A")
        svc.add("Task B")
        svc.complete_all()

        # Reload from disk
        svc2 = TaskService(Storage(tmp_store._path))
        for t in svc2.list_all():
            assert t.status is Status.DONE


# ===========================================================================
# TaskService — purge_completed()
# ===========================================================================


class TestTaskServicePurgeCompleted:
    def test_deletes_done_tasks_and_returns_count(self, task_svc: TaskService):
        task_svc.add("Done A", status=Status.DONE)
        task_svc.add("Done B", status=Status.DONE)
        task_svc.add("Pending task")
        count = task_svc.purge_completed()
        assert count == 2
        remaining = task_svc.list_all()
        assert len(remaining) == 1
        assert remaining[0].title == "Pending task"

    def test_no_done_tasks_returns_zero(self, task_svc: TaskService):
        task_svc.add("Pending task")
        count = task_svc.purge_completed()
        assert count == 0

    def test_project_scoped_purge(self, tmp_store: Storage):
        svc = TaskService(tmp_store)
        p1 = tmp_store.create_project("P1")
        p2 = tmp_store.create_project("P2")
        svc.add("P1 done", status=Status.DONE, project_id=p1.id)
        svc.add("P2 done", status=Status.DONE, project_id=p2.id)

        count = svc.purge_completed(project_id=p1.id)
        assert count == 1
        # P2 task should still exist
        p2_tasks = svc.list_all(project_id=p2.id)
        assert len(p2_tasks) == 1

    def test_nonexistent_project_raises_not_found(self, task_svc: TaskService):
        with pytest.raises(NotFoundError):
            task_svc.purge_completed(project_id="nonexistent-project-id")

    def test_purge_persists(self, tmp_store: Storage):
        svc = TaskService(tmp_store)
        svc.add("Done task", status=Status.DONE)
        svc.add("Pending task")
        svc.purge_completed()

        svc2 = TaskService(Storage(tmp_store._path))
        all_tasks = svc2.list_all()
        assert len(all_tasks) == 1
        assert all_tasks[0].title == "Pending task"


# ===========================================================================
# TaskService — move_to_project()
# ===========================================================================


class TestTaskServiceMoveToProject:
    def test_move_to_another_project(self, tmp_store: Storage):
        svc = TaskService(tmp_store)
        p1 = tmp_store.create_project("P1")
        p2 = tmp_store.create_project("P2")
        task = svc.add("Moveable task", project_id=p1.id)

        updated = svc.move_to_project(task.id, p2.id)
        assert updated.project_id == p2.id

    def test_unlink_from_project(self, tmp_store: Storage):
        svc = TaskService(tmp_store)
        p1 = tmp_store.create_project("P1")
        task = svc.add("Linked task", project_id=p1.id)

        updated = svc.move_to_project(task.id, None)
        assert updated.project_id is None

    def test_move_missing_task_raises_not_found(self, tmp_store: Storage):
        svc = TaskService(tmp_store)
        p1 = tmp_store.create_project("P1")
        with pytest.raises(NotFoundError):
            svc.move_to_project("nonexistent-task-id", p1.id)

    def test_move_to_nonexistent_project_raises_validation_error(
        self, tmp_store: Storage
    ):
        svc = TaskService(tmp_store)
        task = svc.add("Task")
        with pytest.raises(ValidationError):
            svc.move_to_project(task.id, "nonexistent-project-id")


# ===========================================================================
# TaskService — remove()
# ===========================================================================


class TestTaskServiceRemove:
    def test_removes_task(self, task_svc: TaskService):
        task = task_svc.add("Delete me")
        task_svc.remove(task.id)
        with pytest.raises(NotFoundError):
            task_svc.get(task.id)

    def test_remove_missing_task_raises_not_found(self, task_svc: TaskService):
        with pytest.raises(NotFoundError):
            task_svc.remove("nonexistent-id")

    def test_remove_persists(self, tmp_store: Storage):
        svc = TaskService(tmp_store)
        task = svc.add("Delete me")
        svc.remove(task.id)

        svc2 = TaskService(Storage(tmp_store._path))
        assert svc2.list_all() == []


# ===========================================================================
# TaskService — overdue_report()
# ===========================================================================


class TestTaskServiceOverdueReport:
    def test_empty_store_returns_empty_list(self, task_svc: TaskService):
        assert task_svc.overdue_report() == []

    def test_no_overdue_tasks_returns_empty_list(self, task_svc: TaskService):
        task_svc.add("Future task", due_date=_future())
        task_svc.add("No due date task")
        assert task_svc.overdue_report() == []

    def test_returns_only_overdue_tasks(self, task_svc: TaskService):
        task_svc.add("Overdue task", due_date=_past())
        task_svc.add("Future task", due_date=_future())
        report = task_svc.overdue_report()
        assert len(report) == 1
        assert report[0]["title"] == "Overdue task"

    def test_done_tasks_not_included(self, task_svc: TaskService):
        task_svc.add("Done overdue", due_date=_past(), status=Status.DONE)
        task_svc.add("Active overdue", due_date=_past())
        report = task_svc.overdue_report()
        assert len(report) == 1
        assert report[0]["title"] == "Active overdue"

    def test_report_dict_shape(self, task_svc: TaskService):
        task_svc.add("Overdue task", due_date=_past(), priority="high")
        report = task_svc.overdue_report()
        assert len(report) == 1
        entry = report[0]
        assert set(entry.keys()) == {"id", "title", "due_date", "priority", "project_id"}
        assert entry["title"] == "Overdue task"
        assert entry["due_date"] == _past()
        assert entry["priority"] == "high"
        assert entry["project_id"] is None

    def test_report_ordered_by_due_date_earliest_first(self, task_svc: TaskService):
        task_svc.add("Older overdue", due_date=_past(10))
        task_svc.add("Newer overdue", due_date=_past(2))
        report = task_svc.overdue_report()
        assert len(report) == 2
        assert report[0]["due_date"] < report[1]["due_date"]

    def test_report_includes_project_id(self, tmp_store: Storage):
        svc = TaskService(tmp_store)
        project = tmp_store.create_project("My Project")
        svc.add("Overdue in project", due_date=_past(), project_id=project.id)
        report = svc.overdue_report()
        assert report[0]["project_id"] == project.id


# ===========================================================================
# TaskService — format_task()
# ===========================================================================


class TestTaskServiceFormatTask:
    def test_basic_format(self, task_svc: TaskService):
        task = task_svc.add("My task")
        formatted = task_svc.format_task(task)
        assert "[TODO]" in formatted
        assert "My task" in formatted
        assert "medium" in formatted

    def test_includes_due_date_when_present(self, task_svc: TaskService):
        task = task_svc.add("Due task", due_date="2030-12-31")
        formatted = task_svc.format_task(task)
        assert "2030-12-31" in formatted

    def test_no_due_date_section_when_absent(self, task_svc: TaskService):
        task = task_svc.add("No due date task")
        formatted = task_svc.format_task(task)
        assert "due:" not in formatted

    def test_overdue_marker_when_overdue(self, task_svc: TaskService):
        task = task_svc.add("Overdue task", due_date=_past())
        formatted = task_svc.format_task(task)
        assert "overdue!" in formatted

    def test_no_overdue_marker_for_future_task(self, task_svc: TaskService):
        task = task_svc.add("Future task", due_date=_future())
        formatted = task_svc.format_task(task)
        assert "overdue!" not in formatted

    def test_done_status_in_format(self, task_svc: TaskService):
        task = task_svc.add("Done task", status=Status.DONE)
        formatted = task_svc.format_task(task)
        assert "[DONE]" in formatted

    def test_in_progress_status_in_format(self, task_svc: TaskService):
        task = task_svc.add("In progress task", status=Status.IN_PROGRESS)
        formatted = task_svc.format_task(task)
        assert "[IN_PROGRESS]" in formatted

    def test_high_priority_in_format(self, task_svc: TaskService):
        task = task_svc.add("High priority task", priority=Priority.HIGH)
        formatted = task_svc.format_task(task)
        assert "high" in formatted


# ===========================================================================
# ProjectService — create()
# ===========================================================================


class TestProjectServiceCreate:
    def test_returns_project_instance(self, proj_svc: ProjectService):
        project = proj_svc.create("My Project")
        assert isinstance(project, Project)

    def test_name_stored(self, proj_svc: ProjectService):
        project = proj_svc.create("My Project")
        assert project.name == "My Project"

    def test_description_stored(self, proj_svc: ProjectService):
        project = proj_svc.create("My Project", description="A description")
        assert project.description == "A description"

    def test_empty_name_raises_validation_error(self, proj_svc: ProjectService):
        with pytest.raises(ValidationError, match="non-empty"):
            proj_svc.create("")

    def test_whitespace_only_name_raises_validation_error(self, proj_svc: ProjectService):
        with pytest.raises(ValidationError, match="non-empty"):
            proj_svc.create("   ")


# ===========================================================================
# ProjectService — get()
# ===========================================================================


class TestProjectServiceGet:
    def test_returns_correct_project(self, proj_svc: ProjectService):
        created = proj_svc.create("Fetch me")
        fetched = proj_svc.get(created.id)
        assert fetched.id == created.id
        assert fetched.name == "Fetch me"

    def test_missing_id_raises_not_found(self, proj_svc: ProjectService):
        with pytest.raises(NotFoundError, match="not found"):
            proj_svc.get("nonexistent-id")


# ===========================================================================
# ProjectService — list_all()
# ===========================================================================


class TestProjectServiceListAll:
    def test_empty_store_returns_empty_list(self, proj_svc: ProjectService):
        assert proj_svc.list_all() == []

    def test_returns_all_projects(self, proj_svc: ProjectService):
        proj_svc.create("A")
        proj_svc.create("B")
        proj_svc.create("C")
        assert len(proj_svc.list_all()) == 3

    def test_ordered_by_created_at(self, proj_svc: ProjectService):
        p1 = proj_svc.create("First")
        p2 = proj_svc.create("Second")
        p3 = proj_svc.create("Third")
        ids = [p.id for p in proj_svc.list_all()]
        assert ids == [p1.id, p2.id, p3.id]


# ===========================================================================
# ProjectService — find_by_name()
# ===========================================================================


class TestProjectServiceFindByName:
    def test_exact_match(self, proj_svc: ProjectService):
        proj_svc.create("Alpha Project")
        results = proj_svc.find_by_name("Alpha Project")
        assert len(results) == 1
        assert results[0].name == "Alpha Project"

    def test_substring_match(self, proj_svc: ProjectService):
        proj_svc.create("Alpha Project")
        proj_svc.create("Beta Project")
        proj_svc.create("Gamma Work")
        results = proj_svc.find_by_name("Project")
        assert len(results) == 2

    def test_case_insensitive(self, proj_svc: ProjectService):
        proj_svc.create("Alpha Project")
        results = proj_svc.find_by_name("alpha")
        assert len(results) == 1

    def test_no_match_returns_empty(self, proj_svc: ProjectService):
        proj_svc.create("Alpha Project")
        results = proj_svc.find_by_name("xyzzy_no_match")
        assert results == []

    def test_empty_name_returns_all(self, proj_svc: ProjectService):
        proj_svc.create("Alpha")
        proj_svc.create("Beta")
        results = proj_svc.find_by_name("")
        assert len(results) == 2


# ===========================================================================
# ProjectService — rename()
# ===========================================================================


class TestProjectServiceRename:
    def test_renames_project(self, proj_svc: ProjectService):
        project = proj_svc.create("Old Name")
        updated = proj_svc.rename(project.id, "New Name")
        assert updated.name == "New Name"

    def test_rename_persists(self, tmp_store: Storage):
        svc = ProjectService(tmp_store)
        project = svc.create("Old Name")
        svc.rename(project.id, "New Name")

        svc2 = ProjectService(Storage(tmp_store._path))
        fetched = svc2.get(project.id)
        assert fetched.name == "New Name"

    def test_empty_name_raises_validation_error(self, proj_svc: ProjectService):
        project = proj_svc.create("My Project")
        with pytest.raises(ValidationError, match="non-empty"):
            proj_svc.rename(project.id, "")

    def test_missing_id_raises_not_found(self, proj_svc: ProjectService):
        with pytest.raises(NotFoundError):
            proj_svc.rename("nonexistent-id", "New Name")


# ===========================================================================
# ProjectService — update_description()
# ===========================================================================


class TestProjectServiceUpdateDescription:
    def test_updates_description(self, proj_svc: ProjectService):
        project = proj_svc.create("My Project")
        updated = proj_svc.update_description(project.id, "New description")
        assert updated.description == "New description"

    def test_clear_description_with_empty_string(self, proj_svc: ProjectService):
        project = proj_svc.create("My Project", description="Old description")
        updated = proj_svc.update_description(project.id, "")
        assert updated.description == ""

    def test_missing_id_raises_not_found(self, proj_svc: ProjectService):
        with pytest.raises(NotFoundError):
            proj_svc.update_description("nonexistent-id", "New description")


# ===========================================================================
# ProjectService — remove()
# ===========================================================================


class TestProjectServiceRemove:
    def test_removes_project(self, proj_svc: ProjectService):
        project = proj_svc.create("Delete me")
        proj_svc.remove(project.id)
        with pytest.raises(NotFoundError):
            proj_svc.get(project.id)

    def test_remove_missing_project_raises_not_found(self, proj_svc: ProjectService):
        with pytest.raises(NotFoundError):
            proj_svc.remove("nonexistent-id")

    def test_cascade_false_unlinks_tasks(self, tmp_store: Storage):
        task_svc = TaskService(tmp_store)
        proj_svc = ProjectService(tmp_store)
        project = proj_svc.create("My Project")
        task = task_svc.add("Linked task", project_id=project.id)

        proj_svc.remove(project.id, cascade=False)

        # Task should still exist but be unlinked
        fetched = task_svc.get(task.id)
        assert fetched.project_id is None

    def test_cascade_true_deletes_tasks(self, tmp_store: Storage):
        task_svc = TaskService(tmp_store)
        proj_svc = ProjectService(tmp_store)
        project = proj_svc.create("My Project")
        task = task_svc.add("Linked task", project_id=project.id)

        proj_svc.remove(project.id, cascade=True)

        # Task should be deleted
        with pytest.raises(NotFoundError):
            task_svc.get(task.id)

    def test_cascade_false_is_default(self, tmp_store: Storage):
        task_svc = TaskService(tmp_store)
        proj_svc = ProjectService(tmp_store)
        project = proj_svc.create("My Project")
        task = task_svc.add("Linked task", project_id=project.id)

        proj_svc.remove(project.id)  # default cascade=False

        # Task should still exist
        fetched = task_svc.get(task.id)
        assert fetched.project_id is None


# ===========================================================================
# ProjectService — summary()
# ===========================================================================


class TestProjectServiceSummary:
    def test_summary_shape(self, tmp_store: Storage):
        proj_svc = ProjectService(tmp_store)
        project = proj_svc.create("My Project")
        summary = proj_svc.summary(project.id)
        expected_keys = {
            "project_id", "name", "description",
            "total_tasks", "done", "in_progress", "todo",
            "overdue", "completion_pct",
        }
        assert set(summary.keys()) == expected_keys

    def test_summary_zero_tasks(self, tmp_store: Storage):
        proj_svc = ProjectService(tmp_store)
        project = proj_svc.create("Empty Project")
        summary = proj_svc.summary(project.id)
        assert summary["total_tasks"] == 0
        assert summary["done"] == 0
        assert summary["in_progress"] == 0
        assert summary["todo"] == 0
        assert summary["overdue"] == 0
        assert summary["completion_pct"] == 0.0

    def test_summary_completion_pct(self, tmp_store: Storage):
        task_svc = TaskService(tmp_store)
        proj_svc = ProjectService(tmp_store)
        project = proj_svc.create("My Project")
        task_svc.add("Task A", project_id=project.id, status=Status.DONE)
        task_svc.add("Task B", project_id=project.id, status=Status.DONE)
        task_svc.add("Task C", project_id=project.id, status=Status.TODO)
        task_svc.add("Task D", project_id=project.id, status=Status.TODO)

        summary = proj_svc.summary(project.id)
        assert summary["total_tasks"] == 4
        assert summary["done"] == 2
        assert summary["todo"] == 2
        assert summary["completion_pct"] == 50.0

    def test_summary_all_done_gives_100_pct(self, tmp_store: Storage):
        task_svc = TaskService(tmp_store)
        proj_svc = ProjectService(tmp_store)
        project = proj_svc.create("My Project")
        task_svc.add("Task A", project_id=project.id, status=Status.DONE)
        task_svc.add("Task B", project_id=project.id, status=Status.DONE)

        summary = proj_svc.summary(project.id)
        assert summary["completion_pct"] == 100.0

    def test_summary_overdue_count(self, tmp_store: Storage):
        task_svc = TaskService(tmp_store)
        proj_svc = ProjectService(tmp_store)
        project = proj_svc.create("My Project")
        task_svc.add("Overdue task", project_id=project.id, due_date=_past())
        task_svc.add("Future task", project_id=project.id, due_date=_future())

        summary = proj_svc.summary(project.id)
        assert summary["overdue"] == 1

    def test_summary_project_id_and_name(self, tmp_store: Storage):
        proj_svc = ProjectService(tmp_store)
        project = proj_svc.create("Named Project", description="A description")
        summary = proj_svc.summary(project.id)
        assert summary["project_id"] == project.id
        assert summary["name"] == "Named Project"
        assert summary["description"] == "A description"

    def test_summary_missing_project_raises_not_found(self, proj_svc: ProjectService):
        with pytest.raises(NotFoundError):
            proj_svc.summary("nonexistent-id")

    def test_summary_only_counts_tasks_in_project(self, tmp_store: Storage):
        """Tasks from other projects must not appear in the summary."""
        task_svc = TaskService(tmp_store)
        proj_svc = ProjectService(tmp_store)
        p1 = proj_svc.create("P1")
        p2 = proj_svc.create("P2")
        task_svc.add("P1 task", project_id=p1.id, status=Status.DONE)
        task_svc.add("P2 task", project_id=p2.id, status=Status.TODO)

        summary = proj_svc.summary(p1.id)
        assert summary["total_tasks"] == 1
        assert summary["done"] == 1


# ===========================================================================
# ProjectService — all_summaries()
# ===========================================================================


class TestProjectServiceAllSummaries:
    def test_empty_store_returns_empty_list(self, proj_svc: ProjectService):
        assert proj_svc.all_summaries() == []

    def test_returns_one_entry_per_project(self, tmp_store: Storage):
        proj_svc = ProjectService(tmp_store)
        proj_svc.create("P1")
        proj_svc.create("P2")
        proj_svc.create("P3")
        summaries = proj_svc.all_summaries()
        assert len(summaries) == 3

    def test_each_entry_has_correct_shape(self, tmp_store: Storage):
        proj_svc = ProjectService(tmp_store)
        proj_svc.create("P1")
        summaries = proj_svc.all_summaries()
        expected_keys = {
            "project_id", "name", "description",
            "total_tasks", "done", "in_progress", "todo",
            "overdue", "completion_pct",
        }
        assert set(summaries[0].keys()) == expected_keys

    def test_ordered_by_project_creation_time(self, tmp_store: Storage):
        proj_svc = ProjectService(tmp_store)
        p1 = proj_svc.create("First")
        p2 = proj_svc.create("Second")
        p3 = proj_svc.create("Third")
        summaries = proj_svc.all_summaries()
        ids = [s["project_id"] for s in summaries]
        assert ids == [p1.id, p2.id, p3.id]


# ===========================================================================
# ProjectService — format_project()
# ===========================================================================


class TestProjectServiceFormatProject:
    def test_format_contains_name(self, proj_svc: ProjectService):
        project = proj_svc.create("My Project")
        formatted = proj_svc.format_project(project)
        assert "My Project" in formatted

    def test_format_contains_id(self, proj_svc: ProjectService):
        project = proj_svc.create("My Project")
        formatted = proj_svc.format_project(project)
        assert project.id in formatted

    def test_format_string_structure(self, proj_svc: ProjectService):
        project = proj_svc.create("Sprint 1")
        formatted = proj_svc.format_project(project)
        assert formatted == f"Sprint 1 [id: {project.id}]"


# ===========================================================================
# Integration — full end-to-end workflow
# ===========================================================================


class TestServicesIntegration:
    def test_full_project_task_lifecycle(self, tmp_store: Storage):
        """Create project → add tasks → update → complete → purge → verify."""
        task_svc = TaskService(tmp_store)
        proj_svc = ProjectService(tmp_store)

        # 1. Create a project
        project = proj_svc.create("Sprint 1", description="First sprint")
        assert project.name == "Sprint 1"

        # 2. Add tasks with various priorities
        t1 = task_svc.add("Implement login", project_id=project.id, priority="high")
        t2 = task_svc.add("Write tests", project_id=project.id, priority="medium")
        t3 = task_svc.add("Deploy", project_id=project.id, priority="critical")

        # 3. Verify summary shows 3 todo tasks
        summary = proj_svc.summary(project.id)
        assert summary["total_tasks"] == 3
        assert summary["todo"] == 3
        assert summary["completion_pct"] == 0.0

        # 4. Start and complete some tasks
        task_svc.start(t1.id)
        task_svc.complete(t1.id)
        task_svc.complete(t2.id)

        # 5. Verify updated summary
        summary = proj_svc.summary(project.id)
        assert summary["done"] == 2
        assert summary["todo"] == 1
        assert round(summary["completion_pct"], 1) == round(2 / 3 * 100, 1)

        # 6. Search for tasks
        results = task_svc.search("test", project_id=project.id)
        assert len(results) == 1
        assert results[0].title == "Write tests"

        # 7. Purge completed tasks
        purged = task_svc.purge_completed(project_id=project.id)
        assert purged == 2

        # 8. Verify only 1 task remains
        remaining = task_svc.list_all(project_id=project.id)
        assert len(remaining) == 1
        assert remaining[0].title == "Deploy"

        # 9. Complete all remaining tasks
        completed = task_svc.complete_all(project_id=project.id)
        assert len(completed) == 1

        # 10. Delete project with cascade
        proj_svc.remove(project.id, cascade=True)
        assert task_svc.list_all() == []
        assert proj_svc.list_all() == []

    def test_overdue_workflow_across_projects(self, tmp_store: Storage):
        """Overdue tasks are correctly identified across multiple projects."""
        task_svc = TaskService(tmp_store)
        proj_svc = ProjectService(tmp_store)

        p1 = proj_svc.create("P1")
        p2 = proj_svc.create("P2")

        task_svc.add("P1 overdue", project_id=p1.id, due_date=_past(3))
        task_svc.add("P2 overdue", project_id=p2.id, due_date=_past(1))
        task_svc.add("P1 future", project_id=p1.id, due_date=_future(5))

        report = task_svc.overdue_report()
        assert len(report) == 2
        # Ordered by due_date earliest first
        assert report[0]["due_date"] < report[1]["due_date"]

        # Completing an overdue task removes it from the report
        task_svc.complete(task_svc.list_all(project_id=p1.id, overdue_only=True)[0].id)
        report_after = task_svc.overdue_report()
        assert len(report_after) == 1
        assert report_after[0]["project_id"] == p2.id

    def test_move_tasks_between_projects(self, tmp_store: Storage):
        """Tasks can be moved between projects and unlinked."""
        task_svc = TaskService(tmp_store)
        proj_svc = ProjectService(tmp_store)

        p1 = proj_svc.create("P1")
        p2 = proj_svc.create("P2")

        task = task_svc.add("Moveable task", project_id=p1.id)
        assert task.project_id == p1.id

        # Move to P2
        task = task_svc.move_to_project(task.id, p2.id)
        assert task.project_id == p2.id
        assert len(task_svc.list_all(project_id=p1.id)) == 0
        assert len(task_svc.list_all(project_id=p2.id)) == 1

        # Unlink from any project
        task = task_svc.move_to_project(task.id, None)
        assert task.project_id is None
        assert len(task_svc.list_all(project_id=p2.id)) == 0

    def test_reload_from_disk_preserves_all_state(self, tmp_store: Storage):
        """All service-layer changes survive a reload from disk."""
        task_svc = TaskService(tmp_store)
        proj_svc = ProjectService(tmp_store)

        project = proj_svc.create("Persistent Project")
        t1 = task_svc.add("Task A", project_id=project.id, priority="high")
        t2 = task_svc.add("Task B", project_id=project.id, due_date=_future())
        task_svc.complete(t1.id)

        # Reload from disk
        store2 = Storage(tmp_store._path)
        task_svc2 = TaskService(store2)
        proj_svc2 = ProjectService(store2)

        assert len(proj_svc2.list_all()) == 1
        assert proj_svc2.get(project.id).name == "Persistent Project"

        tasks = task_svc2.list_all(project_id=project.id)
        assert len(tasks) == 2

        done_tasks = task_svc2.list_all(status=Status.DONE)
        assert len(done_tasks) == 1
        assert done_tasks[0].id == t1.id
        assert done_tasks[0].priority is Priority.HIGH
