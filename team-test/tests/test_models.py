"""Tests for team-test/models.py.

Covers:
- Status and Priority enum values and str-mixin behaviour
- Task creation with defaults and explicit values
- Task.to_dict() / Task.from_dict() round-trip
- Task.is_overdue() logic (past, future, no due date, DONE status)
- Project creation with defaults and explicit values
- Project.to_dict() / Project.from_dict() round-trip
- Edge cases: missing optional fields in from_dict, unknown status/priority values
"""
from __future__ import annotations

import importlib.util
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone

import pytest

# ---------------------------------------------------------------------------
# Make sure the team-test package is importable when running pytest from the
# workspace root (team-test/ is not a standard package name due to the hyphen,
# so we add its *parent* directory to sys.path and import via the directory
# name aliased as a module).  We also add the team-test directory itself so
# that plain ``import models`` works inside the package tests.
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEAM_TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

for _p in (WORKSPACE_ROOT, TEAM_TEST_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models import Priority, Project, Status, Task  # noqa: E402  (after sys.path setup)


def _utcnow() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


# ===========================================================================
# Status enum
# ===========================================================================


class TestStatusEnum:
    def test_values(self):
        assert Status.TODO.value == "todo"
        assert Status.IN_PROGRESS.value == "in_progress"
        assert Status.DONE.value == "done"

    def test_str_mixin_equality(self):
        """str, Enum mixin means Status.TODO == 'todo'."""
        assert Status.TODO == "todo"
        assert Status.IN_PROGRESS == "in_progress"
        assert Status.DONE == "done"

    def test_construct_from_string(self):
        assert Status("todo") is Status.TODO
        assert Status("in_progress") is Status.IN_PROGRESS
        assert Status("done") is Status.DONE

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            Status("invalid_status")

    def test_all_members(self):
        members = {s.value for s in Status}
        assert members == {"todo", "in_progress", "done"}


# ===========================================================================
# Priority enum
# ===========================================================================


class TestPriorityEnum:
    def test_values(self):
        assert Priority.LOW.value == "low"
        assert Priority.MEDIUM.value == "medium"
        assert Priority.HIGH.value == "high"
        assert Priority.CRITICAL.value == "critical"

    def test_str_mixin_equality(self):
        assert Priority.LOW == "low"
        assert Priority.MEDIUM == "medium"
        assert Priority.HIGH == "high"
        assert Priority.CRITICAL == "critical"

    def test_construct_from_string(self):
        assert Priority("low") is Priority.LOW
        assert Priority("medium") is Priority.MEDIUM
        assert Priority("high") is Priority.HIGH
        assert Priority("critical") is Priority.CRITICAL

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            Priority("urgent")

    def test_all_members(self):
        members = {p.value for p in Priority}
        assert members == {"low", "medium", "high", "critical"}


# ===========================================================================
# Task – creation and defaults
# ===========================================================================


class TestTaskCreation:
    def test_minimal_creation(self):
        task = Task(title="Write tests")
        assert task.title == "Write tests"
        assert task.description == ""
        assert task.status is Status.TODO
        assert task.priority is Priority.MEDIUM
        assert task.due_date is None
        assert task.project_id is None

    def test_auto_generated_id_is_uuid(self):
        task = Task(title="Check UUID")
        # Should not raise
        parsed = uuid.UUID(task.id)
        assert str(parsed) == task.id

    def test_two_tasks_have_different_ids(self):
        t1 = Task(title="Task A")
        t2 = Task(title="Task B")
        assert t1.id != t2.id

    def test_created_at_is_iso_string(self):
        task = Task(title="Timestamp test")
        # Should parse without error
        dt = datetime.fromisoformat(task.created_at)
        assert isinstance(dt, datetime)

    def test_updated_at_is_iso_string(self):
        task = Task(title="Timestamp test")
        dt = datetime.fromisoformat(task.updated_at)
        assert isinstance(dt, datetime)

    def test_explicit_values(self):
        fixed_id = str(uuid.uuid4())
        task = Task(
            title="Explicit task",
            id=fixed_id,
            description="A description",
            status=Status.IN_PROGRESS,
            priority=Priority.HIGH,
            due_date="2026-12-31",
            project_id="proj-123",
        )
        assert task.id == fixed_id
        assert task.description == "A description"
        assert task.status is Status.IN_PROGRESS
        assert task.priority is Priority.HIGH
        assert task.due_date == "2026-12-31"
        assert task.project_id == "proj-123"


# ===========================================================================
# Task – to_dict / from_dict round-trip
# ===========================================================================


class TestTaskSerialization:
    def _make_task(self) -> Task:
        return Task(
            title="Serialise me",
            description="Some description",
            status=Status.IN_PROGRESS,
            priority=Priority.HIGH,
            due_date="2026-06-15",
            project_id="proj-abc",
        )

    def test_to_dict_keys(self):
        d = self._make_task().to_dict()
        expected_keys = {
            "id", "title", "description", "status", "priority",
            "due_date", "project_id", "created_at", "updated_at",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_enum_values_are_strings(self):
        d = self._make_task().to_dict()
        assert d["status"] == "in_progress"
        assert d["priority"] == "high"
        assert isinstance(d["status"], str)
        assert isinstance(d["priority"], str)

    def test_to_dict_values(self):
        task = self._make_task()
        d = task.to_dict()
        assert d["id"] == task.id
        assert d["title"] == "Serialise me"
        assert d["description"] == "Some description"
        assert d["due_date"] == "2026-06-15"
        assert d["project_id"] == "proj-abc"

    def test_round_trip(self):
        original = self._make_task()
        restored = Task.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.title == original.title
        assert restored.description == original.description
        assert restored.status is original.status
        assert restored.priority is original.priority
        assert restored.due_date == original.due_date
        assert restored.project_id == original.project_id
        assert restored.created_at == original.created_at
        assert restored.updated_at == original.updated_at

    def test_from_dict_defaults_for_missing_optional_fields(self):
        """from_dict must tolerate records that lack optional fields."""
        minimal = {
            "id": str(uuid.uuid4()),
            "title": "Minimal record",
        }
        task = Task.from_dict(minimal)
        assert task.title == "Minimal record"
        assert task.description == ""
        assert task.status is Status.TODO
        assert task.priority is Priority.MEDIUM
        assert task.due_date is None
        assert task.project_id is None

    def test_from_dict_status_and_priority_are_enum_instances(self):
        d = self._make_task().to_dict()
        task = Task.from_dict(d)
        assert isinstance(task.status, Status)
        assert isinstance(task.priority, Priority)

    def test_to_dict_none_fields_preserved(self):
        task = Task(title="No due date")
        d = task.to_dict()
        assert d["due_date"] is None
        assert d["project_id"] is None


# ===========================================================================
# Task – is_overdue
# ===========================================================================


class TestTaskIsOverdue:
    def test_no_due_date_not_overdue(self):
        task = Task(title="No due date")
        assert task.is_overdue() is False

    def test_future_due_date_not_overdue(self):
        future = (_utcnow() + timedelta(days=30)).strftime("%Y-%m-%d")
        task = Task(title="Future task", due_date=future)
        assert task.is_overdue() is False

    def test_past_due_date_is_overdue(self):
        past = (_utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        task = Task(title="Past task", due_date=past)
        assert task.is_overdue() is True

    def test_done_task_not_overdue_even_with_past_due_date(self):
        past = (_utcnow() - timedelta(days=5)).strftime("%Y-%m-%d")
        task = Task(title="Done task", due_date=past, status=Status.DONE)
        assert task.is_overdue() is False

    def test_in_progress_task_with_past_due_date_is_overdue(self):
        past = (_utcnow() - timedelta(days=2)).strftime("%Y-%m-%d")
        task = Task(title="In progress overdue", due_date=past, status=Status.IN_PROGRESS)
        assert task.is_overdue() is True

    def test_todo_task_with_past_due_date_is_overdue(self):
        past = (_utcnow() - timedelta(days=10)).strftime("%Y-%m-%d")
        task = Task(title="Todo overdue", due_date=past, status=Status.TODO)
        assert task.is_overdue() is True


# ===========================================================================
# Project – creation and defaults
# ===========================================================================


class TestProjectCreation:
    def test_minimal_creation(self):
        project = Project(name="My Project")
        assert project.name == "My Project"
        assert project.description == ""

    def test_auto_generated_id_is_uuid(self):
        project = Project(name="UUID check")
        parsed = uuid.UUID(project.id)
        assert str(parsed) == project.id

    def test_two_projects_have_different_ids(self):
        p1 = Project(name="Alpha")
        p2 = Project(name="Beta")
        assert p1.id != p2.id

    def test_created_at_is_iso_string(self):
        project = Project(name="Timestamp test")
        dt = datetime.fromisoformat(project.created_at)
        assert isinstance(dt, datetime)

    def test_explicit_values(self):
        fixed_id = str(uuid.uuid4())
        project = Project(
            name="Explicit project",
            id=fixed_id,
            description="A project description",
        )
        assert project.id == fixed_id
        assert project.description == "A project description"


# ===========================================================================
# Project – to_dict / from_dict round-trip
# ===========================================================================


class TestProjectSerialization:
    def _make_project(self) -> Project:
        return Project(name="Test Project", description="Testing serialisation")

    def test_to_dict_keys(self):
        d = self._make_project().to_dict()
        assert set(d.keys()) == {"id", "name", "description", "created_at"}

    def test_to_dict_values(self):
        project = self._make_project()
        d = project.to_dict()
        assert d["id"] == project.id
        assert d["name"] == "Test Project"
        assert d["description"] == "Testing serialisation"
        assert d["created_at"] == project.created_at

    def test_round_trip(self):
        original = self._make_project()
        restored = Project.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.description == original.description
        assert restored.created_at == original.created_at

    def test_from_dict_defaults_for_missing_optional_fields(self):
        minimal = {
            "id": str(uuid.uuid4()),
            "name": "Minimal project",
        }
        project = Project.from_dict(minimal)
        assert project.name == "Minimal project"
        assert project.description == ""

    def test_from_dict_created_at_fallback(self):
        """If created_at is absent, from_dict should supply a timestamp."""
        data = {
            "id": str(uuid.uuid4()),
            "name": "No timestamp",
        }
        project = Project.from_dict(data)
        # Should be a valid ISO datetime string
        dt = datetime.fromisoformat(project.created_at)
        assert isinstance(dt, datetime)


# ===========================================================================
# Package __init__ version
# ===========================================================================


class TestPackageInit:
    def test_version_string(self):
        spec = importlib.util.spec_from_file_location(
            "__init__",
            os.path.join(TEAM_TEST_DIR, "__init__.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "__version__")
        assert mod.__version__ == "0.1.0"
