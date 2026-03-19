"""
Comprehensive tests for team-test/models.py

Covers:
  - Priority enum: values, weights, from_string, error handling
  - Status enum: values, is_terminal, from_string, error handling
  - Task: construction, validation, update, mark_done/cancelled,
          is_overdue, serialisation (to_dict/from_dict/to_json/from_json)
  - TaskList: add, get_by_id, get_by_short_id, remove, update,
              filter_by_status, filter_by_priority, filter_by_tag,
              filter_overdue, search, sorting, serialisation, summary
"""

import importlib.util
import json
import pathlib
import sys
import time
from datetime import date, timedelta

import pytest

# ---------------------------------------------------------------------------
# Load models.py dynamically so we don't need an __init__.py or package setup
# ---------------------------------------------------------------------------
_models_path = pathlib.Path(__file__).parent / "models.py"
_spec = importlib.util.spec_from_file_location("task_models", _models_path)
_mod = importlib.util.module_from_spec(_spec)          # type: ignore[arg-type]
sys.modules["task_models"] = _mod                      # register before exec so @dataclass works
_spec.loader.exec_module(_mod)                          # type: ignore[union-attr]

Priority = _mod.Priority
Status = _mod.Status
Task = _mod.Task
TaskList = _mod.TaskList


# ===========================================================================
# Priority
# ===========================================================================

class TestPriority:
    def test_values(self):
        assert Priority.LOW.value == "low"
        assert Priority.MEDIUM.value == "medium"
        assert Priority.HIGH.value == "high"
        assert Priority.CRITICAL.value == "critical"

    def test_weights_ordered(self):
        assert Priority.LOW.weight < Priority.MEDIUM.weight
        assert Priority.MEDIUM.weight < Priority.HIGH.weight
        assert Priority.HIGH.weight < Priority.CRITICAL.weight

    def test_from_string_exact(self):
        assert Priority.from_string("low") is Priority.LOW
        assert Priority.from_string("medium") is Priority.MEDIUM
        assert Priority.from_string("high") is Priority.HIGH
        assert Priority.from_string("critical") is Priority.CRITICAL

    def test_from_string_case_insensitive(self):
        assert Priority.from_string("LOW") is Priority.LOW
        assert Priority.from_string("High") is Priority.HIGH
        assert Priority.from_string("CRITICAL") is Priority.CRITICAL

    def test_from_string_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid priority"):
            Priority.from_string("urgent")

    def test_str_comparison(self):
        # Because Priority inherits from str, it can be compared to strings
        assert Priority.LOW == "low"
        assert Priority.HIGH == "high"


# ===========================================================================
# Status
# ===========================================================================

class TestStatus:
    def test_values(self):
        assert Status.TODO.value == "todo"
        assert Status.IN_PROGRESS.value == "in_progress"
        assert Status.DONE.value == "done"
        assert Status.CANCELLED.value == "cancelled"

    def test_is_terminal_done(self):
        assert Status.DONE.is_terminal is True

    def test_is_terminal_cancelled(self):
        assert Status.CANCELLED.is_terminal is True

    def test_is_terminal_active(self):
        assert Status.TODO.is_terminal is False
        assert Status.IN_PROGRESS.is_terminal is False

    def test_from_string_exact(self):
        assert Status.from_string("todo") is Status.TODO
        assert Status.from_string("in_progress") is Status.IN_PROGRESS
        assert Status.from_string("done") is Status.DONE
        assert Status.from_string("cancelled") is Status.CANCELLED

    def test_from_string_case_insensitive(self):
        assert Status.from_string("TODO") is Status.TODO
        assert Status.from_string("Done") is Status.DONE

    def test_from_string_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid status"):
            Status.from_string("pending")


# ===========================================================================
# Task – construction & validation
# ===========================================================================

class TestTaskConstruction:
    def test_minimal_task(self):
        t = Task(title="Buy milk")
        assert t.title == "Buy milk"
        assert t.description == ""
        assert t.priority is Priority.MEDIUM
        assert t.status is Status.TODO
        assert t.due_date is None
        assert t.tags == []
        assert len(t.id) == 36  # UUID4 format
        assert t.created_at
        assert t.updated_at

    def test_full_task(self):
        t = Task(
            title="Deploy service",
            description="Push to production",
            priority=Priority.HIGH,
            status=Status.IN_PROGRESS,
            due_date="2030-12-31",
            tags=["devops", "prod"],
        )
        assert t.title == "Deploy service"
        assert t.description == "Push to production"
        assert t.priority is Priority.HIGH
        assert t.status is Status.IN_PROGRESS
        assert t.due_date == "2030-12-31"
        assert "devops" in t.tags
        assert "prod" in t.tags

    def test_empty_title_raises(self):
        with pytest.raises(ValueError, match="title must not be empty"):
            Task(title="")

    def test_whitespace_only_title_raises(self):
        with pytest.raises(ValueError, match="title must not be empty"):
            Task(title="   ")

    def test_string_priority_coerced(self):
        t = Task(title="Test", priority="high")
        assert t.priority is Priority.HIGH

    def test_string_status_coerced(self):
        t = Task(title="Test", status="done")
        assert t.status is Status.DONE

    def test_invalid_priority_raises(self):
        with pytest.raises(ValueError):
            Task(title="Test", priority="urgent")

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError):
            Task(title="Test", status="pending")

    def test_invalid_due_date_raises(self):
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            Task(title="Test", due_date="31-12-2030")

    def test_tags_deduplicated(self):
        t = Task(title="Test", tags=["a", "b", "a", "b"])
        assert t.tags == ["a", "b"]

    def test_tags_stripped(self):
        t = Task(title="Test", tags=["  alpha  ", " beta"])
        assert "alpha" in t.tags
        assert "beta" in t.tags

    def test_empty_tags_filtered(self):
        t = Task(title="Test", tags=["", "  ", "valid"])
        assert t.tags == ["valid"]

    def test_tags_not_list_raises(self):
        with pytest.raises(ValueError, match="tags must be a list"):
            Task(title="Test", tags="not-a-list")  # type: ignore[arg-type]

    def test_auto_id_unique(self):
        t1 = Task(title="A")
        t2 = Task(title="B")
        assert t1.id != t2.id

    def test_short_id(self):
        t = Task(title="Test")
        assert t.short_id == t.id[:8]
        assert len(t.short_id) == 8

    def test_repr_contains_title(self):
        t = Task(title="My Task")
        assert "My Task" in repr(t)


# ===========================================================================
# Task – update
# ===========================================================================

class TestTaskUpdate:
    def test_update_title(self):
        t = Task(title="Old")
        t.update(title="New")
        assert t.title == "New"

    def test_update_empty_title_raises(self):
        t = Task(title="Old")
        with pytest.raises(ValueError, match="title must not be empty"):
            t.update(title="")

    def test_update_description(self):
        t = Task(title="T")
        t.update(description="Some notes")
        assert t.description == "Some notes"

    def test_update_priority(self):
        t = Task(title="T")
        t.update(priority=Priority.CRITICAL)
        assert t.priority is Priority.CRITICAL

    def test_update_priority_string(self):
        t = Task(title="T")
        t.update(priority="low")
        assert t.priority is Priority.LOW

    def test_update_status(self):
        t = Task(title="T")
        t.update(status=Status.DONE)
        assert t.status is Status.DONE

    def test_update_status_string(self):
        t = Task(title="T")
        t.update(status="in_progress")
        assert t.status is Status.IN_PROGRESS

    def test_update_due_date(self):
        t = Task(title="T")
        t.update(due_date="2030-06-15")
        assert t.due_date == "2030-06-15"

    def test_update_due_date_clear(self):
        t = Task(title="T", due_date="2030-06-15")
        t.update(due_date="")
        assert t.due_date is None

    def test_update_invalid_due_date_raises(self):
        t = Task(title="T")
        with pytest.raises(ValueError, match="YYYY-MM-DD"):
            t.update(due_date="not-a-date")

    def test_update_tags(self):
        t = Task(title="T", tags=["old"])
        t.update(tags=["new1", "new2"])
        assert t.tags == ["new1", "new2"]

    def test_update_refreshes_updated_at(self):
        t = Task(title="T")
        old_ts = t.updated_at
        time.sleep(0.01)
        t.update(title="T2")
        assert t.updated_at >= old_ts

    def test_update_no_args_still_refreshes_updated_at(self):
        """Calling update() with no args should still bump updated_at."""
        t = Task(title="T")
        old_ts = t.updated_at
        time.sleep(0.01)
        t.update()
        assert t.updated_at >= old_ts

    def test_mark_done(self):
        t = Task(title="T")
        t.mark_done()
        assert t.status is Status.DONE

    def test_mark_cancelled(self):
        t = Task(title="T")
        t.mark_cancelled()
        assert t.status is Status.CANCELLED


# ===========================================================================
# Task – is_overdue
# ===========================================================================

class TestTaskIsOverdue:
    def test_no_due_date_not_overdue(self):
        t = Task(title="T")
        assert t.is_overdue is False

    def test_future_due_date_not_overdue(self):
        future = (date.today() + timedelta(days=30)).isoformat()
        t = Task(title="T", due_date=future)
        assert t.is_overdue is False

    def test_past_due_date_overdue(self):
        past = (date.today() - timedelta(days=1)).isoformat()
        t = Task(title="T", due_date=past)
        assert t.is_overdue is True

    def test_done_task_not_overdue(self):
        past = (date.today() - timedelta(days=1)).isoformat()
        t = Task(title="T", due_date=past, status=Status.DONE)
        assert t.is_overdue is False

    def test_cancelled_task_not_overdue(self):
        past = (date.today() - timedelta(days=1)).isoformat()
        t = Task(title="T", due_date=past, status=Status.CANCELLED)
        assert t.is_overdue is False


# ===========================================================================
# Task – serialisation
# ===========================================================================

class TestTaskSerialisation:
    def _make_task(self) -> Task:
        return Task(
            title="Write tests",
            description="Cover all edge cases",
            priority=Priority.HIGH,
            status=Status.IN_PROGRESS,
            due_date="2030-01-01",
            tags=["testing", "quality"],
        )

    def test_to_dict_keys(self):
        d = self._make_task().to_dict()
        for key in ("id", "title", "description", "priority", "status",
                    "due_date", "tags", "created_at", "updated_at"):
            assert key in d, f"Missing key: {key}"

    def test_to_dict_enum_values_are_strings(self):
        d = self._make_task().to_dict()
        assert d["priority"] == "high"
        assert d["status"] == "in_progress"

    def test_to_dict_tags_list(self):
        d = self._make_task().to_dict()
        assert isinstance(d["tags"], list)
        assert "testing" in d["tags"]

    def test_from_dict_roundtrip(self):
        original = self._make_task()
        restored = Task.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.title == original.title
        assert restored.description == original.description
        assert restored.priority is original.priority
        assert restored.status is original.status
        assert restored.due_date == original.due_date
        assert restored.tags == original.tags

    def test_from_dict_missing_title_raises(self):
        with pytest.raises(KeyError, match="title"):
            Task.from_dict({"description": "no title here"})

    def test_from_dict_defaults(self):
        t = Task.from_dict({"title": "Minimal"})
        assert t.priority is Priority.MEDIUM
        assert t.status is Status.TODO
        assert t.due_date is None
        assert t.tags == []

    def test_to_json_is_valid_json(self):
        j = self._make_task().to_json()
        parsed = json.loads(j)
        assert parsed["title"] == "Write tests"

    def test_from_json_roundtrip(self):
        original = self._make_task()
        restored = Task.from_json(original.to_json())
        assert restored.title == original.title
        assert restored.priority is original.priority
        assert restored.status is original.status


# ===========================================================================
# TaskList – CRUD
# ===========================================================================

class TestTaskListCRUD:
    def _make_list(self) -> TaskList:
        tl = TaskList()
        tl.add(Task(title="Task A", priority=Priority.LOW, status=Status.TODO))
        tl.add(Task(title="Task B", priority=Priority.HIGH, status=Status.IN_PROGRESS))
        tl.add(Task(title="Task C", priority=Priority.CRITICAL, status=Status.DONE))
        return tl

    def test_add_and_len(self):
        tl = self._make_list()
        assert len(tl) == 3

    def test_add_duplicate_id_raises(self):
        tl = TaskList()
        t = Task(title="T")
        tl.add(t)
        with pytest.raises(ValueError, match="already exists"):
            tl.add(t)

    def test_get_by_id_found(self):
        tl = self._make_list()
        task = tl.all()[0]
        found = tl.get_by_id(task.id)
        assert found is task

    def test_get_by_id_not_found(self):
        tl = self._make_list()
        assert tl.get_by_id("nonexistent-id") is None

    def test_get_by_short_id(self):
        tl = self._make_list()
        task = tl.all()[1]
        found = tl.get_by_short_id(task.id[:8])
        assert found is task

    def test_get_by_short_id_not_found(self):
        tl = self._make_list()
        assert tl.get_by_short_id("00000000") is None

    def test_remove_existing(self):
        tl = self._make_list()
        task = tl.all()[0]
        removed = tl.remove(task.id)
        assert removed is task
        assert len(tl) == 2
        assert tl.get_by_id(task.id) is None

    def test_remove_nonexistent_raises(self):
        tl = self._make_list()
        with pytest.raises(KeyError):
            tl.remove("no-such-id")

    def test_update_task(self):
        tl = self._make_list()
        task = tl.all()[0]
        tl.update(task.id, title="Updated Title", priority="critical")
        assert task.title == "Updated Title"
        assert task.priority is Priority.CRITICAL

    def test_update_nonexistent_raises(self):
        tl = self._make_list()
        with pytest.raises(KeyError):
            tl.update("no-such-id", title="X")

    def test_iter(self):
        tl = self._make_list()
        titles = [t.title for t in tl]
        assert "Task A" in titles
        assert "Task B" in titles
        assert "Task C" in titles

    def test_all_returns_copy(self):
        tl = self._make_list()
        copy = tl.all()
        copy.clear()
        assert len(tl) == 3  # original unaffected

    def test_repr(self):
        tl = self._make_list()
        assert "3" in repr(tl)


# ===========================================================================
# TaskList – filtering
# ===========================================================================

class TestTaskListFiltering:
    def _make_list(self) -> TaskList:
        past = (date.today() - timedelta(days=1)).isoformat()
        future = (date.today() + timedelta(days=30)).isoformat()
        tl = TaskList()
        tl.add(Task(title="Alpha", priority=Priority.LOW,
                    status=Status.TODO, tags=["work"], due_date=past))
        tl.add(Task(title="Beta", priority=Priority.HIGH,
                    status=Status.IN_PROGRESS, tags=["work", "urgent"]))
        tl.add(Task(title="Gamma", priority=Priority.HIGH,
                    status=Status.DONE, tags=["personal"], due_date=future))
        tl.add(Task(title="Delta", priority=Priority.CRITICAL,
                    status=Status.TODO, tags=["urgent"]))
        return tl

    def test_filter_by_status_todo(self):
        result = self._make_list().filter_by_status(Status.TODO)
        assert len(result) == 2
        titles = {t.title for t in result}
        assert titles == {"Alpha", "Delta"}

    def test_filter_by_status_string(self):
        result = self._make_list().filter_by_status("done")
        assert len(result) == 1
        assert result.all()[0].title == "Gamma"

    def test_filter_by_status_invalid_raises(self):
        with pytest.raises(ValueError):
            self._make_list().filter_by_status("unknown")

    def test_filter_by_priority_high(self):
        result = self._make_list().filter_by_priority(Priority.HIGH)
        assert len(result) == 2
        titles = {t.title for t in result}
        assert titles == {"Beta", "Gamma"}

    def test_filter_by_priority_string(self):
        result = self._make_list().filter_by_priority("critical")
        assert len(result) == 1
        assert result.all()[0].title == "Delta"

    def test_filter_by_tag_work(self):
        result = self._make_list().filter_by_tag("work")
        assert len(result) == 2

    def test_filter_by_tag_urgent(self):
        result = self._make_list().filter_by_tag("urgent")
        assert len(result) == 2

    def test_filter_by_tag_no_match(self):
        result = self._make_list().filter_by_tag("nonexistent")
        assert len(result) == 0

    def test_filter_overdue(self):
        result = self._make_list().filter_overdue()
        # Only "Alpha" has a past due date and is not terminal
        assert len(result) == 1
        assert result.all()[0].title == "Alpha"

    def test_search_by_title(self):
        result = self._make_list().search("alpha")
        assert len(result) == 1
        assert result.all()[0].title == "Alpha"

    def test_search_case_insensitive(self):
        result = self._make_list().search("BETA")
        assert len(result) == 1

    def test_search_no_match(self):
        result = self._make_list().search("zzznomatch")
        assert len(result) == 0

    def test_filter_returns_new_tasklist(self):
        tl = self._make_list()
        result = tl.filter_by_status(Status.TODO)
        assert result is not tl


# ===========================================================================
# TaskList – sorting
# ===========================================================================

class TestTaskListSorting:
    def _make_list(self) -> TaskList:
        tl = TaskList()
        tl.add(Task(title="Low", priority=Priority.LOW,
                    due_date="2030-03-01"))
        tl.add(Task(title="Critical", priority=Priority.CRITICAL,
                    due_date="2030-01-01"))
        tl.add(Task(title="High", priority=Priority.HIGH,
                    due_date="2030-02-01"))
        tl.add(Task(title="NoDue", priority=Priority.MEDIUM))
        return tl

    def test_sorted_by_priority_descending(self):
        result = self._make_list().sorted_by_priority(descending=True)
        titles = [t.title for t in result]
        assert titles[0] == "Critical"
        assert titles[-1] == "Low"

    def test_sorted_by_priority_ascending(self):
        result = self._make_list().sorted_by_priority(descending=False)
        titles = [t.title for t in result]
        assert titles[0] == "Low"
        assert titles[-1] == "Critical"

    def test_sorted_by_due_date_ascending(self):
        result = self._make_list().sorted_by_due_date(ascending=True)
        titles = [t.title for t in result]
        # NoDue should be last
        assert titles[-1] == "NoDue"
        assert titles[0] == "Critical"  # earliest date

    def test_sorted_by_due_date_descending(self):
        result = self._make_list().sorted_by_due_date(ascending=False)
        titles = [t.title for t in result]
        # NoDue should be last when descending (date.min sentinel)
        assert titles[-1] == "NoDue"
        assert titles[0] == "Low"  # latest date

    def test_sorted_by_created_at(self):
        result = self._make_list().sorted_by_created_at(ascending=True)
        # Just verify it returns a TaskList of the same length
        assert len(result) == 4

    def test_sort_returns_new_tasklist(self):
        tl = self._make_list()
        result = tl.sorted_by_priority()
        assert result is not tl


# ===========================================================================
# TaskList – serialisation
# ===========================================================================

class TestTaskListSerialisation:
    def _make_list(self) -> TaskList:
        tl = TaskList()
        tl.add(Task(title="T1", priority=Priority.LOW))
        tl.add(Task(title="T2", priority=Priority.HIGH, status=Status.DONE))
        return tl

    def test_to_dict_has_tasks_key(self):
        d = self._make_list().to_dict()
        assert "tasks" in d
        assert len(d["tasks"]) == 2

    def test_from_dict_roundtrip(self):
        original = self._make_list()
        restored = TaskList.from_dict(original.to_dict())
        assert len(restored) == 2
        titles = {t.title for t in restored}
        assert titles == {"T1", "T2"}

    def test_from_dict_empty(self):
        tl = TaskList.from_dict({})
        assert len(tl) == 0

    def test_to_json_valid(self):
        j = self._make_list().to_json()
        parsed = json.loads(j)
        assert len(parsed["tasks"]) == 2

    def test_from_json_roundtrip(self):
        original = self._make_list()
        restored = TaskList.from_json(original.to_json())
        assert len(restored) == 2
        ids_original = {t.id for t in original}
        ids_restored = {t.id for t in restored}
        assert ids_original == ids_restored

    def test_empty_tasklist_serialises(self):
        tl = TaskList()
        j = tl.to_json()
        restored = TaskList.from_json(j)
        assert len(restored) == 0


# ===========================================================================
# TaskList – summary
# ===========================================================================

class TestTaskListSummary:
    def test_summary_counts(self):
        tl = TaskList()
        tl.add(Task(title="A", status=Status.TODO))
        tl.add(Task(title="B", status=Status.TODO))
        tl.add(Task(title="C", status=Status.IN_PROGRESS))
        tl.add(Task(title="D", status=Status.DONE))
        tl.add(Task(title="E", status=Status.CANCELLED))

        s = tl.summary()
        assert s["todo"] == 2
        assert s["in_progress"] == 1
        assert s["done"] == 1
        assert s["cancelled"] == 1
        assert s["total"] == 5

    def test_summary_empty(self):
        s = TaskList().summary()
        assert s["total"] == 0
        assert s["todo"] == 0


# ===========================================================================
# Integration – end-to-end workflow
# ===========================================================================

class TestEndToEndWorkflow:
    """Simulate a realistic CLI session: create, update, filter, serialise."""

    def test_full_workflow(self):
        # 1. Create a fresh task list
        tl = TaskList()

        # 2. Add several tasks
        t1 = Task(title="Design schema", priority=Priority.HIGH,
                  tags=["backend"], due_date="2030-06-01")
        t2 = Task(title="Write unit tests", priority=Priority.MEDIUM,
                  tags=["testing"])
        t3 = Task(title="Deploy to staging", priority=Priority.CRITICAL,
                  tags=["devops", "backend"],
                  due_date=(date.today() - timedelta(days=2)).isoformat())
        tl.add(t1)
        tl.add(t2)
        tl.add(t3)
        assert len(tl) == 3

        # 3. Mark t1 as in-progress
        tl.update(t1.id, status="in_progress")
        assert t1.status is Status.IN_PROGRESS

        # 4. Complete t2
        t2.mark_done()
        assert t2.status is Status.DONE

        # 5. Filter: only active tasks (TODO)
        active = tl.filter_by_status(Status.TODO)
        assert len(active) == 1
        assert active.all()[0].title == "Deploy to staging"

        # 6. Overdue check
        overdue = tl.filter_overdue()
        assert len(overdue) == 1
        assert overdue.all()[0].title == "Deploy to staging"

        # 7. Filter by tag
        backend_tasks = tl.filter_by_tag("backend")
        assert len(backend_tasks) == 2

        # 8. Sort by priority (critical first)
        sorted_tl = tl.sorted_by_priority(descending=True)
        assert sorted_tl.all()[0].title == "Deploy to staging"

        # 9. Serialise and restore
        json_str = tl.to_json()
        restored = TaskList.from_json(json_str)
        assert len(restored) == 3
        restored_ids = {t.id for t in restored}
        original_ids = {t.id for t in tl}
        assert restored_ids == original_ids

        # 10. Summary
        s = tl.summary()
        assert s["total"] == 3
        assert s["in_progress"] == 1
        assert s["done"] == 1
        assert s["todo"] == 1

        # 11. Remove a task
        tl.remove(t3.id)
        assert len(tl) == 2
        assert tl.get_by_id(t3.id) is None
