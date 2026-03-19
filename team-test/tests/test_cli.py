"""Tests for team-test/cli.py.

Strategy: invoke ``cli.main(argv)`` directly (not via subprocess) so we can
capture stdout/stderr with capsys and inspect return codes.  A fresh temporary
Storage file is passed via ``--data`` for every test to ensure isolation.

Covers:
- ``task add``: happy path, all options, validation errors (empty title,
  bad priority, bad due date, nonexistent project)
- ``task list``: empty store, multiple tasks, --status filter, --priority
  filter, --overdue filter, --project filter, --verbose flag
- ``task show``: happy path, not-found error
- ``task update``: title, description, status, priority, due date, clear-due,
  project, unlink-project, no-fields error, not-found error
- ``task complete``: happy path, not-found error
- ``task start``: happy path, not-found error
- ``task reopen``: happy path, not-found error
- ``task delete``: happy path, not-found error
- ``task search``: keyword match, no match, --project scoping, --verbose flag
- ``task purge``: deletes DONE tasks, nothing to purge, --project scoping
- ``task complete-all``: marks pending tasks done, nothing pending
- ``task overdue``: lists overdue tasks, no overdue tasks
- ``project add``: happy path, validation error
- ``project list``: empty store, multiple projects, --verbose flag
- ``project show``: happy path, not-found error
- ``project rename``: happy path, validation error, not-found error
- ``project delete``: happy path, --cascade flag, not-found error
- ``project summary``: happy path, not-found error
- ``project summaries``: multiple projects, empty store
- Integration: full end-to-end workflow
- ``--help`` output for top-level and sub-commands
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

from cli import main  # noqa: E402
from storage import Storage  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _past(days: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")


def _future(days: int = 5) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).strftime("%Y-%m-%d")


def _run(tmp_path: Path, *args: str) -> int:
    """Run ``main()`` with ``--data <tmp_file> <args>`` and return exit code."""
    data_file = str(tmp_path / "data.json")
    return main(["--data", data_file, *args])


def _run_capture(tmp_path: Path, capsys, *args: str):
    """Run ``main()`` and return (exit_code, stdout, stderr).

    Drains any previously buffered capsys output before running so that
    callers always get only the output from this single invocation.
    """
    capsys.readouterr()  # drain any prior output
    code = _run(tmp_path, *args)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _fresh_store(tmp_path: Path) -> Storage:
    """Return a new Storage instance that re-reads the data file from disk."""
    return Storage(str(tmp_path / "data.json"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path: Path) -> Storage:
    """Return a fresh Storage backed by a temp file."""
    return Storage(str(tmp_path / "data.json"))


@pytest.fixture()
def data_file(tmp_path: Path) -> str:
    """Return the path to the temp data file."""
    return str(tmp_path / "data.json")


# ===========================================================================
# --help
# ===========================================================================

class TestHelp:
    def test_top_level_help(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "taskman" in out
        assert "task" in out
        assert "project" in out

    def test_task_help(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["task", "--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "add" in out
        assert "list" in out
        assert "complete" in out

    def test_project_help(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["project", "--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "add" in out
        assert "list" in out
        assert "summary" in out


# ===========================================================================
# task add
# ===========================================================================

class TestTaskAdd:
    def test_happy_path_returns_zero(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "task", "add", "Write docs")
        assert code == 0
        assert err == ""

    def test_output_contains_task_id(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "task", "add", "Write docs")
        assert code == 0
        assert "Task created:" in out

    def test_output_contains_title(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "task", "add", "My Task Title")
        assert code == 0
        assert "My Task Title" in out

    def test_with_priority(self, tmp_path, capsys):
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "add", "High prio", "--priority", "high"
        )
        assert code == 0
        assert "high" in out

    def test_with_due_date(self, tmp_path, capsys):
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "add", "Due task", "--due", "2030-06-15"
        )
        assert code == 0
        assert "2030-06-15" in out

    def test_with_description(self, tmp_path, capsys):
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "add", "Described", "--description", "A long desc"
        )
        assert code == 0
        assert "A long desc" in out

    def test_with_project(self, tmp_path, capsys, store):
        project = store.create_project("My Project")
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "add", "Project task", "--project", project.id
        )
        assert code == 0
        assert project.id in out

    def test_empty_title_returns_one(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "task", "add", "")
        assert code == 1
        assert "Error:" in err

    def test_invalid_priority_rejected_by_argparse(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["--data", str(tmp_path / "data.json"), "task", "add", "T", "--priority", "urgent"])
        assert exc_info.value.code != 0

    def test_invalid_due_date_returns_one(self, tmp_path, capsys):
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "add", "Bad date", "--due", "not-a-date"
        )
        assert code == 1
        assert "Error:" in err

    def test_nonexistent_project_returns_one(self, tmp_path, capsys):
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "add", "Orphan", "--project", "nonexistent-id"
        )
        assert code == 1
        assert "Error:" in err


# ===========================================================================
# task list
# ===========================================================================

class TestTaskList:
    def test_empty_store(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "task", "list")
        assert code == 0
        assert "No tasks found" in out

    def test_lists_tasks(self, tmp_path, capsys, store):
        store.create_task("Task Alpha")
        store.create_task("Task Beta")
        code, out, err = _run_capture(tmp_path, capsys, "task", "list")
        assert code == 0
        assert "Task Alpha" in out
        assert "Task Beta" in out

    def test_shows_task_count(self, tmp_path, capsys, store):
        store.create_task("Task A")
        store.create_task("Task B")
        store.create_task("Task C")
        code, out, err = _run_capture(tmp_path, capsys, "task", "list")
        assert code == 0
        assert "3" in out

    def test_filter_by_status(self, tmp_path, capsys, store):
        from models import Status
        store.create_task("Todo task")
        store.create_task("Done task", status=Status.DONE)
        code, out, err = _run_capture(tmp_path, capsys, "task", "list", "--status", "todo")
        assert code == 0
        assert "Todo task" in out
        assert "Done task" not in out

    def test_filter_by_priority(self, tmp_path, capsys, store):
        from models import Priority
        store.create_task("Low task", priority=Priority.LOW)
        store.create_task("High task", priority=Priority.HIGH)
        code, out, err = _run_capture(tmp_path, capsys, "task", "list", "--priority", "high")
        assert code == 0
        assert "High task" in out
        assert "Low task" not in out

    def test_filter_overdue(self, tmp_path, capsys, store):
        store.create_task("Overdue task", due_date=_past(3))
        store.create_task("Future task", due_date=_future(3))
        code, out, err = _run_capture(tmp_path, capsys, "task", "list", "--overdue")
        assert code == 0
        assert "Overdue task" in out
        assert "Future task" not in out

    def test_filter_by_project(self, tmp_path, capsys, store):
        p1 = store.create_project("P1")
        p2 = store.create_project("P2")
        store.create_task("P1 task", project_id=p1.id)
        store.create_task("P2 task", project_id=p2.id)
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "list", "--project", p1.id
        )
        assert code == 0
        assert "P1 task" in out
        assert "P2 task" not in out

    def test_verbose_shows_id(self, tmp_path, capsys, store):
        task = store.create_task("Verbose task")
        code, out, err = _run_capture(tmp_path, capsys, "task", "list", "--verbose")
        assert code == 0
        assert task.id in out

    def test_verbose_shows_description(self, tmp_path, capsys, store):
        store.create_task("Described task", description="My description here")
        code, out, err = _run_capture(tmp_path, capsys, "task", "list", "--verbose")
        assert code == 0
        assert "My description here" in out


# ===========================================================================
# task show
# ===========================================================================

class TestTaskShow:
    def test_shows_task_details(self, tmp_path, capsys, store):
        task = store.create_task("Show me", description="Detail here")
        code, out, err = _run_capture(tmp_path, capsys, "task", "show", task.id)
        assert code == 0
        assert "Show me" in out
        assert task.id in out
        assert "Detail here" in out

    def test_not_found_returns_one(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "task", "show", "nonexistent-id")
        assert code == 1
        assert "Error:" in err

    def test_not_found_error_message_content(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "task", "show", "bad-id")
        assert "bad-id" in err or "not found" in err.lower()


# ===========================================================================
# task update
# ===========================================================================

class TestTaskUpdate:
    def test_update_title(self, tmp_path, capsys, store):
        task = store.create_task("Old title")
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "update", task.id, "--title", "New title"
        )
        assert code == 0
        assert "New title" in out

    def test_update_description(self, tmp_path, capsys, store):
        task = store.create_task("Task")
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "update", task.id, "--description", "Updated desc"
        )
        assert code == 0
        assert "Updated desc" in out

    def test_update_status(self, tmp_path, capsys, store):
        task = store.create_task("Status task")
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "update", task.id, "--status", "done"
        )
        assert code == 0
        assert "DONE" in out

    def test_update_priority(self, tmp_path, capsys, store):
        task = store.create_task("Priority task")
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "update", task.id, "--priority", "critical"
        )
        assert code == 0
        assert "critical" in out

    def test_update_due_date(self, tmp_path, capsys, store):
        task = store.create_task("Due task")
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "update", task.id, "--due", "2030-01-01"
        )
        assert code == 0
        assert "2030-01-01" in out

    def test_clear_due_date(self, tmp_path, capsys, store):
        task = store.create_task("Due task", due_date="2030-01-01")
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "update", task.id, "--clear-due"
        )
        assert code == 0
        # After clearing, due date should not appear in verbose output
        assert "2030-01-01" not in out

    def test_no_fields_returns_one(self, tmp_path, capsys, store):
        task = store.create_task("No update")
        code, out, err = _run_capture(tmp_path, capsys, "task", "update", task.id)
        assert code == 1
        assert "Error:" in err

    def test_not_found_returns_one(self, tmp_path, capsys):
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "update", "nonexistent-id", "--title", "X"
        )
        assert code == 1
        assert "Error:" in err

    def test_update_success_message(self, tmp_path, capsys, store):
        task = store.create_task("Update me")
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "update", task.id, "--title", "Updated"
        )
        assert code == 0
        assert "Task updated:" in out


# ===========================================================================
# task complete
# ===========================================================================

class TestTaskComplete:
    def test_marks_task_done(self, tmp_path, capsys, store):
        task = store.create_task("Complete me")
        code, out, err = _run_capture(tmp_path, capsys, "task", "complete", task.id)
        assert code == 0
        assert "DONE" in out
        assert "Complete me" in out

    def test_not_found_returns_one(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "task", "complete", "bad-id")
        assert code == 1
        assert "Error:" in err

    def test_success_message_contains_title(self, tmp_path, capsys, store):
        task = store.create_task("My Task")
        code, out, err = _run_capture(tmp_path, capsys, "task", "complete", task.id)
        assert "My Task" in out


# ===========================================================================
# task start
# ===========================================================================

class TestTaskStart:
    def test_marks_task_in_progress(self, tmp_path, capsys, store):
        task = store.create_task("Start me")
        code, out, err = _run_capture(tmp_path, capsys, "task", "start", task.id)
        assert code == 0
        assert "IN_PROGRESS" in out
        assert "Start me" in out

    def test_not_found_returns_one(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "task", "start", "bad-id")
        assert code == 1
        assert "Error:" in err


# ===========================================================================
# task reopen
# ===========================================================================

class TestTaskReopen:
    def test_resets_task_to_todo(self, tmp_path, capsys, store):
        from models import Status
        task = store.create_task("Reopen me", status=Status.DONE)
        code, out, err = _run_capture(tmp_path, capsys, "task", "reopen", task.id)
        assert code == 0
        assert "TODO" in out
        assert "Reopen me" in out

    def test_not_found_returns_one(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "task", "reopen", "bad-id")
        assert code == 1
        assert "Error:" in err


# ===========================================================================
# task delete
# ===========================================================================

class TestTaskDelete:
    def test_deletes_task(self, tmp_path, capsys, store):
        task = store.create_task("Delete me")
        code, out, err = _run_capture(tmp_path, capsys, "task", "delete", task.id)
        assert code == 0
        assert "Delete me" in out
        assert "deleted" in out.lower()

    def test_not_found_returns_one(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "task", "delete", "bad-id")
        assert code == 1
        assert "Error:" in err

    def test_task_no_longer_exists_after_delete(self, tmp_path, capsys, store):
        task = store.create_task("Gone task")
        _run(tmp_path, "task", "delete", task.id)
        # Trying to show it should now fail
        code, out, err = _run_capture(tmp_path, capsys, "task", "show", task.id)
        assert code == 1


# ===========================================================================
# task search
# ===========================================================================

class TestTaskSearch:
    def test_finds_by_title_keyword(self, tmp_path, capsys, store):
        store.create_task("Fix the login bug")
        store.create_task("Write unit tests")
        code, out, err = _run_capture(tmp_path, capsys, "task", "search", "login")
        assert code == 0
        assert "Fix the login bug" in out
        assert "Write unit tests" not in out

    def test_finds_by_description_keyword(self, tmp_path, capsys, store):
        store.create_task("Task A", description="Contains the secret word")
        store.create_task("Task B", description="Nothing special")
        code, out, err = _run_capture(tmp_path, capsys, "task", "search", "secret")
        assert code == 0
        assert "Task A" in out
        assert "Task B" not in out

    def test_case_insensitive(self, tmp_path, capsys, store):
        store.create_task("UPPERCASE TASK")
        code, out, err = _run_capture(tmp_path, capsys, "task", "search", "uppercase")
        assert code == 0
        assert "UPPERCASE TASK" in out

    def test_no_match_message(self, tmp_path, capsys, store):
        store.create_task("Unrelated task")
        code, out, err = _run_capture(tmp_path, capsys, "task", "search", "zzznomatch")
        assert code == 0
        assert "zzznomatch" in out  # message includes the keyword

    def test_project_scoping(self, tmp_path, capsys, store):
        p1 = store.create_project("P1")
        p2 = store.create_project("P2")
        store.create_task("Alpha task", project_id=p1.id)
        store.create_task("Alpha task in P2", project_id=p2.id)
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "search", "alpha", "--project", p1.id
        )
        assert code == 0
        assert "Alpha task" in out
        # The P2 task should not appear
        assert "Alpha task in P2" not in out

    def test_verbose_shows_id(self, tmp_path, capsys, store):
        task = store.create_task("Verbose search task")
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "search", "verbose", "--verbose"
        )
        assert code == 0
        assert task.id in out

    def test_shows_match_count(self, tmp_path, capsys, store):
        store.create_task("Match one")
        store.create_task("Match two")
        store.create_task("Unrelated task")
        code, out, err = _run_capture(tmp_path, capsys, "task", "search", "match")
        assert code == 0
        assert "2" in out


# ===========================================================================
# task purge
# ===========================================================================

class TestTaskPurge:
    def test_purges_done_tasks(self, tmp_path, capsys, store):
        from models import Status
        store.create_task("Done task 1", status=Status.DONE)
        store.create_task("Done task 2", status=Status.DONE)
        store.create_task("Active task")
        code, out, err = _run_capture(tmp_path, capsys, "task", "purge")
        assert code == 0
        assert "2" in out
        assert "Purged" in out

    def test_nothing_to_purge(self, tmp_path, capsys, store):
        store.create_task("Active task")
        code, out, err = _run_capture(tmp_path, capsys, "task", "purge")
        assert code == 0
        assert "No completed tasks" in out

    def test_project_scoped_purge(self, tmp_path, capsys, store):
        from models import Status
        p1 = store.create_project("P1")
        p2 = store.create_project("P2")
        store.create_task("P1 done", status=Status.DONE, project_id=p1.id)
        store.create_task("P2 done", status=Status.DONE, project_id=p2.id)
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "purge", "--project", p1.id
        )
        assert code == 0
        assert "1" in out
        # P2 task should still exist — reload from disk
        remaining = _fresh_store(tmp_path).list_tasks(project_id=p2.id)
        assert len(remaining) == 1

    def test_nonexistent_project_returns_one(self, tmp_path, capsys):
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "purge", "--project", "nonexistent"
        )
        assert code == 1
        assert "Error:" in err


# ===========================================================================
# task complete-all
# ===========================================================================

class TestTaskCompleteAll:
    def test_marks_all_pending_done(self, tmp_path, capsys, store):
        store.create_task("Task A")
        store.create_task("Task B")
        code, out, err = _run_capture(tmp_path, capsys, "task", "complete-all")
        assert code == 0
        assert "2" in out

    def test_nothing_pending(self, tmp_path, capsys, store):
        from models import Status
        store.create_task("Already done", status=Status.DONE)
        code, out, err = _run_capture(tmp_path, capsys, "task", "complete-all")
        assert code == 0
        assert "No pending tasks" in out

    def test_project_scoped(self, tmp_path, capsys, store):
        p1 = store.create_project("P1")
        store.create_task("P1 task", project_id=p1.id)
        store.create_task("Standalone task")
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "complete-all", "--project", p1.id
        )
        assert code == 0
        assert "1" in out
        # Standalone task should still be TODO — reload from disk
        from models import Status
        all_tasks = _fresh_store(tmp_path).list_tasks()
        todo_standalone = [
            t for t in all_tasks
            if t.status == Status.TODO and t.project_id is None
        ]
        assert len(todo_standalone) == 1


# ===========================================================================
# task overdue
# ===========================================================================

class TestTaskOverdue:
    def test_lists_overdue_tasks(self, tmp_path, capsys, store):
        store.create_task("Overdue task", due_date=_past(3))
        store.create_task("Future task", due_date=_future(3))
        code, out, err = _run_capture(tmp_path, capsys, "task", "overdue")
        assert code == 0
        assert "Overdue task" in out
        assert "Future task" not in out

    def test_no_overdue_tasks(self, tmp_path, capsys, store):
        store.create_task("Future task", due_date=_future(3))
        code, out, err = _run_capture(tmp_path, capsys, "task", "overdue")
        assert code == 0
        assert "No overdue tasks" in out

    def test_overdue_report_shows_due_date(self, tmp_path, capsys, store):
        past_date = _past(2)
        store.create_task("Overdue", due_date=past_date)
        code, out, err = _run_capture(tmp_path, capsys, "task", "overdue")
        assert code == 0
        assert past_date in out

    def test_overdue_report_shows_count(self, tmp_path, capsys, store):
        store.create_task("Overdue 1", due_date=_past(1))
        store.create_task("Overdue 2", due_date=_past(2))
        code, out, err = _run_capture(tmp_path, capsys, "task", "overdue")
        assert code == 0
        assert "2" in out


# ===========================================================================
# project add
# ===========================================================================

class TestProjectAdd:
    def test_happy_path(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "project", "add", "Sprint 1")
        assert code == 0
        assert "Sprint 1" in out
        assert "Project created:" in out

    def test_with_description(self, tmp_path, capsys):
        code, out, err = _run_capture(
            tmp_path, capsys, "project", "add", "Sprint 2", "--description", "Second sprint"
        )
        assert code == 0
        assert "Sprint 2" in out
        assert "Second sprint" in out

    def test_empty_name_returns_one(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "project", "add", "")
        assert code == 1
        assert "Error:" in err

    def test_output_contains_project_id(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "project", "add", "My Project")
        assert code == 0
        assert "id:" in out


# ===========================================================================
# project list
# ===========================================================================

class TestProjectList:
    def test_empty_store(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "project", "list")
        assert code == 0
        assert "No projects found" in out

    def test_lists_projects(self, tmp_path, capsys, store):
        store.create_project("Alpha")
        store.create_project("Beta")
        code, out, err = _run_capture(tmp_path, capsys, "project", "list")
        assert code == 0
        assert "Alpha" in out
        assert "Beta" in out

    def test_shows_count(self, tmp_path, capsys, store):
        store.create_project("P1")
        store.create_project("P2")
        store.create_project("P3")
        code, out, err = _run_capture(tmp_path, capsys, "project", "list")
        assert code == 0
        assert "3" in out

    def test_verbose_shows_description(self, tmp_path, capsys, store):
        store.create_project("Verbose project", description="My description")
        code, out, err = _run_capture(tmp_path, capsys, "project", "list", "--verbose")
        assert code == 0
        assert "My description" in out

    def test_verbose_shows_created_at(self, tmp_path, capsys, store):
        store.create_project("Timestamped project")
        code, out, err = _run_capture(tmp_path, capsys, "project", "list", "--verbose")
        assert code == 0
        assert "Created:" in out


# ===========================================================================
# project show
# ===========================================================================

class TestProjectShow:
    def test_shows_project_details(self, tmp_path, capsys, store):
        project = store.create_project("Show me", description="Detail here")
        code, out, err = _run_capture(tmp_path, capsys, "project", "show", project.id)
        assert code == 0
        assert "Show me" in out
        assert project.id in out
        assert "Detail here" in out

    def test_not_found_returns_one(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "project", "show", "bad-id")
        assert code == 1
        assert "Error:" in err

    def test_not_found_error_message(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "project", "show", "bad-id")
        assert "bad-id" in err or "not found" in err.lower()


# ===========================================================================
# project rename
# ===========================================================================

class TestProjectRename:
    def test_renames_project(self, tmp_path, capsys, store):
        project = store.create_project("Old Name")
        code, out, err = _run_capture(
            tmp_path, capsys, "project", "rename", project.id, "New Name"
        )
        assert code == 0
        assert "New Name" in out

    def test_not_found_returns_one(self, tmp_path, capsys):
        code, out, err = _run_capture(
            tmp_path, capsys, "project", "rename", "bad-id", "New Name"
        )
        assert code == 1
        assert "Error:" in err

    def test_empty_name_returns_one(self, tmp_path, capsys, store):
        project = store.create_project("Valid Name")
        code, out, err = _run_capture(
            tmp_path, capsys, "project", "rename", project.id, ""
        )
        assert code == 1
        assert "Error:" in err

    def test_success_message(self, tmp_path, capsys, store):
        project = store.create_project("Old")
        code, out, err = _run_capture(
            tmp_path, capsys, "project", "rename", project.id, "New"
        )
        assert "renamed" in out.lower()
        assert "New" in out


# ===========================================================================
# project delete
# ===========================================================================

class TestProjectDelete:
    def test_deletes_project(self, tmp_path, capsys, store):
        project = store.create_project("Delete me")
        code, out, err = _run_capture(
            tmp_path, capsys, "project", "delete", project.id
        )
        assert code == 0
        assert "Delete me" in out
        assert "deleted" in out.lower()

    def test_not_found_returns_one(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "project", "delete", "bad-id")
        assert code == 1
        assert "Error:" in err

    def test_cascade_deletes_tasks(self, tmp_path, capsys, store):
        project = store.create_project("Cascade project")
        store.create_task("Task in project", project_id=project.id)
        code, out, err = _run_capture(
            tmp_path, capsys, "project", "delete", project.id, "--cascade"
        )
        assert code == 0
        assert "cascade" in out.lower() or "tasks" in out.lower()
        # Reload from disk to verify tasks are gone
        remaining = _fresh_store(tmp_path).list_tasks()
        assert len(remaining) == 0

    def test_without_cascade_unlinks_tasks(self, tmp_path, capsys, store):
        project = store.create_project("Unlink project")
        store.create_task("Linked task", project_id=project.id)
        code, out, err = _run_capture(
            tmp_path, capsys, "project", "delete", project.id
        )
        assert code == 0
        # Reload from disk to verify task still exists but is unlinked
        remaining = _fresh_store(tmp_path).list_tasks()
        assert len(remaining) == 1
        assert remaining[0].project_id is None


# ===========================================================================
# project summary
# ===========================================================================

class TestProjectSummary:
    def test_shows_summary(self, tmp_path, capsys, store):
        from models import Status
        project = store.create_project("My Project")
        store.create_task("Task 1", project_id=project.id)
        store.create_task("Task 2", project_id=project.id, status=Status.DONE)
        code, out, err = _run_capture(
            tmp_path, capsys, "project", "summary", project.id
        )
        assert code == 0
        assert "My Project" in out
        assert "Total tasks:" in out
        assert "2" in out

    def test_shows_completion_percentage(self, tmp_path, capsys, store):
        from models import Status
        project = store.create_project("Pct Project")
        store.create_task("Done task", project_id=project.id, status=Status.DONE)
        store.create_task("Todo task", project_id=project.id)
        code, out, err = _run_capture(
            tmp_path, capsys, "project", "summary", project.id
        )
        assert code == 0
        assert "50.0%" in out

    def test_not_found_returns_one(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "project", "summary", "bad-id")
        assert code == 1
        assert "Error:" in err

    def test_shows_done_count(self, tmp_path, capsys, store):
        from models import Status
        project = store.create_project("Count Project")
        store.create_task("Done 1", project_id=project.id, status=Status.DONE)
        store.create_task("Done 2", project_id=project.id, status=Status.DONE)
        store.create_task("Todo", project_id=project.id)
        code, out, err = _run_capture(
            tmp_path, capsys, "project", "summary", project.id
        )
        assert code == 0
        assert "Done:" in out

    def test_shows_overdue_when_present(self, tmp_path, capsys, store):
        project = store.create_project("Overdue Project")
        store.create_task("Overdue task", project_id=project.id, due_date=_past(2))
        code, out, err = _run_capture(
            tmp_path, capsys, "project", "summary", project.id
        )
        assert code == 0
        assert "Overdue:" in out


# ===========================================================================
# project summaries
# ===========================================================================

class TestProjectSummaries:
    def test_empty_store(self, tmp_path, capsys):
        code, out, err = _run_capture(tmp_path, capsys, "project", "summaries")
        assert code == 0
        assert "No projects found" in out

    def test_shows_all_projects(self, tmp_path, capsys, store):
        store.create_project("Alpha")
        store.create_project("Beta")
        code, out, err = _run_capture(tmp_path, capsys, "project", "summaries")
        assert code == 0
        assert "Alpha" in out
        assert "Beta" in out

    def test_shows_count(self, tmp_path, capsys, store):
        store.create_project("P1")
        store.create_project("P2")
        code, out, err = _run_capture(tmp_path, capsys, "project", "summaries")
        assert code == 0
        assert "2" in out

    def test_shows_completion_pct(self, tmp_path, capsys, store):
        project = store.create_project("Pct Project")
        from models import Status
        store.create_task("Done", project_id=project.id, status=Status.DONE)
        code, out, err = _run_capture(tmp_path, capsys, "project", "summaries")
        assert code == 0
        assert "%" in out


# ===========================================================================
# Integration: full end-to-end workflow
# ===========================================================================

class TestIntegration:
    def test_full_workflow(self, tmp_path, capsys):
        """Create a project, add tasks, update them, complete them, check summary."""
        # 1. Create a project
        code, out, err = _run_capture(
            tmp_path, capsys, "project", "add", "Integration Sprint",
            "--description", "Full workflow test"
        )
        assert code == 0
        assert "Integration Sprint" in out
        # Extract project ID from output — format: "Project created: <id>"
        project_id = None
        for line in out.splitlines():
            if "Project created:" in line:
                project_id = line.split("Project created:")[-1].strip()
                break
        assert project_id is not None, f"Could not extract project_id from:\n{out}"

        # 2. Add tasks to the project
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "add", "Implement feature",
            "--priority", "high", "--project", project_id
        )
        assert code == 0
        task_id = None
        for line in out.splitlines():
            if "Task created:" in line:
                task_id = line.split("Task created:")[-1].strip()
                break
        assert task_id is not None

        code, out, err = _run_capture(
            tmp_path, capsys, "task", "add", "Write tests",
            "--project", project_id
        )
        assert code == 0

        # 3. List tasks in the project
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "list", "--project", project_id
        )
        assert code == 0
        assert "Implement feature" in out
        assert "Write tests" in out

        # 4. Start the first task
        code, out, err = _run_capture(tmp_path, capsys, "task", "start", task_id)
        assert code == 0
        assert "IN_PROGRESS" in out

        # 5. Complete the first task
        code, out, err = _run_capture(tmp_path, capsys, "task", "complete", task_id)
        assert code == 0
        assert "DONE" in out

        # 6. Check project summary
        code, out, err = _run_capture(
            tmp_path, capsys, "project", "summary", project_id
        )
        assert code == 0
        assert "Integration Sprint" in out
        assert "Total tasks:" in out
        assert "2" in out

        # 7. Complete all remaining tasks
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "complete-all", "--project", project_id
        )
        assert code == 0

        # 8. Purge completed tasks
        code, out, err = _run_capture(
            tmp_path, capsys, "task", "purge", "--project", project_id
        )
        assert code == 0
        assert "Purged" in out

        # 9. Delete the project
        code, out, err = _run_capture(
            tmp_path, capsys, "project", "delete", project_id
        )
        assert code == 0
        assert "Integration Sprint" in out

    def test_task_lifecycle(self, tmp_path, capsys, store):
        """Test the full task lifecycle: add → start → complete → reopen → delete."""
        task = store.create_task("Lifecycle task")

        # Start
        code, out, err = _run_capture(tmp_path, capsys, "task", "start", task.id)
        assert code == 0
        assert "IN_PROGRESS" in out

        # Complete
        code, out, err = _run_capture(tmp_path, capsys, "task", "complete", task.id)
        assert code == 0
        assert "DONE" in out

        # Reopen
        code, out, err = _run_capture(tmp_path, capsys, "task", "reopen", task.id)
        assert code == 0
        assert "TODO" in out

        # Delete
        code, out, err = _run_capture(tmp_path, capsys, "task", "delete", task.id)
        assert code == 0
        assert "Lifecycle task" in out

    def test_search_after_add(self, tmp_path, capsys):
        """Add tasks then search for them.

        Note: _run_capture drains capsys before each call, so the search
        output is isolated from the prior task-add output.
        """
        _run(tmp_path, "task", "add", "Searchable task alpha")
        _run(tmp_path, "task", "add", "Another task beta")
        _run(tmp_path, "task", "add", "Third task alpha")

        # _run_capture drains capsys first, so we only see the search output
        code, out, err = _run_capture(tmp_path, capsys, "task", "search", "alpha")
        assert code == 0
        assert "Searchable task alpha" in out
        assert "Third task alpha" in out
        assert "Another task beta" not in out

    def test_overdue_workflow(self, tmp_path, capsys, store):
        """Add overdue tasks and verify they appear in the overdue report."""
        store.create_task("Past due task", due_date=_past(3))
        store.create_task("Future task", due_date=_future(3))

        code, out, err = _run_capture(tmp_path, capsys, "task", "overdue")
        assert code == 0
        assert "Past due task" in out
        assert "Future task" not in out
        assert _past(3) in out
