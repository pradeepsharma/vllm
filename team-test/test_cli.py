"""
Comprehensive tests for team-test/cli.py

Strategy: invoke the CLI via its ``main()`` function (passing argv lists) so
that we exercise the full argument-parsing → service → output pipeline without
spawning a subprocess.  stdout/stderr are captured with ``capsys``.

Coverage:
  - ``add``    : happy path (minimal + all fields), invalid priority/status/due,
                 empty title, tag parsing
  - ``list``   : no tasks, tasks present, filter by status/priority/tag/overdue,
                 sort options (priority/due_date/created_at/title), --desc flag
  - ``show``   : found, not found
  - ``update`` : each field, --clear-due, nothing-to-update guard, not found
  - ``done``   : happy path, not found
  - ``cancel`` : happy path, not found
  - ``start``  : happy path, not found
  - ``delete`` : --yes flag (no prompt), not found
  - ``search`` : match found, no match
  - ``stats``  : empty store, populated store
  - ``backup`` : success, failure (no file)
  - Global     : --file option routes to correct storage file
  - Integration: full CRUD workflow across multiple main() calls
"""

from __future__ import annotations

import importlib.util
import io
import pathlib
import sys
import tempfile
import os
from datetime import date, timedelta
from typing import List, Optional

import pytest

# ---------------------------------------------------------------------------
# Load modules dynamically
# ---------------------------------------------------------------------------

def _load_module(name: str, filepath: pathlib.Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)                          # type: ignore[union-attr]
    return mod


_here     = pathlib.Path(__file__).parent
_models   = _load_module("task_models",   _here / "models.py")
_storage  = _load_module("task_storage",  _here / "storage.py")
_services = _load_module("task_services", _here / "services.py")
_cli      = _load_module("task_cli",      _here / "cli.py")

main         = _cli.main
TaskStorage  = _storage.TaskStorage
TaskService  = _services.TaskService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _future(days: int = 30) -> str:
    return (date.today() + timedelta(days=days)).isoformat()


def _past(days: int = 5) -> str:
    return (date.today() - timedelta(days=days)).isoformat()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tasks_file(tmp_path) -> str:
    """Return a path string for a temporary tasks JSON file."""
    return str(tmp_path / "tasks.json")


@pytest.fixture()
def run(tasks_file):
    """Return a helper that calls main() with --file pre-set."""
    def _run(argv: List[str], *, extra_file: Optional[str] = None) -> tuple:
        """
        Run main() with the given argv (--file is prepended automatically).
        Returns (exit_code, stdout_text, stderr_text).
        """
        file_arg = extra_file if extra_file is not None else tasks_file
        full_argv = ["--file", file_arg] + argv
        # Capture stdout/stderr
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        try:
            code = main(full_argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 1
        finally:
            out = sys.stdout.getvalue()
            err = sys.stderr.getvalue()
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        return code, out, err
    return _run


# ===========================================================================
# add
# ===========================================================================

class TestAdd:
    def test_add_minimal(self, run):
        code, out, err = run(["add", "Buy milk"])
        assert code == 0
        assert "Task created" in out
        assert "Buy milk" in out

    def test_add_all_fields(self, run):
        code, out, err = run([
            "add", "Deploy service",
            "--description", "Push to prod",
            "--priority", "high",
            "--status", "in_progress",
            "--due", _future(10),
            "--tags", "devops,prod",
        ])
        assert code == 0
        assert "Task created" in out
        assert "Deploy service" in out

    def test_add_persists_to_disk(self, run, tasks_file):
        run(["add", "Persisted task"])
        storage = TaskStorage(tasks_file)
        loaded = storage.load()
        assert len(loaded) == 1
        assert loaded.all()[0].title == "Persisted task"

    def test_add_empty_title_fails(self, run):
        code, out, err = run(["add", ""])
        assert code != 0
        assert "Error" in err

    def test_add_whitespace_title_fails(self, run):
        code, out, err = run(["add", "   "])
        assert code != 0
        assert "Error" in err

    def test_add_invalid_priority_fails(self, run):
        # argparse choices validation catches this
        code, out, err = run(["add", "T", "--priority", "urgent"])
        assert code != 0

    def test_add_invalid_status_fails(self, run):
        code, out, err = run(["add", "T", "--status", "pending"])
        assert code != 0

    def test_add_invalid_due_date_fails(self, run):
        code, out, err = run(["add", "T", "--due", "31-12-2030"])
        assert code != 0
        assert "Error" in err

    def test_add_tags_parsed_correctly(self, run, tasks_file):
        run(["add", "Tagged task", "--tags", "alpha,beta,gamma"])
        storage = TaskStorage(tasks_file)
        task = storage.load().all()[0]
        assert "alpha" in task.tags
        assert "beta" in task.tags
        assert "gamma" in task.tags

    def test_add_short_id_shown_in_output(self, run, tasks_file):
        code, out, err = run(["add", "Short ID task"])
        assert code == 0
        storage = TaskStorage(tasks_file)
        task = storage.load().all()[0]
        assert task.short_id in out

    def test_add_multiple_tasks_increments_count(self, run, tasks_file):
        run(["add", "Task 1"])
        run(["add", "Task 2"])
        run(["add", "Task 3"])
        storage = TaskStorage(tasks_file)
        assert len(storage.load()) == 3


# ===========================================================================
# list / ls
# ===========================================================================

class TestList:
    def test_list_empty(self, run):
        code, out, err = run(["list"])
        assert code == 0
        assert "No tasks found" in out

    def test_list_shows_tasks(self, run):
        run(["add", "Alpha"])
        run(["add", "Beta"])
        code, out, err = run(["list"])
        assert code == 0
        assert "Alpha" in out
        assert "Beta" in out

    def test_list_shows_count(self, run):
        run(["add", "T1"])
        run(["add", "T2"])
        code, out, err = run(["list"])
        assert "2 task(s)" in out

    def test_list_filter_by_status(self, run, tasks_file):
        run(["add", "Todo task"])
        # Add a done task
        storage = TaskStorage(tasks_file)
        service = TaskService(storage)
        tasks = service.list_tasks()
        service.complete_task(tasks[0].id)

        run(["add", "Another todo"])
        code, out, err = run(["list", "--status", "todo"])
        assert code == 0
        assert "Another todo" in out
        assert "Todo task" not in out  # was completed

    def test_list_filter_by_priority(self, run):
        run(["add", "High task", "--priority", "high"])
        run(["add", "Low task", "--priority", "low"])
        code, out, err = run(["list", "--priority", "high"])
        assert code == 0
        assert "High task" in out
        assert "Low task" not in out

    def test_list_filter_by_tag(self, run):
        run(["add", "Tagged", "--tags", "work"])
        run(["add", "Untagged"])
        code, out, err = run(["list", "--tag", "work"])
        assert code == 0
        assert "Tagged" in out
        assert "Untagged" not in out

    def test_list_overdue_only(self, run, tasks_file):
        run(["add", "Overdue task", "--due", _past(3)])
        run(["add", "Future task", "--due", _future(10)])
        code, out, err = run(["list", "--overdue"])
        assert code == 0
        assert "Overdue task" in out
        assert "Future task" not in out

    def test_list_sort_by_priority(self, run):
        run(["add", "Low task", "--priority", "low"])
        run(["add", "Critical task", "--priority", "critical"])
        run(["add", "High task", "--priority", "high"])
        code, out, err = run(["list", "--sort", "priority"])
        assert code == 0
        # Critical should appear before Low in the output
        assert out.index("Critical task") < out.index("Low task")

    def test_list_sort_by_title(self, run):
        run(["add", "Zebra"])
        run(["add", "Apple"])
        run(["add", "Mango"])
        code, out, err = run(["list", "--sort", "title"])
        assert code == 0
        assert out.index("Apple") < out.index("Mango") < out.index("Zebra")

    def test_list_sort_desc(self, run):
        run(["add", "Apple"])
        run(["add", "Zebra"])
        code, out, err = run(["list", "--sort", "title", "--desc"])
        assert code == 0
        assert out.index("Zebra") < out.index("Apple")

    def test_list_alias_ls(self, run):
        run(["add", "Task via ls"])
        code, out, err = run(["ls"])
        assert code == 0
        assert "Task via ls" in out


# ===========================================================================
# show
# ===========================================================================

class TestShow:
    def test_show_found(self, run, tasks_file):
        run(["add", "Show me", "--description", "Some details"])
        storage = TaskStorage(tasks_file)
        task = storage.load().all()[0]

        code, out, err = run(["show", task.short_id])
        assert code == 0
        assert task.id in out
        assert "Show me" in out
        assert "Some details" in out

    def test_show_displays_all_fields(self, run, tasks_file):
        run([
            "add", "Full task",
            "--priority", "high",
            "--status", "in_progress",
            "--due", _future(5),
            "--tags", "alpha,beta",
        ])
        storage = TaskStorage(tasks_file)
        task = storage.load().all()[0]

        code, out, err = run(["show", task.id])
        assert code == 0
        assert "high" in out.lower()
        assert "in_progress" in out.lower() or "in progress" in out.lower()
        assert _future(5) in out
        assert "alpha" in out
        assert "beta" in out

    def test_show_not_found(self, run):
        code, out, err = run(["show", "nonexistent"])
        assert code != 0
        assert "Error" in err
        assert "nonexistent" in err


# ===========================================================================
# update
# ===========================================================================

class TestUpdate:
    def _add_and_get_id(self, run, tasks_file, title="Original") -> str:
        run(["add", title])
        storage = TaskStorage(tasks_file)
        return storage.load().all()[0].short_id

    def test_update_title(self, run, tasks_file):
        tid = self._add_and_get_id(run, tasks_file)
        code, out, err = run(["update", tid, "--title", "Updated title"])
        assert code == 0
        assert "Task updated" in out
        assert "Updated title" in out

    def test_update_description(self, run, tasks_file):
        tid = self._add_and_get_id(run, tasks_file)
        code, out, err = run(["update", tid, "--description", "New desc"])
        assert code == 0
        assert "Task updated" in out

    def test_update_priority(self, run, tasks_file):
        tid = self._add_and_get_id(run, tasks_file)
        code, out, err = run(["update", tid, "--priority", "critical"])
        assert code == 0
        storage = TaskStorage(tasks_file)
        task = storage.load().all()[0]
        assert task.priority.value == "critical"

    def test_update_status(self, run, tasks_file):
        tid = self._add_and_get_id(run, tasks_file)
        code, out, err = run(["update", tid, "--status", "in_progress"])
        assert code == 0
        storage = TaskStorage(tasks_file)
        task = storage.load().all()[0]
        assert task.status.value == "in_progress"

    def test_update_due_date(self, run, tasks_file):
        tid = self._add_and_get_id(run, tasks_file)
        new_due = _future(15)
        code, out, err = run(["update", tid, "--due", new_due])
        assert code == 0
        storage = TaskStorage(tasks_file)
        task = storage.load().all()[0]
        assert task.due_date == new_due

    def test_update_clear_due(self, run, tasks_file):
        run(["add", "Task with due", "--due", _future(5)])
        storage = TaskStorage(tasks_file)
        tid = storage.load().all()[0].short_id
        code, out, err = run(["update", tid, "--clear-due"])
        assert code == 0
        storage2 = TaskStorage(tasks_file)
        task = storage2.load().all()[0]
        assert task.due_date is None

    def test_update_tags(self, run, tasks_file):
        tid = self._add_and_get_id(run, tasks_file)
        code, out, err = run(["update", tid, "--tags", "x,y,z"])
        assert code == 0
        storage = TaskStorage(tasks_file)
        task = storage.load().all()[0]
        assert set(task.tags) == {"x", "y", "z"}

    def test_update_nothing_specified_fails(self, run, tasks_file):
        tid = self._add_and_get_id(run, tasks_file)
        code, out, err = run(["update", tid])
        assert code != 0
        assert "Nothing to update" in err

    def test_update_not_found(self, run):
        code, out, err = run(["update", "nonexistent", "--title", "X"])
        assert code != 0
        assert "Error" in err

    def test_update_invalid_due_date(self, run, tasks_file):
        tid = self._add_and_get_id(run, tasks_file)
        code, out, err = run(["update", tid, "--due", "not-a-date"])
        assert code != 0
        assert "Error" in err


# ===========================================================================
# done
# ===========================================================================

class TestDone:
    def test_done_marks_task(self, run, tasks_file):
        run(["add", "Finish me"])
        storage = TaskStorage(tasks_file)
        tid = storage.load().all()[0].short_id

        code, out, err = run(["done", tid])
        assert code == 0
        assert "done" in out.lower()

        storage2 = TaskStorage(tasks_file)
        task = storage2.load().all()[0]
        assert task.status.value == "done"

    def test_done_not_found(self, run):
        code, out, err = run(["done", "nonexistent"])
        assert code != 0
        assert "Error" in err
        assert "nonexistent" in err


# ===========================================================================
# cancel
# ===========================================================================

class TestCancel:
    def test_cancel_marks_task(self, run, tasks_file):
        run(["add", "Cancel me"])
        storage = TaskStorage(tasks_file)
        tid = storage.load().all()[0].short_id

        code, out, err = run(["cancel", tid])
        assert code == 0
        assert "cancel" in out.lower()

        storage2 = TaskStorage(tasks_file)
        task = storage2.load().all()[0]
        assert task.status.value == "cancelled"

    def test_cancel_not_found(self, run):
        code, out, err = run(["cancel", "nonexistent"])
        assert code != 0
        assert "Error" in err


# ===========================================================================
# start
# ===========================================================================

class TestStart:
    def test_start_marks_task(self, run, tasks_file):
        run(["add", "Start me"])
        storage = TaskStorage(tasks_file)
        tid = storage.load().all()[0].short_id

        code, out, err = run(["start", tid])
        assert code == 0
        assert "in-progress" in out.lower() or "in_progress" in out.lower()

        storage2 = TaskStorage(tasks_file)
        task = storage2.load().all()[0]
        assert task.status.value == "in_progress"

    def test_start_not_found(self, run):
        code, out, err = run(["start", "nonexistent"])
        assert code != 0
        assert "Error" in err


# ===========================================================================
# delete / rm
# ===========================================================================

class TestDelete:
    def test_delete_with_yes_flag(self, run, tasks_file):
        run(["add", "Delete me"])
        storage = TaskStorage(tasks_file)
        tid = storage.load().all()[0].short_id

        code, out, err = run(["delete", tid, "--yes"])
        assert code == 0
        assert "deleted" in out.lower()

        storage2 = TaskStorage(tasks_file)
        assert len(storage2.load()) == 0

    def test_delete_alias_rm(self, run, tasks_file):
        run(["add", "Delete via rm"])
        storage = TaskStorage(tasks_file)
        tid = storage.load().all()[0].short_id

        code, out, err = run(["rm", tid, "--yes"])
        assert code == 0
        assert "deleted" in out.lower()

    def test_delete_not_found(self, run):
        code, out, err = run(["delete", "nonexistent", "--yes"])
        assert code != 0
        assert "Error" in err

    def test_delete_removes_correct_task(self, run, tasks_file):
        run(["add", "Keep me"])
        run(["add", "Delete me"])
        storage = TaskStorage(tasks_file)
        tasks = storage.load().all()
        to_delete = next(t for t in tasks if t.title == "Delete me")

        run(["delete", to_delete.short_id, "--yes"])

        storage2 = TaskStorage(tasks_file)
        remaining = storage2.load().all()
        assert len(remaining) == 1
        assert remaining[0].title == "Keep me"


# ===========================================================================
# search
# ===========================================================================

class TestSearch:
    def test_search_finds_match(self, run):
        run(["add", "Write unit tests"])
        run(["add", "Deploy to production"])
        code, out, err = run(["search", "unit"])
        assert code == 0
        assert "Write unit tests" in out
        assert "Deploy to production" not in out

    def test_search_case_insensitive(self, run):
        run(["add", "Write UNIT tests"])
        code, out, err = run(["search", "unit"])
        assert code == 0
        assert "Write UNIT tests" in out

    def test_search_no_match(self, run):
        run(["add", "Buy groceries"])
        code, out, err = run(["search", "xyzzy"])
        assert code == 0
        assert "No tasks match" in out

    def test_search_shows_count(self, run):
        run(["add", "Alpha task"])
        run(["add", "Alpha another"])
        run(["add", "Beta task"])
        code, out, err = run(["search", "alpha"])
        assert code == 0
        assert "2 task(s)" in out

    def test_search_in_description(self, run):
        run(["add", "My task", "--description", "contains keyword here"])
        code, out, err = run(["search", "keyword"])
        assert code == 0
        assert "My task" in out


# ===========================================================================
# stats
# ===========================================================================

class TestStats:
    def test_stats_empty(self, run):
        code, out, err = run(["stats"])
        assert code == 0
        assert "0" in out
        assert "Statistics" in out or "statistics" in out.lower()

    def test_stats_shows_total(self, run):
        run(["add", "T1"])
        run(["add", "T2"])
        run(["add", "T3"])
        code, out, err = run(["stats"])
        assert code == 0
        assert "3" in out

    def test_stats_shows_status_breakdown(self, run, tasks_file):
        run(["add", "Todo 1"])
        run(["add", "Todo 2"])
        storage = TaskStorage(tasks_file)
        tid = storage.load().all()[0].short_id
        run(["done", tid])

        code, out, err = run(["stats"])
        assert code == 0
        # Should show status labels
        assert "TODO" in out or "todo" in out.lower()
        assert "DONE" in out or "done" in out.lower()

    def test_stats_shows_priority_breakdown(self, run):
        run(["add", "High task", "--priority", "high"])
        run(["add", "Low task", "--priority", "low"])
        code, out, err = run(["stats"])
        assert code == 0
        assert "HIGH" in out or "high" in out.lower()
        assert "LOW" in out or "low" in out.lower()

    def test_stats_shows_overdue_count(self, run):
        run(["add", "Overdue", "--due", _past(3)])
        run(["add", "Future", "--due", _future(10)])
        code, out, err = run(["stats"])
        assert code == 0
        assert "Overdue" in out or "overdue" in out.lower()


# ===========================================================================
# backup
# ===========================================================================

class TestBackup:
    def test_backup_success(self, run, tasks_file):
        run(["add", "Task to backup"])
        code, out, err = run(["backup"])
        assert code == 0
        assert "Backup created" in out
        # Verify the backup file actually exists
        backup_path = pathlib.Path(tasks_file + ".bak")
        assert backup_path.exists()

    def test_backup_custom_suffix(self, run, tasks_file):
        run(["add", "Task"])
        code, out, err = run(["backup", "--suffix", ".backup"])
        assert code == 0
        backup_path = pathlib.Path(tasks_file + ".backup")
        assert backup_path.exists()

    def test_backup_no_file_fails(self, run):
        # No tasks added → no file on disk → backup should fail
        code, out, err = run(["backup"])
        assert code != 0
        assert "Error" in err


# ===========================================================================
# Global --file option
# ===========================================================================

class TestGlobalFileOption:
    def test_different_files_are_independent(self, run, tmp_path):
        file_a = str(tmp_path / "a.json")
        file_b = str(tmp_path / "b.json")

        run(["add", "Task A"], extra_file=file_a)
        run(["add", "Task B"], extra_file=file_b)

        code_a, out_a, _ = run(["list"], extra_file=file_a)
        code_b, out_b, _ = run(["list"], extra_file=file_b)

        assert "Task A" in out_a
        assert "Task B" not in out_a
        assert "Task B" in out_b
        assert "Task A" not in out_b


# ===========================================================================
# Integration: full CRUD workflow
# ===========================================================================

class TestIntegration:
    def test_full_crud_workflow(self, run, tasks_file):
        """Create → list → show → update → done → delete."""
        # 1. Create
        code, out, _ = run(["add", "Integration task", "--priority", "high"])
        assert code == 0
        assert "Integration task" in out

        storage = TaskStorage(tasks_file)
        task = storage.load().all()[0]
        tid = task.short_id

        # 2. List
        code, out, _ = run(["list"])
        assert code == 0
        assert "Integration task" in out

        # 3. Show
        code, out, _ = run(["show", tid])
        assert code == 0
        assert task.id in out

        # 4. Update
        code, out, _ = run(["update", tid, "--title", "Updated integration task"])
        assert code == 0
        assert "Updated integration task" in out

        # 5. Start
        code, out, _ = run(["start", tid])
        assert code == 0

        # 6. Done
        code, out, _ = run(["done", tid])
        assert code == 0

        # 7. Stats
        code, out, _ = run(["stats"])
        assert code == 0
        assert "1" in out

        # 8. Delete
        code, out, _ = run(["delete", tid, "--yes"])
        assert code == 0

        # 9. Verify empty
        code, out, _ = run(["list"])
        assert code == 0
        assert "No tasks found" in out

    def test_search_after_add_and_update(self, run, tasks_file):
        """Add tasks, update one, then search to confirm updated content."""
        run(["add", "Design database schema"])
        run(["add", "Write API endpoints"])
        run(["add", "Deploy to staging"])

        storage = TaskStorage(tasks_file)
        tasks = storage.load().all()
        deploy_id = next(t.short_id for t in tasks if t.title == "Deploy to staging")

        # Update description
        run(["update", deploy_id, "--description", "Use Kubernetes for deployment"])

        code, out, _ = run(["search", "kubernetes"])
        assert code == 0
        assert "Deploy to staging" in out

    def test_overdue_workflow(self, run, tasks_file):
        """Add overdue and future tasks; verify overdue filter works."""
        run(["add", "Past due task", "--due", _past(2)])
        run(["add", "Future task", "--due", _future(10)])
        run(["add", "No due task"])

        # Overdue filter
        code, out, _ = run(["list", "--overdue"])
        assert code == 0
        assert "Past due task" in out
        assert "Future task" not in out
        assert "No due task" not in out

        # Stats should show overdue count
        code, out, _ = run(["stats"])
        assert code == 0
        assert "Overdue" in out or "overdue" in out.lower()

    def test_multiple_status_transitions(self, run, tasks_file):
        """todo → in_progress → done lifecycle."""
        run(["add", "Lifecycle task"])
        storage = TaskStorage(tasks_file)
        tid = storage.load().all()[0].short_id

        # Start
        run(["start", tid])
        storage2 = TaskStorage(tasks_file)
        assert storage2.load().all()[0].status.value == "in_progress"

        # Done
        run(["done", tid])
        storage3 = TaskStorage(tasks_file)
        assert storage3.load().all()[0].status.value == "done"

    def test_tag_filtering_workflow(self, run):
        """Add tasks with different tags; filter by tag."""
        run(["add", "Work task 1", "--tags", "work,urgent"])
        run(["add", "Work task 2", "--tags", "work"])
        run(["add", "Personal task", "--tags", "personal"])

        code, out, _ = run(["list", "--tag", "work"])
        assert code == 0
        assert "Work task 1" in out
        assert "Work task 2" in out
        assert "Personal task" not in out

        code2, out2, _ = run(["list", "--tag", "urgent"])
        assert code2 == 0
        assert "Work task 1" in out2
        assert "Work task 2" not in out2
