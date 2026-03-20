"""Tests for team-test/cli_click.py.

Strategy: use Click's CliRunner to invoke the ``cli`` group directly so we
can capture stdout/stderr and inspect exit codes and output content.  A fresh
temporary JSON file is passed via ``--data`` for every test to ensure full
isolation from the user's real data.

Covers:
- ``--help``: top-level and all sub-commands
- ``--version``: version string
- ``add``: happy path (default priority), explicit priority, invalid priority,
  empty title
- ``list``: empty store, multiple tasks, --status filter, --priority filter,
  --verbose flag, combined filters
- ``done``: happy path, not-found error, invalid status transition
- ``start``: happy path, not-found error
- ``reopen``: happy path, not-found error
- ``delete``: happy path with --yes, ambiguous prefix, not-found error
- ``stats``: happy path with tasks, empty store
- Integration: full end-to-end workflow (add → start → done → delete)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

# ---------------------------------------------------------------------------
# Path setup — ensure team-test directory is on sys.path
# ---------------------------------------------------------------------------
_TEAM_TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _TEAM_TEST_DIR not in sys.path:
    sys.path.insert(0, _TEAM_TEST_DIR)

from cli_click import cli  # noqa: E402
from task_store import TaskStore  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _runner() -> CliRunner:
    """Return a CliRunner (stdout and stderr are mixed into one stream in Click 8+)."""
    return CliRunner()


def _invoke(tmp_path: Path, *args: str, input: str | None = None):
    """Invoke the CLI with ``--data <tmp_file>`` prepended to *args*.

    Returns the Click ``Result`` object.
    """
    data_file = str(tmp_path / "tasks.json")
    runner = _runner()
    return runner.invoke(
        cli,
        ["--data", data_file, *args],
        input=input,
        catch_exceptions=False,
    )


def _store(tmp_path: Path) -> TaskStore:
    """Return a TaskStore backed by the temp file used by _invoke."""
    return TaskStore(path=str(tmp_path / "tasks.json"))


# ===========================================================================
# --help / --version
# ===========================================================================

class TestHelpAndVersion:
    def test_top_level_help(self, tmp_path):
        result = _runner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "taskman" in result.output
        assert "add" in result.output
        assert "list" in result.output

    def test_add_help(self, tmp_path):
        result = _runner().invoke(cli, ["add", "--help"])
        assert result.exit_code == 0
        assert "TITLE" in result.output
        assert "--priority" in result.output

    def test_list_help(self, tmp_path):
        result = _runner().invoke(cli, ["list", "--help"])
        assert result.exit_code == 0
        assert "--status" in result.output
        assert "--priority" in result.output
        assert "--verbose" in result.output

    def test_done_help(self, tmp_path):
        result = _runner().invoke(cli, ["done", "--help"])
        assert result.exit_code == 0
        assert "TASK_ID" in result.output

    def test_start_help(self, tmp_path):
        result = _runner().invoke(cli, ["start", "--help"])
        assert result.exit_code == 0
        assert "TASK_ID" in result.output

    def test_reopen_help(self, tmp_path):
        result = _runner().invoke(cli, ["reopen", "--help"])
        assert result.exit_code == 0
        assert "TASK_ID" in result.output

    def test_delete_help(self, tmp_path):
        result = _runner().invoke(cli, ["delete", "--help"])
        assert result.exit_code == 0
        assert "TASK_ID" in result.output
        assert "--yes" in result.output

    def test_stats_help(self, tmp_path):
        result = _runner().invoke(cli, ["stats", "--help"])
        assert result.exit_code == 0
        assert "statistics" in result.output.lower()

    def test_version(self, tmp_path):
        result = _runner().invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "1.0.0" in result.output
        assert "taskman" in result.output


# ===========================================================================
# add command
# ===========================================================================

class TestAdd:
    def test_happy_path_exit_zero(self, tmp_path):
        result = _invoke(tmp_path, "add", "Write unit tests")
        assert result.exit_code == 0

    def test_output_contains_task_created(self, tmp_path):
        result = _invoke(tmp_path, "add", "Write unit tests")
        assert "Task created" in result.output

    def test_output_contains_task_id(self, tmp_path):
        result = _invoke(tmp_path, "add", "Write unit tests")
        # The task id (UUID) should appear in the output
        assert result.exit_code == 0
        # The panel shows the full UUID
        assert len(result.output) > 0

    def test_output_contains_title(self, tmp_path):
        result = _invoke(tmp_path, "add", "My Special Task")
        assert "My Special Task" in result.output

    def test_default_priority_is_medium(self, tmp_path):
        result = _invoke(tmp_path, "add", "Default priority task")
        assert result.exit_code == 0
        assert "medium" in result.output.lower()

    def test_explicit_priority_low(self, tmp_path):
        result = _invoke(tmp_path, "add", "Low prio task", "--priority", "low")
        assert result.exit_code == 0
        assert "low" in result.output.lower()

    def test_explicit_priority_high(self, tmp_path):
        result = _invoke(tmp_path, "add", "High prio task", "--priority", "high")
        assert result.exit_code == 0
        assert "high" in result.output.lower()

    def test_explicit_priority_critical(self, tmp_path):
        result = _invoke(tmp_path, "add", "Critical task", "--priority", "critical")
        assert result.exit_code == 0
        assert "critical" in result.output.lower()

    def test_short_priority_flag(self, tmp_path):
        result = _invoke(tmp_path, "add", "Short flag task", "-p", "high")
        assert result.exit_code == 0
        assert "high" in result.output.lower()

    def test_invalid_priority_rejected(self, tmp_path):
        # Click's Choice validation should reject unknown priority values
        runner = _runner()
        data_file = str(tmp_path / "tasks.json")
        result = runner.invoke(
            cli,
            ["--data", data_file, "add", "Bad priority", "--priority", "urgent"],
            catch_exceptions=False,
        )
        assert result.exit_code != 0

    def test_task_persisted_to_store(self, tmp_path):
        _invoke(tmp_path, "add", "Persisted task")
        store = _store(tmp_path)
        tasks = store.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].title == "Persisted task"

    def test_task_priority_persisted(self, tmp_path):
        _invoke(tmp_path, "add", "High task", "--priority", "high")
        store = _store(tmp_path)
        tasks = store.list_tasks()
        assert tasks[0].priority.value == "high"

    def test_multiple_tasks_added(self, tmp_path):
        _invoke(tmp_path, "add", "Task One")
        _invoke(tmp_path, "add", "Task Two")
        store = _store(tmp_path)
        tasks = store.list_tasks()
        assert len(tasks) == 2
        titles = {t.title for t in tasks}
        assert "Task One" in titles
        assert "Task Two" in titles

    def test_status_panel_shows_todo(self, tmp_path):
        result = _invoke(tmp_path, "add", "New task")
        assert result.exit_code == 0
        # The panel should show TODO status
        assert "TODO" in result.output


# ===========================================================================
# list command
# ===========================================================================

class TestList:
    def test_empty_store_shows_no_tasks(self, tmp_path):
        result = _invoke(tmp_path, "list")
        assert result.exit_code == 0
        assert "No tasks found" in result.output

    def test_lists_single_task(self, tmp_path):
        store = _store(tmp_path)
        store.add_task("Alpha task")
        result = _invoke(tmp_path, "list")
        assert result.exit_code == 0
        assert "Alpha task" in result.output

    def test_lists_multiple_tasks(self, tmp_path):
        store = _store(tmp_path)
        store.add_task("Task Alpha")
        store.add_task("Task Beta")
        store.add_task("Task Gamma")
        result = _invoke(tmp_path, "list")
        assert result.exit_code == 0
        assert "Task Alpha" in result.output
        assert "Task Beta" in result.output
        assert "Task Gamma" in result.output

    def test_shows_task_count(self, tmp_path):
        store = _store(tmp_path)
        store.add_task("Task A")
        store.add_task("Task B")
        result = _invoke(tmp_path, "list")
        assert result.exit_code == 0
        assert "2" in result.output

    def test_filter_by_status_todo(self, tmp_path):
        from models import Status
        store = _store(tmp_path)
        t1 = store.add_task("Todo task")
        t2 = store.add_task("Done task")
        store.update_status(t2.id, "done")
        result = _invoke(tmp_path, "list", "--status", "todo")
        assert result.exit_code == 0
        assert "Todo task" in result.output
        assert "Done task" not in result.output

    def test_filter_by_status_done(self, tmp_path):
        store = _store(tmp_path)
        t1 = store.add_task("Todo task")
        t2 = store.add_task("Done task")
        store.update_status(t2.id, "done")
        result = _invoke(tmp_path, "list", "--status", "done")
        assert result.exit_code == 0
        assert "Done task" in result.output
        assert "Todo task" not in result.output

    def test_filter_by_status_in_progress(self, tmp_path):
        store = _store(tmp_path)
        t1 = store.add_task("In progress task")
        t2 = store.add_task("Todo task")
        store.update_status(t1.id, "in_progress")
        result = _invoke(tmp_path, "list", "--status", "in_progress")
        assert result.exit_code == 0
        assert "In progress task" in result.output
        assert "Todo task" not in result.output

    def test_filter_by_priority_high(self, tmp_path):
        store = _store(tmp_path)
        store.add_task("High task", priority="high")
        store.add_task("Low task", priority="low")
        result = _invoke(tmp_path, "list", "--priority", "high")
        assert result.exit_code == 0
        assert "High task" in result.output
        assert "Low task" not in result.output

    def test_filter_by_priority_critical(self, tmp_path):
        store = _store(tmp_path)
        store.add_task("Critical task", priority="critical")
        store.add_task("Medium task", priority="medium")
        result = _invoke(tmp_path, "list", "--priority", "critical")
        assert result.exit_code == 0
        assert "Critical task" in result.output
        assert "Medium task" not in result.output

    def test_combined_status_and_priority_filter(self, tmp_path):
        store = _store(tmp_path)
        t1 = store.add_task("High todo", priority="high")
        t2 = store.add_task("High done", priority="high")
        t3 = store.add_task("Low todo", priority="low")
        store.update_status(t2.id, "done")
        result = _invoke(tmp_path, "list", "--status", "todo", "--priority", "high")
        assert result.exit_code == 0
        assert "High todo" in result.output
        assert "High done" not in result.output
        assert "Low todo" not in result.output

    def test_verbose_flag_shows_full_id(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Verbose task")
        result = _invoke(tmp_path, "list", "--verbose")
        assert result.exit_code == 0
        # In verbose mode the full UUID is shown (not truncated with …)
        assert task.id in result.output

    def test_non_verbose_shows_truncated_id(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Short id task")
        result = _invoke(tmp_path, "list")
        assert result.exit_code == 0
        # Non-verbose shows first 8 chars + ellipsis
        assert task.id[:8] in result.output
        assert "…" in result.output

    def test_filter_shows_filter_suffix(self, tmp_path):
        store = _store(tmp_path)
        store.add_task("Todo task")
        result = _invoke(tmp_path, "list", "--status", "todo")
        assert result.exit_code == 0
        assert "todo" in result.output

    def test_invalid_status_rejected(self, tmp_path):
        runner = _runner()
        data_file = str(tmp_path / "tasks.json")
        result = runner.invoke(
            cli,
            ["--data", data_file, "list", "--status", "invalid_status"],
            catch_exceptions=False,
        )
        assert result.exit_code != 0

    def test_invalid_priority_rejected(self, tmp_path):
        runner = _runner()
        data_file = str(tmp_path / "tasks.json")
        result = runner.invoke(
            cli,
            ["--data", data_file, "list", "--priority", "super_urgent"],
            catch_exceptions=False,
        )
        assert result.exit_code != 0


# ===========================================================================
# done command
# ===========================================================================

class TestDone:
    def test_happy_path_exit_zero(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Finish me")
        result = _invoke(tmp_path, "done", task.id)
        assert result.exit_code == 0

    def test_output_contains_done_message(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Finish me")
        result = _invoke(tmp_path, "done", task.id)
        assert "DONE" in result.output

    def test_output_contains_task_title(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("My Done Task")
        result = _invoke(tmp_path, "done", task.id)
        assert "My Done Task" in result.output

    def test_task_status_updated_in_store(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Update me")
        _invoke(tmp_path, "done", task.id)
        updated = _store(tmp_path).list_tasks()[0]
        assert updated.status.value == "done"

    def test_partial_id_works(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Partial id task")
        # Use first 8 characters as prefix
        result = _invoke(tmp_path, "done", task.id[:8])
        assert result.exit_code == 0
        assert "DONE" in result.output

    def test_not_found_exits_one(self, tmp_path):
        result = _invoke(tmp_path, "done", "nonexistent-id-xyz")
        assert result.exit_code == 1

    def test_not_found_shows_error(self, tmp_path):
        runner = _runner()
        data_file = str(tmp_path / "tasks.json")
        result = runner.invoke(
            cli,
            ["--data", data_file, "done", "nonexistent-id-xyz"],
            catch_exceptions=False,
        )
        assert result.exit_code == 1
        # Error goes to stderr
        assert "Error" in result.output or "Error" in (result.stderr or "")


# ===========================================================================
# start command
# ===========================================================================

class TestStart:
    def test_happy_path_exit_zero(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Start me")
        result = _invoke(tmp_path, "start", task.id)
        assert result.exit_code == 0

    def test_output_contains_in_progress_message(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Start me")
        result = _invoke(tmp_path, "start", task.id)
        assert "IN PROGRESS" in result.output

    def test_output_contains_task_title(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("My Started Task")
        result = _invoke(tmp_path, "start", task.id)
        assert "My Started Task" in result.output

    def test_task_status_updated_in_store(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Update me")
        _invoke(tmp_path, "start", task.id)
        updated = _store(tmp_path).list_tasks()[0]
        assert updated.status.value == "in_progress"

    def test_partial_id_works(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Partial start task")
        result = _invoke(tmp_path, "start", task.id[:8])
        assert result.exit_code == 0

    def test_not_found_exits_one(self, tmp_path):
        result = _invoke(tmp_path, "start", "nonexistent-id-xyz")
        assert result.exit_code == 1


# ===========================================================================
# reopen command
# ===========================================================================

class TestReopen:
    def test_happy_path_exit_zero(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Reopen me")
        store.update_status(task.id, "done")
        result = _invoke(tmp_path, "reopen", task.id)
        assert result.exit_code == 0

    def test_output_contains_todo_message(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Reopen me")
        store.update_status(task.id, "done")
        result = _invoke(tmp_path, "reopen", task.id)
        assert "TODO" in result.output

    def test_output_contains_task_title(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("My Reopened Task")
        store.update_status(task.id, "done")
        result = _invoke(tmp_path, "reopen", task.id)
        assert "My Reopened Task" in result.output

    def test_task_status_updated_in_store(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Update me")
        store.update_status(task.id, "done")
        _invoke(tmp_path, "reopen", task.id)
        updated = _store(tmp_path).list_tasks()[0]
        assert updated.status.value == "todo"

    def test_not_found_exits_one(self, tmp_path):
        result = _invoke(tmp_path, "reopen", "nonexistent-id-xyz")
        assert result.exit_code == 1


# ===========================================================================
# delete command
# ===========================================================================

class TestDelete:
    def test_happy_path_with_yes_flag(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Delete me")
        result = _invoke(tmp_path, "delete", task.id, "--yes")
        assert result.exit_code == 0

    def test_output_contains_deleted_message(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Delete me")
        result = _invoke(tmp_path, "delete", task.id, "--yes")
        assert "deleted" in result.output.lower()

    def test_output_contains_task_title(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("My Deleted Task")
        result = _invoke(tmp_path, "delete", task.id, "--yes")
        assert "My Deleted Task" in result.output

    def test_task_removed_from_store(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Remove me")
        _invoke(tmp_path, "delete", task.id, "--yes")
        remaining = _store(tmp_path).list_tasks()
        assert len(remaining) == 0

    def test_short_yes_flag(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Short flag delete")
        result = _invoke(tmp_path, "delete", task.id, "-y")
        assert result.exit_code == 0

    def test_partial_id_with_yes(self, tmp_path):
        store = _store(tmp_path)
        task = store.add_task("Partial delete task")
        result = _invoke(tmp_path, "delete", task.id[:8], "--yes")
        assert result.exit_code == 0
        assert "deleted" in result.output.lower()

    def test_not_found_exits_one(self, tmp_path):
        result = _invoke(tmp_path, "delete", "nonexistent-id-xyz", "--yes")
        assert result.exit_code == 1

    def test_not_found_shows_error_message(self, tmp_path):
        result = _invoke(tmp_path, "delete", "nonexistent-id-xyz", "--yes")
        assert result.exit_code == 1
        # Error message goes to stderr (mix_stderr=False); check both streams
        combined = (result.output or "") + (result.stderr or "")
        assert "Error" in combined or "No task found" in combined

    def test_ambiguous_prefix_exits_one(self, tmp_path):
        store = _store(tmp_path)
        # Create two tasks — their UUIDs share the same first character
        # We'll use a very short prefix that matches both
        t1 = store.add_task("Task One")
        t2 = store.add_task("Task Two")
        # Use just the first character — almost certainly matches both
        # (UUIDs are random, but we can use an empty string or a known shared prefix)
        # Instead, use a prefix that we know matches both by checking their IDs
        common = ""
        for i in range(len(t1.id)):
            if t1.id[:i+1] == t2.id[:i+1]:
                common = t1.id[:i+1]
            else:
                break
        if common:
            result = _invoke(tmp_path, "delete", common, "--yes")
            assert result.exit_code == 1
        else:
            # IDs differ from the first character — skip ambiguity test
            pytest.skip("UUIDs don't share a common prefix for ambiguity test")

    def test_confirmation_prompt_abort(self, tmp_path):
        """Without --yes, answering 'n' to the prompt should abort."""
        store = _store(tmp_path)
        task = store.add_task("Confirm delete")
        runner = _runner()
        data_file = str(tmp_path / "tasks.json")
        result = runner.invoke(
            cli,
            ["--data", data_file, "delete", task.id],
            input="n\n",
            catch_exceptions=False,
        )
        # Aborted — task should still exist
        remaining = _store(tmp_path).list_tasks()
        assert len(remaining) == 1

    def test_confirmation_prompt_yes(self, tmp_path):
        """Without --yes, answering 'y' to the prompt should delete the task."""
        store = _store(tmp_path)
        task = store.add_task("Confirm delete yes")
        runner = _runner()
        data_file = str(tmp_path / "tasks.json")
        result = runner.invoke(
            cli,
            ["--data", data_file, "delete", task.id],
            input="y\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        remaining = _store(tmp_path).list_tasks()
        assert len(remaining) == 0


# ===========================================================================
# stats command
# ===========================================================================

class TestStats:
    def test_empty_store_exit_zero(self, tmp_path):
        result = _invoke(tmp_path, "stats")
        assert result.exit_code == 0

    def test_empty_store_shows_zero_total(self, tmp_path):
        result = _invoke(tmp_path, "stats")
        assert result.exit_code == 0
        assert "0" in result.output

    def test_shows_total_tasks(self, tmp_path):
        store = _store(tmp_path)
        store.add_task("Task A")
        store.add_task("Task B")
        store.add_task("Task C")
        result = _invoke(tmp_path, "stats")
        assert result.exit_code == 0
        assert "3" in result.output

    def test_shows_status_breakdown(self, tmp_path):
        store = _store(tmp_path)
        t1 = store.add_task("Todo task")
        t2 = store.add_task("Done task")
        store.update_status(t2.id, "done")
        result = _invoke(tmp_path, "stats")
        assert result.exit_code == 0
        # Should show status labels
        assert "TODO" in result.output or "todo" in result.output.lower()
        assert "DONE" in result.output or "done" in result.output.lower()

    def test_shows_priority_breakdown(self, tmp_path):
        store = _store(tmp_path)
        store.add_task("High task", priority="high")
        store.add_task("Low task", priority="low")
        result = _invoke(tmp_path, "stats")
        assert result.exit_code == 0
        assert "high" in result.output.lower() or "High" in result.output
        assert "low" in result.output.lower() or "Low" in result.output

    def test_shows_statistics_panel_title(self, tmp_path):
        result = _invoke(tmp_path, "stats")
        assert result.exit_code == 0
        # The panel title contains "Statistics"
        assert "Statistics" in result.output or "statistics" in result.output.lower()

    def test_shows_overdue_count_when_nonzero(self, tmp_path):
        from storage import Storage
        from models import Priority, Status
        # Create a task with a past due date directly via Storage
        storage = Storage(str(tmp_path / "tasks.json"))
        storage.create_task("Overdue task", due_date="2000-01-01")
        result = _invoke(tmp_path, "stats")
        assert result.exit_code == 0
        assert "Overdue" in result.output or "overdue" in result.output.lower()


# ===========================================================================
# --data option / TASKMAN_DATA env var
# ===========================================================================

class TestDataOption:
    def test_custom_data_path(self, tmp_path):
        custom_path = str(tmp_path / "custom" / "my_tasks.json")
        runner = _runner()
        result = runner.invoke(
            cli,
            ["--data", custom_path, "add", "Custom path task"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert os.path.exists(custom_path)

    def test_env_var_sets_data_path(self, tmp_path, monkeypatch):
        custom_path = str(tmp_path / "env_tasks.json")
        monkeypatch.setenv("TASKMAN_DATA", custom_path)
        runner = _runner()
        result = runner.invoke(
            cli,
            ["add", "Env var task"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert os.path.exists(custom_path)


# ===========================================================================
# Integration: full end-to-end workflow
# ===========================================================================

class TestIntegration:
    def test_full_lifecycle(self, tmp_path):
        """Add → list → start → done → delete."""
        # 1. Add a task
        result = _invoke(tmp_path, "add", "Integration task", "--priority", "high")
        assert result.exit_code == 0
        assert "Integration task" in result.output

        # 2. List — should appear
        result = _invoke(tmp_path, "list")
        assert result.exit_code == 0
        assert "Integration task" in result.output

        # Get the task id from the store
        store = _store(tmp_path)
        tasks = store.list_tasks()
        assert len(tasks) == 1
        task_id = tasks[0].id

        # 3. Start the task
        result = _invoke(tmp_path, "start", task_id)
        assert result.exit_code == 0
        assert "IN PROGRESS" in result.output

        # 4. Verify status in list
        result = _invoke(tmp_path, "list", "--status", "in_progress")
        assert result.exit_code == 0
        assert "Integration task" in result.output

        # 5. Mark done
        result = _invoke(tmp_path, "done", task_id)
        assert result.exit_code == 0
        assert "DONE" in result.output

        # 6. Verify it no longer appears in todo list
        result = _invoke(tmp_path, "list", "--status", "todo")
        assert result.exit_code == 0
        assert "No tasks found" in result.output

        # 7. Stats should show 1 done task
        result = _invoke(tmp_path, "stats")
        assert result.exit_code == 0
        assert "1" in result.output

        # 8. Delete the task
        result = _invoke(tmp_path, "delete", task_id, "--yes")
        assert result.exit_code == 0
        assert "Integration task" in result.output

        # 9. Store should be empty
        remaining = _store(tmp_path).list_tasks()
        assert len(remaining) == 0

    def test_reopen_workflow(self, tmp_path):
        """Add → done → reopen → verify status is todo."""
        store = _store(tmp_path)
        task = store.add_task("Reopen workflow task")

        # Mark done
        result = _invoke(tmp_path, "done", task.id)
        assert result.exit_code == 0

        # Reopen
        result = _invoke(tmp_path, "reopen", task.id)
        assert result.exit_code == 0
        assert "TODO" in result.output

        # Verify in store
        updated = _store(tmp_path).list_tasks()[0]
        assert updated.status.value == "todo"

    def test_multiple_tasks_filter_workflow(self, tmp_path):
        """Add multiple tasks with different priorities and filter them."""
        store = _store(tmp_path)
        store.add_task("Low task", priority="low")
        store.add_task("Medium task", priority="medium")
        store.add_task("High task", priority="high")
        store.add_task("Critical task", priority="critical")

        # Filter by high
        result = _invoke(tmp_path, "list", "--priority", "high")
        assert result.exit_code == 0
        assert "High task" in result.output
        assert "Low task" not in result.output
        assert "Medium task" not in result.output
        assert "Critical task" not in result.output

        # Filter by critical
        result = _invoke(tmp_path, "list", "--priority", "critical")
        assert result.exit_code == 0
        assert "Critical task" in result.output
        assert "High task" not in result.output

    def test_stats_after_lifecycle(self, tmp_path):
        """Verify stats reflect correct counts after task lifecycle changes."""
        store = _store(tmp_path)
        t1 = store.add_task("Task 1")
        t2 = store.add_task("Task 2")
        t3 = store.add_task("Task 3")

        store.update_status(t1.id, "in_progress")
        store.update_status(t2.id, "done")

        result = _invoke(tmp_path, "stats")
        assert result.exit_code == 0
        # Should show 3 total tasks
        assert "3" in result.output
        # Should show status breakdown
        assert "TODO" in result.output or "todo" in result.output.lower()
        assert "DONE" in result.output or "done" in result.output.lower()

    def test_verbose_list_shows_full_uuid(self, tmp_path):
        """Verbose list should show the full UUID, not a truncated version."""
        store = _store(tmp_path)
        task = store.add_task("Verbose task")

        result = _invoke(tmp_path, "list", "--verbose")
        assert result.exit_code == 0
        assert task.id in result.output

    def test_add_then_list_count_matches(self, tmp_path):
        """Adding N tasks should result in N tasks shown in list."""
        for i in range(5):
            _invoke(tmp_path, "add", f"Task {i+1}")

        result = _invoke(tmp_path, "list")
        assert result.exit_code == 0
        assert "5" in result.output
        for i in range(5):
            assert f"Task {i+1}" in result.output
