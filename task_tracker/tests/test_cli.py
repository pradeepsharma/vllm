"""
tests/test_cli.py — Integration tests for the task CLI (task_tracker.cli).

16 tests covering:
  - help output
  - add: happy path, default priority, invalid priority, missing title
  - list: all tasks, filter by status, filter by priority, empty result
  - done: happy path, non-existent ID
  - delete: happy path (--yes flag), aborted delete
  - stats: output content
  - full end-to-end flow
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from task_tracker.cli import cli


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner(tmp_path: Path) -> CliRunner:
    """Return a Click CliRunner that uses a temp data file via env var.

    mix_stderr=False so stderr is captured separately (result.stderr works).
    """
    return CliRunner(
        env={"TASK_DATA_FILE": str(tmp_path / "tasks.json")},
        mix_stderr=False,
    )


def invoke(runner: CliRunner, *args: str):
    """Helper: invoke the CLI with the given args and return the result."""
    return runner.invoke(cli, list(args), catch_exceptions=False)


def _extract_short_id(output: str) -> str:
    """Extract the 8-char short ID from 'Task added — id: XXXXXXXX ...' output."""
    match = re.search(r"id:\s+([0-9a-f]{8})", output)
    if match:
        return match.group(1)
    raise ValueError(f"Could not find short ID in output: {output!r}")


# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------


class TestHelp:
    def test_help_exits_zero_and_contains_description(self, runner: CliRunner) -> None:
        """task --help exits 0 and contains the app description."""
        result = invoke(runner, "--help")
        assert result.exit_code == 0
        assert "A simple CLI task tracker" in result.output


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_task_happy_path(self, runner: CliRunner) -> None:
        """task add outputs 'Task added' with title and priority."""
        result = invoke(runner, "add", "--title", "Write tests", "--priority", "high")
        assert result.exit_code == 0
        assert "Task added" in result.output
        # Title may be split across lines in Rich output — check for key words
        assert "Write" in result.output
        assert "high" in result.output

    def test_add_task_default_priority(self, runner: CliRunner) -> None:
        """task add without --priority defaults to medium."""
        result = invoke(runner, "add", "--title", "Deploy to prod")
        assert result.exit_code == 0
        assert "Task added" in result.output
        assert "medium" in result.output

    def test_add_task_invalid_priority_exits_nonzero(self, runner: CliRunner) -> None:
        """task add with an invalid priority exits non-zero."""
        result = runner.invoke(cli, ["add", "--title", "Bad", "--priority", "critical"])
        assert result.exit_code != 0
        # Click's Choice validation produces an error message (goes to stderr)
        combined = result.output + result.stderr
        assert "critical" in combined or "invalid" in combined.lower()

    def test_add_task_missing_title_exits_nonzero(self, runner: CliRunner) -> None:
        """task add without --title exits non-zero."""
        result = runner.invoke(cli, ["add"])
        assert result.exit_code != 0
        combined = result.output + result.stderr
        assert "title" in combined.lower() or "missing" in combined.lower()


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestList:
    def test_list_shows_all_tasks(self, runner: CliRunner) -> None:
        """task list shows all added tasks in a table."""
        invoke(runner, "add", "--title", "Alpha", "--priority", "high")
        invoke(runner, "add", "--title", "Beta", "--priority", "low")
        result = invoke(runner, "list")
        assert result.exit_code == 0
        assert "Alpha" in result.output
        assert "Beta" in result.output

    def test_list_filter_by_status(self, runner: CliRunner) -> None:
        """task list --status todo shows only todo tasks, not done tasks."""
        invoke(runner, "add", "--title", "Todo", "--priority", "medium")
        invoke(runner, "add", "--title", "Also", "--priority", "medium")
        r_done = invoke(runner, "add", "--title", "Done", "--priority", "medium")
        done_id = _extract_short_id(r_done.output)
        invoke(runner, "done", done_id)

        result = invoke(runner, "list", "--status", "todo")
        assert result.exit_code == 0
        assert "Todo" in result.output
        assert "Also" in result.output
        # The done task's short ID should not appear in the todo-filtered list
        assert done_id not in result.output

    def test_list_filter_by_priority(self, runner: CliRunner) -> None:
        """task list --priority high shows only high-priority tasks."""
        r_high = invoke(runner, "add", "--title", "High", "--priority", "high")
        r_low = invoke(runner, "add", "--title", "Low", "--priority", "low")
        high_id = _extract_short_id(r_high.output)
        low_id = _extract_short_id(r_low.output)

        result = invoke(runner, "list", "--priority", "high")
        assert result.exit_code == 0
        assert high_id in result.output
        assert low_id not in result.output

    def test_list_empty_shows_no_tasks_message(self, runner: CliRunner) -> None:
        """task list with no tasks shows a 'No tasks found' message."""
        result = invoke(runner, "list")
        assert result.exit_code == 0
        assert "No tasks found" in result.output


# ---------------------------------------------------------------------------
# done
# ---------------------------------------------------------------------------


class TestDone:
    def test_done_marks_task_as_done(self, runner: CliRunner) -> None:
        """task done marks the task as done and outputs confirmation."""
        r = invoke(runner, "add", "--title", "Finish", "--priority", "high")
        short_id = _extract_short_id(r.output)

        result = invoke(runner, "done", short_id)
        assert result.exit_code == 0
        assert "marked as done" in result.output
        assert short_id in result.output

        # Verify the status changed — the task should appear in done filter
        list_result = invoke(runner, "list", "--status", "done")
        assert short_id in list_result.output

    def test_done_nonexistent_id_exits_nonzero(self, runner: CliRunner) -> None:
        """task done with a non-existent ID exits non-zero with an error."""
        result = runner.invoke(cli, ["done", "zzzzzzz"])
        assert result.exit_code != 0
        # Error message goes to stderr
        assert "No task found" in result.stderr or "Error" in result.stderr


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_with_yes_flag_removes_task(self, runner: CliRunner) -> None:
        """task delete --yes removes the task without prompting."""
        r = invoke(runner, "add", "--title", "Obsolete", "--priority", "low")
        short_id = _extract_short_id(r.output)

        result = invoke(runner, "delete", "--yes", short_id)
        assert result.exit_code == 0
        assert "Task deleted" in result.output
        assert short_id in result.output

        # Verify it's gone
        list_result = invoke(runner, "list")
        assert short_id not in list_result.output

    def test_delete_abort_keeps_task(self, runner: CliRunner) -> None:
        """task delete with 'n' confirmation keeps the task."""
        r = invoke(runner, "add", "--title", "Keep", "--priority", "medium")
        short_id = _extract_short_id(r.output)

        # Provide 'n' as input to the confirmation prompt
        result = runner.invoke(cli, ["delete", short_id], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output

        # Task should still be there
        list_result = invoke(runner, "list")
        assert short_id in list_result.output


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_shows_correct_counts(self, runner: CliRunner) -> None:
        """task stats shows correct todo/done counts and priority breakdown."""
        invoke(runner, "add", "--title", "T1", "--priority", "high")
        r2 = invoke(runner, "add", "--title", "T2", "--priority", "low")
        invoke(runner, "add", "--title", "T3", "--priority", "medium")
        done_id = _extract_short_id(r2.output)
        invoke(runner, "done", done_id)

        result = invoke(runner, "stats")
        assert result.exit_code == 0
        assert "todo: 2" in result.output
        assert "done: 1" in result.output
        assert "high: 1" in result.output
        assert "low: 1" in result.output
        assert "medium: 1" in result.output


# ---------------------------------------------------------------------------
# End-to-end flow
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_task_lifecycle(self, runner: CliRunner) -> None:
        """Full lifecycle: add → list → done → stats → delete → list."""
        # Add three tasks
        r1 = invoke(runner, "add", "--title", "Write", "--priority", "high")
        r2 = invoke(runner, "add", "--title", "Lint", "--priority", "low")
        r3 = invoke(runner, "add", "--title", "Deploy")

        assert "Task added" in r1.output
        assert "Task added" in r2.output
        assert "Task added" in r3.output

        id1 = _extract_short_id(r1.output)
        id2 = _extract_short_id(r2.output)
        id3 = _extract_short_id(r3.output)

        # List all — should show 3 tasks (check by IDs)
        list_all = invoke(runner, "list")
        assert id1 in list_all.output
        assert id2 in list_all.output
        assert id3 in list_all.output

        # Mark task 1 as done
        done_result = invoke(runner, "done", id1)
        assert "marked as done" in done_result.output

        # Stats: 1 done, 2 todo
        stats_result = invoke(runner, "stats")
        assert "done: 1" in stats_result.output
        assert "todo: 2" in stats_result.output

        # Delete task 2
        del_result = invoke(runner, "delete", "--yes", id2)
        assert "Task deleted" in del_result.output

        # List after delete — 2 tasks remain
        final_list = invoke(runner, "list")
        assert id1 in final_list.output
        assert id3 in final_list.output
        assert id2 not in final_list.output
