"""
test_cli.py — Integration tests for the task_tracker CLI.

Uses Click's CliRunner to invoke commands as a real user would, verifying
stdout content, exit codes, and side-effects (file mutations).

Covers:
  - add: happy path (all priorities), default priority, blank title error
  - list: all tasks, status filter (todo/done), empty result, priority filter
  - done: happy path, not-found error
  - delete: confirmed deletion, aborted deletion, not-found error
  - stats: panel output with correct counts
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from task_tracker.cli import task_tracker_cli


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture()
def store_path(tmp_path: Path) -> Path:
    """Return a path to a temp tasks.json file (does not exist yet)."""
    return tmp_path / "tasks.json"


def invoke(runner: CliRunner, store_path: Path, args: list[str], input: str | None = None):
    """Invoke the CLI with TASK_TRACKER_FILE set to the temp store."""
    env = {"TASK_TRACKER_FILE": str(store_path)}
    return runner.invoke(task_tracker_cli, args, env=env, input=input, catch_exceptions=False)


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


class TestAddCommand:
    def test_add_high_priority(self, runner: CliRunner, store_path: Path) -> None:
        result = invoke(runner, store_path, ["add", "--title", "Write tests", "--priority", "high"])
        assert result.exit_code == 0
        assert "✅ Added task" in result.output

    def test_add_low_priority(self, runner: CliRunner, store_path: Path) -> None:
        result = invoke(runner, store_path, ["add", "--title", "Fix bug", "--priority", "low"])
        assert result.exit_code == 0
        assert "✅ Added task" in result.output

    def test_add_default_priority_medium(self, runner: CliRunner, store_path: Path) -> None:
        result = invoke(runner, store_path, ["add", "--title", "Review PR"])
        assert result.exit_code == 0
        assert "✅ Added task" in result.output
        # Verify the stored task has medium priority
        data = json.loads(store_path.read_text())
        assert data[0]["priority"] == "medium"

    def test_add_outputs_uuid(self, runner: CliRunner, store_path: Path) -> None:
        result = invoke(runner, store_path, ["add", "--title", "UUID check"])
        assert result.exit_code == 0
        # The output should contain a 32-char hex UUID
        data = json.loads(store_path.read_text())
        task_id = data[0]["id"]
        assert task_id in result.output
        assert len(task_id) == 32

    def test_add_missing_title_fails(self, runner: CliRunner, store_path: Path) -> None:
        result = runner.invoke(
            task_tracker_cli,
            ["add"],
            env={"TASK_TRACKER_FILE": str(store_path)},
        )
        assert result.exit_code != 0

    def test_add_three_tasks_creates_three_records(self, runner: CliRunner, store_path: Path) -> None:
        invoke(runner, store_path, ["add", "--title", "Task 1", "--priority", "high"])
        invoke(runner, store_path, ["add", "--title", "Task 2", "--priority", "low"])
        invoke(runner, store_path, ["add", "--title", "Task 3"])
        data = json.loads(store_path.read_text())
        assert len(data) == 3


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestListCommand:
    def _add_three(self, runner: CliRunner, store_path: Path) -> list[str]:
        """Add 3 tasks and return their IDs."""
        ids = []
        for title, prio in [("Write tests", "high"), ("Fix bug", "low"), ("Review PR", "medium")]:
            result = invoke(runner, store_path, ["add", "--title", title, "--priority", prio])
            # Extract UUID from output line
            data = json.loads(store_path.read_text())
            ids.append(data[-1]["id"])
        return ids

    def test_list_shows_three_rows(self, runner: CliRunner, store_path: Path) -> None:
        self._add_three(runner, store_path)
        result = invoke(runner, store_path, ["list"])
        assert result.exit_code == 0
        assert "Write tests" in result.output
        assert "Fix bug" in result.output
        assert "Review PR" in result.output

    def test_list_shows_table_columns(self, runner: CliRunner, store_path: Path) -> None:
        self._add_three(runner, store_path)
        result = invoke(runner, store_path, ["list"])
        assert "ID" in result.output
        assert "Title" in result.output
        assert "Status" in result.output
        assert "Priority" in result.output
        assert "Created" in result.output

    def test_list_status_todo_shows_all(self, runner: CliRunner, store_path: Path) -> None:
        self._add_three(runner, store_path)
        result = invoke(runner, store_path, ["list", "--status", "todo"])
        assert result.exit_code == 0
        assert "Write tests" in result.output
        assert "Fix bug" in result.output
        assert "Review PR" in result.output

    def test_list_status_done_empty(self, runner: CliRunner, store_path: Path) -> None:
        self._add_three(runner, store_path)
        result = invoke(runner, store_path, ["list", "--status", "done"])
        assert result.exit_code == 0
        assert "No tasks found." in result.output

    def test_list_status_done_after_marking(self, runner: CliRunner, store_path: Path) -> None:
        ids = self._add_three(runner, store_path)
        invoke(runner, store_path, ["done", ids[0][:8]])
        result = invoke(runner, store_path, ["list", "--status", "done"])
        assert result.exit_code == 0
        assert "Write tests" in result.output

    def test_list_empty_store(self, runner: CliRunner, store_path: Path) -> None:
        result = invoke(runner, store_path, ["list"])
        assert result.exit_code == 0
        assert "No tasks found." in result.output

    def test_list_priority_filter(self, runner: CliRunner, store_path: Path) -> None:
        self._add_three(runner, store_path)
        result = invoke(runner, store_path, ["list", "--priority", "high"])
        assert result.exit_code == 0
        assert "Write tests" in result.output
        assert "Fix bug" not in result.output


# ---------------------------------------------------------------------------
# done
# ---------------------------------------------------------------------------


class TestDoneCommand:
    def test_done_marks_task(self, runner: CliRunner, store_path: Path) -> None:
        invoke(runner, store_path, ["add", "--title", "Complete me", "--priority", "medium"])
        data = json.loads(store_path.read_text())
        task_id = data[0]["id"]
        result = invoke(runner, store_path, ["done", task_id[:8]])
        assert result.exit_code == 0
        assert "✅ Marked done" in result.output
        assert "Complete me" in result.output

    def test_done_persists_status(self, runner: CliRunner, store_path: Path) -> None:
        invoke(runner, store_path, ["add", "--title", "Persist done"])
        data = json.loads(store_path.read_text())
        task_id = data[0]["id"]
        invoke(runner, store_path, ["done", task_id[:8]])
        updated = json.loads(store_path.read_text())
        assert updated[0]["status"] == "done"

    def test_done_not_found_exits_1(self, runner: CliRunner, store_path: Path) -> None:
        invoke(runner, store_path, ["add", "--title", "Some task"])
        result = invoke(runner, store_path, ["done", "zzzzzzzzz"])
        assert result.exit_code == 1
        assert "No task matching" in result.output

    def test_done_error_message_contains_partial_id(self, runner: CliRunner, store_path: Path) -> None:
        invoke(runner, store_path, ["add", "--title", "Some task"])
        result = invoke(runner, store_path, ["done", "zzzzzzzzz"])
        assert "zzzzzzzzz" in result.output


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDeleteCommand:
    def test_delete_confirmed(self, runner: CliRunner, store_path: Path) -> None:
        invoke(runner, store_path, ["add", "--title", "Delete me"])
        data = json.loads(store_path.read_text())
        task_id = data[0]["id"]
        result = invoke(runner, store_path, ["delete", task_id[:8]], input="y\n")
        assert result.exit_code == 0
        assert "🗑 Deleted" in result.output
        remaining = json.loads(store_path.read_text())
        assert remaining == []

    def test_delete_aborted(self, runner: CliRunner, store_path: Path) -> None:
        invoke(runner, store_path, ["add", "--title", "Keep me"])
        data = json.loads(store_path.read_text())
        task_id = data[0]["id"]
        # Use catch_exceptions=True because click.Abort raises an exception
        result = runner.invoke(
            task_tracker_cli,
            ["delete", task_id[:8]],
            env={"TASK_TRACKER_FILE": str(store_path)},
            input="n\n",
        )
        assert result.exit_code != 0
        remaining = json.loads(store_path.read_text())
        assert len(remaining) == 1  # task was NOT deleted

    def test_delete_not_found_exits_1(self, runner: CliRunner, store_path: Path) -> None:
        invoke(runner, store_path, ["add", "--title", "Some task"])
        result = invoke(runner, store_path, ["delete", "zzzzzzzzz"], input="y\n")
        assert result.exit_code == 1
        assert "No task matching" in result.output


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestStatsCommand:
    def test_stats_panel_title(self, runner: CliRunner, store_path: Path) -> None:
        result = invoke(runner, store_path, ["stats"])
        assert result.exit_code == 0
        assert "Task Statistics" in result.output

    def test_stats_shows_total(self, runner: CliRunner, store_path: Path) -> None:
        invoke(runner, store_path, ["add", "--title", "T1", "--priority", "high"])
        invoke(runner, store_path, ["add", "--title", "T2", "--priority", "low"])
        result = invoke(runner, store_path, ["stats"])
        assert result.exit_code == 0
        assert "Total tasks" in result.output
        assert "2" in result.output

    def test_stats_shows_status_counts(self, runner: CliRunner, store_path: Path) -> None:
        invoke(runner, store_path, ["add", "--title", "T1"])
        invoke(runner, store_path, ["add", "--title", "T2"])
        data = json.loads(store_path.read_text())
        invoke(runner, store_path, ["done", data[0]["id"][:8]])
        result = invoke(runner, store_path, ["stats"])
        assert result.exit_code == 0
        assert "todo" in result.output
        assert "done" in result.output

    def test_stats_shows_priority_counts(self, runner: CliRunner, store_path: Path) -> None:
        invoke(runner, store_path, ["add", "--title", "High task", "--priority", "high"])
        invoke(runner, store_path, ["add", "--title", "Low task", "--priority", "low"])
        result = invoke(runner, store_path, ["stats"])
        assert result.exit_code == 0
        assert "high" in result.output
        assert "low" in result.output

    def test_stats_empty_store(self, runner: CliRunner, store_path: Path) -> None:
        result = invoke(runner, store_path, ["stats"])
        assert result.exit_code == 0
        assert "Task Statistics" in result.output
        assert "0" in result.output
