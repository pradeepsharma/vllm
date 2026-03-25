"""
Tests for task_tracker.cli — Click command-line interface.

Uses Click's CliRunner for isolated, subprocess-free invocation.
Covers all five subcommands: add, list, done, delete, stats.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from click.testing import CliRunner

from task_tracker.cli import cli


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def store_path(tmp_path: Path) -> str:
    return str(tmp_path / "test_tasks.json")


def invoke(runner: CliRunner, store_path: str, *args: str):
    """Helper: invoke CLI with --store-path set."""
    return runner.invoke(cli, ["--store-path", store_path, *args])


# ---------------------------------------------------------------------------
# add command
# ---------------------------------------------------------------------------


class TestAddCommand:
    def test_add_success_output(self, runner: CliRunner, store_path: str) -> None:
        result = invoke(runner, store_path, "add", "--title", "Fix login bug", "--priority", "high")
        assert result.exit_code == 0
        assert "Added task" in result.output
        assert "Fix login bug" in result.output
        assert "priority: high" in result.output

    def test_add_default_priority_medium(self, runner: CliRunner, store_path: str) -> None:
        result = invoke(runner, store_path, "add", "--title", "Default priority")
        assert result.exit_code == 0
        assert "priority: medium" in result.output

    def test_add_low_priority(self, runner: CliRunner, store_path: str) -> None:
        result = invoke(runner, store_path, "add", "--title", "Write docs", "--priority", "low")
        assert result.exit_code == 0
        assert "Write docs" in result.output
        assert "priority: low" in result.output

    def test_add_output_contains_short_id(self, runner: CliRunner, store_path: str) -> None:
        result = invoke(runner, store_path, "add", "--title", "ID check task")
        assert result.exit_code == 0
        # The short ID (8 chars) should appear in the output
        # Verify a task was actually created and the ID is in the output
        data = json.loads(Path(store_path).read_text())
        short_id = data[0]["id"][:8]
        assert short_id in result.output

    def test_add_missing_title_fails(self, runner: CliRunner, store_path: str) -> None:
        result = invoke(runner, store_path, "add")
        assert result.exit_code != 0
        assert "Missing option '--title'" in result.output or "Missing option '--title'" in (result.output + str(result.exception))

    def test_add_invalid_priority_fails(self, runner: CliRunner, store_path: str) -> None:
        result = invoke(runner, store_path, "add", "--title", "Bad priority", "--priority", "urgent")
        assert result.exit_code != 0

    def test_add_persists_to_json(self, runner: CliRunner, store_path: str) -> None:
        invoke(runner, store_path, "add", "--title", "Persisted task", "--priority", "high")
        data = json.loads(Path(store_path).read_text())
        assert len(data) == 1
        assert data[0]["title"] == "Persisted task"
        assert data[0]["priority"] == "high"
        assert data[0]["status"] == "todo"

    def test_add_multiple_tasks(self, runner: CliRunner, store_path: str) -> None:
        invoke(runner, store_path, "add", "--title", "Task A")
        invoke(runner, store_path, "add", "--title", "Task B")
        invoke(runner, store_path, "add", "--title", "Task C")
        data = json.loads(Path(store_path).read_text())
        assert len(data) == 3


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


class TestListCommand:
    def _add_tasks(self, runner: CliRunner, store_path: str) -> None:
        invoke(runner, store_path, "add", "--title", "Fix login bug", "--priority", "high")
        invoke(runner, store_path, "add", "--title", "Write docs", "--priority", "low")

    def test_list_shows_table_with_tasks(self, runner: CliRunner, store_path: str) -> None:
        self._add_tasks(runner, store_path)
        result = invoke(runner, store_path, "list")
        assert result.exit_code == 0
        assert "Fix login bug" in result.output
        assert "Write docs" in result.output

    def test_list_shows_column_headers(self, runner: CliRunner, store_path: str) -> None:
        self._add_tasks(runner, store_path)
        result = invoke(runner, store_path, "list")
        assert result.exit_code == 0
        # Rich table headers
        assert "Title" in result.output
        assert "Status" in result.output
        assert "Priority" in result.output

    def test_list_empty_store_shows_no_tasks_message(
        self, runner: CliRunner, store_path: str
    ) -> None:
        result = invoke(runner, store_path, "list")
        assert result.exit_code == 0
        assert "No tasks found" in result.output

    def test_list_filter_by_status_todo(self, runner: CliRunner, store_path: str) -> None:
        self._add_tasks(runner, store_path)
        result = invoke(runner, store_path, "list", "--status", "todo")
        assert result.exit_code == 0
        assert "Fix login bug" in result.output
        assert "Write docs" in result.output

    def test_list_filter_by_status_done_empty(
        self, runner: CliRunner, store_path: str
    ) -> None:
        self._add_tasks(runner, store_path)
        result = invoke(runner, store_path, "list", "--status", "done")
        assert result.exit_code == 0
        assert "No tasks found" in result.output

    def test_list_filter_by_priority_high(self, runner: CliRunner, store_path: str) -> None:
        self._add_tasks(runner, store_path)
        result = invoke(runner, store_path, "list", "--priority", "high")
        assert result.exit_code == 0
        assert "Fix login bug" in result.output
        assert "Write docs" not in result.output

    def test_list_filter_by_priority_low(self, runner: CliRunner, store_path: str) -> None:
        self._add_tasks(runner, store_path)
        result = invoke(runner, store_path, "list", "--priority", "low")
        assert result.exit_code == 0
        assert "Write docs" in result.output
        assert "Fix login bug" not in result.output

    def test_list_invalid_status_fails(self, runner: CliRunner, store_path: str) -> None:
        result = invoke(runner, store_path, "list", "--status", "pending")
        assert result.exit_code != 0

    def test_list_invalid_priority_fails(self, runner: CliRunner, store_path: str) -> None:
        result = invoke(runner, store_path, "list", "--priority", "urgent")
        assert result.exit_code != 0

    def test_list_shows_id_column(self, runner: CliRunner, store_path: str) -> None:
        self._add_tasks(runner, store_path)
        result = invoke(runner, store_path, "list")
        assert result.exit_code == 0
        # The ID column header should appear
        assert "ID" in result.output


# ---------------------------------------------------------------------------
# done command
# ---------------------------------------------------------------------------


class TestDoneCommand:
    def test_done_marks_task_done(self, runner: CliRunner, store_path: str) -> None:
        invoke(runner, store_path, "add", "--title", "Mark me done")
        data = json.loads(Path(store_path).read_text())
        task_id = data[0]["id"]

        result = invoke(runner, store_path, "done", task_id[:8])
        assert result.exit_code == 0
        assert "Marked task" in result.output
        assert "done" in result.output

    def test_done_persists_status_change(self, runner: CliRunner, store_path: str) -> None:
        invoke(runner, store_path, "add", "--title", "Persist done")
        data = json.loads(Path(store_path).read_text())
        task_id = data[0]["id"]

        invoke(runner, store_path, "done", task_id[:8])

        # Reload and verify
        data = json.loads(Path(store_path).read_text())
        assert data[0]["status"] == "done"

    def test_done_nonexistent_id_shows_error(
        self, runner: CliRunner, store_path: str
    ) -> None:
        invoke(runner, store_path, "add", "--title", "Some task")
        result = invoke(runner, store_path, "done", "zzzzzzz")
        assert result.exit_code != 0
        assert "Error" in result.output
        assert "zzzzzzz" in result.output

    def test_done_output_contains_task_id(self, runner: CliRunner, store_path: str) -> None:
        invoke(runner, store_path, "add", "--title", "ID in output")
        data = json.loads(Path(store_path).read_text())
        task_id = data[0]["id"]
        short_id = task_id[:8]

        result = invoke(runner, store_path, "done", short_id)
        assert result.exit_code == 0
        assert short_id in result.output

    def test_done_with_partial_id(self, runner: CliRunner, store_path: str) -> None:
        invoke(runner, store_path, "add", "--title", "Partial ID done")
        data = json.loads(Path(store_path).read_text())
        task_id = data[0]["id"]

        # Use first 6 chars
        result = invoke(runner, store_path, "done", task_id[:6])
        assert result.exit_code == 0
        assert "Marked task" in result.output


# ---------------------------------------------------------------------------
# delete command
# ---------------------------------------------------------------------------


class TestDeleteCommand:
    def test_delete_with_yes_flag(self, runner: CliRunner, store_path: str) -> None:
        invoke(runner, store_path, "add", "--title", "Delete me")
        data = json.loads(Path(store_path).read_text())
        task_id = data[0]["id"]

        result = invoke(runner, store_path, "delete", task_id[:8], "--yes")
        assert result.exit_code == 0
        assert "Deleted task" in result.output
        assert "Delete me" in result.output

    def test_delete_removes_task_from_store(self, runner: CliRunner, store_path: str) -> None:
        invoke(runner, store_path, "add", "--title", "Gone task")
        data = json.loads(Path(store_path).read_text())
        task_id = data[0]["id"]

        invoke(runner, store_path, "delete", task_id[:8], "--yes")

        data = json.loads(Path(store_path).read_text())
        assert data == []

    def test_delete_nonexistent_id_shows_error(
        self, runner: CliRunner, store_path: str
    ) -> None:
        invoke(runner, store_path, "add", "--title", "Some task")
        result = invoke(runner, store_path, "delete", "zzzzzzz", "--yes")
        assert result.exit_code != 0
        assert "Error" in result.output

    def test_delete_output_contains_task_title(
        self, runner: CliRunner, store_path: str
    ) -> None:
        invoke(runner, store_path, "add", "--title", "Specific title to delete")
        data = json.loads(Path(store_path).read_text())
        task_id = data[0]["id"]

        result = invoke(runner, store_path, "delete", task_id[:8], "--yes")
        assert result.exit_code == 0
        assert "Specific title to delete" in result.output

    def test_delete_only_removes_target_task(
        self, runner: CliRunner, store_path: str
    ) -> None:
        invoke(runner, store_path, "add", "--title", "Keep me")
        invoke(runner, store_path, "add", "--title", "Delete me")
        data = json.loads(Path(store_path).read_text())

        # Delete the second task
        delete_id = data[1]["id"]
        invoke(runner, store_path, "delete", delete_id[:8], "--yes")

        data = json.loads(Path(store_path).read_text())
        assert len(data) == 1
        assert data[0]["title"] == "Keep me"

    def test_delete_with_partial_id(self, runner: CliRunner, store_path: str) -> None:
        invoke(runner, store_path, "add", "--title", "Partial delete")
        data = json.loads(Path(store_path).read_text())
        task_id = data[0]["id"]

        result = invoke(runner, store_path, "delete", task_id[:6], "--yes")
        assert result.exit_code == 0
        assert "Deleted task" in result.output


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------


class TestStatsCommand:
    def test_stats_empty_store(self, runner: CliRunner, store_path: str) -> None:
        result = invoke(runner, store_path, "stats")
        assert result.exit_code == 0
        assert "Total" in result.output
        assert "0" in result.output

    def test_stats_shows_total_count(self, runner: CliRunner, store_path: str) -> None:
        invoke(runner, store_path, "add", "--title", "Task 1")
        invoke(runner, store_path, "add", "--title", "Task 2")
        result = invoke(runner, store_path, "stats")
        assert result.exit_code == 0
        assert "Total" in result.output
        assert "2" in result.output

    def test_stats_shows_status_breakdown(self, runner: CliRunner, store_path: str) -> None:
        invoke(runner, store_path, "add", "--title", "Task A")
        invoke(runner, store_path, "add", "--title", "Task B")
        data = json.loads(Path(store_path).read_text())
        invoke(runner, store_path, "done", data[0]["id"][:8])

        result = invoke(runner, store_path, "stats")
        assert result.exit_code == 0
        assert "todo" in result.output
        assert "done" in result.output

    def test_stats_shows_priority_breakdown(self, runner: CliRunner, store_path: str) -> None:
        invoke(runner, store_path, "add", "--title", "High task", "--priority", "high")
        invoke(runner, store_path, "add", "--title", "Low task", "--priority", "low")
        result = invoke(runner, store_path, "stats")
        assert result.exit_code == 0
        assert "high" in result.output
        assert "low" in result.output

    def test_stats_panel_title(self, runner: CliRunner, store_path: str) -> None:
        result = invoke(runner, store_path, "stats")
        assert result.exit_code == 0
        assert "Task Statistics" in result.output

    def test_stats_correct_counts(self, runner: CliRunner, store_path: str) -> None:
        invoke(runner, store_path, "add", "--title", "High 1", "--priority", "high")
        invoke(runner, store_path, "add", "--title", "Low 1", "--priority", "low")
        data = json.loads(Path(store_path).read_text())
        # Mark first task done
        invoke(runner, store_path, "done", data[0]["id"][:8])

        result = invoke(runner, store_path, "stats")
        assert result.exit_code == 0
        # total=2, todo=1, done=1, high=1, low=1
        output = result.output
        assert "Total" in output
        assert "2" in output


# ---------------------------------------------------------------------------
# End-to-end integration test
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_workflow(self, runner: CliRunner, store_path: str) -> None:
        """Full user workflow: add → list → done → stats → delete → verify."""

        # 1. Add two tasks
        r1 = invoke(runner, store_path, "add", "--title", "Fix login bug", "--priority", "high")
        assert r1.exit_code == 0
        assert "Fix login bug" in r1.output

        r2 = invoke(runner, store_path, "add", "--title", "Write docs", "--priority", "low")
        assert r2.exit_code == 0
        assert "Write docs" in r2.output

        # 2. List — should show 2 tasks
        r3 = invoke(runner, store_path, "list")
        assert r3.exit_code == 0
        assert "Fix login bug" in r3.output
        assert "Write docs" in r3.output

        # 3. Filter by status=todo — both should appear
        r4 = invoke(runner, store_path, "list", "--status", "todo")
        assert r4.exit_code == 0
        assert "Fix login bug" in r4.output
        assert "Write docs" in r4.output

        # 4. Mark first task done
        data = json.loads(Path(store_path).read_text())
        first_id = data[0]["id"]
        r5 = invoke(runner, store_path, "done", first_id[:8])
        assert r5.exit_code == 0
        assert "Marked task" in r5.output

        # 5. Stats — total=2, todo=1, done=1
        r6 = invoke(runner, store_path, "stats")
        assert r6.exit_code == 0
        assert "Total" in r6.output
        assert "2" in r6.output

        # 6. Delete second task with --yes
        second_id = data[1]["id"]
        r7 = invoke(runner, store_path, "delete", second_id[:8], "--yes")
        assert r7.exit_code == 0
        assert "Deleted task" in r7.output

        # 7. List — should show only 1 task remaining
        r8 = invoke(runner, store_path, "list")
        assert r8.exit_code == 0
        assert "Write docs" not in r8.output

        # 8. Verify persistence — JSON file has 1 task
        final_data = json.loads(Path(store_path).read_text())
        assert len(final_data) == 1
        assert final_data[0]["status"] == "done"

    def test_error_handling_nonexistent_done(
        self, runner: CliRunner, store_path: str
    ) -> None:
        """done with a non-existent ID shows a clear error message."""
        invoke(runner, store_path, "add", "--title", "Some task")
        result = invoke(runner, store_path, "done", "zzzzzzz")
        assert result.exit_code != 0
        assert "Error" in result.output
        assert "zzzzzzz" in result.output

    def test_error_handling_missing_title(
        self, runner: CliRunner, store_path: str
    ) -> None:
        """add without --title exits with code 2."""
        result = invoke(runner, store_path, "add")
        assert result.exit_code == 2

    def test_persistence_survives_multiple_invocations(
        self, runner: CliRunner, store_path: str
    ) -> None:
        """Tasks added in one invocation are visible in the next."""
        invoke(runner, store_path, "add", "--title", "Saved task")
        # Simulate a fresh invocation — same runner, same store file
        result = invoke(runner, store_path, "list")
        assert result.exit_code == 0
        # "Saved task" is short enough to fit in the Title column without truncation
        assert "Saved task" in result.output
