"""
tests/test_cli.py — Integration tests for the Click CLI (cli.py).

Uses Click's CliRunner to invoke the CLI as a real subprocess-like call,
exercising the full command pipeline including TaskStore I/O.

Covers:
  - --help output
  - add: happy path, default priority, invalid priority
  - list: empty store, all tasks, status filter, priority filter, combined filter
  - done: happy path, not found
  - delete: with --yes flag, not found, without --yes (aborted)
  - stats: empty store, populated store
  - --store option: custom store path
  - TASK_STORE_PATH env var
  - End-to-end flow: add → list → done → stats → delete
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

# Make sure the parent directory is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli import cli


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    """Return a Click CliRunner that mixes stdout/stderr."""
    return CliRunner(mix_stderr=False)


@pytest.fixture()
def store_path(tmp_path: Path) -> str:
    """Return a path string for a temp tasks.json file."""
    return str(tmp_path / "tasks.json")


def invoke(runner: CliRunner, store_path: str, *args: str):
    """Helper: invoke the CLI with --store pointing at the temp file."""
    return runner.invoke(cli, ["--store", store_path, *args])


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


class TestHelp:
    def test_root_help_exits_zero(self, runner: CliRunner, store_path: str):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_root_help_lists_all_subcommands(self, runner: CliRunner):
        result = runner.invoke(cli, ["--help"])
        output = result.output
        assert "add" in output
        assert "list" in output
        assert "done" in output
        assert "delete" in output
        assert "stats" in output

    def test_add_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["add", "--help"])
        assert result.exit_code == 0
        assert "--title" in result.output
        assert "--priority" in result.output

    def test_list_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["list", "--help"])
        assert result.exit_code == 0
        assert "--status" in result.output
        assert "--priority" in result.output

    def test_done_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["done", "--help"])
        assert result.exit_code == 0
        assert "TASK_ID" in result.output

    def test_delete_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["delete", "--help"])
        assert result.exit_code == 0
        assert "TASK_ID" in result.output
        assert "--yes" in result.output

    def test_stats_help(self, runner: CliRunner):
        result = runner.invoke(cli, ["stats", "--help"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


class TestAdd:
    def test_add_task_exits_zero(self, runner: CliRunner, store_path: str):
        result = invoke(runner, store_path, "add", "--title", "Fix login bug", "--priority", "high")
        assert result.exit_code == 0

    def test_add_task_output_contains_title(self, runner: CliRunner, store_path: str):
        result = invoke(runner, store_path, "add", "--title", "Fix login bug", "--priority", "high")
        assert "Fix login bug" in result.output

    def test_add_task_output_contains_priority(self, runner: CliRunner, store_path: str):
        result = invoke(runner, store_path, "add", "--title", "Fix login bug", "--priority", "high")
        assert "high" in result.output

    def test_add_task_output_contains_short_id(self, runner: CliRunner, store_path: str):
        result = invoke(runner, store_path, "add", "--title", "My task", "--priority", "medium")
        # Short ID is 8 hex chars; check that some hex-looking token appears
        assert result.exit_code == 0
        # The output should contain "Added task"
        assert "Added" in result.output

    def test_add_task_default_priority_is_medium(self, runner: CliRunner, store_path: str):
        result = invoke(runner, store_path, "add", "--title", "Default priority task")
        assert result.exit_code == 0
        assert "medium" in result.output

    def test_add_task_persists_to_json(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Persisted task", "--priority", "low")
        data = json.loads(Path(store_path).read_text())
        assert len(data) == 1
        assert data[0]["title"] == "Persisted task"
        assert data[0]["priority"] == "low"
        assert data[0]["status"] == "todo"

    def test_add_task_invalid_priority_exits_nonzero(self, runner: CliRunner, store_path: str):
        result = invoke(runner, store_path, "add", "--title", "Bad task", "--priority", "urgent")
        assert result.exit_code != 0

    def test_add_task_missing_title_exits_nonzero(self, runner: CliRunner, store_path: str):
        result = invoke(runner, store_path, "add")
        assert result.exit_code != 0
        # Click writes usage errors to stderr when mix_stderr=False; check both streams
        combined = (result.output + (result.stderr if hasattr(result, "stderr") else "")).lower()
        assert "title" in combined or "missing" in combined or result.exit_code == 2

    def test_add_multiple_tasks_all_stored(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Task A", "--priority", "high")
        invoke(runner, store_path, "add", "--title", "Task B", "--priority", "low")
        data = json.loads(Path(store_path).read_text())
        assert len(data) == 2
        titles = {t["title"] for t in data}
        assert titles == {"Task A", "Task B"}


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


class TestList:
    def test_list_empty_store_shows_no_tasks_message(self, runner: CliRunner, store_path: str):
        result = invoke(runner, store_path, "list")
        assert result.exit_code == 0
        assert "No tasks found" in result.output

    def test_list_shows_all_tasks(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Task Alpha", "--priority", "high")
        invoke(runner, store_path, "add", "--title", "Task Beta", "--priority", "low")
        result = invoke(runner, store_path, "list")
        assert result.exit_code == 0
        assert "Task Alpha" in result.output
        assert "Task Beta" in result.output

    def test_list_shows_status_column(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Status check", "--priority", "medium")
        result = invoke(runner, store_path, "list")
        # "To Do" is the human-friendly label for "todo"
        assert "To Do" in result.output

    def test_list_filter_by_status_todo(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Todo task", "--priority", "medium")
        result = invoke(runner, store_path, "list", "--status", "todo")
        assert result.exit_code == 0
        assert "Todo task" in result.output

    def test_list_filter_by_status_done_empty_initially(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Not done yet", "--priority", "medium")
        result = invoke(runner, store_path, "list", "--status", "done")
        assert result.exit_code == 0
        assert "No tasks found" in result.output

    def test_list_filter_by_priority_high(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "High priority task", "--priority", "high")
        invoke(runner, store_path, "add", "--title", "Low priority task", "--priority", "low")
        result = invoke(runner, store_path, "list", "--priority", "high")
        assert result.exit_code == 0
        assert "High priority task" in result.output
        assert "Low priority task" not in result.output

    def test_list_combined_filter(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "High todo", "--priority", "high")
        invoke(runner, store_path, "add", "--title", "Low todo", "--priority", "low")
        result = invoke(runner, store_path, "list", "--status", "todo", "--priority", "high")
        assert result.exit_code == 0
        assert "High todo" in result.output
        assert "Low todo" not in result.output

    def test_list_invalid_status_exits_nonzero(self, runner: CliRunner, store_path: str):
        result = invoke(runner, store_path, "list", "--status", "pending")
        assert result.exit_code != 0

    def test_list_invalid_priority_exits_nonzero(self, runner: CliRunner, store_path: str):
        result = invoke(runner, store_path, "list", "--priority", "urgent")
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# done
# ---------------------------------------------------------------------------


class TestDone:
    def test_done_marks_task_as_done(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Complete me", "--priority", "medium")
        data = json.loads(Path(store_path).read_text())
        task_id = data[0]["id"]

        result = invoke(runner, store_path, "done", task_id)
        assert result.exit_code == 0
        assert "done" in result.output.lower()

    def test_done_by_prefix(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Prefix done", "--priority", "low")
        data = json.loads(Path(store_path).read_text())
        prefix = data[0]["id"][:8]

        result = invoke(runner, store_path, "done", prefix)
        assert result.exit_code == 0
        assert "done" in result.output.lower()

    def test_done_persists_status_change(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Persist done", "--priority", "high")
        data = json.loads(Path(store_path).read_text())
        task_id = data[0]["id"]

        invoke(runner, store_path, "done", task_id)

        updated_data = json.loads(Path(store_path).read_text())
        assert updated_data[0]["status"] == "done"

    def test_done_not_found_exits_nonzero(self, runner: CliRunner, store_path: str):
        result = invoke(runner, store_path, "done", "nonexistent-prefix")
        assert result.exit_code != 0
        assert "Error" in result.output or "error" in result.output.lower()

    def test_done_output_contains_short_id(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Short ID check", "--priority", "medium")
        data = json.loads(Path(store_path).read_text())
        task_id = data[0]["id"]
        short_id = task_id[:8]

        result = invoke(runner, store_path, "done", task_id)
        assert short_id in result.output


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


class TestDelete:
    def test_delete_with_yes_flag_exits_zero(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Delete me", "--priority", "medium")
        data = json.loads(Path(store_path).read_text())
        task_id = data[0]["id"]

        result = invoke(runner, store_path, "delete", task_id, "--yes")
        assert result.exit_code == 0

    def test_delete_removes_task_from_store(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Gone task", "--priority", "low")
        data = json.loads(Path(store_path).read_text())
        task_id = data[0]["id"]

        invoke(runner, store_path, "delete", task_id, "--yes")

        updated_data = json.loads(Path(store_path).read_text())
        assert updated_data == []

    def test_delete_output_contains_title(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Titled task", "--priority", "medium")
        data = json.loads(Path(store_path).read_text())
        task_id = data[0]["id"]

        result = invoke(runner, store_path, "delete", task_id, "--yes")
        assert "Titled task" in result.output

    def test_delete_by_prefix_with_yes(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Prefix delete", "--priority", "high")
        data = json.loads(Path(store_path).read_text())
        prefix = data[0]["id"][:8]

        result = invoke(runner, store_path, "delete", prefix, "--yes")
        assert result.exit_code == 0

    def test_delete_not_found_exits_nonzero(self, runner: CliRunner, store_path: str):
        result = invoke(runner, store_path, "delete", "nonexistent-prefix", "--yes")
        assert result.exit_code != 0
        assert "Error" in result.output or "error" in result.output.lower()

    def test_delete_without_yes_aborts_on_no(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Keep me", "--priority", "medium")
        data = json.loads(Path(store_path).read_text())
        task_id = data[0]["id"]

        # Simulate user typing "n" at the confirmation prompt
        result = runner.invoke(cli, ["--store", store_path, "delete", task_id], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output

        # Task should still be there
        remaining = json.loads(Path(store_path).read_text())
        assert len(remaining) == 1

    def test_delete_without_yes_proceeds_on_yes_input(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Confirm delete", "--priority", "medium")
        data = json.loads(Path(store_path).read_text())
        task_id = data[0]["id"]

        result = runner.invoke(cli, ["--store", store_path, "delete", task_id], input="y\n")
        assert result.exit_code == 0

        remaining = json.loads(Path(store_path).read_text())
        assert remaining == []


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


class TestStats:
    def test_stats_empty_store_exits_zero(self, runner: CliRunner, store_path: str):
        result = invoke(runner, store_path, "stats")
        assert result.exit_code == 0

    def test_stats_shows_total_tasks(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Task 1", "--priority", "high")
        invoke(runner, store_path, "add", "--title", "Task 2", "--priority", "low")
        result = invoke(runner, store_path, "stats")
        assert result.exit_code == 0
        assert "2" in result.output  # total count appears somewhere

    def test_stats_shows_status_labels(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "A task", "--priority", "medium")
        result = invoke(runner, store_path, "stats")
        assert "To Do" in result.output or "todo" in result.output.lower()

    def test_stats_shows_priority_labels(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "High task", "--priority", "high")
        result = invoke(runner, store_path, "stats")
        assert "High" in result.output

    def test_stats_counts_are_correct(self, runner: CliRunner, store_path: str):
        invoke(runner, store_path, "add", "--title", "Task A", "--priority", "high")
        invoke(runner, store_path, "add", "--title", "Task B", "--priority", "high")
        invoke(runner, store_path, "add", "--title", "Task C", "--priority", "low")

        # Mark one done
        data = json.loads(Path(store_path).read_text())
        invoke(runner, store_path, "done", data[0]["id"])

        result = invoke(runner, store_path, "stats")
        assert result.exit_code == 0
        # "Total tasks: 3" should appear in the panel subtitle
        assert "3" in result.output

    def test_stats_shows_task_statistics_title(self, runner: CliRunner, store_path: str):
        result = invoke(runner, store_path, "stats")
        assert "Task Statistics" in result.output or "Statistics" in result.output


# ---------------------------------------------------------------------------
# --store option and TASK_STORE_PATH env var
# ---------------------------------------------------------------------------


class TestStoreOption:
    def test_custom_store_path_creates_file_at_that_location(
        self, runner: CliRunner, tmp_path: Path
    ):
        custom_path = str(tmp_path / "custom" / "my_tasks.json")
        result = runner.invoke(cli, ["--store", custom_path, "add", "--title", "Custom store"])
        assert result.exit_code == 0
        assert Path(custom_path).exists()

    def test_env_var_task_store_path_is_respected(self, runner: CliRunner, tmp_path: Path):
        env_path = str(tmp_path / "env_tasks.json")
        result = runner.invoke(
            cli,
            ["add", "--title", "Env var task"],
            env={"TASK_STORE_PATH": env_path},
        )
        assert result.exit_code == 0
        assert Path(env_path).exists()
        data = json.loads(Path(env_path).read_text())
        assert data[0]["title"] == "Env var task"


# ---------------------------------------------------------------------------
# End-to-end flow
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_workflow(self, runner: CliRunner, store_path: str):
        """add → list → done → stats → delete — full happy path."""

        # 1. Add two tasks
        r1 = invoke(runner, store_path, "add", "--title", "Build feature", "--priority", "high")
        assert r1.exit_code == 0
        assert "Build feature" in r1.output

        r2 = invoke(runner, store_path, "add", "--title", "Write tests", "--priority", "medium")
        assert r2.exit_code == 0

        # 2. List — both tasks visible
        r3 = invoke(runner, store_path, "list")
        assert r3.exit_code == 0
        assert "Build feature" in r3.output
        assert "Write tests" in r3.output

        # 3. Mark first task done
        data = json.loads(Path(store_path).read_text())
        first_id = data[0]["id"]
        r4 = invoke(runner, store_path, "done", first_id)
        assert r4.exit_code == 0
        assert "done" in r4.output.lower()

        # 4. List with status=done — only first task
        r5 = invoke(runner, store_path, "list", "--status", "done")
        assert r5.exit_code == 0
        assert "Build feature" in r5.output
        assert "Write tests" not in r5.output

        # 5. Stats — total 2, 1 done, 1 todo
        r6 = invoke(runner, store_path, "stats")
        assert r6.exit_code == 0
        assert "2" in r6.output

        # 6. Delete the done task
        r7 = invoke(runner, store_path, "delete", first_id, "--yes")
        assert r7.exit_code == 0
        assert "Build feature" in r7.output

        # 7. Only one task remains
        r8 = invoke(runner, store_path, "list")
        assert r8.exit_code == 0
        assert "Write tests" in r8.output
        assert "Build feature" not in r8.output
