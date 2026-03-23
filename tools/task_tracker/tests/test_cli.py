"""
tests/test_cli.py — Integration tests for the Click CLI (cli.py).

Uses Click's CliRunner to invoke the CLI as a real subprocess-like call,
exercising the full command pipeline including TaskStore I/O.

All tests that interact with the store use the TASK_STORE_PATH environment
variable (via CliRunner's env parameter) so each test gets an isolated store.

Covers:
  - add: happy path, missing title, invalid priority, default priority
  - list: empty store, shows tasks, status filter, priority filter
  - done: success, not found
  - delete: with --yes flag, confirmation abort, not found
  - stats: empty store, populated store
  - --store global option: custom store path
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

# Ensure the task_tracker package directory is importable.
sys.path.insert(0, str(Path(__file__).parent.parent))

from cli import cli  # noqa: E402
from task_store import TaskStore  # noqa: E402


# ---------------------------------------------------------------------------
# add command
# ---------------------------------------------------------------------------


def test_add_command_success(runner: CliRunner, tmp_store_path: str) -> None:
    """cli add --title ... --priority high exits 0 and prints 'Added task'."""
    result = runner.invoke(
        cli,
        ["add", "--title", "Fix bug", "--priority", "high"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    assert result.exit_code == 0
    assert "Added task" in result.output


def test_add_command_missing_title(runner: CliRunner, tmp_store_path: str) -> None:
    """cli add without --title exits with a non-zero code."""
    result = runner.invoke(
        cli,
        ["add", "--priority", "high"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    assert result.exit_code != 0


def test_add_command_invalid_priority(runner: CliRunner, tmp_store_path: str) -> None:
    """cli add with an unrecognised priority exits with a non-zero code."""
    result = runner.invoke(
        cli,
        ["add", "--title", "X", "--priority", "urgent"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    assert result.exit_code != 0


def test_add_command_default_priority(runner: CliRunner, tmp_store_path: str) -> None:
    """cli add without --priority exits 0 (defaults to medium)."""
    result = runner.invoke(
        cli,
        ["add", "--title", "X"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


def test_list_command_empty(runner: CliRunner, tmp_store_path: str) -> None:
    """cli list on an empty store prints 'No tasks found'."""
    result = runner.invoke(
        cli,
        ["list"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    assert result.exit_code == 0
    assert "No tasks found" in result.output


def test_list_command_shows_tasks(runner: CliRunner, tmp_store_path: str) -> None:
    """cli list shows the title of a previously added task."""
    # Add a task first
    runner.invoke(
        cli,
        ["add", "--title", "My listed task", "--priority", "medium"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )

    result = runner.invoke(
        cli,
        ["list"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    assert result.exit_code == 0
    assert "My listed task" in result.output


def test_list_command_filter_status(runner: CliRunner, tmp_store_path: str) -> None:
    """cli list --status todo shows only todo tasks."""
    # Add two tasks; both start as todo
    runner.invoke(
        cli,
        ["add", "--title", "Todo task", "--priority", "medium"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    runner.invoke(
        cli,
        ["add", "--title", "Another task", "--priority", "low"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )

    # Mark the second task done via the store directly
    store = TaskStore(tmp_store_path)
    tasks = store.list_tasks()
    second_id = tasks[1]["id"]
    store.update_status(second_id[:8], "done")

    result = runner.invoke(
        cli,
        ["list", "--status", "todo"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    assert result.exit_code == 0
    assert "Todo task" in result.output
    assert "Another task" not in result.output


def test_list_command_filter_priority(runner: CliRunner, tmp_store_path: str) -> None:
    """cli list --priority high shows only high-priority tasks."""
    runner.invoke(
        cli,
        ["add", "--title", "High task", "--priority", "high"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    runner.invoke(
        cli,
        ["add", "--title", "Low task", "--priority", "low"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )

    result = runner.invoke(
        cli,
        ["list", "--priority", "high"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    assert result.exit_code == 0
    assert "High task" in result.output
    assert "Low task" not in result.output


# ---------------------------------------------------------------------------
# done command
# ---------------------------------------------------------------------------


def test_done_command_success(runner: CliRunner, tmp_store_path: str) -> None:
    """cli done {id[:8]} exits 0 and prints 'marked as done'."""
    # Add a task and retrieve its ID from the JSON file
    runner.invoke(
        cli,
        ["add", "--title", "Complete me", "--priority", "medium"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    data = json.loads(Path(tmp_store_path).read_text())
    task_id = data[0]["id"]

    result = runner.invoke(
        cli,
        ["done", task_id[:8]],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    assert result.exit_code == 0
    assert "marked as done" in result.output


def test_done_command_not_found(runner: CliRunner, tmp_store_path: str) -> None:
    """cli done nonexistent exits non-zero and prints 'Error'."""
    result = runner.invoke(
        cli,
        ["done", "nonexistent"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    assert result.exit_code != 0
    assert "Error" in result.output


# ---------------------------------------------------------------------------
# delete command
# ---------------------------------------------------------------------------


def test_delete_command_with_yes_flag(runner: CliRunner, tmp_store_path: str) -> None:
    """cli delete {id[:8]} --yes exits 0 without prompting."""
    runner.invoke(
        cli,
        ["add", "--title", "Delete me", "--priority", "low"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    data = json.loads(Path(tmp_store_path).read_text())
    task_id = data[0]["id"]

    result = runner.invoke(
        cli,
        ["delete", task_id[:8], "--yes"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    assert result.exit_code == 0


def test_delete_command_confirmation_abort(runner: CliRunner, tmp_store_path: str) -> None:
    """cli delete without --yes and input='n' leaves the task intact."""
    runner.invoke(
        cli,
        ["add", "--title", "Keep me", "--priority", "medium"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    data = json.loads(Path(tmp_store_path).read_text())
    task_id = data[0]["id"]

    # Provide 'n' to the confirmation prompt
    result = runner.invoke(
        cli,
        ["delete", task_id[:8]],
        input="n\n",
        env={"TASK_STORE_PATH": tmp_store_path},
    )

    # Task should still exist
    store = TaskStore(tmp_store_path)
    remaining = store.list_tasks()
    assert len(remaining) == 1
    assert remaining[0]["id"] == task_id


def test_delete_command_not_found(runner: CliRunner, tmp_store_path: str) -> None:
    """cli delete nonexistent --yes exits non-zero."""
    result = runner.invoke(
        cli,
        ["delete", "nonexistent", "--yes"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------


def test_stats_command_empty(runner: CliRunner, tmp_store_path: str) -> None:
    """cli stats on an empty store exits 0 and shows '0' counts."""
    result = runner.invoke(
        cli,
        ["stats"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    assert result.exit_code == 0
    assert "0" in result.output


def test_stats_command_with_tasks(runner: CliRunner, tmp_store_path: str) -> None:
    """cli stats shows correct counts after adding tasks with various statuses/priorities."""
    # Add three tasks
    runner.invoke(
        cli,
        ["add", "--title", "High todo", "--priority", "high"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    runner.invoke(
        cli,
        ["add", "--title", "Medium todo", "--priority", "medium"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    runner.invoke(
        cli,
        ["add", "--title", "Low done", "--priority", "low"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )

    # Mark the third task done
    data = json.loads(Path(tmp_store_path).read_text())
    third_id = data[2]["id"]
    runner.invoke(
        cli,
        ["done", third_id[:8]],
        env={"TASK_STORE_PATH": tmp_store_path},
    )

    result = runner.invoke(
        cli,
        ["stats"],
        env={"TASK_STORE_PATH": tmp_store_path},
    )
    assert result.exit_code == 0
    # Total is 3; counts should appear in the output
    assert "3" in result.output
    # At least one count of 2 (two todo tasks) and one count of 1 (one done)
    assert "2" in result.output
    assert "1" in result.output


# ---------------------------------------------------------------------------
# --store global option
# ---------------------------------------------------------------------------


def test_global_store_option(runner: CliRunner, tmp_path: Path) -> None:
    """cli --store {custom_path} add --title 'X' writes to custom_path."""
    custom_path = str(tmp_path / "custom_tasks.json")

    result = runner.invoke(
        cli,
        ["--store", custom_path, "add", "--title", "Custom store"],
    )
    assert result.exit_code == 0
    assert Path(custom_path).exists()

    data = json.loads(Path(custom_path).read_text())
    assert len(data) == 1
    assert data[0]["title"] == "Custom store"
