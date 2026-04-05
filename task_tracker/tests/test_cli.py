"""
Tests for task_tracker/cli.py

Uses Click's CliRunner to invoke commands in-process without spawning subprocesses.
Each test uses a temporary tasks.json file via a monkeypatched _get_store function.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest
from click.testing import CliRunner

from task_tracker.cli import cli
from task_tracker.task_store import TaskStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    """CliRunner that captures stderr separately."""
    return CliRunner(mix_stderr=False)


@pytest.fixture
def isolated_runner(tmp_path, monkeypatch):
    """
    Return a (CliRunner, TaskStore, filepath) tuple.
    Monkeypatches cli._get_store so all CLI commands use the temp store.
    """
    filepath = str(tmp_path / "tasks.json")
    store = TaskStore(filepath=filepath)

    import task_tracker.cli as cli_module
    monkeypatch.setattr(cli_module, "_get_store", lambda: TaskStore(filepath=filepath))

    return CliRunner(mix_stderr=False), store, filepath


# ---------------------------------------------------------------------------
# add command
# ---------------------------------------------------------------------------


def test_add_command_success(isolated_runner):
    runner, store, _ = isolated_runner
    result = runner.invoke(cli, ["add", "--title", "Fix login bug", "--priority", "high"])
    assert result.exit_code == 0
    assert "Task added" in result.output
    assert "Fix login bug" in result.output
    assert "high" in result.output


def test_add_command_default_priority(isolated_runner):
    runner, store, _ = isolated_runner
    result = runner.invoke(cli, ["add", "--title", "Write docs"])
    assert result.exit_code == 0
    assert "medium" in result.output
    assert "Write docs" in result.output


def test_add_command_low_priority(isolated_runner):
    runner, store, _ = isolated_runner
    result = runner.invoke(cli, ["add", "--title", "Update deps", "--priority", "low"])
    assert result.exit_code == 0
    assert "low" in result.output


def test_add_command_shows_short_id(isolated_runner):
    runner, store, _ = isolated_runner
    result = runner.invoke(cli, ["add", "--title", "ID check"])
    assert result.exit_code == 0
    tasks = store.list_tasks()
    assert len(tasks) == 1
    short_id = tasks[0]["id"][:8]
    assert short_id in result.output


def test_add_command_invalid_priority_exits_2(isolated_runner):
    runner, store, _ = isolated_runner
    result = runner.invoke(cli, ["add", "--title", "Bad", "--priority", "urgent"])
    assert result.exit_code == 2
    # With mix_stderr=False, Click error output goes to stderr
    combined = result.output + (result.stderr if result.stderr_bytes else "")
    assert "Invalid value" in combined or "urgent" in combined


def test_add_command_missing_title_exits_2(isolated_runner):
    runner, store, _ = isolated_runner
    result = runner.invoke(cli, ["add"])
    assert result.exit_code == 2


def test_add_command_persists_to_file(isolated_runner):
    runner, store, filepath = isolated_runner
    runner.invoke(cli, ["add", "--title", "Persist test", "--priority", "high"])
    with open(filepath) as f:
        data = json.load(f)
    assert len(data) == 1
    assert data[0]["title"] == "Persist test"
    assert data[0]["priority"] == "high"


def test_add_command_checkmark_in_output(isolated_runner):
    runner, store, _ = isolated_runner
    result = runner.invoke(cli, ["add", "--title", "Checkmark test"])
    assert result.exit_code == 0
    assert "✅" in result.output or "Task added" in result.output


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


def test_list_command_shows_table(isolated_runner):
    runner, store, _ = isolated_runner
    store.add_task("Task A", priority="high")
    store.add_task("Task B", priority="low")
    result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0
    # Rich table may wrap long titles; check for partial matches
    assert "Task A" in result.output
    assert "Task B" in result.output


def test_list_command_shows_columns(isolated_runner):
    runner, store, _ = isolated_runner
    store.add_task("Column check")
    result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0
    assert "ID" in result.output
    assert "Title" in result.output
    assert "Status" in result.output
    assert "Priority" in result.output
    assert "Created" in result.output


def test_list_command_empty_store(isolated_runner):
    runner, store, _ = isolated_runner
    result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0
    assert "No tasks found" in result.output


def test_list_command_filter_by_status(isolated_runner):
    runner, store, _ = isolated_runner
    t1 = store.add_task("Todo task")
    t2 = store.add_task("Done task")
    store.update_status(t2["id"][:8], "done")

    result = runner.invoke(cli, ["list", "--status", "todo"])
    assert result.exit_code == 0
    assert "Todo task" in result.output
    assert "Done task" not in result.output


def test_list_command_filter_by_priority(isolated_runner):
    runner, store, _ = isolated_runner
    store.add_task("High task", priority="high")
    store.add_task("Low task", priority="low")

    result = runner.invoke(cli, ["list", "--priority", "high"])
    assert result.exit_code == 0
    assert "High task" in result.output
    assert "Low task" not in result.output


def test_list_command_three_tasks_all_present(isolated_runner):
    runner, store, _ = isolated_runner
    store.add_task("Alpha", priority="high")
    store.add_task("Beta")
    store.add_task("Gamma", priority="low")

    result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0
    # Short titles won't be wrapped by Rich
    assert "Alpha" in result.output
    assert "Beta" in result.output
    assert "Gamma" in result.output


def test_list_command_shows_status_values(isolated_runner):
    runner, store, _ = isolated_runner
    t = store.add_task("Status check")
    store.update_status(t["id"][:8], "done")
    result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0
    assert "done" in result.output


def test_list_command_shows_priority_values(isolated_runner):
    runner, store, _ = isolated_runner
    store.add_task("Priority check", priority="high")
    result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0
    assert "high" in result.output


# ---------------------------------------------------------------------------
# done command
# ---------------------------------------------------------------------------


def test_done_command_marks_task_done(isolated_runner):
    runner, store, _ = isolated_runner
    task = store.add_task("Mark me done")
    result = runner.invoke(cli, ["done", task["id"][:8]])
    assert result.exit_code == 0
    assert "Task marked as done" in result.output
    assert "Mark me done" in result.output


def test_done_command_persists_status(isolated_runner):
    runner, store, _ = isolated_runner
    task = store.add_task("Persist done")
    runner.invoke(cli, ["done", task["id"][:8]])
    tasks = store.list_tasks()
    assert tasks[0]["status"] == "done"


def test_done_command_not_found_exits_1(isolated_runner):
    runner, store, _ = isolated_runner
    result = runner.invoke(cli, ["done", "zzzzzzz"])
    assert result.exit_code == 1


def test_done_command_not_found_error_message(isolated_runner):
    runner, store, _ = isolated_runner
    result = runner.invoke(cli, ["done", "zzzzzzz"])
    assert result.exit_code == 1
    # With mix_stderr=False, error goes to stderr bytes
    stderr_text = result.stderr if result.stderr_bytes else ""
    combined = result.output + stderr_text
    assert "zzzzzzz" in combined or "No task found" in combined


def test_done_command_checkmark_in_output(isolated_runner):
    runner, store, _ = isolated_runner
    task = store.add_task("Checkmark done")
    result = runner.invoke(cli, ["done", task["id"][:8]])
    assert result.exit_code == 0
    assert "✅" in result.output or "marked as done" in result.output


# ---------------------------------------------------------------------------
# delete command
# ---------------------------------------------------------------------------


def test_delete_command_with_confirmation(isolated_runner):
    runner, store, _ = isolated_runner
    task = store.add_task("Delete me")
    result = runner.invoke(cli, ["delete", task["id"][:8]], input="y\n")
    assert result.exit_code == 0
    assert "Task deleted" in result.output
    assert len(store.list_tasks()) == 0


def test_delete_command_aborted(isolated_runner):
    runner, store, _ = isolated_runner
    task = store.add_task("Keep me")
    result = runner.invoke(cli, ["delete", task["id"][:8]], input="n\n")
    assert result.exit_code == 0
    assert "Aborted" in result.output
    assert len(store.list_tasks()) == 1


def test_delete_command_shows_title_in_prompt(isolated_runner):
    runner, store, _ = isolated_runner
    task = store.add_task("Prompt title check")
    result = runner.invoke(cli, ["delete", task["id"][:8]], input="n\n")
    assert "Prompt title check" in result.output


def test_delete_command_not_found_exits_1(isolated_runner):
    runner, store, _ = isolated_runner
    result = runner.invoke(cli, ["delete", "zzzzzzz"], input="y\n")
    assert result.exit_code == 1


def test_delete_command_persists_removal(isolated_runner):
    runner, store, _ = isolated_runner
    t1 = store.add_task("Keep me")
    t2 = store.add_task("Delete me")
    runner.invoke(cli, ["delete", t2["id"][:8]], input="y\n")
    remaining = store.list_tasks()
    assert len(remaining) == 1
    assert remaining[0]["id"] == t1["id"]


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------


def test_stats_command_empty_store(isolated_runner):
    runner, store, _ = isolated_runner
    result = runner.invoke(cli, ["stats"])
    assert result.exit_code == 0
    assert "Total" in result.output
    assert "0" in result.output


def test_stats_command_with_tasks(isolated_runner):
    runner, store, _ = isolated_runner
    store.add_task("High task", priority="high")
    t2 = store.add_task("Medium done", priority="medium")
    store.add_task("Low task", priority="low")
    store.update_status(t2["id"][:8], "done")

    result = runner.invoke(cli, ["stats"])
    assert result.exit_code == 0
    assert "Total" in result.output
    assert "3" in result.output
    assert "todo" in result.output
    assert "done" in result.output


def test_stats_command_shows_priority_counts(isolated_runner):
    runner, store, _ = isolated_runner
    store.add_task("High 1", priority="high")
    store.add_task("High 2", priority="high")
    store.add_task("Low 1", priority="low")

    result = runner.invoke(cli, ["stats"])
    assert result.exit_code == 0
    assert "high" in result.output
    assert "low" in result.output


def test_stats_command_shows_panel(isolated_runner):
    runner, store, _ = isolated_runner
    store.add_task("Panel check")
    result = runner.invoke(cli, ["stats"])
    assert result.exit_code == 0
    assert "Task Stats" in result.output


def test_stats_command_shows_in_progress(isolated_runner):
    runner, store, _ = isolated_runner
    t = store.add_task("In progress task")
    store.update_status(t["id"][:8], "in_progress")
    result = runner.invoke(cli, ["stats"])
    assert result.exit_code == 0
    assert "in_progress" in result.output


# ---------------------------------------------------------------------------
# Full end-to-end flow
# ---------------------------------------------------------------------------


def test_full_workflow(isolated_runner):
    """
    End-to-end: add 3 tasks → list → mark one done → check stats → delete one.
    """
    runner, store, _ = isolated_runner

    # Add tasks (use short titles to avoid Rich table wrapping)
    r1 = runner.invoke(cli, ["add", "--title", "Bug fix", "--priority", "high"])
    assert r1.exit_code == 0
    assert "Bug fix" in r1.output
    assert "high" in r1.output

    r2 = runner.invoke(cli, ["add", "--title", "Docs"])
    assert r2.exit_code == 0
    assert "Docs" in r2.output

    r3 = runner.invoke(cli, ["add", "--title", "Deps", "--priority", "low"])
    assert r3.exit_code == 0
    assert "Deps" in r3.output

    # List — should show 3 tasks
    r_list = runner.invoke(cli, ["list"])
    assert r_list.exit_code == 0
    assert "Bug fix" in r_list.output
    assert "Docs" in r_list.output
    assert "Deps" in r_list.output

    # Mark first task done
    tasks = store.list_tasks()
    assert len(tasks) == 3
    first_id = tasks[0]["id"][:8]
    r_done = runner.invoke(cli, ["done", first_id])
    assert r_done.exit_code == 0
    assert "Task marked as done" in r_done.output

    # Stats — total 3, done 1, todo 2
    r_stats = runner.invoke(cli, ["stats"])
    assert r_stats.exit_code == 0
    assert "3" in r_stats.output
    assert "done" in r_stats.output

    # Delete a task
    tasks = store.list_tasks()
    last_id = tasks[-1]["id"][:8]
    r_del = runner.invoke(cli, ["delete", last_id], input="y\n")
    assert r_del.exit_code == 0
    assert "Task deleted" in r_del.output

    # Verify 2 tasks remain
    remaining = store.list_tasks()
    assert len(remaining) == 2
