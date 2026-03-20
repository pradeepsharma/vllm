"""
test_cli.py — CLI tests for the Task Tracker (cli.py).

Uses Click's CliRunner to invoke commands in-process with a temporary
store file, verifying both exit codes and meaningful output content.

Commands tested:
  - cli (group): --help, --store option, TASK_STORE_PATH env var
  - add:    happy path, --priority flag, missing --title, invalid priority
  - list:   no tasks, all tasks, --status filter, --priority filter, combined
  - done:   happy path, partial ID, nonexistent ID
  - delete: --yes flag, confirmation prompt (y/n), nonexistent ID
  - stats:  empty store, populated store counts
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

# Ensure the task_tracker package root is on sys.path so that
# `from cli import cli` and `from task_store import TaskStore` work.
_PACKAGE_ROOT = str(Path(__file__).parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)

from cli import cli  # noqa: E402
from task_store import TaskStore  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_runner() -> CliRunner:
    """Return a CliRunner for testing CLI commands."""
    return CliRunner()


def invoke(runner: CliRunner, store_path: str, *args: str, input: str | None = None):
    """Invoke the CLI with ``--store <store_path>`` prepended to *args*."""
    return runner.invoke(cli, ["--store", store_path, *args], input=input, catch_exceptions=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def runner() -> CliRunner:
    return make_runner()


@pytest.fixture()
def store_file(tmp_path: Path) -> str:
    """Return the path to a temporary tasks.json file (does not exist yet)."""
    return str(tmp_path / "tasks.json")


@pytest.fixture()
def populated_store_file(tmp_path: Path) -> tuple[str, list[dict]]:
    """Return (store_path, tasks) with three pre-added tasks."""
    store_path = str(tmp_path / "tasks.json")
    store = TaskStore(store_path=store_path)
    t1 = store.add_task("Fix login bug", priority="high")
    t2 = store.add_task("Write docs", priority="low")
    t3 = store.add_task("Deploy to prod", priority="medium")
    return store_path, [t1, t2, t3]


# ===========================================================================
# CLI group — help and store option
# ===========================================================================


class TestCliGroup:
    def test_help_exits_zero(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    def test_help_mentions_task_tracker(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        assert "Task Tracker" in result.output

    def test_help_lists_commands(self, runner: CliRunner) -> None:
        result = runner.invoke(cli, ["--help"])
        for cmd in ("add", "list", "done", "delete", "stats"):
            assert cmd in result.output

    def test_store_option_creates_file_at_given_path(
        self, runner: CliRunner, store_file: str
    ) -> None:
        invoke(runner, store_file, "add", "--title", "Test task")
        assert os.path.exists(store_file)

    def test_task_store_path_env_var_is_respected(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        env_store = str(tmp_path / "env_tasks.json")
        result = runner.invoke(
            cli,
            ["add", "--title", "Env task"],
            env={"TASK_STORE_PATH": env_store},
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert os.path.exists(env_store)


# ===========================================================================
# add command
# ===========================================================================


class TestAddCommand:
    def test_add_exits_zero(self, runner: CliRunner, store_file: str) -> None:
        result = invoke(runner, store_file, "add", "--title", "My task")
        assert result.exit_code == 0

    def test_add_output_contains_task_added(
        self, runner: CliRunner, store_file: str
    ) -> None:
        result = invoke(runner, store_file, "add", "--title", "My task")
        assert "Task added" in result.output or "Task Added" in result.output

    def test_add_output_contains_title(self, runner: CliRunner, store_file: str) -> None:
        result = invoke(runner, store_file, "add", "--title", "Fix the bug")
        assert "Fix the bug" in result.output

    def test_add_output_contains_task_id(self, runner: CliRunner, store_file: str) -> None:
        result = invoke(runner, store_file, "add", "--title", "ID check")
        # The full UUID should appear in the output
        store = TaskStore(store_path=store_file)
        tasks = store.list_tasks()
        assert tasks[0]["id"] in result.output

    def test_add_default_priority_is_medium(
        self, runner: CliRunner, store_file: str
    ) -> None:
        invoke(runner, store_file, "add", "--title", "Default prio")
        store = TaskStore(store_path=store_file)
        tasks = store.list_tasks()
        assert tasks[0]["priority"] == "medium"

    def test_add_with_high_priority(self, runner: CliRunner, store_file: str) -> None:
        invoke(runner, store_file, "add", "--title", "Urgent", "--priority", "high")
        store = TaskStore(store_path=store_file)
        tasks = store.list_tasks()
        assert tasks[0]["priority"] == "high"

    def test_add_with_low_priority(self, runner: CliRunner, store_file: str) -> None:
        invoke(runner, store_file, "add", "--title", "Chill", "--priority", "low")
        store = TaskStore(store_path=store_file)
        tasks = store.list_tasks()
        assert tasks[0]["priority"] == "low"

    def test_add_short_flag_t(self, runner: CliRunner, store_file: str) -> None:
        result = invoke(runner, store_file, "add", "-t", "Short flag task")
        assert result.exit_code == 0
        store = TaskStore(store_path=store_file)
        assert store.list_tasks()[0]["title"] == "Short flag task"

    def test_add_short_flag_p(self, runner: CliRunner, store_file: str) -> None:
        result = invoke(runner, store_file, "add", "-t", "Prio task", "-p", "high")
        assert result.exit_code == 0
        store = TaskStore(store_path=store_file)
        assert store.list_tasks()[0]["priority"] == "high"

    def test_add_missing_title_exits_nonzero(
        self, runner: CliRunner, store_file: str
    ) -> None:
        result = runner.invoke(cli, ["--store", store_file, "add"], catch_exceptions=False)
        assert result.exit_code != 0

    def test_add_missing_title_shows_error(
        self, runner: CliRunner, store_file: str
    ) -> None:
        result = runner.invoke(cli, ["--store", store_file, "add"], catch_exceptions=False)
        combined = result.output + (result.stderr or "")
        assert "Missing" in combined or "Error" in combined or "title" in combined.lower()

    def test_add_invalid_priority_exits_nonzero(
        self, runner: CliRunner, store_file: str
    ) -> None:
        result = runner.invoke(
            cli,
            ["--store", store_file, "add", "--title", "Bad prio", "--priority", "urgent"],
            catch_exceptions=False,
        )
        assert result.exit_code != 0

    def test_add_invalid_priority_shows_error(
        self, runner: CliRunner, store_file: str
    ) -> None:
        result = runner.invoke(
            cli,
            ["--store", store_file, "add", "--title", "Bad prio", "--priority", "urgent"],
            catch_exceptions=False,
        )
        combined = result.output + (result.stderr or "")
        assert "invalid" in combined.lower() or "error" in combined.lower() or "choice" in combined.lower()

    def test_add_persists_task_to_json_file(
        self, runner: CliRunner, store_file: str
    ) -> None:
        invoke(runner, store_file, "add", "--title", "Persisted task")
        with open(store_file, encoding="utf-8") as fh:
            data = json.load(fh)
        assert len(data) == 1
        assert data[0]["title"] == "Persisted task"

    def test_add_multiple_tasks_all_persisted(
        self, runner: CliRunner, store_file: str
    ) -> None:
        invoke(runner, store_file, "add", "--title", "Task One")
        invoke(runner, store_file, "add", "--title", "Task Two")
        invoke(runner, store_file, "add", "--title", "Task Three")
        store = TaskStore(store_path=store_file)
        assert len(store.list_tasks()) == 3

    def test_add_output_shows_status_todo(
        self, runner: CliRunner, store_file: str
    ) -> None:
        result = invoke(runner, store_file, "add", "--title", "Status check")
        assert "todo" in result.output


# ===========================================================================
# list command
# ===========================================================================


class TestListCommand:
    def test_list_empty_store_exits_zero(
        self, runner: CliRunner, store_file: str
    ) -> None:
        result = invoke(runner, store_file, "list")
        assert result.exit_code == 0

    def test_list_empty_store_shows_no_tasks_message(
        self, runner: CliRunner, store_file: str
    ) -> None:
        result = invoke(runner, store_file, "list")
        assert "No tasks" in result.output

    def test_list_shows_all_tasks(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        result = invoke(runner, store_path, "list")
        assert result.exit_code == 0
        for task in tasks:
            assert task["title"] in result.output

    def test_list_shows_task_count(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, _ = populated_store_file
        result = invoke(runner, store_path, "list")
        assert "3" in result.output  # "3 task(s) shown"

    def test_list_shows_short_ids(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        result = invoke(runner, store_path, "list")
        for task in tasks:
            assert task["id"][:8] in result.output

    def test_list_filter_by_status_todo(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        result = invoke(runner, store_path, "list", "--status", "todo")
        assert result.exit_code == 0
        # All three tasks are todo initially
        assert "3" in result.output

    def test_list_filter_by_status_done_shows_only_done(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        # Mark first task done via store directly
        store = TaskStore(store_path=store_path)
        store.update_status(tasks[0]["id"], "done")

        result = invoke(runner, store_path, "list", "--status", "done")
        assert result.exit_code == 0
        assert tasks[0]["title"] in result.output
        assert tasks[1]["title"] not in result.output

    def test_list_filter_by_priority_high(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        result = invoke(runner, store_path, "list", "--priority", "high")
        assert result.exit_code == 0
        assert "Fix login bug" in result.output
        assert "Write docs" not in result.output

    def test_list_filter_by_priority_low(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        result = invoke(runner, store_path, "list", "--priority", "low")
        assert result.exit_code == 0
        assert "Write docs" in result.output
        assert "Fix login bug" not in result.output

    def test_list_combined_filter_no_match(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, _ = populated_store_file
        result = invoke(runner, store_path, "list", "--status", "done", "--priority", "high")
        assert result.exit_code == 0
        assert "No tasks" in result.output

    def test_list_combined_filter_with_match(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        result = invoke(runner, store_path, "list", "--status", "todo", "--priority", "medium")
        assert result.exit_code == 0
        assert "Deploy to prod" in result.output

    def test_list_short_flags(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, _ = populated_store_file
        result = invoke(runner, store_path, "list", "-s", "todo", "-p", "high")
        assert result.exit_code == 0
        assert "Fix login bug" in result.output

    def test_list_help_exits_zero(self, runner: CliRunner, store_file: str) -> None:
        result = invoke(runner, store_file, "list", "--help")
        assert result.exit_code == 0

    def test_list_shows_priority_values(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, _ = populated_store_file
        result = invoke(runner, store_path, "list")
        # At least one of the priority values should appear
        assert any(p in result.output for p in ("high", "low", "medium"))


# ===========================================================================
# done command
# ===========================================================================


class TestDoneCommand:
    def test_done_exits_zero(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        result = invoke(runner, store_path, "done", tasks[0]["id"])
        assert result.exit_code == 0

    def test_done_output_confirms_task_marked_done(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        result = invoke(runner, store_path, "done", tasks[0]["id"])
        assert "done" in result.output.lower()

    def test_done_output_shows_task_title(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        result = invoke(runner, store_path, "done", tasks[0]["id"])
        assert tasks[0]["title"] in result.output

    def test_done_output_shows_task_id(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        result = invoke(runner, store_path, "done", tasks[0]["id"])
        assert tasks[0]["id"] in result.output

    def test_done_actually_updates_status_in_store(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        invoke(runner, store_path, "done", tasks[0]["id"])
        store = TaskStore(store_path=store_path)
        updated = store.list_tasks(status="done")
        assert len(updated) == 1
        assert updated[0]["id"] == tasks[0]["id"]

    def test_done_with_partial_id_prefix(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        prefix = tasks[1]["id"][:8]
        result = invoke(runner, store_path, "done", prefix)
        assert result.exit_code == 0
        store = TaskStore(store_path=store_path)
        done_tasks = store.list_tasks(status="done")
        assert done_tasks[0]["id"] == tasks[1]["id"]

    def test_done_nonexistent_id_exits_nonzero(
        self, runner: CliRunner, store_file: str
    ) -> None:
        # Add a task so the store is not empty
        store = TaskStore(store_path=store_file)
        store.add_task("Some task")
        result = runner.invoke(
            cli,
            ["--store", store_file, "done", "zzz-no-match-zzz"],
            catch_exceptions=False,
        )
        assert result.exit_code != 0

    def test_done_nonexistent_id_shows_error_message(
        self, runner: CliRunner, store_file: str
    ) -> None:
        store = TaskStore(store_path=store_file)
        store.add_task("Some task")
        result = runner.invoke(
            cli,
            ["--store", store_file, "done", "zzz-no-match-zzz"],
            catch_exceptions=False,
        )
        assert "Error" in result.output or "No task" in result.output

    def test_done_help_exits_zero(self, runner: CliRunner, store_file: str) -> None:
        result = invoke(runner, store_file, "done", "--help")
        assert result.exit_code == 0


# ===========================================================================
# delete command
# ===========================================================================


class TestDeleteCommand:
    def test_delete_with_yes_flag_exits_zero(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        result = invoke(runner, store_path, "delete", tasks[0]["id"], "--yes")
        assert result.exit_code == 0

    def test_delete_with_yes_flag_shows_deleted_message(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        result = invoke(runner, store_path, "delete", tasks[0]["id"], "--yes")
        assert "deleted" in result.output.lower() or "Deleted" in result.output

    def test_delete_with_yes_flag_shows_task_title(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        result = invoke(runner, store_path, "delete", tasks[0]["id"], "--yes")
        assert tasks[0]["title"] in result.output

    def test_delete_with_yes_flag_removes_task_from_store(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        invoke(runner, store_path, "delete", tasks[0]["id"], "--yes")
        store = TaskStore(store_path=store_path)
        remaining = store.list_tasks()
        assert len(remaining) == 2
        assert all(t["id"] != tasks[0]["id"] for t in remaining)

    def test_delete_short_yes_flag(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        result = invoke(runner, store_path, "delete", tasks[2]["id"], "-y")
        assert result.exit_code == 0
        store = TaskStore(store_path=store_path)
        assert len(store.list_tasks()) == 2

    def test_delete_confirmation_prompt_yes(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        # Simulate user typing "y" at the confirmation prompt
        result = runner.invoke(
            cli,
            ["--store", store_path, "delete", tasks[0]["id"]],
            input="y\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        store = TaskStore(store_path=store_path)
        assert len(store.list_tasks()) == 2

    def test_delete_confirmation_prompt_no_cancels(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        # Simulate user typing "n" at the confirmation prompt
        result = runner.invoke(
            cli,
            ["--store", store_path, "delete", tasks[0]["id"]],
            input="n\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        # Task should still be present
        store = TaskStore(store_path=store_path)
        assert len(store.list_tasks()) == 3

    def test_delete_cancellation_shows_cancelled_message(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        result = runner.invoke(
            cli,
            ["--store", store_path, "delete", tasks[0]["id"]],
            input="n\n",
            catch_exceptions=False,
        )
        assert "cancel" in result.output.lower() or "Deletion cancelled" in result.output

    def test_delete_partial_id_with_yes_flag(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        prefix = tasks[1]["id"][:8]
        result = invoke(runner, store_path, "delete", prefix, "--yes")
        assert result.exit_code == 0
        store = TaskStore(store_path=store_path)
        remaining_ids = {t["id"] for t in store.list_tasks()}
        assert tasks[1]["id"] not in remaining_ids

    def test_delete_nonexistent_id_exits_nonzero(
        self, runner: CliRunner, store_file: str
    ) -> None:
        store = TaskStore(store_path=store_file)
        store.add_task("Existing task")
        result = runner.invoke(
            cli,
            ["--store", store_file, "delete", "zzz-no-match-zzz", "--yes"],
            catch_exceptions=False,
        )
        assert result.exit_code != 0

    def test_delete_nonexistent_id_shows_error_message(
        self, runner: CliRunner, store_file: str
    ) -> None:
        store = TaskStore(store_path=store_file)
        store.add_task("Existing task")
        result = runner.invoke(
            cli,
            ["--store", store_file, "delete", "zzz-no-match-zzz", "--yes"],
            catch_exceptions=False,
        )
        assert "Error" in result.output or "No task" in result.output

    def test_delete_help_exits_zero(self, runner: CliRunner, store_file: str) -> None:
        result = invoke(runner, store_file, "delete", "--help")
        assert result.exit_code == 0


# ===========================================================================
# stats command
# ===========================================================================


class TestStatsCommand:
    def test_stats_empty_store_exits_zero(
        self, runner: CliRunner, store_file: str
    ) -> None:
        result = invoke(runner, store_file, "stats")
        assert result.exit_code == 0

    def test_stats_shows_statistics_heading(
        self, runner: CliRunner, store_file: str
    ) -> None:
        result = invoke(runner, store_file, "stats")
        assert "Statistics" in result.output or "stats" in result.output.lower()

    def test_stats_shows_status_labels(
        self, runner: CliRunner, store_file: str
    ) -> None:
        result = invoke(runner, store_file, "stats")
        assert "todo" in result.output
        assert "done" in result.output

    def test_stats_shows_priority_labels(
        self, runner: CliRunner, store_file: str
    ) -> None:
        result = invoke(runner, store_file, "stats")
        assert "low" in result.output
        assert "medium" in result.output
        assert "high" in result.output

    def test_stats_shows_zero_counts_for_empty_store(
        self, runner: CliRunner, store_file: str
    ) -> None:
        result = invoke(runner, store_file, "stats")
        # All counts should be 0
        assert "0" in result.output

    def test_stats_shows_total_task_count(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, _ = populated_store_file
        result = invoke(runner, store_path, "stats")
        assert result.exit_code == 0
        assert "3" in result.output  # total tasks

    def test_stats_counts_by_status_correctly(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, tasks = populated_store_file
        # Mark one task done, one in_progress
        store = TaskStore(store_path=store_path)
        store.update_status(tasks[0]["id"], "done")
        store.update_status(tasks[1]["id"], "in_progress")

        result = invoke(runner, store_path, "stats")
        assert result.exit_code == 0
        # Output should contain the counts: 1 todo, 1 in_progress, 1 done
        assert "1" in result.output

    def test_stats_counts_by_priority_correctly(
        self,
        runner: CliRunner,
        populated_store_file: tuple[str, list[dict]],
    ) -> None:
        store_path, _ = populated_store_file
        result = invoke(runner, store_path, "stats")
        assert result.exit_code == 0
        # populated_store has 1 high, 1 low, 1 medium — each count is 1
        assert "1" in result.output

    def test_stats_help_exits_zero(self, runner: CliRunner, store_file: str) -> None:
        result = invoke(runner, store_file, "stats", "--help")
        assert result.exit_code == 0


# ===========================================================================
# End-to-end integration test — full user workflow
# ===========================================================================


class TestEndToEndWorkflow:
    def test_full_workflow_add_list_done_delete(
        self, runner: CliRunner, store_file: str
    ) -> None:
        """
        Full user workflow:
          1. Add two tasks
          2. List all tasks — verify both appear
          3. Mark first task done
          4. List with status=done — verify only first task appears
          5. Delete second task with --yes
          6. List all — verify only first (done) task remains
          7. Stats — verify counts reflect final state
        """
        # 1. Add two tasks
        r1 = invoke(runner, store_file, "add", "--title", "Task Alpha", "--priority", "high")
        assert r1.exit_code == 0
        r2 = invoke(runner, store_file, "add", "--title", "Task Beta", "--priority", "low")
        assert r2.exit_code == 0

        store = TaskStore(store_path=store_file)
        tasks = store.list_tasks()
        assert len(tasks) == 2
        alpha_id = tasks[0]["id"]
        beta_id = tasks[1]["id"]

        # 2. List all tasks
        r3 = invoke(runner, store_file, "list")
        assert r3.exit_code == 0
        assert "Task Alpha" in r3.output
        assert "Task Beta" in r3.output

        # 3. Mark first task done
        r4 = invoke(runner, store_file, "done", alpha_id)
        assert r4.exit_code == 0
        assert "done" in r4.output.lower()

        # 4. List with status=done
        r5 = invoke(runner, store_file, "list", "--status", "done")
        assert r5.exit_code == 0
        assert "Task Alpha" in r5.output
        assert "Task Beta" not in r5.output

        # 5. Delete second task
        r6 = invoke(runner, store_file, "delete", beta_id, "--yes")
        assert r6.exit_code == 0

        # 6. List all — only Task Alpha remains
        r7 = invoke(runner, store_file, "list")
        assert r7.exit_code == 0
        assert "Task Alpha" in r7.output
        assert "Task Beta" not in r7.output

        # 7. Stats — 1 total, 1 done, 0 todo, 0 in_progress
        r8 = invoke(runner, store_file, "stats")
        assert r8.exit_code == 0
        assert "1" in r8.output  # at least one count of 1

    def test_add_then_list_shows_correct_priority_and_status(
        self, runner: CliRunner, store_file: str
    ) -> None:
        invoke(runner, store_file, "add", "--title", "High prio task", "--priority", "high")
        result = invoke(runner, store_file, "list")
        assert "High prio task" in result.output
        assert "high" in result.output
        assert "todo" in result.output

    def test_done_then_stats_reflects_updated_counts(
        self, runner: CliRunner, store_file: str
    ) -> None:
        invoke(runner, store_file, "add", "--title", "Task 1")
        invoke(runner, store_file, "add", "--title", "Task 2")
        store = TaskStore(store_path=store_file)
        tasks = store.list_tasks()
        invoke(runner, store_file, "done", tasks[0]["id"])

        result = invoke(runner, store_file, "stats")
        assert result.exit_code == 0
        # Should show 1 done and 1 todo
        assert "1" in result.output
