"""
test_cli.py — Tests for the CLI interface (cli.py).

Strategy:
  - Tests call main() with an injected --store path so no real tasks.json
    is touched.
  - stdout/stderr are captured via capsys.
  - Return codes are asserted alongside output content.

Coverage:
  - add: happy path (default priority, explicit priority), blank title,
         invalid priority
  - list: empty store, all tasks, --status filter, --priority filter,
          combined filters, no-match filter, invalid filter values
  - update: happy path, invalid status, no match
  - delete: happy path, no match
  - no sub-command: prints help, exits 0
  - integration: full add → list → update → delete flow
  - CLI invoked as a subprocess (real binary invocation)
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# We need to import from the package directory.
# pytest is run from tools/task_tracker so the module is on sys.path.
import cli as cli_module
from cli import main


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def run(args: list[str], store_path: str) -> tuple[int, str, str]:
    """Call main() with *args* prepended by --store <store_path>.

    Returns (exit_code, stdout, stderr).
    """
    full_args = ["--store", store_path] + args
    rc = main(full_args)
    return rc


def run_cap(
    args: list[str], store_path: str, capsys
) -> tuple[int, str, str]:
    """Like run() but also captures and returns (rc, stdout, stderr)."""
    rc = run(args, store_path)
    captured = capsys.readouterr()
    return rc, captured.out, captured.err


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store_file(tmp_path: Path) -> str:
    return str(tmp_path / "tasks.json")


# ===========================================================================
# add sub-command
# ===========================================================================


class TestCmdAdd:
    def test_add_returns_exit_code_0(self, store_file: str, capsys) -> None:
        rc, out, err = run_cap(["add", "My first task"], store_file, capsys)
        assert rc == 0

    def test_add_prints_task_added_message(self, store_file: str, capsys) -> None:
        rc, out, err = run_cap(["add", "My first task"], store_file, capsys)
        assert "Task added" in out

    def test_add_prints_task_title_in_output(self, store_file: str, capsys) -> None:
        rc, out, err = run_cap(["add", "Write unit tests"], store_file, capsys)
        assert "Write unit tests" in out

    def test_add_prints_short_id_in_output(self, store_file: str, capsys) -> None:
        rc, out, err = run_cap(["add", "Short ID check"], store_file, capsys)
        # Short ID is 8 hex chars from a UUID — just verify something bracket-like
        assert "[" in out and "]" in out

    def test_add_default_priority_medium(self, store_file: str, capsys) -> None:
        rc, out, err = run_cap(["add", "Default prio"], store_file, capsys)
        assert "MEDIUM" in out.upper()

    def test_add_explicit_priority_high(self, store_file: str, capsys) -> None:
        rc, out, err = run_cap(
            ["add", "Urgent task", "--priority", "high"], store_file, capsys
        )
        assert rc == 0
        assert "HIGH" in out.upper()

    def test_add_explicit_priority_low(self, store_file: str, capsys) -> None:
        rc, out, err = run_cap(
            ["add", "Chill task", "--priority", "low"], store_file, capsys
        )
        assert rc == 0
        assert "LOW" in out.upper()

    def test_add_blank_title_returns_exit_code_1(
        self, store_file: str, capsys
    ) -> None:
        rc, out, err = run_cap(["add", "   "], store_file, capsys)
        assert rc == 1

    def test_add_blank_title_prints_error_to_stderr(
        self, store_file: str, capsys
    ) -> None:
        rc, out, err = run_cap(["add", "   "], store_file, capsys)
        assert "Error" in err
        assert "empty" in err.lower()

    def test_add_invalid_priority_exits_nonzero(
        self, store_file: str, capsys
    ) -> None:
        # argparse will reject invalid choices before reaching our handler
        with pytest.raises(SystemExit) as exc_info:
            main(["--store", store_file, "add", "Task", "--priority", "critical"])
        assert exc_info.value.code != 0


# ===========================================================================
# list sub-command
# ===========================================================================


class TestCmdList:
    def test_list_empty_store_returns_exit_code_0(
        self, store_file: str, capsys
    ) -> None:
        rc, out, err = run_cap(["list"], store_file, capsys)
        assert rc == 0

    def test_list_empty_store_prints_no_tasks_message(
        self, store_file: str, capsys
    ) -> None:
        rc, out, err = run_cap(["list"], store_file, capsys)
        assert "No tasks found" in out

    def test_list_shows_added_tasks(self, store_file: str, capsys) -> None:
        run(["add", "Alpha task"], store_file)
        run(["add", "Beta task"], store_file)
        rc, out, err = run_cap(["list"], store_file, capsys)
        assert rc == 0
        assert "Alpha task" in out
        assert "Beta task" in out

    def test_list_prints_table_header(self, store_file: str, capsys) -> None:
        run(["add", "Some task"], store_file)
        rc, out, err = run_cap(["list"], store_file, capsys)
        # Header contains column names
        assert "ID" in out
        assert "STATUS" in out
        assert "TITLE" in out

    def test_list_prints_task_count(self, store_file: str, capsys) -> None:
        run(["add", "Task 1"], store_file)
        run(["add", "Task 2"], store_file)
        rc, out, err = run_cap(["list"], store_file, capsys)
        assert "2 tasks" in out

    def test_list_single_task_uses_singular_noun(
        self, store_file: str, capsys
    ) -> None:
        run(["add", "Only task"], store_file)
        rc, out, err = run_cap(["list"], store_file, capsys)
        assert "1 task" in out
        assert "1 tasks" not in out

    def test_list_filter_by_status_todo(self, store_file: str, capsys) -> None:
        run(["add", "Todo task"], store_file)
        # Add a second task and update it so we can get its ID from the store
        from task_store import TaskStore
        ts = TaskStore(store_path=store_file)
        t2 = ts.add_task("Done task")
        ts.update_status(t2["id"], "done")

        rc, out, err = run_cap(["list", "--status", "todo"], store_file, capsys)
        assert rc == 0
        assert "Todo task" in out
        assert "Done task" not in out

    def test_list_filter_by_priority_high(self, store_file: str, capsys) -> None:
        run(["add", "High task", "--priority", "high"], store_file)
        run(["add", "Low task", "--priority", "low"], store_file)
        capsys.readouterr()  # flush add output before asserting on list output
        rc, out, err = run_cap(["list", "--priority", "high"], store_file, capsys)
        assert rc == 0
        assert "High task" in out
        assert "Low task" not in out

    def test_list_combined_filters(self, store_file: str, capsys) -> None:
        from task_store import TaskStore
        ts = TaskStore(store_path=store_file)
        t1 = ts.add_task("High todo", priority="high")
        t2 = ts.add_task("High done", priority="high")
        ts.update_status(t2["id"], "done")
        ts.add_task("Low todo", priority="low")

        rc, out, err = run_cap(
            ["list", "--status", "todo", "--priority", "high"],
            store_file,
            capsys,
        )
        assert rc == 0
        assert "High todo" in out
        assert "High done" not in out
        assert "Low todo" not in out

    def test_list_no_match_filter_shows_no_tasks_message(
        self, store_file: str, capsys
    ) -> None:
        run(["add", "Low task", "--priority", "low"], store_file)
        rc, out, err = run_cap(["list", "--priority", "high"], store_file, capsys)
        assert rc == 0
        assert "No tasks found" in out
        assert "priority=high" in out

    def test_list_no_match_shows_filter_in_message(
        self, store_file: str, capsys
    ) -> None:
        run(["add", "Some task"], store_file)
        rc, out, err = run_cap(["list", "--status", "done"], store_file, capsys)
        assert "status=done" in out

    def test_list_invalid_status_exits_nonzero(
        self, store_file: str, capsys
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--store", store_file, "list", "--status", "pending"])
        assert exc_info.value.code != 0

    def test_list_invalid_priority_exits_nonzero(
        self, store_file: str, capsys
    ) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--store", store_file, "list", "--priority", "urgent"])
        assert exc_info.value.code != 0


# ===========================================================================
# update sub-command
# ===========================================================================


class TestCmdUpdate:
    def _add_and_get_id(self, store_file: str) -> str:
        from task_store import TaskStore
        ts = TaskStore(store_path=store_file)
        task = ts.add_task("Task to update")
        return task["id"]

    def test_update_returns_exit_code_0(self, store_file: str, capsys) -> None:
        tid = self._add_and_get_id(store_file)
        rc, out, err = run_cap(["update", tid, "in_progress"], store_file, capsys)
        assert rc == 0

    def test_update_prints_task_updated_message(
        self, store_file: str, capsys
    ) -> None:
        tid = self._add_and_get_id(store_file)
        rc, out, err = run_cap(["update", tid, "done"], store_file, capsys)
        assert "Task updated" in out

    def test_update_prints_new_status_in_output(
        self, store_file: str, capsys
    ) -> None:
        tid = self._add_and_get_id(store_file)
        rc, out, err = run_cap(["update", tid, "done"], store_file, capsys)
        assert "DONE" in out.upper()

    def test_update_prints_task_title_in_output(
        self, store_file: str, capsys
    ) -> None:
        tid = self._add_and_get_id(store_file)
        rc, out, err = run_cap(["update", tid, "in_progress"], store_file, capsys)
        assert "Task to update" in out

    def test_update_with_partial_id(self, store_file: str, capsys) -> None:
        tid = self._add_and_get_id(store_file)
        partial = tid[:8]
        rc, out, err = run_cap(["update", partial, "done"], store_file, capsys)
        assert rc == 0
        assert "Task updated" in out

    def test_update_no_match_returns_exit_code_1(
        self, store_file: str, capsys
    ) -> None:
        self._add_and_get_id(store_file)
        rc, out, err = run_cap(
            ["update", "nonexistent-xyz-999", "done"], store_file, capsys
        )
        assert rc == 1

    def test_update_no_match_prints_error_to_stderr(
        self, store_file: str, capsys
    ) -> None:
        self._add_and_get_id(store_file)
        rc, out, err = run_cap(
            ["update", "nonexistent-xyz-999", "done"], store_file, capsys
        )
        assert "Error" in err
        assert "No task found" in err

    def test_update_invalid_status_exits_nonzero(
        self, store_file: str, capsys
    ) -> None:
        tid = self._add_and_get_id(store_file)
        with pytest.raises(SystemExit) as exc_info:
            main(["--store", store_file, "update", tid, "pending"])
        assert exc_info.value.code != 0

    def test_update_persists_new_status(self, store_file: str, capsys) -> None:
        from task_store import TaskStore
        tid = self._add_and_get_id(store_file)
        run(["update", tid, "done"], store_file)
        ts = TaskStore(store_path=store_file)
        tasks = ts.list_tasks(status="done")
        assert len(tasks) == 1
        assert tasks[0]["id"] == tid


# ===========================================================================
# delete sub-command
# ===========================================================================


class TestCmdDelete:
    def _add_and_get_id(self, store_file: str, title: str = "Task to delete") -> str:
        from task_store import TaskStore
        ts = TaskStore(store_path=store_file)
        task = ts.add_task(title)
        return task["id"]

    def test_delete_returns_exit_code_0(self, store_file: str, capsys) -> None:
        tid = self._add_and_get_id(store_file)
        rc, out, err = run_cap(["delete", tid], store_file, capsys)
        assert rc == 0

    def test_delete_prints_task_deleted_message(
        self, store_file: str, capsys
    ) -> None:
        tid = self._add_and_get_id(store_file)
        rc, out, err = run_cap(["delete", tid], store_file, capsys)
        assert "Task deleted" in out

    def test_delete_prints_task_title_in_output(
        self, store_file: str, capsys
    ) -> None:
        tid = self._add_and_get_id(store_file, "Goodbye task")
        rc, out, err = run_cap(["delete", tid], store_file, capsys)
        assert "Goodbye task" in out

    def test_delete_with_partial_id(self, store_file: str, capsys) -> None:
        tid = self._add_and_get_id(store_file)
        partial = tid[:8]
        rc, out, err = run_cap(["delete", partial], store_file, capsys)
        assert rc == 0
        assert "Task deleted" in out

    def test_delete_removes_task_from_store(self, store_file: str, capsys) -> None:
        from task_store import TaskStore
        tid = self._add_and_get_id(store_file)
        run(["delete", tid], store_file)
        ts = TaskStore(store_path=store_file)
        assert ts.list_tasks() == []

    def test_delete_no_match_returns_exit_code_1(
        self, store_file: str, capsys
    ) -> None:
        self._add_and_get_id(store_file)
        rc, out, err = run_cap(["delete", "nonexistent-xyz-999"], store_file, capsys)
        assert rc == 1

    def test_delete_no_match_prints_error_to_stderr(
        self, store_file: str, capsys
    ) -> None:
        self._add_and_get_id(store_file)
        rc, out, err = run_cap(["delete", "nonexistent-xyz-999"], store_file, capsys)
        assert "Error" in err
        assert "No task found" in err

    def test_delete_leaves_other_tasks_intact(
        self, store_file: str, capsys
    ) -> None:
        from task_store import TaskStore
        tid1 = self._add_and_get_id(store_file, "Keep me")
        tid2 = self._add_and_get_id(store_file, "Delete me")
        run(["delete", tid2], store_file)
        ts = TaskStore(store_path=store_file)
        remaining = ts.list_tasks()
        assert len(remaining) == 1
        assert remaining[0]["id"] == tid1


# ===========================================================================
# No sub-command
# ===========================================================================


class TestNoSubCommand:
    def test_no_subcommand_returns_exit_code_0(
        self, store_file: str, capsys
    ) -> None:
        rc = main(["--store", store_file])
        assert rc == 0

    def test_no_subcommand_prints_help(self, store_file: str, capsys) -> None:
        main(["--store", store_file])
        captured = capsys.readouterr()
        assert "task-tracker" in captured.out
        assert "COMMAND" in captured.out


# ===========================================================================
# Integration: full end-to-end flow
# ===========================================================================


class TestIntegrationFlow:
    def test_full_add_list_update_delete_flow(
        self, store_file: str, capsys
    ) -> None:
        """
        Full user flow:
          1. Add two tasks.
          2. List all — verify both appear.
          3. Update one to 'done'.
          4. List with status=done — verify only the updated one appears.
          5. Delete the done task.
          6. List all — verify only the remaining task is present.
        """
        from task_store import TaskStore

        # 1. Add tasks
        rc1 = run(["add", "First task", "--priority", "high"], store_file)
        rc2 = run(["add", "Second task", "--priority", "low"], store_file)
        assert rc1 == 0
        assert rc2 == 0

        # 2. List all
        rc, out, err = run_cap(["list"], store_file, capsys)
        assert rc == 0
        assert "First task" in out
        assert "Second task" in out
        assert "2 tasks" in out

        # 3. Update first task to done
        ts = TaskStore(store_path=store_file)
        tasks = ts.list_tasks()
        first_id = next(t["id"] for t in tasks if t["title"] == "First task")
        rc, out, err = run_cap(["update", first_id, "done"], store_file, capsys)
        assert rc == 0
        assert "Task updated" in out

        # 4. List with status=done
        rc, out, err = run_cap(["list", "--status", "done"], store_file, capsys)
        assert rc == 0
        assert "First task" in out
        assert "Second task" not in out

        # 5. Delete the done task
        rc, out, err = run_cap(["delete", first_id], store_file, capsys)
        assert rc == 0
        assert "Task deleted" in out
        assert "First task" in out

        # 6. List all — only Second task remains
        rc, out, err = run_cap(["list"], store_file, capsys)
        assert rc == 0
        assert "Second task" in out
        assert "First task" not in out
        assert "1 task" in out


# ===========================================================================
# Subprocess invocation (real binary test)
# ===========================================================================


class TestSubprocessInvocation:
    """Invoke cli.py as a real subprocess to verify the entry point works."""

    def _cli_path(self) -> str:
        """Return the absolute path to cli.py."""
        return str(Path(__file__).parent.parent / "cli.py")

    def test_subprocess_add_exits_0(self, tmp_path: Path) -> None:
        store = str(tmp_path / "tasks.json")
        result = subprocess.run(
            [sys.executable, self._cli_path(), "--store", store, "add", "Subprocess task"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Task added" in result.stdout
        assert "Subprocess task" in result.stdout

    def test_subprocess_list_shows_added_task(self, tmp_path: Path) -> None:
        store = str(tmp_path / "tasks.json")
        # Add
        subprocess.run(
            [sys.executable, self._cli_path(), "--store", store, "add", "Listed task"],
            capture_output=True,
            text=True,
        )
        # List
        result = subprocess.run(
            [sys.executable, self._cli_path(), "--store", store, "list"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Listed task" in result.stdout

    def test_subprocess_no_args_exits_0(self, tmp_path: Path) -> None:
        store = str(tmp_path / "tasks.json")
        result = subprocess.run(
            [sys.executable, self._cli_path(), "--store", store],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "task-tracker" in result.stdout

    def test_subprocess_delete_nonexistent_exits_1(self, tmp_path: Path) -> None:
        store = str(tmp_path / "tasks.json")
        # Add a task first so the store exists
        subprocess.run(
            [sys.executable, self._cli_path(), "--store", store, "add", "Dummy"],
            capture_output=True,
            text=True,
        )
        result = subprocess.run(
            [
                sys.executable,
                self._cli_path(),
                "--store",
                store,
                "delete",
                "nonexistent-xyz",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "Error" in result.stderr
