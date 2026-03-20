# Build a Python CLI Task Tracker Using Click and Rich

**Generated:** 2026-03-20  
**Workspace:** `/Users/pradeepsharma/sasva/projects/vllm`  
**Target Output Directory:** `task_tracker/` (new subdirectory within the workspace)

---

## Workspace Context

This workspace is the **vLLM** project — a high-throughput LLM inference engine. Key observations relevant to this plan:

- **Python version:** `>=3.10,<3.14` (from `pyproject.toml`)
- **Test framework:** `pytest` with `pytest-asyncio`, `pytest-cov` (from `requirements/test.in`)
- **Existing CLI pattern:** vLLM uses `argparse`-based CLI in `vllm/entrypoints/cli/` with a `CLISubcommand` base class pattern (`vllm/entrypoints/cli/types.py`). Our new tool uses **Click** instead — a cleaner, decorator-based approach.
- **Rich usage:** Already used in `benchmarks/attention_benchmarks/common.py` via `rich.console.Console` and `rich.table.Table` — confirming Rich is available or easily installable.
- **Data modeling:** vLLM uses `@dataclass` (e.g., `vllm/outputs.py`) and `TypedDict` (e.g., `vllm/inputs/data.py`) — we'll use `dataclass` for the Task model.
- **JSON persistence:** `docker/versions.json` confirms JSON is the standard for config/data files in this repo.
- **No existing `tasks.json`, `task_store.py`, or `cli.py`** — these are net-new files.
- **Test structure:** Tests live in `tests/` with `conftest.py` at each level. We'll create `tests/task_tracker/` to stay consistent with the project layout.

---

## Architecture Overview

```
task_tracker/
├── __init__.py          # Package marker
├── task_store.py        # TaskStore class — data layer (JSON persistence)
└── cli.py               # Click CLI — commands: add, list, done, delete, stats

tests/task_tracker/
├── __init__.py          # Package marker
├── conftest.py          # Shared fixtures (tmp_path store, runner)
├── test_store.py        # Unit tests for TaskStore
└── test_cli.py          # CLI tests using Click's CliRunner
```

**Data file:** `tasks.json` (created at runtime in the working directory, or configurable via env var `TASK_TRACKER_FILE`)

---

## Task Schema

```python
# Each task is a dict serialized to JSON:
{
    "id": "550e8400-e29b-41d4-a716-446655440000",  # uuid4 string
    "title": "Write unit tests",                    # str
    "status": "todo",                               # "todo" | "in_progress" | "done"
    "priority": "medium",                           # "low" | "medium" | "high"
    "created_at": "2026-03-20T12:00:00.000000"      # ISO 8601 timestamp
}
```

---

## Execution Phases

### Phase 1 — Project Scaffolding & Dependencies

**Goal:** Create the package structure and declare dependencies.

#### Step 1.1 — Create `task_tracker/__init__.py`

```python
# task_tracker/__init__.py
"""Python CLI Task Tracker — Click + Rich."""
```

#### Step 1.2 — Create `tests/task_tracker/__init__.py`

Empty file to make the test directory a Python package.

#### Step 1.3 — Create `task_tracker/requirements.txt`

```
click>=8.1.0
rich>=13.0.0
```

> **Note:** These are the only two new runtime dependencies. `pytest` is already in `requirements/test.in`. No changes to the root `pyproject.toml` are required since this is a self-contained sub-tool within the workspace.

---

### Phase 2 — Core Data Layer: `task_tracker/task_store.py`

**Goal:** Implement `TaskStore` — all persistence and business logic lives here.

#### Full Implementation

```python
# task_tracker/task_store.py
"""
TaskStore: reads/writes tasks to a local tasks.json file.

Task schema:
    id         : str  — UUID4
    title      : str  — human-readable description
    status     : str  — "todo" | "in_progress" | "done"
    priority   : str  — "low" | "medium" | "high"
    created_at : str  — ISO 8601 timestamp (datetime.utcnow().isoformat())
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

# ── Constants ────────────────────────────────────────────────────────────────

STATUS_VALUES  = ("todo", "in_progress", "done")
PRIORITY_VALUES = ("low", "medium", "high")

DEFAULT_TASKS_FILE = os.environ.get("TASK_TRACKER_FILE", "tasks.json")

# ── Data model ───────────────────────────────────────────────────────────────

@dataclass
class Task:
    id: str
    title: str
    status: Literal["todo", "in_progress", "done"]
    priority: Literal["low", "medium", "high"]
    created_at: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(
            id=data["id"],
            title=data["title"],
            status=data["status"],
            priority=data["priority"],
            created_at=data["created_at"],
        )

# ── TaskStore ─────────────────────────────────────────────────────────────────

class TaskStore:
    """Manages task persistence in a local JSON file."""

    def __init__(self, filepath: str = DEFAULT_TASKS_FILE) -> None:
        self.filepath = Path(filepath)
        self._tasks: list[Task] = []
        self._load()

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load(self) -> None:
        """Load tasks from disk. Creates an empty store if file doesn't exist."""
        if self.filepath.exists():
            with self.filepath.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
            self._tasks = [Task.from_dict(t) for t in raw]
        else:
            self._tasks = []

    def _save(self) -> None:
        """Persist current task list to disk (atomic write via temp file)."""
        tmp = self.filepath.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump([t.to_dict() for t in self._tasks], fh, indent=2)
        tmp.replace(self.filepath)

    def _find_by_partial_id(self, partial_id: str) -> Optional[Task]:
        """Return the first task whose id starts with partial_id."""
        matches = [t for t in self._tasks if t.id.startswith(partial_id)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous partial id '{partial_id}' matches {len(matches)} tasks."
            )
        return None

    # ── Public API ────────────────────────────────────────────────────────────

    def add_task(
        self,
        title: str,
        priority: Literal["low", "medium", "high"] = "medium",
    ) -> Task:
        """Create a new task and persist it. Returns the created Task."""
        if priority not in PRIORITY_VALUES:
            raise ValueError(f"Invalid priority '{priority}'. Choose from {PRIORITY_VALUES}.")
        task = Task(
            id=str(uuid.uuid4()),
            title=title,
            status="todo",
            priority=priority,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        self._tasks.append(task)
        self._save()
        return task

    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> list[Task]:
        """Return tasks, optionally filtered by status and/or priority."""
        if status and status not in STATUS_VALUES:
            raise ValueError(f"Invalid status '{status}'. Choose from {STATUS_VALUES}.")
        if priority and priority not in PRIORITY_VALUES:
            raise ValueError(f"Invalid priority '{priority}'. Choose from {PRIORITY_VALUES}.")

        result = self._tasks
        if status:
            result = [t for t in result if t.status == status]
        if priority:
            result = [t for t in result if t.priority == priority]
        return result

    def update_status(
        self,
        partial_id: str,
        new_status: Literal["todo", "in_progress", "done"],
    ) -> Task:
        """Update a task's status by partial id match. Returns updated Task."""
        if new_status not in STATUS_VALUES:
            raise ValueError(f"Invalid status '{new_status}'. Choose from {STATUS_VALUES}.")
        task = self._find_by_partial_id(partial_id)
        if task is None:
            raise KeyError(f"No task found matching id prefix '{partial_id}'.")
        task.status = new_status
        self._save()
        return task

    def delete_task(self, partial_id: str) -> Task:
        """Delete a task by partial id match. Returns the deleted Task."""
        task = self._find_by_partial_id(partial_id)
        if task is None:
            raise KeyError(f"No task found matching id prefix '{partial_id}'.")
        self._tasks.remove(task)
        self._save()
        return task
```

**Key design decisions:**
- `_load()` is called in `__init__` so the store is always in sync with disk on construction.
- `_save()` uses an atomic write (write to `.tmp`, then `replace`) to avoid data corruption.
- `_find_by_partial_id()` raises `ValueError` on ambiguous matches and returns `None` on no match — callers handle the `None` case.
- `created_at` uses `datetime.now(timezone.utc).isoformat()` for timezone-aware ISO 8601 strings.

---

### Phase 3 — CLI Interface: `task_tracker/cli.py`

**Goal:** Implement all five Click commands with Rich display.

#### Full Implementation

```python
# task_tracker/cli.py
"""
CLI Task Tracker — commands: add, list, done, delete, stats.

Usage:
    python -m task_tracker.cli add --title "My task" --priority high
    python -m task_tracker.cli list --status todo --priority high
    python -m task_tracker.cli done <partial-id>
    python -m task_tracker.cli delete <partial-id>
    python -m task_tracker.cli stats
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from task_tracker.task_store import (
    DEFAULT_TASKS_FILE,
    PRIORITY_VALUES,
    STATUS_VALUES,
    TaskStore,
)

console = Console()

# ── Priority / Status color maps ─────────────────────────────────────────────

PRIORITY_COLORS = {"low": "green", "medium": "yellow", "high": "red"}
STATUS_COLORS   = {"todo": "cyan", "in_progress": "magenta", "done": "green"}

# ── CLI group ─────────────────────────────────────────────────────────────────

@click.group()
@click.option(
    "--file",
    "filepath",
    default=DEFAULT_TASKS_FILE,
    envvar="TASK_TRACKER_FILE",
    show_default=True,
    help="Path to the tasks JSON file.",
)
@click.pass_context
def cli(ctx: click.Context, filepath: str) -> None:
    """A simple CLI task tracker powered by Click and Rich."""
    ctx.ensure_object(dict)
    ctx.obj["store"] = TaskStore(filepath=filepath)


# ── add ───────────────────────────────────────────────────────────────────────

@cli.command("add")
@click.option("--title",    required=True,  help="Task title.")
@click.option(
    "--priority",
    default="medium",
    show_default=True,
    type=click.Choice(PRIORITY_VALUES, case_sensitive=False),
    help="Task priority.",
)
@click.pass_context
def add_cmd(ctx: click.Context, title: str, priority: str) -> None:
    """Add a new task."""
    store: TaskStore = ctx.obj["store"]
    task = store.add_task(title=title, priority=priority)
    console.print(
        f"[bold green]✓ Task added[/bold green] "
        f"[dim](id: {task.id[:8]}…)[/dim]  "
        f"[bold]{task.title}[/bold]  "
        f"priority=[{PRIORITY_COLORS[task.priority]}]{task.priority}[/{PRIORITY_COLORS[task.priority]}]"
    )


# ── list ──────────────────────────────────────────────────────────────────────

@cli.command("list")
@click.option(
    "--status",
    default=None,
    type=click.Choice(STATUS_VALUES, case_sensitive=False),
    help="Filter by status.",
)
@click.option(
    "--priority",
    default=None,
    type=click.Choice(PRIORITY_VALUES, case_sensitive=False),
    help="Filter by priority.",
)
@click.pass_context
def list_cmd(ctx: click.Context, status: str | None, priority: str | None) -> None:
    """List tasks, optionally filtered by status and/or priority."""
    store: TaskStore = ctx.obj["store"]
    tasks = store.list_tasks(status=status, priority=priority)

    if not tasks:
        console.print("[dim]No tasks found.[/dim]")
        return

    table = Table(
        title="Tasks",
        show_header=True,
        header_style="bold blue",
        border_style="dim",
    )
    table.add_column("ID (short)", style="dim", width=10)
    table.add_column("Title",      min_width=20)
    table.add_column("Status",     width=12)
    table.add_column("Priority",   width=10)
    table.add_column("Created At", width=26)

    for task in tasks:
        status_text   = Text(task.status,   style=STATUS_COLORS.get(task.status, "white"))
        priority_text = Text(task.priority, style=PRIORITY_COLORS.get(task.priority, "white"))
        table.add_row(
            task.id[:8] + "…",
            task.title,
            status_text,
            priority_text,
            task.created_at,
        )

    console.print(table)


# ── done ──────────────────────────────────────────────────────────────────────

@cli.command("done")
@click.argument("partial_id")
@click.pass_context
def done_cmd(ctx: click.Context, partial_id: str) -> None:
    """Mark a task as done by partial ID match."""
    store: TaskStore = ctx.obj["store"]
    try:
        task = store.update_status(partial_id, "done")
        console.print(
            f"[bold green]✓ Marked done[/bold green] "
            f"[dim](id: {task.id[:8]}…)[/dim]  [bold]{task.title}[/bold]"
        )
    except KeyError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1)


# ── delete ────────────────────────────────────────────────────────────────────

@cli.command("delete")
@click.argument("partial_id")
@click.confirmation_option(
    prompt="Are you sure you want to delete this task?",
    help="Skip confirmation prompt.",
)
@click.pass_context
def delete_cmd(ctx: click.Context, partial_id: str) -> None:
    """Delete a task by partial ID match (with confirmation)."""
    store: TaskStore = ctx.obj["store"]
    try:
        task = store.delete_task(partial_id)
        console.print(
            f"[bold red]✗ Task deleted[/bold red] "
            f"[dim](id: {task.id[:8]}…)[/dim]  [bold]{task.title}[/bold]"
        )
    except KeyError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1)


# ── stats ─────────────────────────────────────────────────────────────────────

@cli.command("stats")
@click.pass_context
def stats_cmd(ctx: click.Context) -> None:
    """Show task statistics in a Rich panel."""
    store: TaskStore = ctx.obj["store"]
    tasks = store.list_tasks()

    total = len(tasks)

    # Status counts
    status_counts = {s: sum(1 for t in tasks if t.status == s) for s in STATUS_VALUES}
    # Priority counts
    priority_counts = {p: sum(1 for t in tasks if t.priority == p) for p in PRIORITY_VALUES}

    lines = [
        f"[bold]Total tasks:[/bold] {total}",
        "",
        "[bold underline]By Status[/bold underline]",
    ]
    for s, count in status_counts.items():
        color = STATUS_COLORS.get(s, "white")
        lines.append(f"  [{color}]{s:<12}[/{color}]  {count}")

    lines += ["", "[bold underline]By Priority[/bold underline]"]
    for p, count in priority_counts.items():
        color = PRIORITY_COLORS.get(p, "white")
        lines.append(f"  [{color}]{p:<12}[/{color}]  {count}")

    panel_content = "\n".join(lines)
    console.print(Panel(panel_content, title="[bold blue]Task Statistics[/bold blue]", expand=False))


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli()
```

**Key design decisions:**
- `@click.pass_context` + `ctx.obj["store"]` pattern shares a single `TaskStore` instance across all commands, respecting the `--file` option.
- `@click.confirmation_option` on `delete` provides the `--yes` flag for non-interactive use (critical for tests).
- `click.Choice` on `--status` and `--priority` provides built-in validation with helpful error messages.
- Rich `Table` for `list`, Rich `Panel` for `stats` — consistent with the existing usage in `benchmarks/attention_benchmarks/common.py`.

---

### Phase 4 — Tests: `tests/task_tracker/conftest.py`

**Goal:** Shared fixtures for both test modules.

```python
# tests/task_tracker/conftest.py
"""Shared pytest fixtures for task_tracker tests."""

import pytest
from click.testing import CliRunner

from task_tracker.task_store import TaskStore


@pytest.fixture
def tmp_store(tmp_path):
    """A TaskStore backed by a temporary file — isolated per test."""
    return TaskStore(filepath=str(tmp_path / "tasks.json"))


@pytest.fixture
def runner():
    """Click CliRunner with mix_stderr=False for clean output capture."""
    return CliRunner(mix_stderr=False)


@pytest.fixture
def tmp_cli_env(tmp_path):
    """Returns (runner, env dict) pointing CLI at a temp tasks.json."""
    r = CliRunner(mix_stderr=False)
    env = {"TASK_TRACKER_FILE": str(tmp_path / "tasks.json")}
    return r, env
```

---

### Phase 5 — Tests: `tests/task_tracker/test_store.py`

**Goal:** Full unit test coverage of `TaskStore`.

```python
# tests/task_tracker/test_store.py
"""Unit tests for TaskStore (task_tracker/task_store.py)."""

import json

import pytest

from task_tracker.task_store import TaskStore


# ── add_task ──────────────────────────────────────────────────────────────────

class TestAddTask:
    def test_add_returns_task(self, tmp_store):
        task = tmp_store.add_task("Write docs")
        assert task.title == "Write docs"
        assert task.status == "todo"
        assert task.priority == "medium"
        assert len(task.id) == 36  # UUID4 string length

    def test_add_with_priority(self, tmp_store):
        task = tmp_store.add_task("Urgent fix", priority="high")
        assert task.priority == "high"

    def test_add_invalid_priority_raises(self, tmp_store):
        with pytest.raises(ValueError, match="Invalid priority"):
            tmp_store.add_task("Bad task", priority="critical")

    def test_add_increments_count(self, tmp_store):
        tmp_store.add_task("Task 1")
        tmp_store.add_task("Task 2")
        assert len(tmp_store.list_tasks()) == 2

    def test_add_persists_to_disk(self, tmp_path):
        """Verify the task survives a store reload."""
        filepath = str(tmp_path / "tasks.json")
        store1 = TaskStore(filepath=filepath)
        task = store1.add_task("Persistent task")

        store2 = TaskStore(filepath=filepath)
        tasks = store2.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].id == task.id
        assert tasks[0].title == "Persistent task"


# ── list_tasks ────────────────────────────────────────────────────────────────

class TestListTasks:
    def test_list_empty(self, tmp_store):
        assert tmp_store.list_tasks() == []

    def test_list_all(self, tmp_store):
        tmp_store.add_task("A")
        tmp_store.add_task("B")
        assert len(tmp_store.list_tasks()) == 2

    def test_filter_by_status(self, tmp_store):
        t1 = tmp_store.add_task("Todo task")
        t2 = tmp_store.add_task("Done task")
        tmp_store.update_status(t2.id, "done")

        todo_tasks = tmp_store.list_tasks(status="todo")
        assert len(todo_tasks) == 1
        assert todo_tasks[0].id == t1.id

    def test_filter_by_priority(self, tmp_store):
        tmp_store.add_task("Low task", priority="low")
        tmp_store.add_task("High task", priority="high")

        high_tasks = tmp_store.list_tasks(priority="high")
        assert len(high_tasks) == 1
        assert high_tasks[0].priority == "high"

    def test_filter_by_status_and_priority(self, tmp_store):
        tmp_store.add_task("Low todo", priority="low")
        tmp_store.add_task("High todo", priority="high")
        t3 = tmp_store.add_task("High done", priority="high")
        tmp_store.update_status(t3.id, "done")

        result = tmp_store.list_tasks(status="todo", priority="high")
        assert len(result) == 1
        assert result[0].title == "High todo"

    def test_filter_invalid_status_raises(self, tmp_store):
        with pytest.raises(ValueError, match="Invalid status"):
            tmp_store.list_tasks(status="pending")

    def test_filter_invalid_priority_raises(self, tmp_store):
        with pytest.raises(ValueError, match="Invalid priority"):
            tmp_store.list_tasks(priority="urgent")


# ── update_status ─────────────────────────────────────────────────────────────

class TestUpdateStatus:
    def test_update_to_done(self, tmp_store):
        task = tmp_store.add_task("Finish me")
        updated = tmp_store.update_status(task.id, "done")
        assert updated.status == "done"

    def test_update_to_in_progress(self, tmp_store):
        task = tmp_store.add_task("Start me")
        updated = tmp_store.update_status(task.id[:8], "in_progress")
        assert updated.status == "in_progress"

    def test_update_persists(self, tmp_path):
        filepath = str(tmp_path / "tasks.json")
        store1 = TaskStore(filepath=filepath)
        task = store1.add_task("Persist status")
        store1.update_status(task.id, "done")

        store2 = TaskStore(filepath=filepath)
        assert store2.list_tasks()[0].status == "done"

    def test_update_not_found_raises(self, tmp_store):
        with pytest.raises(KeyError, match="No task found"):
            tmp_store.update_status("nonexistent", "done")

    def test_update_invalid_status_raises(self, tmp_store):
        task = tmp_store.add_task("Task")
        with pytest.raises(ValueError, match="Invalid status"):
            tmp_store.update_status(task.id, "finished")

    def test_update_ambiguous_id_raises(self, tmp_store):
        """Two tasks with same id prefix should raise ValueError."""
        # Force two tasks with the same short prefix by mocking uuid
        import unittest.mock as mock
        with mock.patch("uuid.uuid4", side_effect=["aaa00001-0000-0000-0000-000000000000",
                                                    "aaa00002-0000-0000-0000-000000000000"]):
            tmp_store.add_task("Task A")
            tmp_store.add_task("Task B")
        with pytest.raises(ValueError, match="Ambiguous"):
            tmp_store.update_status("aaa", "done")


# ── delete_task ───────────────────────────────────────────────────────────────

class TestDeleteTask:
    def test_delete_removes_task(self, tmp_store):
        task = tmp_store.add_task("Delete me")
        tmp_store.delete_task(task.id)
        assert len(tmp_store.list_tasks()) == 0

    def test_delete_returns_task(self, tmp_store):
        task = tmp_store.add_task("Return on delete")
        deleted = tmp_store.delete_task(task.id[:8])
        assert deleted.id == task.id

    def test_delete_persists(self, tmp_path):
        filepath = str(tmp_path / "tasks.json")
        store1 = TaskStore(filepath=filepath)
        task = store1.add_task("Ephemeral")
        store1.delete_task(task.id)

        store2 = TaskStore(filepath=filepath)
        assert store2.list_tasks() == []

    def test_delete_not_found_raises(self, tmp_store):
        with pytest.raises(KeyError, match="No task found"):
            tmp_store.delete_task("nonexistent")


# ── Persistence (save/load round-trip) ────────────────────────────────────────

class TestPersistence:
    def test_json_file_is_valid(self, tmp_path):
        filepath = str(tmp_path / "tasks.json")
        store = TaskStore(filepath=filepath)
        store.add_task("Check JSON", priority="low")

        with open(filepath) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1
        assert set(data[0].keys()) == {"id", "title", "status", "priority", "created_at"}

    def test_reload_preserves_all_fields(self, tmp_path):
        filepath = str(tmp_path / "tasks.json")
        store1 = TaskStore(filepath=filepath)
        original = store1.add_task("Full round-trip", priority="high")

        store2 = TaskStore(filepath=filepath)
        reloaded = store2.list_tasks()[0]

        assert reloaded.id == original.id
        assert reloaded.title == original.title
        assert reloaded.status == original.status
        assert reloaded.priority == original.priority
        assert reloaded.created_at == original.created_at

    def test_no_file_starts_empty(self, tmp_path):
        store = TaskStore(filepath=str(tmp_path / "nonexistent.json"))
        assert store.list_tasks() == []
```

---

### Phase 6 — Tests: `tests/task_tracker/test_cli.py`

**Goal:** Full CLI command coverage using Click's `CliRunner`.

```python
# tests/task_tracker/test_cli.py
"""CLI tests for task_tracker/cli.py using Click's CliRunner."""

import json

import pytest
from click.testing import CliRunner

from task_tracker.cli import cli
from task_tracker.task_store import TaskStore


# ── Helpers ───────────────────────────────────────────────────────────────────

def invoke(runner, tmp_path, *args, input=None):
    """Invoke CLI with TASK_TRACKER_FILE pointing to tmp_path."""
    env = {"TASK_TRACKER_FILE": str(tmp_path / "tasks.json")}
    return runner.invoke(cli, list(args), env=env, input=input, catch_exceptions=False)


# ── add command ───────────────────────────────────────────────────────────────

class TestAddCommand:
    def test_add_basic(self, runner, tmp_path):
        result = invoke(runner, tmp_path, "add", "--title", "My first task")
        assert result.exit_code == 0
        assert "Task added" in result.output
        assert "My first task" in result.output

    def test_add_with_priority(self, runner, tmp_path):
        result = invoke(runner, tmp_path, "add", "--title", "Urgent", "--priority", "high")
        assert result.exit_code == 0
        assert "high" in result.output

    def test_add_default_priority_is_medium(self, runner, tmp_path):
        invoke(runner, tmp_path, "add", "--title", "Default priority task")
        # Verify via list
        result = invoke(runner, tmp_path, "list")
        assert "medium" in result.output

    def test_add_invalid_priority_fails(self, runner, tmp_path):
        result = runner.invoke(
            cli,
            ["add", "--title", "Bad", "--priority", "critical"],
            env={"TASK_TRACKER_FILE": str(tmp_path / "tasks.json")},
        )
        assert result.exit_code != 0

    def test_add_missing_title_fails(self, runner, tmp_path):
        result = runner.invoke(
            cli,
            ["add"],
            env={"TASK_TRACKER_FILE": str(tmp_path / "tasks.json")},
        )
        assert result.exit_code != 0

    def test_add_persists_to_file(self, runner, tmp_path):
        invoke(runner, tmp_path, "add", "--title", "Persisted task")
        tasks_file = tmp_path / "tasks.json"
        assert tasks_file.exists()
        data = json.loads(tasks_file.read_text())
        assert len(data) == 1
        assert data[0]["title"] == "Persisted task"


# ── list command ──────────────────────────────────────────────────────────────

class TestListCommand:
    def test_list_empty(self, runner, tmp_path):
        result = invoke(runner, tmp_path, "list")
        assert result.exit_code == 0
        assert "No tasks found" in result.output

    def test_list_shows_tasks(self, runner, tmp_path):
        invoke(runner, tmp_path, "add", "--title", "Task Alpha")
        invoke(runner, tmp_path, "add", "--title", "Task Beta")
        result = invoke(runner, tmp_path, "list")
        assert "Task Alpha" in result.output
        assert "Task Beta" in result.output

    def test_list_filter_by_status(self, runner, tmp_path):
        invoke(runner, tmp_path, "add", "--title", "Todo task")
        # Add and mark done
        r = invoke(runner, tmp_path, "add", "--title", "Done task")
        # Extract short id from output
        store = TaskStore(filepath=str(tmp_path / "tasks.json"))
        done_task = [t for t in store.list_tasks() if t.title == "Done task"][0]
        invoke(runner, tmp_path, "done", done_task.id[:8])

        result = invoke(runner, tmp_path, "list", "--status", "todo")
        assert "Todo task" in result.output
        assert "Done task" not in result.output

    def test_list_filter_by_priority(self, runner, tmp_path):
        invoke(runner, tmp_path, "add", "--title", "Low task", "--priority", "low")
        invoke(runner, tmp_path, "add", "--title", "High task", "--priority", "high")

        result = invoke(runner, tmp_path, "list", "--priority", "high")
        assert "High task" in result.output
        assert "Low task" not in result.output

    def test_list_invalid_status_fails(self, runner, tmp_path):
        result = runner.invoke(
            cli,
            ["list", "--status", "pending"],
            env={"TASK_TRACKER_FILE": str(tmp_path / "tasks.json")},
        )
        assert result.exit_code != 0

    def test_list_shows_rich_table_headers(self, runner, tmp_path):
        invoke(runner, tmp_path, "add", "--title", "Table test")
        result = invoke(runner, tmp_path, "list")
        assert "Title" in result.output
        assert "Status" in result.output
        assert "Priority" in result.output


# ── done command ──────────────────────────────────────────────────────────────

class TestDoneCommand:
    def test_done_marks_task(self, runner, tmp_path):
        invoke(runner, tmp_path, "add", "--title", "Finish this")
        store = TaskStore(filepath=str(tmp_path / "tasks.json"))
        task = store.list_tasks()[0]

        result = invoke(runner, tmp_path, "done", task.id[:8])
        assert result.exit_code == 0
        assert "Marked done" in result.output

        store2 = TaskStore(filepath=str(tmp_path / "tasks.json"))
        assert store2.list_tasks()[0].status == "done"

    def test_done_not_found(self, runner, tmp_path):
        result = invoke(runner, tmp_path, "done", "nonexistent")
        assert result.exit_code == 1
        assert "Error" in result.output

    def test_done_full_id(self, runner, tmp_path):
        invoke(runner, tmp_path, "add", "--title", "Full ID test")
        store = TaskStore(filepath=str(tmp_path / "tasks.json"))
        task = store.list_tasks()[0]

        result = invoke(runner, tmp_path, "done", task.id)
        assert result.exit_code == 0


# ── delete command ────────────────────────────────────────────────────────────

class TestDeleteCommand:
    def test_delete_with_confirmation(self, runner, tmp_path):
        invoke(runner, tmp_path, "add", "--title", "Delete me")
        store = TaskStore(filepath=str(tmp_path / "tasks.json"))
        task = store.list_tasks()[0]

        # Provide "y" as input for confirmation prompt
        result = runner.invoke(
            cli,
            ["delete", task.id[:8]],
            env={"TASK_TRACKER_FILE": str(tmp_path / "tasks.json")},
            input="y\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "Task deleted" in result.output

        store2 = TaskStore(filepath=str(tmp_path / "tasks.json"))
        assert store2.list_tasks() == []

    def test_delete_abort_on_no(self, runner, tmp_path):
        invoke(runner, tmp_path, "add", "--title", "Keep me")
        store = TaskStore(filepath=str(tmp_path / "tasks.json"))
        task = store.list_tasks()[0]

        result = runner.invoke(
            cli,
            ["delete", task.id[:8]],
            env={"TASK_TRACKER_FILE": str(tmp_path / "tasks.json")},
            input="n\n",
            catch_exceptions=False,
        )
        # Task should still exist
        store2 = TaskStore(filepath=str(tmp_path / "tasks.json"))
        assert len(store2.list_tasks()) == 1

    def test_delete_not_found(self, runner, tmp_path):
        result = runner.invoke(
            cli,
            ["delete", "nonexistent"],
            env={"TASK_TRACKER_FILE": str(tmp_path / "tasks.json")},
            input="y\n",
            catch_exceptions=False,
        )
        assert result.exit_code == 1

    def test_delete_yes_flag_skips_prompt(self, runner, tmp_path):
        invoke(runner, tmp_path, "add", "--title", "Skip prompt")
        store = TaskStore(filepath=str(tmp_path / "tasks.json"))
        task = store.list_tasks()[0]

        result = runner.invoke(
            cli,
            ["delete", "--yes", task.id[:8]],
            env={"TASK_TRACKER_FILE": str(tmp_path / "tasks.json")},
            catch_exceptions=False,
        )
        assert result.exit_code == 0


# ── stats command ─────────────────────────────────────────────────────────────

class TestStatsCommand:
    def test_stats_empty(self, runner, tmp_path):
        result = invoke(runner, tmp_path, "stats")
        assert result.exit_code == 0
        assert "Total tasks" in result.output
        assert "0" in result.output

    def test_stats_counts(self, runner, tmp_path):
        invoke(runner, tmp_path, "add", "--title", "T1", "--priority", "high")
        invoke(runner, tmp_path, "add", "--title", "T2", "--priority", "low")
        invoke(runner, tmp_path, "add", "--title", "T3")

        store = TaskStore(filepath=str(tmp_path / "tasks.json"))
        t1 = store.list_tasks()[0]
        invoke(runner, tmp_path, "done", t1.id[:8])

        result = invoke(runner, tmp_path, "stats")
        assert result.exit_code == 0
        assert "Total tasks" in result.output
        assert "3" in result.output
        assert "By Status" in result.output
        assert "By Priority" in result.output

    def test_stats_shows_panel(self, runner, tmp_path):
        result = invoke(runner, tmp_path, "stats")
        assert "Task Statistics" in result.output
```

---

### Phase 7 — Final Integration & Verification

**Goal:** Ensure all files are in place, dependencies are installable, and all tests pass.

#### Step 7.1 — Install dependencies

```bash
cd /Users/pradeepsharma/sasva/projects/vllm
pip install click>=8.1.0 rich>=13.0.0
```

#### Step 7.2 — Run all tests

```bash
cd /Users/pradeepsharma/sasva/projects/vllm
python -m pytest tests/task_tracker/ -v
```

Expected output:
```
tests/task_tracker/test_store.py::TestAddTask::test_add_returns_task PASSED
tests/task_tracker/test_store.py::TestAddTask::test_add_with_priority PASSED
tests/task_tracker/test_store.py::TestAddTask::test_add_invalid_priority_raises PASSED
tests/task_tracker/test_store.py::TestAddTask::test_add_increments_count PASSED
tests/task_tracker/test_store.py::TestAddTask::test_add_persists_to_disk PASSED
tests/task_tracker/test_store.py::TestListTasks::test_list_empty PASSED
... (all tests PASSED)
tests/task_tracker/test_cli.py::TestStatsCommand::test_stats_shows_panel PASSED

============================= N passed in X.XXs ==============================
```

#### Step 7.3 — Manual smoke test

```bash
# Add tasks
python -m task_tracker.cli add --title "Write unit tests" --priority high
python -m task_tracker.cli add --title "Update README" --priority low
python -m task_tracker.cli add --title "Fix bug #42"

# List all
python -m task_tracker.cli list

# Filter
python -m task_tracker.cli list --status todo --priority high

# Mark done (use first 8 chars of id from list output)
python -m task_tracker.cli done <partial-id>

# Stats
python -m task_tracker.cli stats

# Delete (will prompt for confirmation)
python -m task_tracker.cli delete <partial-id>
```

---

## File Summary

| File | Purpose | Phase |
|------|---------|-------|
| `task_tracker/__init__.py` | Package marker | 1 |
| `task_tracker/requirements.txt` | Runtime deps (click, rich) | 1 |
| `task_tracker/task_store.py` | `Task` dataclass + `TaskStore` class | 2 |
| `task_tracker/cli.py` | Click CLI: add, list, done, delete, stats | 3 |
| `tests/task_tracker/__init__.py` | Test package marker | 4 |
| `tests/task_tracker/conftest.py` | Shared fixtures | 4 |
| `tests/task_tracker/test_store.py` | TaskStore unit tests | 5 |
| `tests/task_tracker/test_cli.py` | CLI integration tests | 6 |

---

## Dependency Map

```
cli.py
  └── task_store.py          (TaskStore, Task, constants)
  └── click                  (external: >=8.1.0)
  └── rich                   (external: >=13.0.0)

task_store.py
  └── stdlib: json, uuid, dataclasses, datetime, pathlib, os, typing

test_store.py
  └── task_store.py
  └── pytest

test_cli.py
  └── cli.py
  └── task_store.py          (for setup/verification)
  └── click.testing.CliRunner
  └── pytest
```

---

## Verification Criteria

### CLI Commands to Run

```bash
# 1. Install dependencies
pip install click>=8.1.0 rich>=13.0.0

# 2. Run the full test suite — ALL tests must pass
cd /Users/pradeepsharma/sasva/projects/vllm
python -m pytest tests/task_tracker/ -v --tb=short

# Expected: All tests PASSED, exit code 0
# Minimum expected test count: 30+ tests across test_store.py and test_cli.py

# 3. Smoke test: add command
python -m task_tracker.cli add --title "Verify add works" --priority high
# Expected output contains: "✓ Task added" and "Verify add works" and "high"

# 4. Smoke test: list command
python -m task_tracker.cli list
# Expected: Rich table with columns ID, Title, Status, Priority, Created At
# Expected: Row containing "Verify add works"

# 5. Smoke test: list with filter
python -m task_tracker.cli list --status todo
# Expected: "Verify add works" appears (status is todo by default)

python -m task_tracker.cli list --status done
# Expected: "No tasks found." (nothing is done yet)

# 6. Smoke test: done command
# Get partial id from: python -m task_tracker.cli list
python -m task_tracker.cli done <first-8-chars-of-id>
# Expected output contains: "✓ Marked done"

# 7. Smoke test: stats command
python -m task_tracker.cli stats
# Expected: Rich panel titled "Task Statistics"
# Expected: "Total tasks: 1", "done" count = 1, "high" count = 1

# 8. Smoke test: delete command
python -m task_tracker.cli delete <first-8-chars-of-id>
# Prompts: "Are you sure you want to delete this task? [y/N]:"
# After "y": output contains "✗ Task deleted"

# 9. Verify tasks.json is valid JSON
python -c "import json; data=json.load(open('tasks.json')); print(f'Valid JSON, {len(data)} tasks')"
# Expected: "Valid JSON, 0 tasks" (after deletion)

# 10. Verify --yes flag skips confirmation
python -m task_tracker.cli add --title "Skip confirm test"
python -m task_tracker.cli delete --yes <partial-id>
# Expected: No prompt, immediate deletion output
```

### Test Pass Criteria

- `python -m pytest tests/task_tracker/test_store.py -v` → **all tests PASSED**
- `python -m pytest tests/task_tracker/test_cli.py -v` → **all tests PASSED**
- `python -m pytest tests/task_tracker/ -v` → **exit code 0**, no failures, no errors
- No existing vLLM tests broken: `python -m pytest tests/test_config.py -v` still passes

### Functional Acceptance Criteria

| Criterion | How to Verify |
|-----------|--------------|
| `add` creates task with uuid, title, status=todo, priority, created_at | Check `tasks.json` after `add` |
| `list` renders Rich table with all columns | Run `list` and inspect output |
| `list --status` filters correctly | Run `list --status done` after marking one done |
| `list --priority` filters correctly | Run `list --priority high` with mixed priorities |
| `done` accepts partial id (8 chars) | Run `done <8-char-prefix>` |
| `delete` prompts for confirmation | Run `delete <id>` without `--yes` |
| `delete --yes` skips prompt | Run `delete --yes <id>` |
| `stats` shows counts per status and priority | Run `stats` and verify all 3 statuses and 3 priorities appear |
| `tasks.json` survives store reload | Add task, create new `TaskStore`, verify task present |
| Atomic write (no corruption) | Check `.tmp` file is cleaned up after save |
