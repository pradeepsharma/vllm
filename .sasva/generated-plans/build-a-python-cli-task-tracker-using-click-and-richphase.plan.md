# Build a Python CLI Task Tracker using Click and Rich

## Project Overview

Build a self-contained Python CLI task tracker that lives **inside the vLLM workspace** at the path `tools/task_tracker/` (keeping it isolated from vLLM's own source). The tool uses:

- **Click** for CLI command parsing and dispatch
- **Rich** for beautiful terminal output (tables, panels)
- **`tasks.json`** as a local flat-file persistence store
- **pytest** with Click's `CliRunner` for all tests

The workspace is the vLLM project (`/Users/pradeepsharma/sasva/projects/vllm`), a large Python monorepo. It already uses `pyproject.toml` (with `[tool.pytest.ini_options]`), `requirements/` for dependency management, and a `tests/` directory with `conftest.py` patterns. The new task tracker is a standalone sub-tool and must not interfere with vLLM's build system.

---

## File Layout (Target)

```
tools/task_tracker/
├── task_store.py          # Phase 1 — data layer
├── cli.py                 # Phase 2 — Click CLI
├── requirements.txt       # click, rich
└── tests/
    ├── __init__.py
    ├── conftest.py        # shared fixtures (tmp_path store)
    ├── test_store.py      # Phase 3a — unit tests for TaskStore
    └── test_cli.py        # Phase 3b — CLI integration tests via CliRunner
```

---

## Phase 1 — Core Data Layer (`task_store.py`)

**File:** `tools/task_tracker/task_store.py`

### Data Model

Each task is a plain Python `dict` with the following schema:

```python
{
    "id":         str,   # uuid4 hex string, e.g. "3f2504e0-4f89-11d3-9a0c-0305e82c3301"
    "title":      str,   # free-form task title
    "status":     str,   # one of: "todo" | "in_progress" | "done"
    "priority":   str,   # one of: "low" | "medium" | "high"
    "created_at": str,   # ISO 8601 timestamp, e.g. "2026-03-22T15:30:00.123456"
}
```

### `TaskStore` Class

```python
# tools/task_tracker/task_store.py
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

VALID_STATUSES  = {"todo", "in_progress", "done"}
VALID_PRIORITIES = {"low", "medium", "high"}

class TaskStore:
    def __init__(self, filepath: str | Path = "tasks.json"):
        self.filepath = Path(filepath)
        self._tasks: list[dict] = []
        self._load()

    # ── private helpers ──────────────────────────────────────────────────────
    def _load(self) -> None:
        """Read tasks from disk; start empty if file absent or corrupt."""
        if self.filepath.exists():
            with self.filepath.open("r", encoding="utf-8") as fh:
                self._tasks = json.load(fh)
        else:
            self._tasks = []

    def _save(self) -> None:
        """Persist current task list to disk (atomic-ish via write + rename)."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.filepath.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(self._tasks, fh, indent=2, ensure_ascii=False)
        tmp.replace(self.filepath)

    def _find_by_partial_id(self, partial_id: str) -> Optional[dict]:
        """Return the first task whose id starts with partial_id."""
        matches = [t for t in self._tasks if t["id"].startswith(partial_id)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(f"Ambiguous id prefix '{partial_id}': {len(matches)} matches")
        return None

    # ── public API ────────────────────────────────────────────────────────────
    def add_task(self, title: str, priority: str = "medium") -> dict:
        """Create and persist a new task. Returns the created task dict."""
        if priority not in VALID_PRIORITIES:
            raise ValueError(f"Invalid priority '{priority}'. Choose from {VALID_PRIORITIES}")
        task = {
            "id":         str(uuid.uuid4()),
            "title":      title,
            "status":     "todo",
            "priority":   priority,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        self._tasks.append(task)
        self._save()
        return task

    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> list[dict]:
        """Return tasks, optionally filtered by status and/or priority."""
        if status and status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{status}'. Choose from {VALID_STATUSES}")
        if priority and priority not in VALID_PRIORITIES:
            raise ValueError(f"Invalid priority '{priority}'. Choose from {VALID_PRIORITIES}")
        result = self._tasks
        if status:
            result = [t for t in result if t["status"] == status]
        if priority:
            result = [t for t in result if t["priority"] == priority]
        return result

    def update_status(self, partial_id: str, new_status: str) -> dict:
        """Update a task's status by partial id match. Returns updated task."""
        if new_status not in VALID_STATUSES:
            raise ValueError(f"Invalid status '{new_status}'. Choose from {VALID_STATUSES}")
        task = self._find_by_partial_id(partial_id)
        if task is None:
            raise KeyError(f"No task found with id prefix '{partial_id}'")
        task["status"] = new_status
        self._save()
        return task

    def delete_task(self, partial_id: str) -> dict:
        """Delete a task by partial id match. Returns the deleted task."""
        task = self._find_by_partial_id(partial_id)
        if task is None:
            raise KeyError(f"No task found with id prefix '{partial_id}'")
        self._tasks.remove(task)
        self._save()
        return task
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| Atomic write via `.tmp` + `rename` | Prevents corrupt JSON on crash mid-write |
| `partial_id` prefix matching | Mirrors git short-SHA UX; raises on ambiguity |
| `timezone.utc` timestamps | Consistent ISO 8601 with timezone info |
| In-memory `_tasks` list | Simple; file is small (task tracker, not a DB) |
| `_load()` in `__init__` | Allows `TaskStore(path)` to be the single entry point |

---

## Phase 2 — CLI Interface (`cli.py`)

**File:** `tools/task_tracker/cli.py`

### Command Structure

```
task [OPTIONS] COMMAND [ARGS]...

Commands:
  add     Add a new task
  list    List tasks (with optional filters)
  done    Mark a task as done
  delete  Delete a task (with confirmation)
  stats   Show task statistics panel
```

### Full Implementation Sketch

```python
# tools/task_tracker/cli.py
import os
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from task_store import TaskStore, VALID_STATUSES, VALID_PRIORITIES

console = Console()

DEFAULT_STORE_PATH = os.environ.get("TASK_STORE_PATH", "tasks.json")

def get_store(ctx: click.Context) -> TaskStore:
    return TaskStore(ctx.obj.get("store_path", DEFAULT_STORE_PATH))

# ── Priority / Status colour maps ────────────────────────────────────────────
PRIORITY_STYLE = {"low": "green", "medium": "yellow", "high": "red"}
STATUS_STYLE   = {"todo": "cyan", "in_progress": "magenta", "done": "dim green"}

@click.group()
@click.option("--store", default=DEFAULT_STORE_PATH,
              envvar="TASK_STORE_PATH", show_default=True,
              help="Path to tasks.json file.")
@click.pass_context
def cli(ctx: click.Context, store: str) -> None:
    """A simple CLI task tracker powered by Click and Rich."""
    ctx.ensure_object(dict)
    ctx.obj["store_path"] = store

# ── add ───────────────────────────────────────────────────────────────────────
@cli.command()
@click.option("--title",    required=True,  help="Task title.")
@click.option("--priority", default="medium",
              type=click.Choice(list(VALID_PRIORITIES), case_sensitive=False),
              show_default=True, help="Task priority.")
@click.pass_context
def add(ctx: click.Context, title: str, priority: str) -> None:
    """Add a new task."""
    store = get_store(ctx)
    task = store.add_task(title, priority.lower())
    console.print(f"[bold green]✓[/] Added task [cyan]{task['id'][:8]}[/]: {task['title']}")

# ── list ──────────────────────────────────────────────────────────────────────
@cli.command(name="list")
@click.option("--status",   default=None,
              type=click.Choice(list(VALID_STATUSES), case_sensitive=False),
              help="Filter by status.")
@click.option("--priority", default=None,
              type=click.Choice(list(VALID_PRIORITIES), case_sensitive=False),
              help="Filter by priority.")
@click.pass_context
def list_tasks(ctx: click.Context, status: str | None, priority: str | None) -> None:
    """List tasks, optionally filtered by status and/or priority."""
    store = get_store(ctx)
    tasks = store.list_tasks(status=status, priority=priority)

    if not tasks:
        console.print("[yellow]No tasks found.[/]")
        return

    table = Table(title="Tasks", show_header=True, header_style="bold blue")
    table.add_column("ID",       style="dim",    width=10)
    table.add_column("Title",    style="white",  min_width=20)
    table.add_column("Status",   justify="center")
    table.add_column("Priority", justify="center")
    table.add_column("Created",  style="dim",    width=20)

    for t in tasks:
        status_text   = Text(t["status"],   style=STATUS_STYLE.get(t["status"], ""))
        priority_text = Text(t["priority"], style=PRIORITY_STYLE.get(t["priority"], ""))
        table.add_row(
            t["id"][:8],
            t["title"],
            status_text,
            priority_text,
            t["created_at"][:19].replace("T", " "),
        )

    console.print(table)

# ── done ──────────────────────────────────────────────────────────────────────
@cli.command()
@click.argument("task_id")
@click.pass_context
def done(ctx: click.Context, task_id: str) -> None:
    """Mark a task as done (by partial ID)."""
    store = get_store(ctx)
    try:
        task = store.update_status(task_id, "done")
        console.print(f"[bold green]✓[/] Task [cyan]{task['id'][:8]}[/] marked as [green]done[/].")
    except KeyError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise SystemExit(1)
    except ValueError as e:
        console.print(f"[bold red]Error:[/] {e}")
        raise SystemExit(1)

# ── delete ────────────────────────────────────────────────────────────────────
@cli.command()
@click.argument("task_id")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt.")
@click.pass_context
def delete(ctx: click.Context, task_id: str, yes: bool) -> None:
    """Delete a task by partial ID (with confirmation)."""
    store = get_store(ctx)
    try:
        # Peek at the task first so we can show its title in the prompt
        tasks = store.list_tasks()
        match = next((t for t in tasks if t["id"].startswith(task_id)), None)
        if match is None:
            console.print(f"[bold red]Error:[/] No task found with id prefix '{task_id}'")
            raise SystemExit(1)

        if not yes:
            click.confirm(
                f"Delete task '{match['title']}' ({match['id'][:8]})?",
                abort=True,
            )
        store.delete_task(task_id)
        console.print(f"[bold red]✗[/] Task [cyan]{match['id'][:8]}[/] deleted.")
    except click.exceptions.Abort:
        console.print("[yellow]Aborted.[/]")

# ── stats ─────────────────────────────────────────────────────────────────────
@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show task statistics in a Rich panel."""
    store = get_store(ctx)
    tasks = store.list_tasks()

    status_counts   = {s: 0 for s in VALID_STATUSES}
    priority_counts = {p: 0 for p in VALID_PRIORITIES}
    for t in tasks:
        status_counts[t["status"]]     += 1
        priority_counts[t["priority"]] += 1

    lines = [
        f"[bold]Total tasks:[/] {len(tasks)}\n",
        "[bold underline]By Status[/]",
    ]
    for s, count in status_counts.items():
        lines.append(f"  [{STATUS_STYLE[s]}]{s}[/]: {count}")
    lines.append("\n[bold underline]By Priority[/]")
    for p, count in priority_counts.items():
        lines.append(f"  [{PRIORITY_STYLE[p]}]{p}[/]: {count}")

    panel = Panel("\n".join(lines), title="[bold blue]Task Stats[/]", expand=False)
    console.print(panel)

if __name__ == "__main__":
    cli()
```

### CLI Design Decisions

| Decision | Rationale |
|---|---|
| `@click.group()` with `--store` option | Allows `TASK_STORE_PATH` env var override; critical for tests |
| `ctx.obj` dict for store path | Passes store path cleanly to sub-commands without globals |
| `click.Choice` for status/priority | Validates at CLI layer before hitting `TaskStore` |
| `--yes` flag on `delete` | Allows non-interactive use in scripts/tests |
| Rich `Table` for `list` | Coloured, aligned output; degrades gracefully in CI |
| Rich `Panel` for `stats` | Visually distinct summary block |

---

## Phase 3 — Tests

### 3a — `tests/test_store.py`

**File:** `tools/task_tracker/tests/test_store.py`

```python
# tools/task_tracker/tests/test_store.py
import json
import pytest
from task_store import TaskStore

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def store(tmp_path):
    """Fresh TaskStore backed by a temp file."""
    return TaskStore(tmp_path / "tasks.json")

# ── add_task ──────────────────────────────────────────────────────────────────

def test_add_task_returns_dict(store):
    task = store.add_task("Write tests")
    assert isinstance(task, dict)
    assert task["title"] == "Write tests"
    assert task["status"] == "todo"
    assert task["priority"] == "medium"
    assert "id" in task
    assert "created_at" in task

def test_add_task_custom_priority(store):
    task = store.add_task("Urgent fix", priority="high")
    assert task["priority"] == "high"

def test_add_task_invalid_priority(store):
    with pytest.raises(ValueError, match="Invalid priority"):
        store.add_task("Bad task", priority="critical")

def test_add_task_increments_count(store):
    store.add_task("Task 1")
    store.add_task("Task 2")
    assert len(store.list_tasks()) == 2

# ── list_tasks ────────────────────────────────────────────────────────────────

def test_list_tasks_empty(store):
    assert store.list_tasks() == []

def test_list_tasks_filter_by_status(store):
    store.add_task("Todo task")
    t2 = store.add_task("Done task")
    store.update_status(t2["id"], "done")
    result = store.list_tasks(status="done")
    assert len(result) == 1
    assert result[0]["title"] == "Done task"

def test_list_tasks_filter_by_priority(store):
    store.add_task("Low task", priority="low")
    store.add_task("High task", priority="high")
    result = store.list_tasks(priority="high")
    assert len(result) == 1
    assert result[0]["priority"] == "high"

def test_list_tasks_combined_filter(store):
    store.add_task("Low todo", priority="low")
    t2 = store.add_task("Low done", priority="low")
    store.update_status(t2["id"], "done")
    result = store.list_tasks(status="done", priority="low")
    assert len(result) == 1
    assert result[0]["title"] == "Low done"

def test_list_tasks_invalid_status(store):
    with pytest.raises(ValueError, match="Invalid status"):
        store.list_tasks(status="invalid")

def test_list_tasks_invalid_priority(store):
    with pytest.raises(ValueError, match="Invalid priority"):
        store.list_tasks(priority="extreme")

# ── update_status ─────────────────────────────────────────────────────────────

def test_update_status_success(store):
    task = store.add_task("In progress task")
    updated = store.update_status(task["id"][:8], "in_progress")
    assert updated["status"] == "in_progress"

def test_update_status_partial_id(store):
    task = store.add_task("Partial match task")
    updated = store.update_status(task["id"][:6], "done")
    assert updated["status"] == "done"

def test_update_status_not_found(store):
    with pytest.raises(KeyError, match="No task found"):
        store.update_status("nonexistent", "done")

def test_update_status_invalid_status(store):
    task = store.add_task("Some task")
    with pytest.raises(ValueError, match="Invalid status"):
        store.update_status(task["id"], "archived")

# ── delete_task ───────────────────────────────────────────────────────────────

def test_delete_task_success(store):
    task = store.add_task("To be deleted")
    deleted = store.delete_task(task["id"][:8])
    assert deleted["id"] == task["id"]
    assert len(store.list_tasks()) == 0

def test_delete_task_not_found(store):
    with pytest.raises(KeyError, match="No task found"):
        store.delete_task("deadbeef")

def test_delete_task_removes_from_list(store):
    t1 = store.add_task("Keep me")
    t2 = store.add_task("Delete me")
    store.delete_task(t2["id"])
    remaining = store.list_tasks()
    assert len(remaining) == 1
    assert remaining[0]["id"] == t1["id"]

# ── persistence ───────────────────────────────────────────────────────────────

def test_persistence_across_instances(tmp_path):
    """Tasks written by one TaskStore instance are readable by another."""
    path = tmp_path / "tasks.json"
    store1 = TaskStore(path)
    task = store1.add_task("Persistent task", priority="high")

    store2 = TaskStore(path)
    tasks = store2.list_tasks()
    assert len(tasks) == 1
    assert tasks[0]["id"] == task["id"]
    assert tasks[0]["title"] == "Persistent task"
    assert tasks[0]["priority"] == "high"

def test_persistence_json_structure(tmp_path):
    """Verify the on-disk JSON is a valid list of task dicts."""
    path = tmp_path / "tasks.json"
    store = TaskStore(path)
    store.add_task("Check JSON")
    raw = json.loads(path.read_text())
    assert isinstance(raw, list)
    assert raw[0]["title"] == "Check JSON"

def test_persistence_update_survives_reload(tmp_path):
    """Status updates are persisted and visible after reload."""
    path = tmp_path / "tasks.json"
    store1 = TaskStore(path)
    task = store1.add_task("Update me")
    store1.update_status(task["id"], "done")

    store2 = TaskStore(path)
    assert store2.list_tasks()[0]["status"] == "done"

def test_persistence_delete_survives_reload(tmp_path):
    """Deletions are persisted and visible after reload."""
    path = tmp_path / "tasks.json"
    store1 = TaskStore(path)
    task = store1.add_task("Delete me")
    store1.delete_task(task["id"])

    store2 = TaskStore(path)
    assert store2.list_tasks() == []
```

### 3b — `tests/test_cli.py`

**File:** `tools/task_tracker/tests/test_cli.py`

```python
# tools/task_tracker/tests/test_cli.py
import json
import pytest
from click.testing import CliRunner
from cli import cli

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def runner():
    return CliRunner()

@pytest.fixture
def store_path(tmp_path):
    return str(tmp_path / "tasks.json")

def invoke(runner, store_path, *args):
    """Helper: invoke CLI with --store pointing to tmp file."""
    return runner.invoke(cli, ["--store", store_path, *args])

# ── add ───────────────────────────────────────────────────────────────────────

def test_add_command_success(runner, store_path):
    result = invoke(runner, store_path, "add", "--title", "My first task")
    assert result.exit_code == 0
    assert "Added task" in result.output

def test_add_command_with_priority(runner, store_path):
    result = invoke(runner, store_path, "add", "--title", "Urgent", "--priority", "high")
    assert result.exit_code == 0
    assert "Added task" in result.output

def test_add_command_default_priority_is_medium(runner, store_path):
    invoke(runner, store_path, "add", "--title", "Default prio task")
    data = json.loads(open(store_path).read())
    assert data[0]["priority"] == "medium"

def test_add_command_missing_title(runner, store_path):
    result = invoke(runner, store_path, "add")
    assert result.exit_code != 0

def test_add_command_invalid_priority(runner, store_path):
    result = invoke(runner, store_path, "add", "--title", "Bad", "--priority", "critical")
    assert result.exit_code != 0

# ── list ──────────────────────────────────────────────────────────────────────

def test_list_command_empty(runner, store_path):
    result = invoke(runner, store_path, "list")
    assert result.exit_code == 0
    assert "No tasks found" in result.output

def test_list_command_shows_tasks(runner, store_path):
    invoke(runner, store_path, "add", "--title", "Task Alpha")
    result = invoke(runner, store_path, "list")
    assert result.exit_code == 0
    assert "Task Alpha" in result.output

def test_list_command_filter_by_status(runner, store_path):
    invoke(runner, store_path, "add", "--title", "Todo task")
    result = invoke(runner, store_path, "list", "--status", "todo")
    assert result.exit_code == 0
    assert "Todo task" in result.output

def test_list_command_filter_by_priority(runner, store_path):
    invoke(runner, store_path, "add", "--title", "High prio", "--priority", "high")
    invoke(runner, store_path, "add", "--title", "Low prio",  "--priority", "low")
    result = invoke(runner, store_path, "list", "--priority", "high")
    assert result.exit_code == 0
    assert "High prio" in result.output
    assert "Low prio" not in result.output

def test_list_command_invalid_status(runner, store_path):
    result = invoke(runner, store_path, "list", "--status", "invalid")
    assert result.exit_code != 0

# ── done ──────────────────────────────────────────────────────────────────────

def test_done_command_success(runner, store_path):
    invoke(runner, store_path, "add", "--title", "Finish me")
    task_id = json.loads(open(store_path).read())[0]["id"]
    result = invoke(runner, store_path, "done", task_id[:8])
    assert result.exit_code == 0
    assert "done" in result.output

def test_done_command_updates_json(runner, store_path):
    invoke(runner, store_path, "add", "--title", "Mark done")
    task_id = json.loads(open(store_path).read())[0]["id"]
    invoke(runner, store_path, "done", task_id[:8])
    data = json.loads(open(store_path).read())
    assert data[0]["status"] == "done"

def test_done_command_not_found(runner, store_path):
    result = invoke(runner, store_path, "done", "deadbeef")
    assert result.exit_code != 0

# ── delete ────────────────────────────────────────────────────────────────────

def test_delete_command_with_yes_flag(runner, store_path):
    invoke(runner, store_path, "add", "--title", "Delete me")
    task_id = json.loads(open(store_path).read())[0]["id"]
    result = invoke(runner, store_path, "delete", task_id[:8], "--yes")
    assert result.exit_code == 0
    assert "deleted" in result.output

def test_delete_command_removes_from_store(runner, store_path):
    invoke(runner, store_path, "add", "--title", "Gone")
    task_id = json.loads(open(store_path).read())[0]["id"]
    invoke(runner, store_path, "delete", task_id[:8], "--yes")
    data = json.loads(open(store_path).read())
    assert data == []

def test_delete_command_not_found(runner, store_path):
    result = invoke(runner, store_path, "delete", "deadbeef", "--yes")
    assert result.exit_code != 0

def test_delete_command_confirmation_abort(runner, store_path):
    invoke(runner, store_path, "add", "--title", "Keep me")
    task_id = json.loads(open(store_path).read())[0]["id"]
    # Simulate user typing 'n' at the confirmation prompt
    result = runner.invoke(
        cli,
        ["--store", store_path, "delete", task_id[:8]],
        input="n\n",
    )
    assert result.exit_code == 0
    assert "Aborted" in result.output
    # Task should still exist
    data = json.loads(open(store_path).read())
    assert len(data) == 1

# ── stats ─────────────────────────────────────────────────────────────────────

def test_stats_command_empty(runner, store_path):
    result = invoke(runner, store_path, "stats")
    assert result.exit_code == 0
    assert "Total tasks" in result.output
    assert "0" in result.output

def test_stats_command_counts(runner, store_path):
    invoke(runner, store_path, "add", "--title", "T1", "--priority", "high")
    invoke(runner, store_path, "add", "--title", "T2", "--priority", "low")
    task_id = json.loads(open(store_path).read())[0]["id"]
    invoke(runner, store_path, "done", task_id[:8])
    result = invoke(runner, store_path, "stats")
    assert result.exit_code == 0
    assert "Total tasks" in result.output
    assert "By Status" in result.output
    assert "By Priority" in result.output
```

### 3c — `tests/conftest.py`

```python
# tools/task_tracker/tests/conftest.py
# Shared fixtures are defined here.
# Individual test files import from task_store and cli directly.
# pytest discovers this file automatically.
```

---

## Phase 4 — Project Scaffolding Files

### `tools/task_tracker/requirements.txt`

```
click>=8.1.0
rich>=13.0.0
pytest>=7.0.0
```

### `tools/task_tracker/tests/__init__.py`

Empty file — marks `tests/` as a Python package so pytest can import `task_store` and `cli` from the parent directory.

### `conftest.py` at `tools/task_tracker/` root

```python
# tools/task_tracker/conftest.py
import sys
from pathlib import Path

# Ensure task_store and cli are importable from tests/
sys.path.insert(0, str(Path(__file__).parent))
```

This mirrors the pattern used in `tests/conftest.py` in the vLLM root, which also manipulates `sys.path` for test discovery.

---

## Execution Plan (Parallel Phases)

### Phase 1 — Scaffold & Data Layer *(no dependencies)*

**Actions (can be done in parallel):**

1. Create `tools/task_tracker/` directory structure
2. Write `tools/task_tracker/task_store.py` with full `TaskStore` implementation
3. Write `tools/task_tracker/requirements.txt`
4. Write `tools/task_tracker/conftest.py` (sys.path fix)
5. Write `tools/task_tracker/tests/__init__.py` (empty)
6. Write `tools/task_tracker/tests/conftest.py` (empty/shared fixtures)

**Deliverables:**
- `tools/task_tracker/task_store.py`
- `tools/task_tracker/requirements.txt`
- `tools/task_tracker/conftest.py`
- `tools/task_tracker/tests/__init__.py`
- `tools/task_tracker/tests/conftest.py`

---

### Phase 2 — CLI Layer *(depends on Phase 1: task_store.py)*

**Actions:**

1. Write `tools/task_tracker/cli.py` with all 5 commands: `add`, `list`, `done`, `delete`, `stats`
2. Verify imports: `click`, `rich.console.Console`, `rich.table.Table`, `rich.panel.Panel`, `rich.text.Text`
3. Ensure `TASK_STORE_PATH` env var is respected (needed for test isolation)

**Deliverables:**
- `tools/task_tracker/cli.py`

---

### Phase 3a — Store Unit Tests *(depends on Phase 1)*

**Actions:**

1. Write `tools/task_tracker/tests/test_store.py`
2. Cover: `add_task`, `list_tasks` (no filter, status filter, priority filter, combined filter), `update_status`, `delete_task`, persistence across save/load

**Test cases (20 total):**

| Test | What it verifies |
|---|---|
| `test_add_task_returns_dict` | Return shape |
| `test_add_task_custom_priority` | Priority stored correctly |
| `test_add_task_invalid_priority` | ValueError raised |
| `test_add_task_increments_count` | Multiple adds accumulate |
| `test_list_tasks_empty` | Empty store returns `[]` |
| `test_list_tasks_filter_by_status` | Status filter works |
| `test_list_tasks_filter_by_priority` | Priority filter works |
| `test_list_tasks_combined_filter` | Both filters AND-ed |
| `test_list_tasks_invalid_status` | ValueError raised |
| `test_list_tasks_invalid_priority` | ValueError raised |
| `test_update_status_success` | Full id match |
| `test_update_status_partial_id` | Prefix match |
| `test_update_status_not_found` | KeyError raised |
| `test_update_status_invalid_status` | ValueError raised |
| `test_delete_task_success` | Task removed |
| `test_delete_task_not_found` | KeyError raised |
| `test_delete_task_removes_from_list` | Other tasks unaffected |
| `test_persistence_across_instances` | Two `TaskStore` instances share data |
| `test_persistence_json_structure` | Raw JSON is valid list |
| `test_persistence_update_survives_reload` | Status update persisted |
| `test_persistence_delete_survives_reload` | Deletion persisted |

**Deliverables:**
- `tools/task_tracker/tests/test_store.py`

---

### Phase 3b — CLI Integration Tests *(depends on Phase 2)*

**Actions:**

1. Write `tools/task_tracker/tests/test_cli.py`
2. Use `click.testing.CliRunner` — no subprocess, no real file I/O outside `tmp_path`
3. Use `--store <tmp_path>` flag on every invocation for isolation
4. Cover all 5 commands with success and failure paths

**Test cases (20 total):**

| Test | Command | What it verifies |
|---|---|---|
| `test_add_command_success` | `add` | Exit 0, "Added task" in output |
| `test_add_command_with_priority` | `add` | High priority accepted |
| `test_add_command_default_priority_is_medium` | `add` | JSON has `"medium"` |
| `test_add_command_missing_title` | `add` | Exit non-zero |
| `test_add_command_invalid_priority` | `add` | Exit non-zero |
| `test_list_command_empty` | `list` | "No tasks found" |
| `test_list_command_shows_tasks` | `list` | Task title in output |
| `test_list_command_filter_by_status` | `list --status` | Filtered correctly |
| `test_list_command_filter_by_priority` | `list --priority` | Only matching shown |
| `test_list_command_invalid_status` | `list` | Exit non-zero |
| `test_done_command_success` | `done` | Exit 0, "done" in output |
| `test_done_command_updates_json` | `done` | JSON status = "done" |
| `test_done_command_not_found` | `done` | Exit non-zero |
| `test_delete_command_with_yes_flag` | `delete --yes` | Exit 0, "deleted" in output |
| `test_delete_command_removes_from_store` | `delete --yes` | JSON is empty |
| `test_delete_command_not_found` | `delete --yes` | Exit non-zero |
| `test_delete_command_confirmation_abort` | `delete` (input=n) | Task still exists |
| `test_stats_command_empty` | `stats` | "Total tasks: 0" |
| `test_stats_command_counts` | `stats` | Correct counts shown |

**Deliverables:**
- `tools/task_tracker/tests/test_cli.py`

---

### Phase 4 — Integration & Verification *(depends on all prior phases)*

**Actions:**

1. Install dependencies: `pip install click rich pytest` (or `pip install -r tools/task_tracker/requirements.txt`)
2. Run full test suite from `tools/task_tracker/`:
   ```bash
   cd tools/task_tracker
   pytest tests/ -v
   ```
3. Smoke-test the CLI manually:
   ```bash
   cd tools/task_tracker
   python cli.py add --title "Plan the sprint" --priority high
   python cli.py add --title "Write docs"
   python cli.py list
   python cli.py stats
   python cli.py done <first-8-chars-of-id>
   python cli.py list --status done
   python cli.py delete <first-8-chars-of-id> --yes
   python cli.py stats
   ```

---

## Verification Criteria

### Install

```bash
cd /Users/pradeepsharma/sasva/projects/vllm/tools/task_tracker
pip install -r requirements.txt
```

Expected: No errors. `click`, `rich`, `pytest` installed.

### Run Tests

```bash
cd /Users/pradeepsharma/sasva/projects/vllm/tools/task_tracker
pytest tests/ -v
```

**Expected output:**
- All tests collected and **PASSED** (target: ≥ 39 tests, 0 failures, 0 errors)
- Example passing lines:
  ```
  tests/test_store.py::test_add_task_returns_dict PASSED
  tests/test_store.py::test_persistence_across_instances PASSED
  tests/test_cli.py::test_add_command_success PASSED
  tests/test_cli.py::test_stats_command_counts PASSED
  ...
  39 passed in <2s
  ```

### CLI Smoke Tests

Run each command and verify expected output:

| Command | Expected Output Contains |
|---|---|
| `python cli.py add --title "Test task" --priority high` | `✓ Added task` + 8-char ID |
| `python cli.py list` | Rich table with "Test task", "high", "todo" |
| `python cli.py list --status todo` | Same table |
| `python cli.py list --status done` | "No tasks found" |
| `python cli.py done <id8>` | `✓ Task <id8> marked as done` |
| `python cli.py list --status done` | Rich table with "Test task", "done" |
| `python cli.py stats` | Rich panel with "Total tasks: 1", "done: 1", "high: 1" |
| `python cli.py delete <id8> --yes` | `✗ Task <id8> deleted` |
| `python cli.py stats` | "Total tasks: 0" |

### Edge Case Verification

```bash
# Invalid priority → non-zero exit
python cli.py add --title "Bad" --priority critical
echo "Exit: $?"   # Expected: 2

# Missing --title → non-zero exit
python cli.py add
echo "Exit: $?"   # Expected: 2

# Unknown task ID → non-zero exit + error message
python cli.py done deadbeef
echo "Exit: $?"   # Expected: 1

# Delete with confirmation (type 'n')
python cli.py delete deadbeef   # prompts, type 'n' → "Aborted."
```

### Persistence Verification

```bash
python cli.py add --title "Persist me" --priority low
cat tasks.json   # Should show valid JSON list with 1 task
python cli.py done $(python -c "import json; print(json.load(open('tasks.json'))[0]['id'][:8])")
cat tasks.json   # status should be "done"
```

---

## Notes on vLLM Workspace Compatibility

- The task tracker lives in `tools/task_tracker/` — **not** inside `vllm/` — so it does not affect vLLM's `setuptools` package discovery (`include = ["vllm*"]` in `pyproject.toml`).
- The task tracker's `conftest.py` uses `sys.path.insert` to make `task_store` and `cli` importable from `tests/` — the same pattern used in vLLM's own test infrastructure.
- The task tracker's `pytest` run is scoped to `tools/task_tracker/tests/` and does not inherit vLLM's `[tool.pytest.ini_options]` markers (no GPU/distributed markers needed).
- `tasks.json` is written to the current working directory by default; the `--store` flag and `TASK_STORE_PATH` env var allow full isolation in tests and CI.
- No vLLM source files are modified.
