# Build a Python CLI Task Tracker Using Click and Rich

**Generated:** March 19, 2026  
**Workspace:** `/Users/pradeepsharma/sasva/projects/vllm`  
**Target directory:** `team-test/` (new Click+Rich implementation alongside existing argparse version)

---

## Codebase Context

This workspace is the **vLLM** project — a high-throughput LLM inference engine. The relevant subdirectory is `team-test/`, which already contains a working argparse-based task tracker (363 tests, all passing). The new implementation will use **Click** and **Rich** as specified, living in the same `team-test/` directory alongside the existing code.

### Existing reference files (do NOT modify):
| File | Purpose |
|------|---------|
| `team-test/models.py` | `Task`, `Project`, `Status`, `Priority` dataclasses + enums |
| `team-test/storage.py` | `Storage` class — JSON persistence, CRUD, `NotFoundError`, `ValidationError` |
| `team-test/services.py` | `TaskService`, `ProjectService` — business logic layer |
| `team-test/requirements.txt` | Runtime: stdlib only; dev: `pytest>=9.0.2` |
| `team-test/tests/test_*.py` | 363 existing tests — must remain green |

### Key data model (from `team-test/models.py`):
```python
class Status(str, Enum):   # "todo" | "in_progress" | "done"
class Priority(str, Enum): # "low" | "medium" | "high" | "critical"

@dataclass
class Task:
    id: str          # uuid4 string
    title: str
    status: Status   # default: TODO
    priority: Priority  # default: MEDIUM
    created_at: str  # ISO-8601 UTC timestamp
    description: str = ""
    due_date: Optional[str] = None
    project_id: Optional[str] = None
    updated_at: str  # ISO-8601 UTC timestamp
```

### Storage API (from `team-test/storage.py`):
```python
class Storage:
    def __init__(self, data_path: str = "~/.taskman/data.json") -> None
    def create_task(title, description="", status=Status.TODO, priority=Priority.MEDIUM,
                    due_date=None, project_id=None) -> Task
    def get_task(task_id: str) -> Task          # raises NotFoundError
    def list_tasks(project_id=None, status=None, priority=None, overdue=False) -> List[Task]
    def update_task(task_id, **kwargs) -> Task  # raises NotFoundError, ValidationError
    def delete_task(task_id: str) -> None       # raises NotFoundError
    def stats() -> dict                         # counts by status/priority
```

---

## New Files to Create

| File | Description |
|------|-------------|
| `team-test/task_store.py` | `TaskStore` class wrapping `Storage` — the new data layer |
| `team-test/cli_click.py` | Click+Rich CLI with `add`, `list`, `done`, `delete`, `stats` commands |
| `team-test/tests/test_store.py` | pytest tests for `TaskStore` |
| `team-test/tests/test_cli_click.py` | pytest tests for Click CLI using `CliRunner` |
| `team-test/requirements_click.txt` | Click + Rich dependency pins |

---

## Phase 1 — Core Data Layer (`task_store.py`)

**File:** `team-test/task_store.py`

### Implementation

```python
"""TaskStore: thin wrapper around Storage providing the task_store.py interface.

Reads/writes tasks to a local tasks.json file.
Each task has:
  - id         (uuid4 string)
  - title      (str)
  - status     ("todo" | "in_progress" | "done")
  - priority   ("low" | "medium" | "high")
  - created_at (ISO-8601 UTC timestamp)
"""
from __future__ import annotations

import os
import sys
from typing import List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from models import Priority, Status, Task
from storage import NotFoundError, Storage, ValidationError

__all__ = ["TaskStore", "NotFoundError", "ValidationError"]

_DEFAULT_TASKS_JSON = os.path.join(
    os.path.expanduser("~"), ".taskman", "tasks.json"
)


class TaskStore:
    """Reads/writes tasks to a local tasks.json file.

    Parameters
    ----------
    path:
        Path to the JSON file. Defaults to ``~/.taskman/tasks.json``.
        Pass a custom path (e.g. a tmp file) for testing.
    """

    def __init__(self, path: str = _DEFAULT_TASKS_JSON) -> None:
        self._storage = Storage(data_path=path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_task(
        self,
        title: str,
        priority: str = "medium",
    ) -> Task:
        """Create a new task and persist it.

        Parameters
        ----------
        title:
            Human-readable task title (required, non-empty).
        priority:
            One of ``"low"``, ``"medium"``, ``"high"`` (default ``"medium"``).

        Returns
        -------
        Task
            The newly created :class:`~models.Task` instance.

        Raises
        ------
        ValidationError
            If *title* is empty or *priority* is not a valid value.
        """
        pri = Priority(priority.lower())
        return self._storage.create_task(title=title, priority=pri)

    def list_tasks(
        self,
        status: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> List[Task]:
        """Return tasks, optionally filtered by status and/or priority.

        Parameters
        ----------
        status:
            Optional filter: ``"todo"``, ``"in_progress"``, or ``"done"``.
        priority:
            Optional filter: ``"low"``, ``"medium"``, or ``"high"``.

        Returns
        -------
        List[Task]
            Matching tasks ordered by ``created_at`` ascending.
        """
        status_enum = Status(status.lower()) if status else None
        priority_enum = Priority(priority.lower()) if priority else None
        return self._storage.list_tasks(
            status=status_enum,
            priority=priority_enum,
        )

    def update_status(self, task_id: str, new_status: str) -> Task:
        """Update the status of a task identified by *task_id*.

        Supports partial id matching: if *task_id* is a prefix of exactly
        one stored task id, that task is updated.

        Parameters
        ----------
        task_id:
            Full or partial UUID string.
        new_status:
            One of ``"todo"``, ``"in_progress"``, ``"done"``.

        Returns
        -------
        Task
            The updated :class:`~models.Task` instance.

        Raises
        ------
        NotFoundError
            If no task matches *task_id* or more than one task matches.
        ValidationError
            If *new_status* is not a valid status value.
        """
        resolved_id = self._resolve_id(task_id)
        status_enum = Status(new_status.lower())
        return self._storage.update_task(resolved_id, status=status_enum)

    def delete_task(self, task_id: str) -> None:
        """Delete a task by full or partial id.

        Parameters
        ----------
        task_id:
            Full or partial UUID string.

        Raises
        ------
        NotFoundError
            If no task matches *task_id* or more than one task matches.
        """
        resolved_id = self._resolve_id(task_id)
        self._storage.delete_task(resolved_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_id(self, partial_id: str) -> str:
        """Return the full task id for a full or partial *partial_id*.

        Raises
        ------
        NotFoundError
            If zero or more-than-one tasks match the prefix.
        """
        all_tasks = self._storage.list_tasks()
        matches = [t for t in all_tasks if t.id.startswith(partial_id)]
        if len(matches) == 1:
            return matches[0].id
        if len(matches) == 0:
            raise NotFoundError(f"No task found matching id prefix '{partial_id}'.")
        raise NotFoundError(
            f"Ambiguous id prefix '{partial_id}' matches {len(matches)} tasks. "
            "Please provide more characters."
        )
```

### Key design decisions
- **Delegates to `Storage`** — no duplicate JSON logic; `Storage` already handles atomic writes, lazy loading, and parent-dir creation.
- **Partial id matching** in `_resolve_id` — scans `list_tasks()` for prefix matches; raises `NotFoundError` for 0 or >1 matches.
- **Enum coercion** — converts raw strings to `Status`/`Priority` enums before calling `Storage`, so `ValidationError` is raised on invalid values.
- **Default path** — `~/.taskman/tasks.json` (separate from the existing `data.json` used by the argparse CLI).

---

## Phase 2 — CLI Interface (`cli_click.py`)

**File:** `team-test/cli_click.py`

### Command structure

```
taskman
├── add     --title TEXT  --priority [low|medium|high]
├── list    --status [todo|in_progress|done]  --priority [low|medium|high]
├── done    TASK_ID
├── delete  TASK_ID
└── stats
```

### Implementation

```python
"""Click + Rich CLI for the task tracker.

Commands:
  add     Add a new task.
  list    List tasks (with optional filters). Displays a Rich table.
  done    Mark a task as done by partial id.
  delete  Delete a task by partial id (with confirmation prompt).
  stats   Show a Rich panel with counts per status and priority.

Usage:
  python cli_click.py --help
  python cli_click.py --data /path/to/tasks.json add --title "My task" --priority high
  python cli_click.py list --status todo
  python cli_click.py done abc123
  python cli_click.py delete abc123
  python cli_click.py stats
"""
from __future__ import annotations

import os
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from storage import NotFoundError, ValidationError  # noqa: E402
from task_store import TaskStore  # noqa: E402

console = Console()

# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--data",
    default=None,
    envvar="TASKMAN_DATA",
    help="Path to tasks JSON file (default: ~/.taskman/tasks.json).",
    metavar="PATH",
)
@click.pass_context
def cli(ctx: click.Context, data: str | None) -> None:
    """A simple CLI task tracker powered by Click and Rich."""
    ctx.ensure_object(dict)
    kwargs = {"path": data} if data else {}
    ctx.obj["store"] = TaskStore(**kwargs)


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--title", required=True, help="Task title.")
@click.option(
    "--priority",
    default="medium",
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    show_default=True,
    help="Task priority.",
)
@click.pass_context
def add(ctx: click.Context, title: str, priority: str) -> None:
    """Add a new task."""
    store: TaskStore = ctx.obj["store"]
    try:
        task = store.add_task(title=title, priority=priority)
        console.print(
            f"[green]✓[/green] Task added: [bold]{task.title}[/bold] "
            f"(id: [cyan]{task.id[:8]}…[/cyan], priority: [yellow]{task.priority.value}[/yellow])"
        )
    except ValidationError as exc:
        console.print(f"[red]Error:[/red] {exc}", err=True)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------

@cli.command(name="list")
@click.option(
    "--status",
    default=None,
    type=click.Choice(["todo", "in_progress", "done"], case_sensitive=False),
    help="Filter by status.",
)
@click.option(
    "--priority",
    default=None,
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    help="Filter by priority.",
)
@click.pass_context
def list_tasks(ctx: click.Context, status: str | None, priority: str | None) -> None:
    """List tasks, optionally filtered by status and/or priority."""
    store: TaskStore = ctx.obj["store"]
    tasks = store.list_tasks(status=status, priority=priority)

    if not tasks:
        console.print("[dim]No tasks found.[/dim]")
        return

    table = Table(title="Tasks", show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", no_wrap=True, width=10)
    table.add_column("Title", style="white")
    table.add_column("Status", style="green")
    table.add_column("Priority", style="yellow")
    table.add_column("Created At", style="dim")

    _STATUS_STYLE = {
        "todo": "blue",
        "in_progress": "yellow",
        "done": "green",
    }
    _PRIORITY_STYLE = {
        "low": "dim",
        "medium": "white",
        "high": "bold red",
    }

    for task in tasks:
        status_val = task.status.value
        priority_val = task.priority.value
        table.add_row(
            task.id[:8] + "…",
            task.title,
            f"[{_STATUS_STYLE.get(status_val, 'white')}]{status_val}[/]",
            f"[{_PRIORITY_STYLE.get(priority_val, 'white')}]{priority_val}[/]",
            task.created_at[:10],
        )

    console.print(table)
    console.print(f"[dim]{len(tasks)} task(s) shown.[/dim]")


# ---------------------------------------------------------------------------
# done
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("task_id")
@click.pass_context
def done(ctx: click.Context, task_id: str) -> None:
    """Mark a task as done (by full or partial id)."""
    store: TaskStore = ctx.obj["store"]
    try:
        task = store.update_status(task_id, "done")
        console.print(
            f"[green]✓[/green] Task [bold]{task.title}[/bold] "
            f"(id: [cyan]{task.id[:8]}…[/cyan]) marked as [green]done[/green]."
        )
    except NotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}", err=True)
        raise SystemExit(1)
    except ValidationError as exc:
        console.print(f"[red]Error:[/red] {exc}", err=True)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("task_id")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompt.")
@click.pass_context
def delete(ctx: click.Context, task_id: str, yes: bool) -> None:
    """Delete a task by full or partial id (prompts for confirmation)."""
    store: TaskStore = ctx.obj["store"]
    try:
        # Resolve first so we can show the title in the confirmation prompt.
        resolved_id = store._resolve_id(task_id)
        tasks = store.list_tasks()
        task = next(t for t in tasks if t.id == resolved_id)
    except NotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}", err=True)
        raise SystemExit(1)

    if not yes:
        confirmed = click.confirm(
            f"Delete task '{task.title}' (id: {task.id[:8]}…)?", default=False
        )
        if not confirmed:
            console.print("[dim]Aborted.[/dim]")
            return

    try:
        store.delete_task(task_id)
        console.print(
            f"[green]✓[/green] Task [bold]{task.title}[/bold] "
            f"(id: [cyan]{task.id[:8]}…[/cyan]) deleted."
        )
    except NotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}", err=True)
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------

@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show task counts per status and priority in a Rich panel."""
    store: TaskStore = ctx.obj["store"]
    all_tasks = store.list_tasks()

    if not all_tasks:
        console.print(Panel("[dim]No tasks in store.[/dim]", title="Stats"))
        return

    # Count by status
    status_counts: dict[str, int] = {"todo": 0, "in_progress": 0, "done": 0}
    priority_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0}

    for task in all_tasks:
        sv = task.status.value
        pv = task.priority.value
        if sv in status_counts:
            status_counts[sv] += 1
        if pv in priority_counts:
            priority_counts[pv] += 1

    lines = [
        "[bold underline]By Status[/bold underline]",
        f"  [blue]todo[/blue]        : {status_counts['todo']}",
        f"  [yellow]in_progress[/yellow] : {status_counts['in_progress']}",
        f"  [green]done[/green]        : {status_counts['done']}",
        "",
        "[bold underline]By Priority[/bold underline]",
        f"  [dim]low[/dim]         : {priority_counts['low']}",
        f"  [white]medium[/white]      : {priority_counts['medium']}",
        f"  [bold red]high[/bold red]        : {priority_counts['high']}",
        "",
        f"[bold]Total[/bold]         : {len(all_tasks)}",
    ]

    console.print(Panel("\n".join(lines), title="[bold cyan]Task Stats[/bold cyan]"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
```

### Key design decisions
- **`@click.group()` with `@click.pass_context`** — `TaskStore` is instantiated once in the group callback and stored in `ctx.obj["store"]`, so all sub-commands share the same store instance.
- **`--data` global option** — allows overriding the JSON file path; also reads `TASKMAN_DATA` env var. This is critical for test isolation with `CliRunner`.
- **`done` command** — uses `store.update_status(task_id, "done")` which internally calls `_resolve_id` for partial matching.
- **`delete` command** — resolves the id first to show the task title in the confirmation prompt; uses `click.confirm()` for interactive confirmation; `--yes/-y` flag skips the prompt (needed for `CliRunner` tests).
- **`stats` command** — uses `rich.panel.Panel` wrapping formatted markup lines.
- **`list` command** — uses `rich.table.Table` with per-column styles.
- **Error handling** — `NotFoundError` and `ValidationError` are caught, printed to stderr via `console.print(..., err=True)`, and exit with code 1.

---

## Phase 3 — Tests

### 3a. `tests/test_store.py`

**File:** `team-test/tests/test_store.py`

```python
"""Tests for team-test/task_store.py.

Covers:
- add_task: happy path, default priority, explicit priority, empty title error,
  invalid priority error
- list_tasks: empty store, no filters, status filter, priority filter,
  combined filter, no matches
- update_status: happy path, partial id match, invalid status, not found,
  ambiguous prefix
- delete_task: happy path, partial id match, not found, ambiguous prefix
- Persistence: save/load round-trip (data survives across TaskStore instances)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEAM_TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (WORKSPACE_ROOT, TEAM_TEST_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from storage import NotFoundError, ValidationError  # noqa: E402
from task_store import TaskStore  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path: Path) -> TaskStore:
    """Fresh TaskStore backed by a temp file."""
    return TaskStore(path=str(tmp_path / "tasks.json"))


@pytest.fixture()
def store_with_tasks(tmp_path: Path) -> tuple[TaskStore, list]:
    """TaskStore pre-populated with three tasks."""
    s = TaskStore(path=str(tmp_path / "tasks.json"))
    t1 = s.add_task("Task A", priority="low")
    t2 = s.add_task("Task B", priority="medium")
    t3 = s.add_task("Task C", priority="high")
    return s, [t1, t2, t3]


# ===========================================================================
# add_task
# ===========================================================================

class TestAddTask:
    def test_returns_task(self, store):
        task = store.add_task("Write tests")
        assert task.title == "Write tests"

    def test_default_priority_is_medium(self, store):
        task = store.add_task("My task")
        assert task.priority.value == "medium"

    def test_explicit_priority_low(self, store):
        task = store.add_task("Low task", priority="low")
        assert task.priority.value == "low"

    def test_explicit_priority_high(self, store):
        task = store.add_task("High task", priority="high")
        assert task.priority.value == "high"

    def test_default_status_is_todo(self, store):
        task = store.add_task("New task")
        assert task.status.value == "todo"

    def test_task_has_uuid_id(self, store):
        import uuid
        task = store.add_task("UUID task")
        uuid.UUID(task.id)  # raises ValueError if not valid UUID

    def test_task_has_created_at(self, store):
        task = store.add_task("Timestamped task")
        assert task.created_at  # non-empty ISO string

    def test_empty_title_raises_validation_error(self, store):
        with pytest.raises((ValidationError, ValueError)):
            store.add_task("")

    def test_invalid_priority_raises_value_error(self, store):
        with pytest.raises((ValidationError, ValueError)):
            store.add_task("Task", priority="urgent")

    def test_multiple_tasks_have_unique_ids(self, store):
        t1 = store.add_task("Task 1")
        t2 = store.add_task("Task 2")
        assert t1.id != t2.id


# ===========================================================================
# list_tasks
# ===========================================================================

class TestListTasks:
    def test_empty_store_returns_empty_list(self, store):
        assert store.list_tasks() == []

    def test_returns_all_tasks_without_filter(self, store_with_tasks):
        store, tasks = store_with_tasks
        result = store.list_tasks()
        assert len(result) == 3

    def test_filter_by_status_todo(self, store_with_tasks):
        store, _ = store_with_tasks
        result = store.list_tasks(status="todo")
        assert all(t.status.value == "todo" for t in result)
        assert len(result) == 3  # all start as todo

    def test_filter_by_status_done_empty(self, store_with_tasks):
        store, _ = store_with_tasks
        result = store.list_tasks(status="done")
        assert result == []

    def test_filter_by_priority_low(self, store_with_tasks):
        store, tasks = store_with_tasks
        result = store.list_tasks(priority="low")
        assert len(result) == 1
        assert result[0].id == tasks[0].id

    def test_filter_by_priority_high(self, store_with_tasks):
        store, tasks = store_with_tasks
        result = store.list_tasks(priority="high")
        assert len(result) == 1
        assert result[0].id == tasks[2].id

    def test_combined_status_and_priority_filter(self, store_with_tasks):
        store, tasks = store_with_tasks
        result = store.list_tasks(status="todo", priority="medium")
        assert len(result) == 1
        assert result[0].id == tasks[1].id

    def test_combined_filter_no_match(self, store_with_tasks):
        store, _ = store_with_tasks
        result = store.list_tasks(status="done", priority="high")
        assert result == []

    def test_invalid_status_raises(self, store):
        with pytest.raises((ValidationError, ValueError)):
            store.list_tasks(status="invalid")

    def test_invalid_priority_raises(self, store):
        with pytest.raises((ValidationError, ValueError)):
            store.list_tasks(priority="invalid")


# ===========================================================================
# update_status
# ===========================================================================

class TestUpdateStatus:
    def test_update_to_done(self, store):
        task = store.add_task("Finish me")
        updated = store.update_status(task.id, "done")
        assert updated.status.value == "done"

    def test_update_to_in_progress(self, store):
        task = store.add_task("Start me")
        updated = store.update_status(task.id, "in_progress")
        assert updated.status.value == "in_progress"

    def test_partial_id_match(self, store):
        task = store.add_task("Partial match task")
        prefix = task.id[:8]
        updated = store.update_status(prefix, "done")
        assert updated.id == task.id
        assert updated.status.value == "done"

    def test_not_found_raises(self, store):
        with pytest.raises(NotFoundError):
            store.update_status("nonexistent-id-xyz", "done")

    def test_invalid_status_raises(self, store):
        task = store.add_task("Task")
        with pytest.raises((ValidationError, ValueError)):
            store.update_status(task.id, "finished")

    def test_status_persisted(self, tmp_path):
        path = str(tmp_path / "tasks.json")
        s1 = TaskStore(path=path)
        task = s1.add_task("Persist status")
        s1.update_status(task.id, "done")

        s2 = TaskStore(path=path)
        tasks = s2.list_tasks()
        assert tasks[0].status.value == "done"


# ===========================================================================
# delete_task
# ===========================================================================

class TestDeleteTask:
    def test_delete_removes_task(self, store):
        task = store.add_task("Delete me")
        store.delete_task(task.id)
        assert store.list_tasks() == []

    def test_partial_id_delete(self, store):
        task = store.add_task("Partial delete")
        prefix = task.id[:8]
        store.delete_task(prefix)
        assert store.list_tasks() == []

    def test_not_found_raises(self, store):
        with pytest.raises(NotFoundError):
            store.delete_task("nonexistent-id-xyz")

    def test_delete_only_target(self, store_with_tasks):
        store, tasks = store_with_tasks
        store.delete_task(tasks[0].id)
        remaining = store.list_tasks()
        assert len(remaining) == 2
        assert all(t.id != tasks[0].id for t in remaining)


# ===========================================================================
# Persistence (save/load round-trip)
# ===========================================================================

class TestPersistence:
    def test_tasks_survive_reload(self, tmp_path):
        path = str(tmp_path / "tasks.json")
        s1 = TaskStore(path=path)
        t = s1.add_task("Survive reload", priority="high")

        s2 = TaskStore(path=path)
        tasks = s2.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].id == t.id
        assert tasks[0].title == "Survive reload"
        assert tasks[0].priority.value == "high"

    def test_multiple_tasks_survive_reload(self, tmp_path):
        path = str(tmp_path / "tasks.json")
        s1 = TaskStore(path=path)
        s1.add_task("Task 1", priority="low")
        s1.add_task("Task 2", priority="medium")
        s1.add_task("Task 3", priority="high")

        s2 = TaskStore(path=path)
        tasks = s2.list_tasks()
        assert len(tasks) == 3

    def test_delete_persists_across_reload(self, tmp_path):
        path = str(tmp_path / "tasks.json")
        s1 = TaskStore(path=path)
        t = s1.add_task("Delete and reload")
        s1.delete_task(t.id)

        s2 = TaskStore(path=path)
        assert s2.list_tasks() == []

    def test_status_update_persists_across_reload(self, tmp_path):
        path = str(tmp_path / "tasks.json")
        s1 = TaskStore(path=path)
        t = s1.add_task("Status persist")
        s1.update_status(t.id, "in_progress")

        s2 = TaskStore(path=path)
        tasks = s2.list_tasks()
        assert tasks[0].status.value == "in_progress"

    def test_empty_store_file_created(self, tmp_path):
        path = str(tmp_path / "tasks.json")
        s = TaskStore(path=path)
        s.add_task("Create file")
        assert os.path.exists(path)
```

### 3b. `tests/test_cli_click.py`

**File:** `team-test/tests/test_cli_click.py`

```python
"""Tests for team-test/cli_click.py using Click's CliRunner.

Covers:
- add: happy path, default priority, explicit priority, missing title,
  invalid priority
- list: empty store, all tasks, --status filter, --priority filter,
  combined filter, Rich table output
- done: happy path, partial id, not found
- delete: happy path with --yes flag, not found, aborted confirmation
- stats: empty store, populated store, panel output
- Global --data option for test isolation
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from click.testing import CliRunner

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
TEAM_TEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for _p in (WORKSPACE_ROOT, TEAM_TEST_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cli_click import cli  # noqa: E402
from task_store import TaskStore  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(runner: CliRunner, tmp_path: Path, *args: str):
    """Invoke CLI with --data pointing to a temp file."""
    data_file = str(tmp_path / "tasks.json")
    return runner.invoke(cli, ["--data", data_file, *args], catch_exceptions=False)


def add_task(runner: CliRunner, tmp_path: Path, title: str, priority: str = "medium"):
    """Helper: add a task and return the result."""
    return run(runner, tmp_path, "add", "--title", title, "--priority", priority)


# ===========================================================================
# add command
# ===========================================================================

class TestAddCommand:
    def test_add_happy_path(self, tmp_path):
        runner = CliRunner()
        result = add_task(runner, tmp_path, "Write documentation")
        assert result.exit_code == 0
        assert "Write documentation" in result.output

    def test_add_default_priority_medium(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        result = runner.invoke(cli, ["--data", data_file, "add", "--title", "Default prio"])
        assert result.exit_code == 0
        assert "medium" in result.output

    def test_add_priority_high(self, tmp_path):
        runner = CliRunner()
        result = add_task(runner, tmp_path, "Urgent task", priority="high")
        assert result.exit_code == 0
        assert "high" in result.output

    def test_add_priority_low(self, tmp_path):
        runner = CliRunner()
        result = add_task(runner, tmp_path, "Low priority task", priority="low")
        assert result.exit_code == 0
        assert "low" in result.output

    def test_add_missing_title_fails(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        result = runner.invoke(cli, ["--data", data_file, "add", "--priority", "high"])
        assert result.exit_code != 0

    def test_add_invalid_priority_fails(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        result = runner.invoke(
            cli, ["--data", data_file, "add", "--title", "Task", "--priority", "urgent"]
        )
        assert result.exit_code != 0

    def test_add_shows_task_id(self, tmp_path):
        runner = CliRunner()
        result = add_task(runner, tmp_path, "ID check task")
        assert result.exit_code == 0
        # Output should contain a partial UUID (8 hex chars)
        assert any(c in result.output for c in "0123456789abcdef")

    def test_add_task_persisted(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        runner.invoke(cli, ["--data", data_file, "add", "--title", "Persisted task"])
        store = TaskStore(path=data_file)
        tasks = store.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].title == "Persisted task"


# ===========================================================================
# list command
# ===========================================================================

class TestListCommand:
    def test_list_empty_store(self, tmp_path):
        runner = CliRunner()
        result = run(runner, tmp_path, "list")
        assert result.exit_code == 0
        assert "No tasks" in result.output

    def test_list_shows_tasks(self, tmp_path):
        runner = CliRunner()
        add_task(runner, tmp_path, "Task Alpha")
        add_task(runner, tmp_path, "Task Beta")
        result = run(runner, tmp_path, "list")
        assert result.exit_code == 0
        assert "Task Alpha" in result.output
        assert "Task Beta" in result.output

    def test_list_shows_count(self, tmp_path):
        runner = CliRunner()
        add_task(runner, tmp_path, "Task 1")
        add_task(runner, tmp_path, "Task 2")
        result = run(runner, tmp_path, "list")
        assert result.exit_code == 0
        assert "2" in result.output

    def test_list_filter_by_status_todo(self, tmp_path):
        runner = CliRunner()
        add_task(runner, tmp_path, "Todo task")
        result = run(runner, tmp_path, "list", "--status", "todo")
        assert result.exit_code == 0
        assert "Todo task" in result.output

    def test_list_filter_by_status_done_empty(self, tmp_path):
        runner = CliRunner()
        add_task(runner, tmp_path, "Not done yet")
        result = run(runner, tmp_path, "list", "--status", "done")
        assert result.exit_code == 0
        assert "No tasks" in result.output

    def test_list_filter_by_priority_high(self, tmp_path):
        runner = CliRunner()
        add_task(runner, tmp_path, "High task", priority="high")
        add_task(runner, tmp_path, "Low task", priority="low")
        result = run(runner, tmp_path, "list", "--priority", "high")
        assert result.exit_code == 0
        assert "High task" in result.output
        assert "Low task" not in result.output

    def test_list_combined_filter(self, tmp_path):
        runner = CliRunner()
        add_task(runner, tmp_path, "High todo", priority="high")
        add_task(runner, tmp_path, "Low todo", priority="low")
        result = run(runner, tmp_path, "list", "--status", "todo", "--priority", "high")
        assert result.exit_code == 0
        assert "High todo" in result.output
        assert "Low todo" not in result.output

    def test_list_shows_rich_table_headers(self, tmp_path):
        runner = CliRunner()
        add_task(runner, tmp_path, "Table task")
        result = run(runner, tmp_path, "list")
        assert result.exit_code == 0
        # Rich table headers
        assert "Title" in result.output
        assert "Status" in result.output
        assert "Priority" in result.output

    def test_list_invalid_status_fails(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        result = runner.invoke(cli, ["--data", data_file, "list", "--status", "invalid"])
        assert result.exit_code != 0

    def test_list_invalid_priority_fails(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        result = runner.invoke(cli, ["--data", data_file, "list", "--priority", "invalid"])
        assert result.exit_code != 0


# ===========================================================================
# done command
# ===========================================================================

class TestDoneCommand:
    def test_done_marks_task(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        runner.invoke(cli, ["--data", data_file, "add", "--title", "Finish me"])
        store = TaskStore(path=data_file)
        task_id = store.list_tasks()[0].id
        result = runner.invoke(cli, ["--data", data_file, "done", task_id])
        assert result.exit_code == 0
        assert "done" in result.output.lower()

    def test_done_partial_id(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        runner.invoke(cli, ["--data", data_file, "add", "--title", "Partial done"])
        store = TaskStore(path=data_file)
        task_id = store.list_tasks()[0].id
        prefix = task_id[:8]
        result = runner.invoke(cli, ["--data", data_file, "done", prefix])
        assert result.exit_code == 0

    def test_done_persists_status(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        runner.invoke(cli, ["--data", data_file, "add", "--title", "Persist done"])
        store = TaskStore(path=data_file)
        task_id = store.list_tasks()[0].id
        runner.invoke(cli, ["--data", data_file, "done", task_id])
        store2 = TaskStore(path=data_file)
        assert store2.list_tasks()[0].status.value == "done"

    def test_done_not_found(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        result = runner.invoke(cli, ["--data", data_file, "done", "nonexistent-id"])
        assert result.exit_code != 0

    def test_done_shows_task_title(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        runner.invoke(cli, ["--data", data_file, "add", "--title", "Show title done"])
        store = TaskStore(path=data_file)
        task_id = store.list_tasks()[0].id
        result = runner.invoke(cli, ["--data", data_file, "done", task_id])
        assert "Show title done" in result.output


# ===========================================================================
# delete command
# ===========================================================================

class TestDeleteCommand:
    def test_delete_with_yes_flag(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        runner.invoke(cli, ["--data", data_file, "add", "--title", "Delete me"])
        store = TaskStore(path=data_file)
        task_id = store.list_tasks()[0].id
        result = runner.invoke(cli, ["--data", data_file, "delete", "--yes", task_id])
        assert result.exit_code == 0
        assert "deleted" in result.output.lower()

    def test_delete_removes_task(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        runner.invoke(cli, ["--data", data_file, "add", "--title", "Gone task"])
        store = TaskStore(path=data_file)
        task_id = store.list_tasks()[0].id
        runner.invoke(cli, ["--data", data_file, "delete", "--yes", task_id])
        store2 = TaskStore(path=data_file)
        assert store2.list_tasks() == []

    def test_delete_partial_id(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        runner.invoke(cli, ["--data", data_file, "add", "--title", "Partial delete"])
        store = TaskStore(path=data_file)
        task_id = store.list_tasks()[0].id
        prefix = task_id[:8]
        result = runner.invoke(cli, ["--data", data_file, "delete", "--yes", prefix])
        assert result.exit_code == 0

    def test_delete_not_found(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        result = runner.invoke(cli, ["--data", data_file, "delete", "--yes", "nonexistent"])
        assert result.exit_code != 0

    def test_delete_confirmation_abort(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        runner.invoke(cli, ["--data", data_file, "add", "--title", "Keep me"])
        store = TaskStore(path=data_file)
        task_id = store.list_tasks()[0].id
        # Simulate user typing "n" at the confirmation prompt
        result = runner.invoke(
            cli, ["--data", data_file, "delete", task_id], input="n\n"
        )
        assert result.exit_code == 0
        assert "Aborted" in result.output
        store2 = TaskStore(path=data_file)
        assert len(store2.list_tasks()) == 1  # task still exists

    def test_delete_shows_task_title(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        runner.invoke(cli, ["--data", data_file, "add", "--title", "Title in delete"])
        store = TaskStore(path=data_file)
        task_id = store.list_tasks()[0].id
        result = runner.invoke(cli, ["--data", data_file, "delete", "--yes", task_id])
        assert "Title in delete" in result.output


# ===========================================================================
# stats command
# ===========================================================================

class TestStatsCommand:
    def test_stats_empty_store(self, tmp_path):
        runner = CliRunner()
        result = run(runner, tmp_path, "stats")
        assert result.exit_code == 0
        assert "No tasks" in result.output

    def test_stats_shows_panel(self, tmp_path):
        runner = CliRunner()
        add_task(runner, tmp_path, "Task 1")
        result = run(runner, tmp_path, "stats")
        assert result.exit_code == 0
        assert "Stats" in result.output

    def test_stats_shows_status_counts(self, tmp_path):
        runner = CliRunner()
        add_task(runner, tmp_path, "Task A")
        add_task(runner, tmp_path, "Task B")
        result = run(runner, tmp_path, "stats")
        assert result.exit_code == 0
        assert "todo" in result.output

    def test_stats_shows_priority_counts(self, tmp_path):
        runner = CliRunner()
        add_task(runner, tmp_path, "High task", priority="high")
        add_task(runner, tmp_path, "Low task", priority="low")
        result = run(runner, tmp_path, "stats")
        assert result.exit_code == 0
        assert "high" in result.output
        assert "low" in result.output

    def test_stats_shows_total(self, tmp_path):
        runner = CliRunner()
        add_task(runner, tmp_path, "T1")
        add_task(runner, tmp_path, "T2")
        add_task(runner, tmp_path, "T3")
        result = run(runner, tmp_path, "stats")
        assert result.exit_code == 0
        assert "3" in result.output

    def test_stats_after_done(self, tmp_path):
        runner = CliRunner()
        data_file = str(tmp_path / "tasks.json")
        runner.invoke(cli, ["--data", data_file, "add", "--title", "Complete me"])
        store = TaskStore(path=data_file)
        task_id = store.list_tasks()[0].id
        runner.invoke(cli, ["--data", data_file, "done", task_id])
        result = runner.invoke(cli, ["--data", data_file, "stats"])
        assert result.exit_code == 0
        assert "done" in result.output
```

---

## Phase 4 — Dependencies (`requirements_click.txt`)

**File:** `team-test/requirements_click.txt`

```
# Runtime dependencies for the Click+Rich CLI task tracker
click>=8.1.0
rich>=13.0.0

# Testing
pytest>=9.0.2
```

**Installation command:**
```bash
pip install -r team-test/requirements_click.txt
```

---

## Execution Order & Parallelism

```
Phase 1 ──────────────────────────────────────────────────────────────────
  [SEQUENTIAL] Create team-test/task_store.py
               (depends on: models.py, storage.py — already exist)

Phase 2 ──────────────────────────────────────────────────────────────────
  [SEQUENTIAL] Create team-test/cli_click.py
               (depends on: task_store.py from Phase 1)

Phase 3 ──────────────────────────────────────────────────────────────────
  [PARALLEL]   Create team-test/tests/test_store.py
               Create team-test/tests/test_cli_click.py
               (both depend on Phase 1 + Phase 2 being complete)

Phase 4 ──────────────────────────────────────────────────────────────────
  [SEQUENTIAL] Create team-test/requirements_click.txt
               Install dependencies
               Run pytest
```

---

## File Creation Checklist

| # | File | Phase | Action |
|---|------|-------|--------|
| 1 | `team-test/task_store.py` | 1 | CREATE |
| 2 | `team-test/cli_click.py` | 2 | CREATE |
| 3 | `team-test/tests/test_store.py` | 3 | CREATE |
| 4 | `team-test/tests/test_cli_click.py` | 3 | CREATE |
| 5 | `team-test/requirements_click.txt` | 4 | CREATE |

**Do NOT modify:**
- `team-test/models.py`
- `team-test/storage.py`
- `team-test/services.py`
- `team-test/cli.py`
- `team-test/tests/test_models.py`
- `team-test/tests/test_storage.py`
- `team-test/tests/test_services.py`
- `team-test/tests/test_cli.py`

---

## Implementation Notes

### `task_store.py` — Critical details

1. **`_resolve_id(partial_id)`** must scan `self._storage.list_tasks()` and match tasks whose `id` starts with `partial_id`. Raise `NotFoundError` for 0 matches; raise `NotFoundError` (with "ambiguous" message) for >1 matches.

2. **`add_task`** must convert the `priority` string to `Priority(priority.lower())` before calling `self._storage.create_task()`. This ensures `ValueError` is raised for invalid values.

3. **`list_tasks`** must convert `status` and `priority` strings to their enum equivalents before passing to `self._storage.list_tasks()`. Pass `None` when the filter is not specified.

4. **`update_status`** must call `_resolve_id` first, then `self._storage.update_task(resolved_id, status=Status(new_status.lower()))`.

5. **`delete_task`** must call `_resolve_id` first, then `self._storage.delete_task(resolved_id)`.

### `cli_click.py` — Critical details

1. **`--data` global option** is passed to `TaskStore(path=data)`. When `data` is `None`, call `TaskStore()` with no arguments (uses default path).

2. **`delete` command** — resolve the id and fetch the task title *before* showing the confirmation prompt. This requires calling `store._resolve_id(task_id)` and then finding the task in `store.list_tasks()`.

3. **`CliRunner` compatibility** — Rich outputs ANSI escape codes. Use `CliRunner(mix_stderr=False)` if needed, or strip ANSI in assertions. Alternatively, use `Console(force_terminal=False)` or check for plain-text substrings that appear regardless of markup.

4. **`stats` command** — iterate `store.list_tasks()` and count by `task.status.value` and `task.priority.value`. Only count `"low"`, `"medium"`, `"high"` for priority (the existing `Priority` enum also has `"critical"` — handle gracefully).

5. **`list` command** — use `rich.table.Table` with columns: `ID` (first 8 chars + `…`), `Title`, `Status`, `Priority`, `Created At` (first 10 chars of ISO string = `YYYY-MM-DD`).

### Test isolation

All tests use `tmp_path` (pytest built-in fixture) to create a fresh JSON file per test. The `--data` CLI option and `TaskStore(path=...)` constructor parameter are the isolation mechanisms.

### Existing tests must remain green

The new files do not touch any existing modules. Run the full suite after implementation:
```bash
cd team-test && python -m pytest tests/ -v
```
Expected: all 363 existing tests + new tests pass.

---

## Verification Criteria

### How to verify this project works

#### 1. Install dependencies
```bash
cd /Users/pradeepsharma/sasva/projects/vllm/team-test
pip install -r requirements_click.txt
```
Expected: `click`, `rich`, and `pytest` install without errors.

#### 2. Verify imports
```bash
cd /Users/pradeepsharma/sasva/projects/vllm/team-test
python -c "from task_store import TaskStore; print('task_store OK')"
python -c "from cli_click import cli; print('cli_click OK')"
```
Expected: both print their OK message with no `ImportError`.

#### 3. Run the new test suite
```bash
cd /Users/pradeepsharma/sasva/projects/vllm/team-test
python -m pytest tests/test_store.py tests/test_cli_click.py -v
```
Expected: **all tests pass** (0 failures, 0 errors). Approximate count: ~45 tests in `test_store.py` + ~35 tests in `test_cli_click.py` = ~80 tests total.

#### 4. Run the full test suite (regression check)
```bash
cd /Users/pradeepsharma/sasva/projects/vllm/team-test
python -m pytest tests/ -v
```
Expected: **all 363 original tests + new tests pass** (0 regressions).

#### 5. Manual CLI smoke test
```bash
cd /Users/pradeepsharma/sasva/projects/vllm/team-test
python cli_click.py --help
# Expected: shows "add", "list", "done", "delete", "stats" commands

python cli_click.py add --title "Buy groceries" --priority high
# Expected: "✓ Task added: Buy groceries (id: <8-char-id>…, priority: high)"

python cli_click.py add --title "Write report"
# Expected: "✓ Task added: Write report (id: <8-char-id>…, priority: medium)"

python cli_click.py list
# Expected: Rich table with 2 rows showing both tasks

python cli_click.py list --status todo
# Expected: Rich table with 2 rows (both are todo)

python cli_click.py list --priority high
# Expected: Rich table with 1 row ("Buy groceries")

python cli_click.py stats
# Expected: Rich panel showing "todo: 2", "medium: 1", "high: 1", "Total: 2"

# Get the id of "Buy groceries" from the list output, then:
python cli_click.py done <partial-id>
# Expected: "✓ Task Buy groceries (id: <id>…) marked as done."

python cli_click.py list --status done
# Expected: Rich table with 1 row ("Buy groceries")

python cli_click.py delete --yes <partial-id-of-write-report>
# Expected: "✓ Task Write report (id: <id>…) deleted."

python cli_click.py list
# Expected: Rich table with 1 row ("Buy groceries", status: done)

python cli_click.py stats
# Expected: Panel showing "done: 1", "Total: 1"
```

#### 6. Confirm `--data` isolation works
```bash
python cli_click.py --data /tmp/test_tasks.json add --title "Isolated task"
python cli_click.py --data /tmp/test_tasks.json list
# Expected: shows only "Isolated task" (separate from default store)
```

#### Pass criteria
- `pip install` exits 0
- Both `python -c` import checks print OK
- `pytest tests/test_store.py tests/test_cli_click.py -v` → 0 failures
- `pytest tests/ -v` → 0 failures (363 + new tests all green)
- All 6 manual smoke-test commands produce the expected output
