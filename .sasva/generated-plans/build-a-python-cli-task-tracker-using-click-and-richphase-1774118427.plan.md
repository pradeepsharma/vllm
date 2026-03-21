# Plan: Build a Python CLI Task Tracker using Click and Rich

**Workspace:** `/Users/pradeepsharma/sasva/projects/vllm`  
**Target directory:** `tools/task_tracker/`  
**Date:** March 21, 2026

---

## Codebase Context

This workspace is the **vLLM** project — a high-throughput LLM inference engine. The task tracker is a self-contained developer utility that lives under `tools/task_tracker/`. An earlier implementation already exists at that path with the following files:

| File | Status | Notes |
|------|--------|-------|
| `tools/task_tracker/task_store.py` | **Exists** | `TaskStore` class, full CRUD, JSON persistence |
| `tools/task_tracker/cli.py` | **Exists** | Click group with `add`, `list`, `done`, `delete`, `stats` |
| `tools/task_tracker/setup.py` | **Exists** | Package metadata |
| `tools/task_tracker/requirements.txt` | **Exists** | `click`, `rich` pinned |
| `tools/task_tracker/tests/test_store.py` | **Exists** | Comprehensive unit tests for `TaskStore` |
| `tools/task_tracker/tests/test_cli.py` | **Exists** | CLI tests using `CliRunner` |
| `tools/task_tracker/VERIFICATION_REPORT.md` | **Exists** | Prior verification run |

The plan below **creates or overwrites** each file with a complete, correct implementation, then verifies all tests pass with `pytest`.

---

## Architecture Overview

```
tools/task_tracker/
├── task_store.py          # Phase 1 — data layer
├── cli.py                 # Phase 2 — CLI interface
├── requirements.txt       # Phase 1 — dependencies
├── setup.py               # Phase 1 — package metadata
└── tests/
    ├── __init__.py        # Phase 3 — test package marker
    ├── test_store.py      # Phase 3 — TaskStore unit tests
    └── test_cli.py        # Phase 3 — CLI integration tests
```

### Task data schema

```python
{
    "id":         str,   # uuid.uuid4() — e.g. "3f2a1b4c-..."
    "title":      str,   # non-empty, stripped
    "status":     str,   # "todo" | "in_progress" | "done"
    "priority":   str,   # "low" | "medium" | "high"
    "created_at": str,   # ISO-8601 UTC — datetime.now(timezone.utc).isoformat()
}
```

---

## Phase 1 — Core Data Layer

**Goal:** Create `tools/task_tracker/task_store.py` with a `TaskStore` class that reads/writes tasks to `tasks.json`, plus supporting project files.

### 1.1 — `tools/task_tracker/task_store.py`

**Create/overwrite** this file with the complete `TaskStore` implementation.

#### Module-level constants

```python
VALID_STATUSES  = ("todo", "in_progress", "done")
VALID_PRIORITIES = ("low", "medium", "high")
```

#### Class: `TaskStore`

```python
class TaskStore:
    def __init__(self, store_path: str = "tasks.json") -> None:
        self.store_path = store_path
```

#### Private helpers

| Method | Signature | Behaviour |
|--------|-----------|-----------|
| `_load` | `() -> list[dict]` | Opens `self.store_path`; returns `[]` if missing, corrupted JSON, or non-list root |
| `_save` | `(tasks: list[dict]) -> None` | Writes `json.dump(tasks, fh, indent=2, ensure_ascii=False)` |
| `_find_by_partial_id` | `(partial_id: str, tasks: list[dict]) -> dict` | Strips whitespace; matches tasks where `partial_id in task["id"]`; raises `ValueError` if 0 or >1 matches |

#### Public methods

**`add_task(title: str, priority: str = "medium") -> dict`**
- Strip `title`; raise `ValueError("Task title must not be empty.")` if blank
- Raise `ValueError` if `priority not in VALID_PRIORITIES`
- Build task dict: `id=str(uuid.uuid4())`, `status="todo"`, `created_at=datetime.now(timezone.utc).isoformat()`
- `_load()` → append → `_save()` → return task

**`list_tasks(status: Optional[str] = None, priority: Optional[str] = None) -> list[dict]`**
- Validate `status` and `priority` against constants (raise `ValueError` if invalid)
- `_load()` → filter by `status` if given → filter by `priority` if given
- Sort by `created_at` ascending → return

**`update_status(partial_id: str, new_status: str) -> dict`**
- Validate `new_status` against `VALID_STATUSES`
- `_load()` → `_find_by_partial_id()` → mutate `task["status"]` → `_save()` → return task

**`delete_task(partial_id: str) -> dict`**
- `_load()` → `_find_by_partial_id()` → rebuild list excluding matched task → `_save()` → return deleted task

#### Key implementation details
- All file I/O uses `encoding="utf-8"`
- `_load` catches `json.JSONDecodeError` and returns `[]` (graceful corruption handling)
- `_load` checks `isinstance(data, list)` and returns `[]` if not
- `_find_by_partial_id` strips whitespace from `partial_id` before matching

### 1.2 — `tools/task_tracker/requirements.txt`

```
click>=8.1.0
rich>=13.0.0
pytest>=7.0.0
```

### 1.3 — `tools/task_tracker/setup.py`

```python
from setuptools import setup, find_packages

setup(
    name="task-tracker",
    version="0.1.0",
    py_modules=["task_store", "cli"],
    packages=find_packages(),
    install_requires=["click>=8.1.0", "rich>=13.0.0"],
    entry_points={"console_scripts": ["task=cli:cli"]},
    python_requires=">=3.10",
)
```

---

## Phase 2 — CLI Interface

**Goal:** Create `tools/task_tracker/cli.py` using Click with five commands, Rich tables, and Rich panels.

### 2.1 — `tools/task_tracker/cli.py`

#### Imports

```python
from __future__ import annotations
import os, sys
from datetime import datetime, timezone
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from task_store import TaskStore
```

#### Module-level constants

```python
console = Console()

STATUS_STYLES  = {"todo": "cyan", "in_progress": "yellow", "done": "green"}
PRIORITY_STYLES = {"low": "dim white", "medium": "white", "high": "bold red"}
STATUS_LABELS  = {"todo": "todo", "in_progress": "in progress", "done": "done"}
```

#### Helper functions

**`_get_store(ctx: click.Context) -> TaskStore`**  
Returns `ctx.obj["store"]`.

**`_fmt_created_at(iso: str) -> str`**  
Parses ISO timestamp → converts to local time → `strftime("%Y-%m-%d %H:%M")`. Falls back to raw string on error.

**`_build_task_table(tasks: list[dict], title: str = "Tasks") -> Table`**  
Creates a `Table(box=box.ROUNDED, header_style="bold magenta")` with columns:
- `ID (short)` — `task["id"][:8]`, `style="dim"`, `no_wrap=True`
- `Title` — `task["title"]`
- `Status` — coloured via `STATUS_STYLES`
- `Priority` — coloured via `PRIORITY_STYLES`
- `Created` — formatted via `_fmt_created_at`

#### Click group

```python
@click.group()
@click.option("--store", "store_path", default=None,
              envvar="TASK_STORE_PATH",
              help="Path to the tasks JSON file.")
@click.pass_context
def cli(ctx, store_path):
    ctx.ensure_object(dict)
    resolved = store_path or os.environ.get("TASK_STORE_PATH", "tasks.json")
    ctx.obj["store"] = TaskStore(store_path=resolved)
```

#### Command: `add`

```
task add --title TEXT [--priority {low|medium|high}]
```

- `--title / -t` required
- `--priority / -p` default `"medium"`, `type=click.Choice(["low","medium","high"], case_sensitive=False)`
- Calls `store.add_task(title=title, priority=priority.lower())`
- On `ValueError`: print `[bold red]Error:[/bold red] {exc}` → `sys.exit(1)`
- On success: print a `Panel` (border `"green"`) showing full UUID, title, status, priority, created_at

#### Command: `list`

```
task list [--status {todo|in_progress|done}] [--priority {low|medium|high}]
```

- Both options optional, `type=click.Choice(...)`, `case_sensitive=False`
- Calls `store.list_tasks(status=..., priority=...)`
- Empty result → print dim message indicating active filters or "No tasks yet"
- Non-empty → `_build_task_table(tasks, title=...)` where title reflects active filters
- Prints `"{n} task(s) shown."` below table

#### Command: `done`

```
task done PARTIAL_ID
```

- `PARTIAL_ID` is a Click argument
- Calls `store.update_status(partial_id=partial_id, new_status="done")`
- On `ValueError`: print error → `sys.exit(1)`
- On success: print a green `Panel` with task ID and title

#### Command: `delete`

```
task delete PARTIAL_ID [--yes/-y]
```

- `PARTIAL_ID` is a Click argument
- `--yes / -y` is_flag, skips confirmation
- Without `--yes`: resolve task first (to show title in prompt), then `click.confirm(f"Delete '{task['title']}'?", abort=True)`
- Calls `store.delete_task(partial_id=partial_id)`
- On `ValueError`: print error → `sys.exit(1)`
- On success: print a red `Panel` confirming deletion

#### Command: `stats`

```
task stats
```

- Calls `store.list_tasks()` (no filters)
- Counts per status: `{s: 0 for s in ("todo","in_progress","done")}`
- Counts per priority: `{p: 0 for p in ("low","medium","high")}`
- Builds a `Table(box=box.SIMPLE)` with columns `Category`, `Value`, `Count`
- Adds status rows (coloured), `add_section()` separator, then priority rows (coloured)
- Wraps table in a `Panel(title="📊 Task Statistics", subtitle=f"Total tasks: {total}", border_style="magenta")`

#### Entry point guard

```python
if __name__ == "__main__":
    cli()
```

---

## Phase 3 — Tests

**Goal:** Create `tests/test_store.py` and `tests/test_cli.py` under `tools/task_tracker/tests/`. All tests must pass with `pytest`.

### 3.1 — `tools/task_tracker/tests/__init__.py`

```python
# Ensure the package root is importable from tests.
import sys
from pathlib import Path
_root = str(Path(__file__).parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
```

### 3.2 — `tools/task_tracker/tests/test_store.py`

**Fixtures:**

```python
@pytest.fixture()
def tmp_store(tmp_path):
    return TaskStore(store_path=str(tmp_path / "tasks.json"))

@pytest.fixture()
def populated_store(tmp_path):
    store = TaskStore(store_path=str(tmp_path / "tasks.json"))
    store.add_task("Task Alpha", priority="low")
    store.add_task("Task Beta",  priority="medium")
    store.add_task("Task Gamma", priority="high")
    return store
```

**Test classes and cases:**

| Class | Test cases |
|-------|-----------|
| `TestLoad` | missing file → `[]`; corrupted JSON → `[]`; non-list JSON → `[]`; valid file → tasks |
| `TestAddTask` | all 5 fields present; id is valid UUID4; title stored/stripped; status is `"todo"`; priority stored; `created_at` parses as ISO; persisted to file; multiple tasks accumulate; empty title raises `ValueError`; whitespace-only title raises; invalid priority raises; default priority is `"medium"`; unique IDs |
| `TestListTasks` | empty store → `[]`; no filter → all 3; status=`todo` filter; status=`done` filter (after update); status=`in_progress` filter; priority=`low`; priority=`high`; combined status+priority; combined with no match → `[]`; sorted by `created_at`; invalid status raises; invalid priority raises |
| `TestUpdateStatus` | happy path changes status; invalid status raises; no match raises; ambiguous match raises; persisted after update |
| `TestDeleteTask` | happy path removes task; returns deleted task dict; no match raises; ambiguous match raises; persisted after delete; delete all tasks → empty list |
| `TestFindByPartialId` | exact full ID; prefix match; substring match; strips whitespace; no match raises; multiple matches raises `"Ambiguous"` |
| `TestPersistence` | save/load round-trip: add tasks, create fresh `TaskStore` at same path, verify tasks present |

### 3.3 — `tools/task_tracker/tests/test_cli.py`

**Fixtures:**

```python
@pytest.fixture()
def runner():
    return CliRunner()

@pytest.fixture()
def store_file(tmp_path):
    return str(tmp_path / "tasks.json")

@pytest.fixture()
def populated_store_file(tmp_path):
    store_path = str(tmp_path / "tasks.json")
    store = TaskStore(store_path=store_path)
    t1 = store.add_task("Fix login bug",  priority="high")
    t2 = store.add_task("Write docs",     priority="low")
    t3 = store.add_task("Deploy to prod", priority="medium")
    return store_path, [t1, t2, t3]
```

**Helper:**

```python
def invoke(runner, store_path, *args, input=None):
    return runner.invoke(cli, ["--store", store_path, *args],
                         input=input, catch_exceptions=False)
```

**Test classes and cases:**

| Class | Test cases |
|-------|-----------|
| `TestCLIGroup` | `--help` exits 0 and contains "Task Tracker"; `--store` option wires correct path; `TASK_STORE_PATH` env var respected |
| `TestAddCommand` | happy path exits 0 and output contains title; `--priority high` stored; missing `--title` exits non-zero; invalid priority exits non-zero; output contains short ID (first 8 chars of UUID) |
| `TestListCommand` | empty store prints "No tasks"; all tasks shown after add; `--status todo` filters correctly; `--priority high` filters correctly; combined `--status todo --priority high` filters; output contains column headers |
| `TestDoneCommand` | happy path exits 0 and output contains "done"; partial ID (first 8 chars) works; nonexistent ID exits non-zero and prints "Error" |
| `TestDeleteCommand` | `--yes` flag skips prompt, exits 0; confirmation prompt with `y` input deletes task; confirmation prompt with `n` input aborts (task still present); nonexistent ID exits non-zero |
| `TestStatsCommand` | empty store exits 0 and shows "Total tasks: 0"; populated store shows correct counts; output contains "todo", "done", "in progress", "low", "medium", "high" |

---

## Execution Order & Parallelism

```
Phase 1 (task_store.py + requirements.txt + setup.py)
    │
    ▼
Phase 2 (cli.py)          ← depends on task_store.py
    │
    ▼
Phase 3 (tests/)          ← depends on both task_store.py and cli.py
    │
    ▼
Verification
```

Phases 1 and 2 are sequential (cli.py imports task_store). Within Phase 3, `test_store.py` and `test_cli.py` can be written in parallel since they are independent files.

---

## File Manifest

| File to create/overwrite | Phase |
|--------------------------|-------|
| `tools/task_tracker/task_store.py` | 1 |
| `tools/task_tracker/requirements.txt` | 1 |
| `tools/task_tracker/setup.py` | 1 |
| `tools/task_tracker/cli.py` | 2 |
| `tools/task_tracker/tests/__init__.py` | 3 |
| `tools/task_tracker/tests/test_store.py` | 3 |
| `tools/task_tracker/tests/test_cli.py` | 3 |

---

## Dependencies & Environment

Install before running tests:

```bash
cd tools/task_tracker
pip install click>=8.1.0 rich>=13.0.0 pytest>=7.0.0
# or
pip install -r requirements.txt
```

The project uses **Python ≥ 3.10** (matching the vLLM workspace `pyproject.toml` constraint).

No external services, databases, or GPU hardware are required.

---

## Verification Criteria

### How to verify

Run from the `tools/task_tracker/` directory:

```bash
cd /Users/pradeepsharma/sasva/projects/vllm/tools/task_tracker
pip install -r requirements.txt
pytest tests/ -v
```

**Expected outcome:** All tests pass with **0 failures, 0 errors**.

### Specific commands to test manually

```bash
# 1. Add a task (default medium priority)
python cli.py add --title "Fix login bug"
# Expected: Green panel with "Task Added", shows UUID, status=todo, priority=medium

# 2. Add a high-priority task
python cli.py add --title "Deploy hotfix" --priority high
# Expected: Green panel, priority=high in bold red

# 3. List all tasks
python cli.py list
# Expected: Rich table with 2 rows, columns: ID (short), Title, Status, Priority, Created

# 4. List with status filter
python cli.py list --status todo
# Expected: Both tasks shown (both are todo)

# 5. List with priority filter
python cli.py list --priority high
# Expected: Only "Deploy hotfix" shown

# 6. Mark a task done (use first 8 chars of UUID from step 1)
python cli.py done <first-8-chars-of-uuid>
# Expected: Green panel "Task marked as done!"

# 7. Stats
python cli.py stats
# Expected: Magenta panel "📊 Task Statistics", shows todo:1, done:1, high:1, medium:1

# 8. Delete with confirmation
python cli.py delete <first-8-chars-of-uuid>
# Expected: Prompt "Delete 'Deploy hotfix'? [y/N]:" → type y → red panel confirming deletion

# 9. Delete without prompt
python cli.py delete <first-8-chars-of-uuid> --yes
# Expected: Red panel confirming deletion immediately
```

### pytest test count targets

| Test file | Approximate test count |
|-----------|----------------------|
| `tests/test_store.py` | ~40 tests |
| `tests/test_cli.py` | ~25 tests |
| **Total** | **~65 tests, all passing** |

### Exit code expectations

| Scenario | Exit code |
|----------|-----------|
| Successful `add`, `list`, `done`, `delete --yes`, `stats` | `0` |
| `add` with missing `--title` | `2` (Click usage error) |
| `add` with invalid `--priority` | `2` (Click usage error) |
| `done` with nonexistent partial ID | `1` |
| `delete` with nonexistent partial ID | `1` |
| `delete` with confirmation prompt answered `n` | `1` (aborted) |

### Persistence verification

```bash
# Verify tasks.json is created and readable
python cli.py add --title "Persist test"
cat tasks.json
# Expected: valid JSON array with one task object containing all 5 fields

# Verify data survives process restart
python cli.py list
# Expected: same task still shown
```
