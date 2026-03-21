# Build a Python CLI Task Tracker using Click and Rich

**Workspace:** `/Users/pradeepsharma/sasva/projects/vllm`
**Generated:** 2026-03-21
**Target directory:** `tools/task_tracker/`

---

## Codebase Context

This workspace is the **vLLM** project — a high-throughput LLM inference engine. The task tracker is a standalone developer utility that lives under `tools/task_tracker/`. An initial implementation already exists there:

| File | Status | Notes |
|---|---|---|
| `tools/task_tracker/task_store.py` | ✅ Exists | Full `TaskStore` class, 241 lines |
| `tools/task_tracker/cli.py` | ✅ Exists | Full Click CLI, 451 lines |
| `tools/task_tracker/tests/test_store.py` | ✅ Exists | 477-line pytest suite |
| `tools/task_tracker/tests/test_cli.py` | ✅ Exists | 29 KB pytest suite using CliRunner |
| `tools/task_tracker/requirements.txt` | ✅ Exists | `click>=8.1,<9.0`, `rich>=13.0,<15.0`, `pytest>=7.0` |
| `tools/task_tracker/setup.py` | ✅ Exists | Package setup |
| `tools/task_tracker/tests/__init__.py` | ✅ Exists | Path-injection for imports |

The project uses **Python ≥ 3.10** (per `pyproject.toml`), **pytest** (configured in `pyproject.toml` `[tool.pytest.ini_options]`), and **ruff** for linting.

### Key Design Decisions Already in Place

- `TaskStore.__init__(store_path="tasks.json")` — injectable path enables test isolation via `tmp_path`
- `_load()` / `_save()` — private helpers; `_load()` gracefully handles missing/corrupted files
- `_find_by_partial_id(partial_id, tasks)` — raises `ValueError` on zero or ambiguous matches
- CLI uses `@click.pass_context` + `ctx.obj["store"]` to share the `TaskStore` instance
- `--store PATH` / `TASK_STORE_PATH` env var override the default `tasks.json` path
- Rich `Table` (rounded box, magenta header) for `list`; Rich `Panel` for `add`, `done`, `delete`, `stats`
- Tests inject `--store <tmp_path>` via a shared `invoke()` helper to avoid touching real files

---

## Execution Phases

### Phase 1 — Core Data Layer (`task_store.py`)

**Goal:** Create/verify `tools/task_tracker/task_store.py` with the complete `TaskStore` class.

**File:** `tools/task_tracker/task_store.py`

#### Data Schema

Each task is a plain `dict` with exactly five fields:

```python
{
    "id":         str,   # uuid.uuid4() → str
    "title":      str,   # stripped, non-empty
    "status":     str,   # "todo" | "in_progress" | "done"
    "priority":   str,   # "low" | "medium" | "high"
    "created_at": str,   # datetime.now(timezone.utc).isoformat()
}
```

#### Module-level constants

```python
VALID_STATUSES  = ("todo", "in_progress", "done")
VALID_PRIORITIES = ("low", "medium", "high")
```

#### Class: `TaskStore`

```python
class TaskStore:
    def __init__(self, store_path: str = "tasks.json") -> None: ...

    # Private helpers
    def _load(self) -> list[dict]: ...          # returns [] if file missing/corrupt
    def _save(self, tasks: list[dict]) -> None: ...  # json.dump indent=2
    def _find_by_partial_id(self, partial_id: str, tasks: list[dict]) -> dict: ...
        # raises ValueError if 0 or >1 matches

    # Public API
    def add_task(self, title: str, priority: str = "medium") -> dict: ...
    def list_tasks(self, status: Optional[str] = None, priority: Optional[str] = None) -> list[dict]: ...
    def update_status(self, partial_id: str, new_status: str) -> dict: ...
    def delete_task(self, partial_id: str) -> dict: ...
```

#### Method Specifications

**`_load()`**
- If `store_path` does not exist → return `[]`
- On `json.JSONDecodeError` → return `[]` (corrupted file, start fresh)
- If JSON root is not a `list` → return `[]`
- Otherwise return the list

**`_save(tasks)`**
- `json.dump(tasks, fh, indent=2, ensure_ascii=False)` to `store_path` in `"w"` mode, UTF-8

**`_find_by_partial_id(partial_id, tasks)`**
- Strip whitespace from `partial_id`
- `matches = [t for t in tasks if partial_id in t["id"]]`
- `len(matches) == 0` → `raise ValueError(f"No task found matching partial ID '{partial_id}'.")`
- `len(matches) > 1` → `raise ValueError(f"Ambiguous partial ID '{partial_id}' matches {len(matches)} tasks: ...")`
- Return `matches[0]`

**`add_task(title, priority="medium")`**
- Strip `title`; raise `ValueError("Task title must not be empty.")` if blank
- Raise `ValueError` if `priority not in VALID_PRIORITIES`
- Build task dict with `uuid.uuid4()`, `"todo"` status, `datetime.now(timezone.utc).isoformat()`
- `tasks = _load(); tasks.append(task); _save(tasks)`
- Return the new task dict

**`list_tasks(status=None, priority=None)`**
- Validate `status` and `priority` against constants (raise `ValueError` if invalid)
- `tasks = _load()`
- Apply filters: `[t for t in tasks if t.get("status") == status]` etc.
- Sort by `created_at` ascending: `tasks.sort(key=lambda t: t.get("created_at", ""))`
- Return filtered, sorted list

**`update_status(partial_id, new_status)`**
- Validate `new_status`; raise `ValueError` if invalid
- `tasks = _load(); task = _find_by_partial_id(...); task["status"] = new_status; _save(tasks)`
- Return updated task dict

**`delete_task(partial_id)`**
- `tasks = _load(); task = _find_by_partial_id(...)`
- `tasks = [t for t in tasks if t["id"] != task["id"]]; _save(tasks)`
- Return the deleted task dict (snapshot)

---

### Phase 2 — CLI Interface (`cli.py`)

**Goal:** Create/verify `tools/task_tracker/cli.py` using Click + Rich.

**File:** `tools/task_tracker/cli.py`

#### Imports

```python
import os, sys
from datetime import datetime, timezone
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box
from task_store import TaskStore
```

#### Shared State

```python
console = Console()

STATUS_STYLES   = {"todo": "cyan", "in_progress": "yellow", "done": "green"}
PRIORITY_STYLES = {"low": "dim white", "medium": "white", "high": "bold red"}
STATUS_LABELS   = {"todo": "todo", "in_progress": "in progress", "done": "done"}
```

#### Helper Functions

**`_get_store(ctx) -> TaskStore`**
- Returns `ctx.obj["store"]`

**`_fmt_created_at(iso: str) -> str`**
- Parse ISO timestamp → convert to local time → `strftime("%Y-%m-%d %H:%M")`
- Fall back to raw string on parse error

**`_build_task_table(tasks, title="Tasks") -> Table`**
- `Table(box=box.ROUNDED, header_style="bold magenta", expand=False)`
- Columns: `"ID (short)"` (dim, no_wrap), `"Title"`, `"Status"` (center), `"Priority"` (center), `"Created"` (right)
- Each row: `task["id"][:8]`, title, styled status label, styled priority, formatted timestamp

#### CLI Group

```python
@click.group()
@click.option("--store", "store_path", default=None, envvar="TASK_STORE_PATH",
              help="Path to the tasks JSON file.")
@click.pass_context
def cli(ctx, store_path): ...
```
- `ctx.ensure_object(dict)`
- Resolve: `store_path or os.environ.get("TASK_STORE_PATH", "tasks.json")`
- `ctx.obj["store"] = TaskStore(store_path=resolved_path)`

#### Command: `add`

```
task add --title TEXT [--priority {low,medium,high}]
```
- `--title / -t` required
- `--priority / -p` default `"medium"`, `click.Choice(["low","medium","high"], case_sensitive=False)`
- On success: print Rich `Panel` (green border) with full UUID, title, status, priority, created_at
- On `ValueError`: print `[bold red]Error:[/bold red] {exc}` → `sys.exit(1)`

#### Command: `list`

```
task list [--status {todo,in_progress,done}] [--priority {low,medium,high}]
```
- Both options optional, `click.Choice(...)`, `case_sensitive=False`
- Call `store.list_tasks(status=..., priority=...)`
- If empty: print dim "No tasks" message reflecting active filters
- Otherwise: print `_build_task_table(tasks, title=...)` + `"{n} task(s) shown."` footer

#### Command: `done`

```
task done PARTIAL_ID
```
- `PARTIAL_ID` is a Click argument
- Call `store.update_status(partial_id, "done")`
- On success: print green Panel with full UUID and title
- On `ValueError`: error + `sys.exit(1)`

#### Command: `delete`

```
task delete PARTIAL_ID [--yes/-y]
```
- `PARTIAL_ID` argument + `--yes / -y` flag (skip confirmation)
- Resolve task first (to show title in prompt); on `ValueError` → error + exit
- If not `--yes`: `click.confirm(f"Delete task '{task['title']}'?", abort=True)`
- On success: print red Panel confirming deletion
- On `ValueError`: error + `sys.exit(1)`

#### Command: `stats`

```
task stats
```
- Call `store.list_tasks()` (no filters)
- Count per status: `{s: 0 for s in ("todo","in_progress","done")}`
- Count per priority: `{p: 0 for p in ("low","medium","high")}`
- Build `Table(box=box.SIMPLE)` with columns: Category, Value (styled), Count
- Wrap in `Panel(title="📊 Task Statistics", subtitle="Total tasks: N", border_style="magenta")`

#### Entry Point

```python
if __name__ == "__main__":
    cli()
```

---

### Phase 3 — Tests

**Goal:** Create/verify both test files so that `pytest tools/task_tracker/tests/` passes with zero failures.

#### File: `tools/task_tracker/tests/__init__.py`

```python
import sys
from pathlib import Path

_PACKAGE_ROOT = str(Path(__file__).parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)
```

This ensures `from task_store import TaskStore` and `from cli import cli` resolve correctly regardless of how pytest is invoked.

#### File: `tools/task_tracker/tests/test_store.py`

**Test classes and coverage:**

```
TestLoad
  test_returns_empty_list_when_file_missing
  test_returns_empty_list_for_corrupted_json
  test_returns_empty_list_when_json_is_not_a_list
  test_returns_tasks_from_valid_file

TestAddTask
  test_returns_task_dict_with_all_fields
  test_id_is_valid_uuid4_string
  test_title_is_stored_correctly
  test_title_is_stripped_of_whitespace
  test_initial_status_is_todo
  test_priority_is_stored_correctly
  test_created_at_is_iso8601_string
  test_task_is_persisted_to_file          ← persistence across save/load
  test_multiple_tasks_accumulate
  test_raises_value_error_for_empty_title
  test_raises_value_error_for_whitespace_only_title
  test_raises_value_error_for_invalid_priority
  test_default_priority_is_medium
  test_each_task_gets_unique_id

TestListTasks
  test_returns_empty_list_when_no_tasks
  test_returns_all_tasks_without_filters
  test_filter_by_status_todo
  test_filter_by_status_done_returns_only_done_tasks
  test_filter_by_status_in_progress
  test_filter_by_priority_low
  test_filter_by_priority_high
  test_combined_status_and_priority_filter
  test_combined_filter_returns_empty_when_no_match
  test_tasks_sorted_by_created_at
  test_raises_value_error_for_invalid_status
  test_raises_value_error_for_invalid_priority

TestUpdateStatus
  test_updates_status_to_done
  test_updates_status_to_in_progress
  test_returns_updated_task_dict
  test_persists_status_change              ← persistence across save/load
  test_raises_for_invalid_status
  test_raises_for_no_match
  test_raises_for_ambiguous_match

TestDeleteTask
  test_deletes_task_and_returns_it
  test_task_no_longer_in_store             ← persistence across save/load
  test_raises_for_no_match
  test_raises_for_ambiguous_match
  test_delete_all_tasks_leaves_empty_store

TestFindByPartialId
  test_exact_full_id_match
  test_prefix_match
  test_strips_whitespace_from_partial_id
  test_raises_when_no_match
  test_raises_when_multiple_matches
```

**Fixtures:**
- `tmp_store(tmp_path)` → `TaskStore(store_path=str(tmp_path / "tasks.json"))`
- `populated_store(tmp_path)` → store with 3 tasks: Alpha(low), Beta(medium), Gamma(high)

#### File: `tools/task_tracker/tests/test_cli.py`

**Shared helpers:**
```python
def make_runner() -> CliRunner: return CliRunner()

def invoke(runner, store_path, *args, input=None):
    return runner.invoke(cli, ["--store", store_path, *args],
                         input=input, catch_exceptions=False)
```

**Fixtures:**
- `runner()` → `CliRunner()`
- `store_file(tmp_path)` → `str(tmp_path / "tasks.json")`
- `populated_store_file(tmp_path)` → `(store_path, [t1, t2, t3])` with 3 pre-added tasks

**Test classes and coverage:**

```
TestCliGroup
  test_help_exits_zero
  test_help_mentions_task_tracker
  test_help_lists_commands (add, list, done, delete, stats)
  test_store_option_creates_file_at_given_path
  test_env_var_task_store_path_is_respected

TestAddCommand
  test_add_exits_zero
  test_add_prints_task_added
  test_add_prints_full_uuid
  test_add_default_priority_is_medium
  test_add_with_explicit_priority_high
  test_add_with_explicit_priority_low
  test_add_missing_title_exits_nonzero
  test_add_invalid_priority_exits_nonzero
  test_add_empty_title_exits_nonzero
  test_add_persists_to_store_file

TestListCommand
  test_list_empty_store_prints_no_tasks_message
  test_list_shows_all_tasks
  test_list_shows_task_titles
  test_list_filter_by_status_todo
  test_list_filter_by_status_done
  test_list_filter_by_priority_high
  test_list_filter_by_priority_low
  test_list_combined_filter
  test_list_combined_filter_no_match
  test_list_shows_task_count

TestDoneCommand
  test_done_exits_zero
  test_done_prints_marked_as_done
  test_done_by_partial_id
  test_done_nonexistent_id_exits_nonzero
  test_done_updates_status_in_store

TestDeleteCommand
  test_delete_with_yes_flag_exits_zero
  test_delete_with_yes_flag_prints_deleted
  test_delete_confirms_with_prompt_yes
  test_delete_confirms_with_prompt_no_aborts
  test_delete_nonexistent_id_exits_nonzero
  test_delete_removes_task_from_store

TestStatsCommand
  test_stats_empty_store_exits_zero
  test_stats_shows_task_statistics_panel
  test_stats_shows_correct_todo_count
  test_stats_shows_correct_done_count
  test_stats_shows_correct_priority_counts
  test_stats_shows_total_count
```

---

### Phase 4 — Project Scaffolding & Dependencies

**Goal:** Ensure all supporting files are correct so the project installs and runs cleanly.

#### File: `tools/task_tracker/requirements.txt`

```
# Runtime
click>=8.1,<9.0
rich>=13.0,<15.0

# Testing
pytest>=7.0,<10.0
```

#### File: `tools/task_tracker/setup.py`

```python
from setuptools import setup, find_packages

setup(
    name="task-tracker",
    version="0.1.0",
    py_modules=["task_store", "cli"],
    install_requires=["click>=8.1,<9.0", "rich>=13.0,<15.0"],
    entry_points={
        "console_scripts": ["task=cli:cli"],
    },
    python_requires=">=3.10",
)
```

#### File: `tools/task_tracker/tests/__init__.py`

```python
import sys
from pathlib import Path

_PACKAGE_ROOT = str(Path(__file__).parent.parent)
if _PACKAGE_ROOT not in sys.path:
    sys.path.insert(0, _PACKAGE_ROOT)
```

---

## File Map (Complete)

```
tools/task_tracker/
├── task_store.py                  # Phase 1 — TaskStore class
├── cli.py                         # Phase 2 — Click CLI (add/list/done/delete/stats)
├── requirements.txt               # Phase 4 — click, rich, pytest
├── setup.py                       # Phase 4 — entry_point: task=cli:cli
└── tests/
    ├── __init__.py                # Phase 4 — sys.path injection
    ├── test_store.py              # Phase 3 — TaskStore unit tests
    └── test_cli.py                # Phase 3 — CLI integration tests (CliRunner)
```

---

## Parallelism Strategy

| Phase | Depends On | Can Run In Parallel With |
|---|---|---|
| Phase 1 (task_store.py) | — | Phase 4 scaffolding |
| Phase 2 (cli.py) | Phase 1 | — |
| Phase 3 (tests) | Phase 1 + Phase 2 | — |
| Phase 4 (scaffolding) | — | Phase 1 |

**Optimal execution order:**
1. **Parallel:** Phase 1 + Phase 4 (no dependencies between them)
2. **Sequential:** Phase 2 (needs Phase 1 complete)
3. **Sequential:** Phase 3 (needs Phase 1 + Phase 2 complete)

---

## Implementation Notes

### Import Resolution
Both `cli.py` and the tests use bare `from task_store import TaskStore`. This works because:
- `tests/__init__.py` inserts `tools/task_tracker/` into `sys.path`
- When running `python cli.py` directly from `tools/task_tracker/`, the CWD is on `sys.path`
- When running `pytest tools/task_tracker/tests/`, the `__init__.py` path injection fires first

### Test Isolation
Every test that touches the filesystem uses `tmp_path` (pytest built-in fixture) to get a unique temporary directory. The `invoke()` helper always passes `--store <tmp_path/tasks.json>` so no test ever reads or writes the real `tasks.json`.

### Click CliRunner Usage
```python
result = runner.invoke(cli, ["--store", store_path, "add", "--title", "My task"])
assert result.exit_code == 0
assert "Task Added" in result.output
```
Use `catch_exceptions=False` to get real tracebacks during test development.

### Rich Output in Tests
Rich renders ANSI escape codes by default. The `CliRunner` captures stdout as plain text. Rich auto-detects non-TTY and strips markup, so assertions like `assert "Task Added" in result.output` work reliably without stripping ANSI codes.

### Partial ID Matching
The `done` and `delete` commands accept any substring of a UUID. The first 8 characters (shown in the table's "ID (short)" column) are sufficient for typical use. The `_find_by_partial_id` method raises `ValueError` on ambiguity, which the CLI surfaces as a red error message.

### Error Handling Pattern
All CLI commands follow this pattern:
```python
try:
    result = store.some_method(...)
except ValueError as exc:
    console.print(f"[bold red]Error:[/bold red] {exc}")
    sys.exit(1)
```

---

## Verification Criteria

### Install Dependencies
```bash
cd tools/task_tracker
pip install -r requirements.txt
```
Expected: clean install of `click`, `rich`, `pytest`.

### Run All Tests
```bash
cd tools/task_tracker
pytest tests/ -v
```
**Expected:** All tests pass. Zero failures, zero errors.
Approximate test count: ~55 tests across `test_store.py` and `test_cli.py`.

### Manual CLI Smoke Tests

Run from `tools/task_tracker/`:

```bash
# 1. Add tasks
python cli.py add --title "Fix login bug" --priority high
python cli.py add --title "Write docs" --priority low
python cli.py add --title "Deploy to prod"

# Expected: Rich Panel printed for each, green border, "Task Added" in title

# 2. List all tasks
python cli.py list
# Expected: Rich Table with 3 rows, rounded box, magenta header
# Columns: ID (short), Title, Status, Priority, Created

# 3. Filter by status
python cli.py list --status todo
# Expected: all 3 tasks (all start as todo)

# 4. Filter by priority
python cli.py list --priority high
# Expected: 1 task ("Fix login bug")

# 5. Mark done by partial ID (use first 8 chars from list output)
python cli.py done <first-8-chars-of-id>
# Expected: green Panel "Task marked as done!"

# 6. Verify status change
python cli.py list --status done
# Expected: 1 task shown

# 7. Stats
python cli.py stats
# Expected: magenta Panel "📊 Task Statistics"
#   Status:   todo=2, in_progress=0, done=1
#   Priority: low=1, medium=1, high=1
#   Total tasks: 3

# 8. Delete with confirmation
python cli.py delete <partial-id>
# Expected: prompt "Delete task '...'? [y/N]:" → type y → red Panel "Task Deleted"

# 9. Delete with --yes flag (no prompt)
python cli.py delete <partial-id> --yes
# Expected: immediate deletion, red Panel

# 10. Error cases
python cli.py add --title ""
# Expected: exit code 1, "[bold red]Error:[/bold red] Task title must not be empty."

python cli.py done zzz-no-match
# Expected: exit code 1, "No task found matching partial ID 'zzz-no-match'."
```

### Pytest Specific Assertions

```bash
# Run only store tests
pytest tools/task_tracker/tests/test_store.py -v
# Expected: ~40 tests, all PASSED

# Run only CLI tests
pytest tools/task_tracker/tests/test_cli.py -v
# Expected: ~35 tests, all PASSED

# Run with coverage (optional)
pytest tools/task_tracker/tests/ --tb=short -q
# Expected: "X passed in Y.YYs" with no failures
```

### Key Assertions Per Command

| Command | Expected exit code | Expected output contains |
|---|---|---|
| `add --title "X"` | 0 | `"Task Added"`, `"✅ Task added successfully!"` |
| `add` (no title) | 2 | `"Missing option '--title'"` |
| `list` (empty) | 0 | `"No tasks yet"` |
| `list` (populated) | 0 | task titles, `"task(s) shown"` |
| `done <valid-id>` | 0 | `"Task marked as done!"` |
| `done <bad-id>` | 1 | `"No task found"` |
| `delete <id> --yes` | 0 | `"Task Deleted"` |
| `delete <id>` + `n` | 1 | `"Aborted"` |
| `stats` | 0 | `"Task Statistics"`, `"Total tasks"` |
