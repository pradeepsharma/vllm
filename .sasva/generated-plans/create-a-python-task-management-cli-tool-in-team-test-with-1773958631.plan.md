# Plan: Python Task Management CLI Tool in `team-test/`

## Overview

Create a fully-featured Python task management CLI tool inside the `team-test/` directory of the vLLM workspace. The tool is organized into clearly separated components: data models, storage layer, business logic services, CLI interface, and comprehensive pytest tests. The workspace is a large Python project (vLLM) using Python ≥ 3.10, `pyproject.toml` for build config, `pytest` for testing, and `ruff` for linting — all conventions we will follow.

---

## Workspace Context

- **Root**: `/Users/pradeepsharma/sasva/projects/vllm`
- **Python version**: ≥ 3.10 (per `pyproject.toml`)
- **Test framework**: `pytest` with `conftest.py` fixtures (see `tests/conftest.py`, `tests/v1/engine/conftest.py`)
- **Existing `team-test/`**: Has empty `api/` and `database/` subdirs with `__pycache__` only — no Python source files yet
- **Dataclass pattern**: Used extensively in `vllm/outputs.py`, `vllm/sampling_params.py`, `vllm/engine/arg_utils.py`
- **Enum pattern**: Used in `vllm/sampling_params.py` (`SamplingType(IntEnum)`), `vllm/scalar_type.py`
- **CLI pattern**: `vllm/entrypoints/cli/main.py` uses `argparse` subparsers with `subparser_init` / `cmd` dispatch pattern
- **JSON storage**: Standard `json` module used throughout vLLM codebase
- **Requirements**: Separate `requirements/*.txt` files per concern (common, dev, test, lint)

---

## Target File Structure

```
team-test/
├── __init__.py                  # Package marker
├── models.py                    # Data models: Task, Project, Status, Priority enums
├── storage.py                   # JSONFileStorage class
├── services.py                  # TaskService, ProjectService business logic
├── cli.py                       # argparse CLI with subcommands
├── requirements.txt             # Runtime + dev dependencies
└── tests/
    ├── __init__.py
    ├── conftest.py              # Shared pytest fixtures
    ├── test_models.py           # Tests for models.py
    ├── test_storage.py          # Tests for storage.py
    └── test_services.py         # Tests for services.py
```

---

## Phase 1 — Data Models (`team-test/models.py` + `team-test/__init__.py`)

**Files**: `team-test/__init__.py`, `team-test/models.py`

### `team-test/__init__.py`
Empty package marker with a version string:
```python
"""Task Management CLI Tool."""
__version__ = "0.1.0"
```

### `team-test/models.py`

**Imports**:
```python
from __future__ import annotations
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
```

**Enums**:

```python
class Status(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
```

Using `str, Enum` mixin so values serialize cleanly to/from JSON strings (same pattern as vLLM's `SamplingType`).

**Task dataclass**:
```python
@dataclass
class Task:
    title: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    status: Status = Status.TODO
    priority: Priority = Priority.MEDIUM
    due_date: Optional[str] = None          # ISO-8601 date string "YYYY-MM-DD"
    project_id: Optional[str] = None
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "status": self.status.value,
            "priority": self.priority.value,
            "due_date": self.due_date,
            "project_id": self.project_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            status=Status(data.get("status", "todo")),
            priority=Priority(data.get("priority", "medium")),
            due_date=data.get("due_date"),
            project_id=data.get("project_id"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            updated_at=data.get("updated_at", datetime.utcnow().isoformat()),
        )

    def is_overdue(self) -> bool:
        if self.due_date is None or self.status == Status.DONE:
            return False
        return datetime.strptime(self.due_date, "%Y-%m-%d").date() < datetime.utcnow().date()
```

**Project dataclass**:
```python
@dataclass
class Project:
    name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Project":
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
        )
```

---

## Phase 2 — Storage Layer (`team-test/storage.py`)

**File**: `team-test/storage.py`

**Imports**:
```python
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional
from team_test.models import Task, Project   # NOTE: package imported as team_test
```

> **Note**: Because the directory is `team-test/` (with a hyphen), Python imports it as `team_test` when installed or when `sys.path` includes the workspace root. Within the package itself, use relative imports: `from .models import Task, Project`.

**Class `JSONFileStorage`**:

```python
class JSONFileStorage:
    """Persists tasks and projects to a single JSON file."""

    DEFAULT_PATH = Path.home() / ".task_manager" / "data.json"

    def __init__(self, filepath: Optional[str | Path] = None) -> None:
        self.filepath = Path(filepath) if filepath else self.DEFAULT_PATH
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_file()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_file(self) -> None:
        """Create an empty store if the file does not exist."""
        if not self.filepath.exists():
            self._write({"tasks": {}, "projects": {}})

    def _read(self) -> dict:
        with open(self.filepath, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, data: dict) -> None:
        with open(self.filepath, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    # ── Task methods ──────────────────────────────────────────────────────────

    def save_task(self, task: Task) -> Task:
        data = self._read()
        data["tasks"][task.id] = task.to_dict()
        self._write(data)
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        data = self._read()
        raw = data["tasks"].get(task_id)
        return Task.from_dict(raw) if raw else None

    def list_tasks(
        self,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> list[Task]:
        data = self._read()
        tasks = [Task.from_dict(t) for t in data["tasks"].values()]
        if project_id:
            tasks = [t for t in tasks if t.project_id == project_id]
        if status:
            tasks = [t for t in tasks if t.status.value == status]
        if priority:
            tasks = [t for t in tasks if t.priority.value == priority]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def delete_task(self, task_id: str) -> bool:
        data = self._read()
        if task_id not in data["tasks"]:
            return False
        del data["tasks"][task_id]
        self._write(data)
        return True

    # ── Project methods ───────────────────────────────────────────────────────

    def save_project(self, project: Project) -> Project:
        data = self._read()
        data["projects"][project.id] = project.to_dict()
        self._write(data)
        return project

    def get_project(self, project_id: str) -> Optional[Project]:
        data = self._read()
        raw = data["projects"].get(project_id)
        return Project.from_dict(raw) if raw else None

    def list_projects(self) -> list[Project]:
        data = self._read()
        return [Project.from_dict(p) for p in data["projects"].values()]

    def delete_project(self, project_id: str) -> bool:
        data = self._read()
        if project_id not in data["projects"]:
            return False
        del data["projects"][project_id]
        self._write(data)
        return True
```

---

## Phase 3 — Business Logic (`team-test/services.py`)

**File**: `team-test/services.py`

**Imports**:
```python
from __future__ import annotations
from datetime import datetime
from typing import Optional
from .models import Task, Project, Status, Priority
from .storage import JSONFileStorage
```

### `TaskService`

```python
class TaskService:
    def __init__(self, storage: JSONFileStorage) -> None:
        self.storage = storage

    def create_task(
        self,
        title: str,
        description: str = "",
        priority: str = "medium",
        due_date: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Task:
        """Create and persist a new task."""
        task = Task(
            title=title,
            description=description,
            priority=Priority(priority),
            due_date=due_date,
            project_id=project_id,
        )
        return self.storage.save_task(task)

    def assign_to_project(self, task_id: str, project_id: str) -> Task:
        """Assign an existing task to a project."""
        task = self.storage.get_task(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found.")
        project = self.storage.get_project(project_id)
        if project is None:
            raise ValueError(f"Project '{project_id}' not found.")
        task.project_id = project_id
        task.updated_at = datetime.utcnow().isoformat()
        return self.storage.save_task(task)

    def change_status(self, task_id: str, new_status: str) -> Task:
        """Change the status of a task."""
        task = self.storage.get_task(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found.")
        task.status = Status(new_status)
        task.updated_at = datetime.utcnow().isoformat()
        return self.storage.save_task(task)

    def update_task(self, task_id: str, **kwargs) -> Task:
        """Update arbitrary fields on a task."""
        task = self.storage.get_task(task_id)
        if task is None:
            raise ValueError(f"Task '{task_id}' not found.")
        for key, value in kwargs.items():
            if key == "status":
                task.status = Status(value)
            elif key == "priority":
                task.priority = Priority(value)
            elif hasattr(task, key):
                setattr(task, key, value)
        task.updated_at = datetime.utcnow().isoformat()
        return self.storage.save_task(task)

    def delete_task(self, task_id: str) -> bool:
        return self.storage.delete_task(task_id)

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.storage.get_task(task_id)

    def list_tasks(
        self,
        project_id: Optional[str] = None,
        status: Optional[str] = None,
        priority: Optional[str] = None,
    ) -> list[Task]:
        return self.storage.list_tasks(
            project_id=project_id, status=status, priority=priority
        )

    def get_overdue_tasks(self) -> list[Task]:
        """Return all tasks that are past their due_date and not done."""
        return [t for t in self.storage.list_tasks() if t.is_overdue()]

    def get_tasks_by_priority(self, priority: str) -> list[Task]:
        """Return tasks filtered by priority level."""
        return self.storage.list_tasks(priority=priority)

    def get_stats(self) -> dict:
        """Return a summary dict of task counts by status and priority."""
        tasks = self.storage.list_tasks()
        stats: dict = {
            "total": len(tasks),
            "by_status": {s.value: 0 for s in Status},
            "by_priority": {p.value: 0 for p in Priority},
            "overdue": 0,
        }
        for task in tasks:
            stats["by_status"][task.status.value] += 1
            stats["by_priority"][task.priority.value] += 1
            if task.is_overdue():
                stats["overdue"] += 1
        return stats
```

### `ProjectService`

```python
class ProjectService:
    def __init__(self, storage: JSONFileStorage) -> None:
        self.storage = storage

    def create_project(self, name: str, description: str = "") -> Project:
        project = Project(name=name, description=description)
        return self.storage.save_project(project)

    def list_projects(self) -> list[Project]:
        return self.storage.list_projects()

    def get_project(self, project_id: str) -> Optional[Project]:
        return self.storage.get_project(project_id)

    def delete_project(self, project_id: str) -> bool:
        return self.storage.delete_project(project_id)
```

---

## Phase 4 — CLI Interface (`team-test/cli.py`)

**File**: `team-test/cli.py`

**Imports**:
```python
from __future__ import annotations
import argparse
import sys
from typing import Optional
from .models import Status, Priority
from .storage import JSONFileStorage
from .services import TaskService, ProjectService
```

**Color helpers** (no external deps — pure ANSI):
```python
class Color:
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    BOLD    = "\033[1m"
    RESET   = "\033[0m"

def colorize(text: str, color: str) -> str:
    return f"{color}{text}{Color.RESET}"

def _priority_color(priority: str) -> str:
    return {
        "critical": Color.RED,
        "high":     Color.MAGENTA,
        "medium":   Color.YELLOW,
        "low":      Color.CYAN,
    }.get(priority, Color.RESET)

def _status_color(status: str) -> str:
    return {
        "done":        Color.GREEN,
        "in_progress": Color.YELLOW,
        "todo":        Color.BLUE,
    }.get(status, Color.RESET)
```

**Table printer**:
```python
def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a simple fixed-width table."""
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            # strip ANSI codes for width calculation
            import re
            plain = re.sub(r"\033\[[0-9;]*m", "", cell)
            col_widths[i] = max(col_widths[i], len(plain))
    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    def fmt_row(cells):
        parts = []
        for cell, w in zip(cells, col_widths):
            import re
            plain_len = len(re.sub(r"\033\[[0-9;]*m", "", cell))
            padding = w - plain_len
            parts.append(f" {cell}{' ' * padding} ")
        return "|" + "|".join(parts) + "|"
    print(sep)
    print(fmt_row([colorize(h, Color.BOLD) for h in headers]))
    print(sep)
    for row in rows:
        print(fmt_row(row))
    print(sep)
```

**Subcommand handlers**:

| Subcommand | Arguments | Action |
|---|---|---|
| `add` | `title`, `--desc`, `--priority`, `--due`, `--project` | `TaskService.create_task()` |
| `list` | `--status`, `--priority`, `--project`, `--overdue` | `TaskService.list_tasks()` / `get_overdue_tasks()` |
| `update` | `task_id`, `--title`, `--desc`, `--status`, `--priority`, `--due` | `TaskService.update_task()` |
| `delete` | `task_id` | `TaskService.delete_task()` |
| `projects` | sub-sub: `add`, `list`, `delete` | `ProjectService.*` |
| `stats` | _(none)_ | `TaskService.get_stats()` |

**`build_parser()` function**:
```python
def build_parser(storage_path: Optional[str] = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="taskman",
        description="Task Management CLI",
    )
    parser.add_argument("--db", default=storage_path, help="Path to JSON storage file")
    sub = parser.add_subparsers(dest="command", required=True)

    # add
    p_add = sub.add_parser("add", help="Add a new task")
    p_add.add_argument("title")
    p_add.add_argument("--desc", default="")
    p_add.add_argument("--priority", choices=[p.value for p in Priority], default="medium")
    p_add.add_argument("--due", dest="due_date", help="Due date YYYY-MM-DD")
    p_add.add_argument("--project", dest="project_id")

    # list
    p_list = sub.add_parser("list", help="List tasks")
    p_list.add_argument("--status", choices=[s.value for s in Status])
    p_list.add_argument("--priority", choices=[p.value for p in Priority])
    p_list.add_argument("--project", dest="project_id")
    p_list.add_argument("--overdue", action="store_true")

    # update
    p_upd = sub.add_parser("update", help="Update a task")
    p_upd.add_argument("task_id")
    p_upd.add_argument("--title")
    p_upd.add_argument("--desc")
    p_upd.add_argument("--status", choices=[s.value for s in Status])
    p_upd.add_argument("--priority", choices=[p.value for p in Priority])
    p_upd.add_argument("--due", dest="due_date")

    # delete
    p_del = sub.add_parser("delete", help="Delete a task")
    p_del.add_argument("task_id")

    # projects
    p_proj = sub.add_parser("projects", help="Manage projects")
    proj_sub = p_proj.add_subparsers(dest="proj_command", required=True)
    p_proj_add = proj_sub.add_parser("add")
    p_proj_add.add_argument("name")
    p_proj_add.add_argument("--desc", default="")
    proj_sub.add_parser("list")
    p_proj_del = proj_sub.add_parser("delete")
    p_proj_del.add_argument("project_id")

    # stats
    sub.add_parser("stats", help="Show task statistics")

    return parser
```

**`main()` entry point**:
```python
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    storage = JSONFileStorage(filepath=args.db)
    task_svc = TaskService(storage)
    proj_svc = ProjectService(storage)

    if args.command == "add":
        task = task_svc.create_task(
            title=args.title,
            description=args.desc,
            priority=args.priority,
            due_date=args.due_date,
            project_id=args.project_id,
        )
        print(colorize(f"✓ Task created: {task.id}", Color.GREEN))

    elif args.command == "list":
        if args.overdue:
            tasks = task_svc.get_overdue_tasks()
        else:
            tasks = task_svc.list_tasks(
                project_id=args.project_id,
                status=args.status,
                priority=args.priority,
            )
        if not tasks:
            print(colorize("No tasks found.", Color.YELLOW))
            return
        rows = [
            [
                t.id[:8],
                t.title,
                colorize(t.status.value, _status_color(t.status.value)),
                colorize(t.priority.value, _priority_color(t.priority.value)),
                t.due_date or "-",
            ]
            for t in tasks
        ]
        print_table(["ID", "Title", "Status", "Priority", "Due Date"], rows)

    elif args.command == "update":
        kwargs = {}
        if args.title:    kwargs["title"] = args.title
        if args.desc:     kwargs["description"] = args.desc
        if args.status:   kwargs["status"] = args.status
        if args.priority: kwargs["priority"] = args.priority
        if args.due_date: kwargs["due_date"] = args.due_date
        task = task_svc.update_task(args.task_id, **kwargs)
        print(colorize(f"✓ Task updated: {task.id}", Color.GREEN))

    elif args.command == "delete":
        ok = task_svc.delete_task(args.task_id)
        if ok:
            print(colorize(f"✓ Task deleted: {args.task_id}", Color.GREEN))
        else:
            print(colorize(f"✗ Task not found: {args.task_id}", Color.RED))
            sys.exit(1)

    elif args.command == "projects":
        if args.proj_command == "add":
            proj = proj_svc.create_project(args.name, args.desc)
            print(colorize(f"✓ Project created: {proj.id}", Color.GREEN))
        elif args.proj_command == "list":
            projects = proj_svc.list_projects()
            if not projects:
                print(colorize("No projects found.", Color.YELLOW))
                return
            rows = [[p.id[:8], p.name, p.description] for p in projects]
            print_table(["ID", "Name", "Description"], rows)
        elif args.proj_command == "delete":
            ok = proj_svc.delete_project(args.project_id)
            msg = f"✓ Project deleted: {args.project_id}" if ok else f"✗ Project not found: {args.project_id}"
            print(colorize(msg, Color.GREEN if ok else Color.RED))

    elif args.command == "stats":
        stats = task_svc.get_stats()
        print(colorize(f"\n{'Task Statistics':^40}", Color.BOLD))
        print(f"  Total tasks : {stats['total']}")
        print(f"  Overdue     : {colorize(str(stats['overdue']), Color.RED)}")
        print(colorize("\nBy Status:", Color.BOLD))
        for status, count in stats["by_status"].items():
            print(f"  {colorize(status, _status_color(status)):<30} {count}")
        print(colorize("\nBy Priority:", Color.BOLD))
        for priority, count in stats["by_priority"].items():
            print(f"  {colorize(priority, _priority_color(priority)):<30} {count}")

if __name__ == "__main__":
    main()
```

---

## Phase 5 — Requirements File (`team-test/requirements.txt`)

**File**: `team-test/requirements.txt`

```
# Runtime — no external deps beyond stdlib (json, argparse, uuid, datetime, pathlib)

# Development / Testing
pytest>=7.4.0
pytest-cov>=4.1.0
```

> The CLI uses only Python stdlib (`argparse`, `json`, `uuid`, `datetime`, `pathlib`, `re`) — zero runtime dependencies. This keeps the tool portable and installable anywhere Python ≥ 3.10 is available.

---

## Phase 6 — Tests

### `team-test/tests/__init__.py`
Empty file.

### `team-test/tests/conftest.py`

```python
"""Shared pytest fixtures for team-test tests."""
from __future__ import annotations
import pytest
import tempfile
import os
from pathlib import Path
from team_test.storage import JSONFileStorage   # or relative: sys.path manipulation
from team_test.services import TaskService, ProjectService
from team_test.models import Task, Project, Status, Priority


@pytest.fixture
def tmp_storage(tmp_path):
    """Return a JSONFileStorage backed by a temp file."""
    db_path = tmp_path / "test_tasks.json"
    return JSONFileStorage(filepath=str(db_path))


@pytest.fixture
def task_service(tmp_storage):
    return TaskService(tmp_storage)


@pytest.fixture
def project_service(tmp_storage):
    return ProjectService(tmp_storage)


@pytest.fixture
def sample_task():
    return Task(
        title="Fix bug #42",
        description="Reproduce and fix the null pointer exception",
        priority=Priority.HIGH,
        due_date="2099-12-31",
    )


@pytest.fixture
def sample_project():
    return Project(name="Alpha Release", description="First public release")


@pytest.fixture
def overdue_task():
    return Task(
        title="Overdue task",
        priority=Priority.CRITICAL,
        due_date="2000-01-01",   # definitely in the past
        status=Status.TODO,
    )
```

### `team-test/tests/test_models.py`

Tests to implement (each as a separate `def test_*` function):

| Test | What it verifies |
|---|---|
| `test_status_enum_values` | `Status.TODO.value == "todo"`, `Status.IN_PROGRESS.value == "in_progress"`, `Status.DONE.value == "done"` |
| `test_priority_enum_values` | All four Priority values match expected strings |
| `test_task_defaults` | `Task(title="X")` gets a UUID `id`, `status=Status.TODO`, `priority=Priority.MEDIUM`, `description=""`, `due_date=None`, `project_id=None`, non-empty `created_at` |
| `test_task_to_dict` | `task.to_dict()` returns a plain dict with string status/priority values |
| `test_task_from_dict_roundtrip` | `Task.from_dict(task.to_dict()) == task` (field-by-field) |
| `test_task_is_overdue_past_date` | Task with `due_date="2000-01-01"` and `status=TODO` → `is_overdue() == True` |
| `test_task_is_overdue_future_date` | Task with `due_date="2099-12-31"` → `is_overdue() == False` |
| `test_task_is_overdue_done_status` | Task with past due_date but `status=DONE` → `is_overdue() == False` |
| `test_task_is_overdue_no_due_date` | Task with `due_date=None` → `is_overdue() == False` |
| `test_project_defaults` | `Project(name="P")` gets UUID `id`, empty `description`, non-empty `created_at` |
| `test_project_to_dict` | Returns dict with all expected keys |
| `test_project_from_dict_roundtrip` | `Project.from_dict(project.to_dict())` matches original |
| `test_status_is_str_enum` | `isinstance(Status.TODO, str) == True` (for JSON serialization) |
| `test_priority_is_str_enum` | `isinstance(Priority.HIGH, str) == True` |

### `team-test/tests/test_storage.py`

Tests to implement:

| Test | What it verifies |
|---|---|
| `test_storage_creates_file` | After `JSONFileStorage(filepath=...)`, the file exists |
| `test_save_and_get_task` | `save_task(task)` then `get_task(task.id)` returns equivalent task |
| `test_get_task_not_found` | `get_task("nonexistent")` returns `None` |
| `test_list_tasks_empty` | Fresh storage returns `[]` |
| `test_list_tasks_multiple` | Save 3 tasks, `list_tasks()` returns all 3 |
| `test_list_tasks_filter_status` | Save todo + done tasks, filter by `status="done"` returns only done |
| `test_list_tasks_filter_priority` | Filter by `priority="high"` returns only high-priority tasks |
| `test_list_tasks_filter_project` | Filter by `project_id` returns only tasks in that project |
| `test_delete_task_existing` | `delete_task(id)` returns `True`, subsequent `get_task` returns `None` |
| `test_delete_task_nonexistent` | `delete_task("bad-id")` returns `False` |
| `test_save_and_get_project` | `save_project(p)` then `get_project(p.id)` returns equivalent project |
| `test_list_projects_empty` | Fresh storage returns `[]` |
| `test_list_projects_multiple` | Save 2 projects, `list_projects()` returns both |
| `test_delete_project_existing` | `delete_project(id)` returns `True` |
| `test_delete_project_nonexistent` | Returns `False` |
| `test_persistence_across_instances` | Save task with instance A, create instance B with same path, `get_task` succeeds |
| `test_storage_file_is_valid_json` | After operations, file is valid JSON with `tasks` and `projects` keys |

### `team-test/tests/test_services.py`

Tests to implement:

| Test | What it verifies |
|---|---|
| `test_create_task_defaults` | `task_service.create_task("T")` returns Task with correct defaults |
| `test_create_task_with_all_fields` | All kwargs are stored correctly |
| `test_create_task_persisted` | Created task can be retrieved via `task_service.get_task(id)` |
| `test_assign_to_project` | Creates task + project, assigns, task.project_id is set |
| `test_assign_to_project_bad_task` | Raises `ValueError` for unknown task_id |
| `test_assign_to_project_bad_project` | Raises `ValueError` for unknown project_id |
| `test_change_status_todo_to_done` | Status changes correctly, `updated_at` is refreshed |
| `test_change_status_invalid` | Raises `ValueError` for unknown status string |
| `test_update_task_title` | `update_task(id, title="New")` changes title |
| `test_update_task_priority` | `update_task(id, priority="critical")` changes priority enum |
| `test_update_task_not_found` | Raises `ValueError` |
| `test_delete_task_existing` | Returns `True` |
| `test_delete_task_nonexistent` | Returns `False` |
| `test_list_tasks_all` | Returns all created tasks |
| `test_list_tasks_by_status` | Filtered correctly |
| `test_get_overdue_tasks` | Returns only tasks with past due_date and non-done status |
| `test_get_overdue_tasks_excludes_done` | Done tasks with past due_date are excluded |
| `test_get_tasks_by_priority` | Returns only tasks matching priority |
| `test_get_stats_empty` | Returns dict with zeros |
| `test_get_stats_counts` | Correct counts after creating tasks with various statuses/priorities |
| `test_project_service_create` | `project_service.create_project("P")` returns Project |
| `test_project_service_list` | Returns all created projects |
| `test_project_service_delete` | Deletes and returns `True` |
| `test_project_service_get` | Returns project by id |

---

## Execution Order & Parallelism

```
Phase 1 (models.py + __init__.py)
    │
    ▼
Phase 2 (storage.py)  ──depends on── Phase 1
    │
    ▼
Phase 3 (services.py) ──depends on── Phase 1 + Phase 2
    │
    ├──► Phase 4 (cli.py)       ──depends on── Phase 3
    │
    └──► Phase 6 (tests/)       ──depends on── Phase 1 + Phase 2 + Phase 3
         ├── conftest.py
         ├── test_models.py     (can run after Phase 1)
         ├── test_storage.py    (can run after Phase 2)
         └── test_services.py   (can run after Phase 3)

Phase 5 (requirements.txt) — independent, can be written at any time
```

**Phases 4, 5, and 6 can be written in parallel** once Phases 1–3 are complete.

---

## Implementation Notes

### Import Strategy
Because `team-test/` uses a hyphen (not valid Python identifier), use **relative imports** within the package:
```python
# Inside storage.py
from .models import Task, Project

# Inside services.py
from .models import Task, Project, Status, Priority
from .storage import JSONFileStorage

# Inside cli.py
from .models import Status, Priority
from .storage import JSONFileStorage
from .services import TaskService, ProjectService
```

For tests, add the workspace root to `sys.path` in `conftest.py` or use `pytest`'s `rootdir` detection:
```python
# team-test/tests/conftest.py
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent))
# Then import as: from team_test.models import ...
# OR use relative: sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
# and import as: from models import ...
```

The safest approach: add `team-test/` itself to `sys.path` and import modules directly (`from models import ...`, `from storage import ...`).

### JSON Schema
The `data.json` file structure:
```json
{
  "tasks": {
    "<uuid>": {
      "id": "<uuid>",
      "title": "...",
      "description": "...",
      "status": "todo",
      "priority": "medium",
      "due_date": "2026-12-31",
      "project_id": null,
      "created_at": "2026-03-19T10:00:00.000000",
      "updated_at": "2026-03-19T10:00:00.000000"
    }
  },
  "projects": {
    "<uuid>": {
      "id": "<uuid>",
      "name": "...",
      "description": "...",
      "created_at": "2026-03-19T10:00:00.000000"
    }
  }
}
```

### Colored Output
Use raw ANSI escape codes (no `colorama` or `rich` dependency) so the tool works on any POSIX terminal. Windows users can enable ANSI via `os.system("color")` or the tool degrades gracefully.

### Error Handling in CLI
- `ValueError` from services → print red error message + `sys.exit(1)`
- Unknown task/project IDs → clear user-facing message
- Invalid enum values → `argparse` `choices=` validation catches these before reaching services

---

## Verification Criteria

After implementation, verify the tool works correctly by running the following:

### 1. Unit Tests
```bash
cd /Users/pradeepsharma/sasva/projects/vllm/team-test
pytest tests/ -v --tb=short
```
**Expected**: All tests pass. Minimum 40 test cases across `test_models.py`, `test_storage.py`, `test_services.py`. Zero failures, zero errors.

```bash
pytest tests/ --cov=. --cov-report=term-missing
```
**Expected**: Coverage ≥ 80% across `models.py`, `storage.py`, `services.py`.

### 2. CLI Smoke Tests
Run from the workspace root or `team-test/` directory:

```bash
# Add a task
python -m team_test.cli --db /tmp/test_tasks.json add "Write unit tests" --priority high --due 2026-12-31
# Expected: "✓ Task created: <uuid>"

# List tasks
python -m team_test.cli --db /tmp/test_tasks.json list
# Expected: Table with 1 row showing the task

# Add a project
python -m team_test.cli --db /tmp/test_tasks.json projects add "My Project" --desc "Test project"
# Expected: "✓ Project created: <uuid>"

# List projects
python -m team_test.cli --db /tmp/test_tasks.json projects list
# Expected: Table with 1 row

# Update task status
python -m team_test.cli --db /tmp/test_tasks.json update <task_id> --status in_progress
# Expected: "✓ Task updated: <task_id>"

# Show stats
python -m team_test.cli --db /tmp/test_tasks.json stats
# Expected: Statistics table showing 1 total task, counts by status and priority

# Delete task
python -m team_test.cli --db /tmp/test_tasks.json delete <task_id>
# Expected: "✓ Task deleted: <task_id>"

# List overdue tasks (none should be overdue since due_date is 2026-12-31)
python -m team_test.cli --db /tmp/test_tasks.json list --overdue
# Expected: "No tasks found." (since the task was deleted)
```

### 3. File Structure Verification
```bash
ls -la /Users/pradeepsharma/sasva/projects/vllm/team-test/
# Expected files: __init__.py, models.py, storage.py, services.py, cli.py, requirements.txt
ls -la /Users/pradeepsharma/sasva/projects/vllm/team-test/tests/
# Expected files: __init__.py, conftest.py, test_models.py, test_storage.py, test_services.py
```

### 4. Import Verification
```bash
cd /Users/pradeepsharma/sasva/projects/vllm/team-test
python -c "from models import Task, Project, Status, Priority; t = Task(title='test'); print(t.id, t.status)"
# Expected: <uuid> Status.TODO

python -c "from storage import JSONFileStorage; s = JSONFileStorage('/tmp/verify.json'); print('OK')"
# Expected: OK

python -c "from services import TaskService; from storage import JSONFileStorage; svc = TaskService(JSONFileStorage('/tmp/verify2.json')); t = svc.create_task('hello'); print(t.title)"
# Expected: hello
```

### 5. Edge Case Verification
```bash
# Attempt to delete non-existent task (should exit with code 1)
python -m team_test.cli --db /tmp/test_tasks.json delete "nonexistent-id"
# Expected: "✗ Task not found: nonexistent-id" + exit code 1

# Invalid priority (argparse should reject)
python -m team_test.cli --db /tmp/test_tasks.json add "Bad task" --priority invalid
# Expected: argparse error message + exit code 2
```
