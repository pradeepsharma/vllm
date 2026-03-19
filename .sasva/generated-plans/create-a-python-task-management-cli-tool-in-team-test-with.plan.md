# Execution Plan: Python Task Management CLI Tool in `team-test/`

**Generated:** March 19, 2026  
**Workspace:** `/Users/pradeepsharma/sasva/projects/vllm`  
**Target Directory:** `team-test/`

---

## Context & Workspace Analysis

This plan targets the `team-test/` directory inside the vLLM monorepo. The workspace is a large Python project (vLLM) using:
- **Python 3.10–3.13** (per `pyproject.toml`)
- **pytest** as the test framework (per `requirements/test.in` and `pyproject.toml` `[tool.pytest.ini_options]`)
- **dataclasses** and **enums** as the standard pattern for data modeling (see `vllm/outputs.py`, `vllm/sampling_params.py`)
- **argparse** for CLI interfaces (see `vllm/entrypoints/cli/main.py`, `vllm/entrypoints/cli/types.py`)
- **conftest.py** fixtures pattern for pytest (see `tests/lora/conftest.py`, `tests/v1/engine/conftest.py`)
- The `team-test/` directory currently has two empty subdirectories: `api/` and `database/` (both with only `__pycache__`)

The new tool is fully self-contained inside `team-test/` and does **not** depend on vLLM internals.

---

## File Structure to Create

```
team-test/
├── __init__.py
├── requirements.txt
├── models.py
├── storage.py
├── services.py
├── cli.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_models.py
    ├── test_storage.py
    └── test_services.py
```

---

## Phase 1: Data Models (`team-test/models.py`)

**Goal:** Define all data structures — enums and dataclasses — that the rest of the system depends on.

### File: `team-test/models.py`

```python
"""
Data models for the Task Management CLI tool.
Defines enums (Status, Priority) and dataclasses (Task, Project).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class Status(str, Enum):
    """Task lifecycle status."""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"


class Priority(str, Enum):
    """Task urgency level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Task:
    """Represents a single task."""
    title: str
    description: str = ""
    status: Status = Status.TODO
    priority: Priority = Priority.MEDIUM
    due_date: Optional[str] = None          # ISO-8601 date string "YYYY-MM-DD"
    project_id: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
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
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Task":
        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            status=Status(data["status"]),
            priority=Priority(data["priority"]),
            due_date=data.get("due_date"),
            project_id=data.get("project_id"),
            created_at=data["created_at"],
        )


@dataclass
class Project:
    """Represents a project that groups tasks."""
    name: str
    description: str = ""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
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
            created_at=data["created_at"],
        )
```

**Key design decisions:**
- `Status` and `Priority` inherit from `str, Enum` so they serialize naturally to JSON strings (same pattern as `vllm/usage/usage_lib.py` `UsageContext`)
- `due_date` stored as ISO-8601 string for JSON portability; comparison done in services layer
- `id` auto-generated via `uuid.uuid4()` using `field(default_factory=...)` — same pattern as vLLM dataclasses
- `to_dict()` / `from_dict()` provide clean JSON round-trip without external dependencies

---

## Phase 2: Storage Layer (`team-test/storage.py`)

**Goal:** Implement `JSONFileStorage` — a file-backed persistence layer that reads/writes a single JSON file.

### File: `team-test/storage.py`

```python
"""
Storage layer for the Task Management CLI tool.
Implements JSONFileStorage for persisting Tasks and Projects to a JSON file.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from team_test.models import Priority, Project, Status, Task

DEFAULT_DB_PATH = Path.home() / ".task_manager" / "db.json"

_EMPTY_DB: dict = {"tasks": {}, "projects": {}}


class JSONFileStorage:
    """
    Persists Task and Project objects to a single JSON file.

    Schema:
        {
            "tasks":    { "<task_id>":    { ...task fields... }, ... },
            "projects": { "<project_id>": { ...project fields... }, ... }
        }
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self._ensure_db()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _ensure_db(self) -> None:
        """Create the DB file and parent directories if they don't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.db_path.exists():
            self._write(_EMPTY_DB.copy())

    def _read(self) -> dict:
        with open(self.db_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _write(self, data: dict) -> None:
        with open(self.db_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------ #
    # Task operations
    # ------------------------------------------------------------------ #

    def save_task(self, task: Task) -> Task:
        """Insert or update a task. Returns the saved task."""
        db = self._read()
        db["tasks"][task.id] = task.to_dict()
        self._write(db)
        return task

    def get_task(self, task_id: str) -> Optional[Task]:
        """Return a Task by id, or None if not found."""
        db = self._read()
        raw = db["tasks"].get(task_id)
        return Task.from_dict(raw) if raw else None

    def list_tasks(
        self,
        status: Optional[Status] = None,
        priority: Optional[Priority] = None,
        project_id: Optional[str] = None,
    ) -> list[Task]:
        """
        Return all tasks, optionally filtered by status, priority,
        or project_id.
        """
        db = self._read()
        tasks = [Task.from_dict(v) for v in db["tasks"].values()]
        if status is not None:
            tasks = [t for t in tasks if t.status == status]
        if priority is not None:
            tasks = [t for t in tasks if t.priority == priority]
        if project_id is not None:
            tasks = [t for t in tasks if t.project_id == project_id]
        return tasks

    def delete_task(self, task_id: str) -> bool:
        """Delete a task by id. Returns True if deleted, False if not found."""
        db = self._read()
        if task_id not in db["tasks"]:
            return False
        del db["tasks"][task_id]
        self._write(db)
        return True

    # ------------------------------------------------------------------ #
    # Project operations
    # ------------------------------------------------------------------ #

    def save_project(self, project: Project) -> Project:
        """Insert or update a project. Returns the saved project."""
        db = self._read()
        db["projects"][project.id] = project.to_dict()
        self._write(db)
        return project

    def get_project(self, project_id: str) -> Optional[Project]:
        """Return a Project by id, or None if not found."""
        db = self._read()
        raw = db["projects"].get(project_id)
        return Project.from_dict(raw) if raw else None

    def list_projects(self) -> list[Project]:
        """Return all projects."""
        db = self._read()
        return [Project.from_dict(v) for v in db["projects"].values()]

    def delete_project(self, project_id: str) -> bool:
        """Delete a project by id. Returns True if deleted, False if not found."""
        db = self._read()
        if project_id not in db["projects"]:
            return False
        del db["projects"][project_id]
        self._write(db)
        return True
```

**Key design decisions:**
- Single JSON file; atomic read-modify-write pattern (safe for CLI single-user use)
- `DEFAULT_DB_PATH` in `~/.task_manager/db.json` keeps data outside the repo
- Filtering logic in `list_tasks()` is done in-memory after loading — appropriate for CLI-scale data
- `db_path` is injectable (constructor param) to make testing easy with `tmp_path` fixtures

---

## Phase 3: Business Logic (`team-test/services.py`)

**Goal:** Implement `TaskService` and `ProjectService` with domain logic on top of storage.

### File: `team-test/services.py`

```python
"""
Business logic layer for the Task Management CLI tool.
TaskService and ProjectService orchestrate models and storage.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Optional

from team_test.models import Priority, Project, Status, Task
from team_test.storage import JSONFileStorage


class TaskService:
    """High-level operations on Tasks."""

    def __init__(self, storage: JSONFileStorage) -> None:
        self.storage = storage

    def create_task(
        self,
        title: str,
        description: str = "",
        priority: Priority = Priority.MEDIUM,
        due_date: Optional[str] = None,
        project_id: Optional[str] = None,
    ) -> Task:
        """Create and persist a new task."""
        if not title.strip():
            raise ValueError("Task title cannot be empty.")
        task = Task(
            title=title.strip(),
            description=description,
            priority=priority,
            due_date=due_date,
            project_id=project_id,
        )
        return self.storage.save_task(task)

    def get_task(self, task_id: str) -> Optional[Task]:
        return self.storage.get_task(task_id)

    def list_tasks(
        self,
        status: Optional[Status] = None,
        priority: Optional[Priority] = None,
        project_id: Optional[str] = None,
    ) -> list[Task]:
        return self.storage.list_tasks(
            status=status, priority=priority, project_id=project_id
        )

    def change_status(self, task_id: str, new_status: Status) -> Task:
        """Update the status of an existing task."""
        task = self.storage.get_task(task_id)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found.")
        task.status = new_status
        return self.storage.save_task(task)

    def update_task(
        self,
        task_id: str,
        title: Optional[str] = None,
        description: Optional[str] = None,
        priority: Optional[Priority] = None,
        due_date: Optional[str] = None,
    ) -> Task:
        """Partially update task fields."""
        task = self.storage.get_task(task_id)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found.")
        if title is not None:
            task.title = title.strip()
        if description is not None:
            task.description = description
        if priority is not None:
            task.priority = priority
        if due_date is not None:
            task.due_date = due_date
        return self.storage.save_task(task)

    def assign_to_project(self, task_id: str, project_id: str) -> Task:
        """Assign a task to a project."""
        task = self.storage.get_task(task_id)
        if task is None:
            raise KeyError(f"Task '{task_id}' not found.")
        task.project_id = project_id
        return self.storage.save_task(task)

    def delete_task(self, task_id: str) -> bool:
        return self.storage.delete_task(task_id)

    def get_overdue_tasks(self) -> list[Task]:
        """Return tasks whose due_date is in the past and status != DONE."""
        today = date.today().isoformat()
        all_tasks = self.storage.list_tasks()
        return [
            t for t in all_tasks
            if t.due_date is not None
            and t.due_date < today
            and t.status != Status.DONE
        ]

    def get_tasks_by_priority(self, priority: Priority) -> list[Task]:
        """Return all tasks with the given priority."""
        return self.storage.list_tasks(priority=priority)

    def get_stats(self) -> dict:
        """Return a summary dict: counts by status and priority."""
        all_tasks = self.storage.list_tasks()
        stats: dict = {
            "total": len(all_tasks),
            "by_status": {s.value: 0 for s in Status},
            "by_priority": {p.value: 0 for p in Priority},
            "overdue": len(self.get_overdue_tasks()),
        }
        for task in all_tasks:
            stats["by_status"][task.status.value] += 1
            stats["by_priority"][task.priority.value] += 1
        return stats


class ProjectService:
    """High-level operations on Projects."""

    def __init__(self, storage: JSONFileStorage) -> None:
        self.storage = storage

    def create_project(self, name: str, description: str = "") -> Project:
        """Create and persist a new project."""
        if not name.strip():
            raise ValueError("Project name cannot be empty.")
        project = Project(name=name.strip(), description=description)
        return self.storage.save_project(project)

    def get_project(self, project_id: str) -> Optional[Project]:
        return self.storage.get_project(project_id)

    def list_projects(self) -> list[Project]:
        return self.storage.list_projects()

    def delete_project(self, project_id: str) -> bool:
        return self.storage.delete_project(project_id)

    def get_project_tasks(
        self, project_id: str, task_service: TaskService
    ) -> list[Task]:
        """Return all tasks belonging to a project."""
        return task_service.list_tasks(project_id=project_id)
```

**Key design decisions:**
- Services accept `JSONFileStorage` via constructor injection — easy to mock in tests
- `get_overdue_tasks()` compares ISO-8601 date strings lexicographically (valid for `YYYY-MM-DD`)
- `get_stats()` returns a plain dict for easy rendering in the CLI layer
- `change_status()` and `update_task()` raise `KeyError` on missing tasks — CLI layer catches and formats

---

## Phase 4: CLI Interface (`team-test/cli.py`)

**Goal:** Build a user-facing CLI with argparse subcommands, colored output, and formatted tables.

### File: `team-test/cli.py`

```python
"""
CLI interface for the Task Management tool.
Subcommands: add, list, update, delete, projects, stats
Uses ANSI color codes for colored output and formatted tables.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from team_test.models import Priority, Status
from team_test.services import ProjectService, TaskService
from team_test.storage import JSONFileStorage

# ------------------------------------------------------------------ #
# ANSI color helpers
# ------------------------------------------------------------------ #

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[31m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
BLUE   = "\033[34m"
CYAN   = "\033[36m"
WHITE  = "\033[37m"

STATUS_COLORS = {
    Status.TODO:        YELLOW,
    Status.IN_PROGRESS: BLUE,
    Status.DONE:        GREEN,
}

PRIORITY_COLORS = {
    Priority.LOW:      WHITE,
    Priority.MEDIUM:   CYAN,
    Priority.HIGH:     YELLOW,
    Priority.CRITICAL: RED,
}


def colorize(text: str, color: str) -> str:
    return f"{color}{text}{RESET}"


def color_status(status: Status) -> str:
    return colorize(status.value, STATUS_COLORS.get(status, RESET))


def color_priority(priority: Priority) -> str:
    return colorize(priority.value, PRIORITY_COLORS.get(priority, RESET))


# ------------------------------------------------------------------ #
# Table rendering
# ------------------------------------------------------------------ #

def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Print a simple fixed-width table."""
    col_widths = [len(h) for h in headers]
    # Strip ANSI codes for width calculation
    import re
    ansi_escape = re.compile(r'\033\[[0-9;]*m')
    for row in rows:
        for i, cell in enumerate(row):
            visible = ansi_escape.sub("", cell)
            col_widths[i] = max(col_widths[i], len(visible))

    sep = "+" + "+".join("-" * (w + 2) for w in col_widths) + "+"
    header_row = "|" + "|".join(
        f" {BOLD}{h:<{col_widths[i]}}{RESET} " for i, h in enumerate(headers)
    ) + "|"

    print(sep)
    print(header_row)
    print(sep)
    for row in rows:
        ansi_escape2 = re.compile(r'\033\[[0-9;]*m')
        cells = []
        for i, cell in enumerate(row):
            visible_len = len(ansi_escape2.sub("", cell))
            padding = col_widths[i] - visible_len
            cells.append(f" {cell}{' ' * padding} ")
        print("|" + "|".join(cells) + "|")
    print(sep)


# ------------------------------------------------------------------ #
# Subcommand handlers
# ------------------------------------------------------------------ #

def cmd_add(args: argparse.Namespace, task_svc: TaskService) -> None:
    """Handle: task add <title> [options]"""
    priority = Priority(args.priority) if args.priority else Priority.MEDIUM
    task = task_svc.create_task(
        title=args.title,
        description=args.description or "",
        priority=priority,
        due_date=args.due_date,
        project_id=args.project,
    )
    print(colorize(f"✓ Task created: {task.id}", GREEN))
    print(f"  Title:    {task.title}")
    print(f"  Priority: {color_priority(task.priority)}")
    print(f"  Status:   {color_status(task.status)}")
    if task.due_date:
        print(f"  Due:      {task.due_date}")


def cmd_list(args: argparse.Namespace, task_svc: TaskService) -> None:
    """Handle: task list [--status] [--priority] [--project]"""
    status   = Status(args.status)     if args.status   else None
    priority = Priority(args.priority) if args.priority else None
    tasks = task_svc.list_tasks(
        status=status, priority=priority, project_id=args.project
    )
    if not tasks:
        print(colorize("No tasks found.", YELLOW))
        return
    headers = ["ID (short)", "Title", "Status", "Priority", "Due Date", "Project"]
    rows = [
        [
            t.id[:8],
            t.title[:40],
            color_status(t.status),
            color_priority(t.priority),
            t.due_date or "-",
            t.project_id[:8] if t.project_id else "-",
        ]
        for t in tasks
    ]
    print_table(headers, rows)
    print(f"\n{len(tasks)} task(s) found.")


def cmd_update(args: argparse.Namespace, task_svc: TaskService) -> None:
    """Handle: task update <id> [--status] [--priority] [--title] [--due-date]"""
    try:
        if args.status:
            task_svc.change_status(args.id, Status(args.status))
        task = task_svc.update_task(
            task_id=args.id,
            title=args.title,
            description=args.description,
            priority=Priority(args.priority) if args.priority else None,
            due_date=args.due_date,
        )
        print(colorize(f"✓ Task {args.id[:8]} updated.", GREEN))
    except KeyError as e:
        print(colorize(f"✗ Error: {e}", RED), file=sys.stderr)
        sys.exit(1)


def cmd_delete(args: argparse.Namespace, task_svc: TaskService) -> None:
    """Handle: task delete <id>"""
    deleted = task_svc.delete_task(args.id)
    if deleted:
        print(colorize(f"✓ Task {args.id[:8]} deleted.", GREEN))
    else:
        print(colorize(f"✗ Task '{args.id}' not found.", RED), file=sys.stderr)
        sys.exit(1)


def cmd_projects(args: argparse.Namespace, proj_svc: ProjectService) -> None:
    """Handle: task projects [--add <name>] [--list] [--delete <id>]"""
    if args.add:
        proj = proj_svc.create_project(
            name=args.add, description=args.description or ""
        )
        print(colorize(f"✓ Project created: {proj.id}", GREEN))
        print(f"  Name: {proj.name}")
    elif args.delete:
        deleted = proj_svc.delete_project(args.delete)
        if deleted:
            print(colorize(f"✓ Project {args.delete[:8]} deleted.", GREEN))
        else:
            print(colorize(f"✗ Project '{args.delete}' not found.", RED), file=sys.stderr)
            sys.exit(1)
    else:
        projects = proj_svc.list_projects()
        if not projects:
            print(colorize("No projects found.", YELLOW))
            return
        headers = ["ID (short)", "Name", "Description", "Created At"]
        rows = [
            [p.id[:8], p.name, p.description[:40], p.created_at[:10]]
            for p in projects
        ]
        print_table(headers, rows)
        print(f"\n{len(projects)} project(s) found.")


def cmd_stats(args: argparse.Namespace, task_svc: TaskService) -> None:
    """Handle: task stats"""
    stats = task_svc.get_stats()
    print(colorize(f"\n{'='*40}", BOLD))
    print(colorize("  Task Statistics", BOLD))
    print(colorize(f"{'='*40}", BOLD))
    print(f"  Total tasks : {BOLD}{stats['total']}{RESET}")
    print(f"  Overdue     : {colorize(str(stats['overdue']), RED)}")
    print()
    print(colorize("  By Status:", BOLD))
    for status_val, count in stats["by_status"].items():
        color = STATUS_COLORS.get(Status(status_val), RESET)
        print(f"    {colorize(f'{status_val:<12}', color)}: {count}")
    print()
    print(colorize("  By Priority:", BOLD))
    for prio_val, count in stats["by_priority"].items():
        color = PRIORITY_COLORS.get(Priority(prio_val), RESET)
        print(f"    {colorize(f'{prio_val:<12}', color)}: {count}")
    print()


# ------------------------------------------------------------------ #
# Argument parser construction
# ------------------------------------------------------------------ #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="task",
        description="Task Management CLI — manage tasks and projects from the terminal.",
    )
    parser.add_argument(
        "--db", default=None,
        help="Path to the JSON database file (default: ~/.task_manager/db.json)"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- add --
    p_add = subparsers.add_parser("add", help="Add a new task")
    p_add.add_argument("title", help="Task title")
    p_add.add_argument("-d", "--description", default="", help="Task description")
    p_add.add_argument(
        "-p", "--priority",
        choices=[p.value for p in Priority],
        default=Priority.MEDIUM.value,
        help="Task priority (default: medium)",
    )
    p_add.add_argument("--due-date", dest="due_date", help="Due date YYYY-MM-DD")
    p_add.add_argument("--project", help="Project ID to assign to")

    # -- list --
    p_list = subparsers.add_parser("list", help="List tasks")
    p_list.add_argument("--status", choices=[s.value for s in Status], help="Filter by status")
    p_list.add_argument("--priority", choices=[p.value for p in Priority], help="Filter by priority")
    p_list.add_argument("--project", help="Filter by project ID")

    # -- update --
    p_update = subparsers.add_parser("update", help="Update a task")
    p_update.add_argument("id", help="Task ID")
    p_update.add_argument("--title", help="New title")
    p_update.add_argument("-d", "--description", help="New description")
    p_update.add_argument("--status", choices=[s.value for s in Status], help="New status")
    p_update.add_argument("--priority", choices=[p.value for p in Priority], help="New priority")
    p_update.add_argument("--due-date", dest="due_date", help="New due date YYYY-MM-DD")

    # -- delete --
    p_delete = subparsers.add_parser("delete", help="Delete a task")
    p_delete.add_argument("id", help="Task ID")

    # -- projects --
    p_proj = subparsers.add_parser("projects", help="Manage projects")
    proj_group = p_proj.add_mutually_exclusive_group()
    proj_group.add_argument("--add", metavar="NAME", help="Create a new project")
    proj_group.add_argument("--delete", metavar="ID", help="Delete a project by ID")
    p_proj.add_argument("-d", "--description", default="", help="Project description (used with --add)")

    # -- stats --
    subparsers.add_parser("stats", help="Show task statistics")

    return parser


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    from team_test.storage import DEFAULT_DB_PATH
    db_path = args.db if args.db else DEFAULT_DB_PATH
    storage  = JSONFileStorage(db_path=db_path)
    task_svc = TaskService(storage)
    proj_svc = ProjectService(storage)

    dispatch = {
        "add":      lambda: cmd_add(args, task_svc),
        "list":     lambda: cmd_list(args, task_svc),
        "update":   lambda: cmd_update(args, task_svc),
        "delete":   lambda: cmd_delete(args, task_svc),
        "projects": lambda: cmd_projects(args, proj_svc),
        "stats":    lambda: cmd_stats(args, task_svc),
    }
    dispatch[args.command]()


if __name__ == "__main__":
    main()
```

**Key design decisions:**
- Follows the same subcommand dispatch pattern as `vllm/entrypoints/cli/main.py`
- ANSI color codes applied inline — no external color library needed (keeps `requirements.txt` minimal)
- `build_parser()` is a standalone function — makes it testable without side effects
- `main(argv=None)` accepts an optional argv list for easy unit testing
- `--db` flag allows overriding the database path (critical for testing)

---

## Phase 5: Package Init & Requirements

### File: `team-test/__init__.py`

```python
"""Task Management CLI tool package."""
__version__ = "0.1.0"
```

### File: `team-test/requirements.txt`

```
# Runtime dependencies — all stdlib except pytest
pytest>=7.0
pytest-cov>=4.0
```

> **Note:** The tool itself uses only Python stdlib (`argparse`, `json`, `uuid`, `dataclasses`, `enum`, `datetime`, `pathlib`). The `requirements.txt` lists only the test runner.

---

## Phase 6: Tests

### File: `team-test/tests/__init__.py`

```python
```

### File: `team-test/tests/conftest.py`

```python
"""
Shared pytest fixtures for team-test tests.
"""

import json
import pytest
from pathlib import Path

from team_test.storage import JSONFileStorage
from team_test.services import TaskService, ProjectService
from team_test.models import Task, Project, Status, Priority


@pytest.fixture
def tmp_db(tmp_path: Path) -> Path:
    """Return a path to a temporary JSON database file."""
    return tmp_path / "test_db.json"


@pytest.fixture
def storage(tmp_db: Path) -> JSONFileStorage:
    """Return a JSONFileStorage backed by a temp file."""
    return JSONFileStorage(db_path=tmp_db)


@pytest.fixture
def task_service(storage: JSONFileStorage) -> TaskService:
    """Return a TaskService using the temp storage."""
    return TaskService(storage)


@pytest.fixture
def project_service(storage: JSONFileStorage) -> ProjectService:
    """Return a ProjectService using the temp storage."""
    return ProjectService(storage)


@pytest.fixture
def sample_task(task_service: TaskService) -> Task:
    """A pre-created task for use in tests."""
    return task_service.create_task(
        title="Sample Task",
        description="A test task",
        priority=Priority.HIGH,
        due_date="2099-12-31",
    )


@pytest.fixture
def sample_project(project_service: ProjectService) -> Project:
    """A pre-created project for use in tests."""
    return project_service.create_project(
        name="Sample Project",
        description="A test project",
    )
```

---

### File: `team-test/tests/test_models.py`

```python
"""Tests for team_test.models — Task, Project, Status, Priority."""

import pytest
from datetime import datetime

from team_test.models import Task, Project, Status, Priority


class TestStatusEnum:
    def test_values(self):
        assert Status.TODO.value == "todo"
        assert Status.IN_PROGRESS.value == "in_progress"
        assert Status.DONE.value == "done"

    def test_from_string(self):
        assert Status("todo") == Status.TODO
        assert Status("in_progress") == Status.IN_PROGRESS
        assert Status("done") == Status.DONE

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            Status("invalid")


class TestPriorityEnum:
    def test_values(self):
        assert Priority.LOW.value == "low"
        assert Priority.MEDIUM.value == "medium"
        assert Priority.HIGH.value == "high"
        assert Priority.CRITICAL.value == "critical"

    def test_ordering_by_value(self):
        priorities = [p.value for p in Priority]
        assert "low" in priorities
        assert "critical" in priorities


class TestTask:
    def test_default_fields(self):
        task = Task(title="My Task")
        assert task.title == "My Task"
        assert task.description == ""
        assert task.status == Status.TODO
        assert task.priority == Priority.MEDIUM
        assert task.due_date is None
        assert task.project_id is None
        assert task.id is not None
        assert task.created_at is not None

    def test_unique_ids(self):
        t1 = Task(title="A")
        t2 = Task(title="B")
        assert t1.id != t2.id

    def test_to_dict_round_trip(self):
        task = Task(
            title="Round Trip",
            description="desc",
            status=Status.IN_PROGRESS,
            priority=Priority.CRITICAL,
            due_date="2026-06-01",
        )
        d = task.to_dict()
        assert d["status"] == "in_progress"
        assert d["priority"] == "critical"
        restored = Task.from_dict(d)
        assert restored.title == task.title
        assert restored.status == task.status
        assert restored.priority == task.priority
        assert restored.id == task.id

    def test_from_dict_missing_optional_fields(self):
        minimal = {
            "id": "abc",
            "title": "Min",
            "status": "todo",
            "priority": "low",
            "created_at": datetime.utcnow().isoformat(),
        }
        task = Task.from_dict(minimal)
        assert task.description == ""
        assert task.due_date is None
        assert task.project_id is None


class TestProject:
    def test_default_fields(self):
        proj = Project(name="My Project")
        assert proj.name == "My Project"
        assert proj.description == ""
        assert proj.id is not None
        assert proj.created_at is not None

    def test_to_dict_round_trip(self):
        proj = Project(name="Test", description="Desc")
        d = proj.to_dict()
        restored = Project.from_dict(d)
        assert restored.name == proj.name
        assert restored.id == proj.id
        assert restored.description == proj.description
```

---

### File: `team-test/tests/test_storage.py`

```python
"""Tests for team_test.storage — JSONFileStorage."""

import json
import pytest
from pathlib import Path

from team_test.models import Task, Project, Status, Priority
from team_test.storage import JSONFileStorage


class TestJSONFileStorageInit:
    def test_creates_db_file(self, tmp_db: Path):
        _ = JSONFileStorage(db_path=tmp_db)
        assert tmp_db.exists()

    def test_creates_parent_dirs(self, tmp_path: Path):
        nested = tmp_path / "a" / "b" / "c" / "db.json"
        _ = JSONFileStorage(db_path=nested)
        assert nested.exists()

    def test_empty_db_structure(self, tmp_db: Path):
        _ = JSONFileStorage(db_path=tmp_db)
        with open(tmp_db) as f:
            data = json.load(f)
        assert "tasks" in data
        assert "projects" in data
        assert data["tasks"] == {}
        assert data["projects"] == {}


class TestTaskStorage:
    def test_save_and_get_task(self, storage: JSONFileStorage):
        task = Task(title="Test Task")
        storage.save_task(task)
        retrieved = storage.get_task(task.id)
        assert retrieved is not None
        assert retrieved.title == "Test Task"
        assert retrieved.id == task.id

    def test_get_nonexistent_task_returns_none(self, storage: JSONFileStorage):
        assert storage.get_task("nonexistent-id") is None

    def test_list_tasks_empty(self, storage: JSONFileStorage):
        assert storage.list_tasks() == []

    def test_list_tasks_returns_all(self, storage: JSONFileStorage):
        t1 = Task(title="Task 1")
        t2 = Task(title="Task 2")
        storage.save_task(t1)
        storage.save_task(t2)
        tasks = storage.list_tasks()
        assert len(tasks) == 2

    def test_list_tasks_filter_by_status(self, storage: JSONFileStorage):
        t1 = Task(title="Todo", status=Status.TODO)
        t2 = Task(title="Done", status=Status.DONE)
        storage.save_task(t1)
        storage.save_task(t2)
        todo_tasks = storage.list_tasks(status=Status.TODO)
        assert len(todo_tasks) == 1
        assert todo_tasks[0].title == "Todo"

    def test_list_tasks_filter_by_priority(self, storage: JSONFileStorage):
        t1 = Task(title="Low", priority=Priority.LOW)
        t2 = Task(title="High", priority=Priority.HIGH)
        storage.save_task(t1)
        storage.save_task(t2)
        high_tasks = storage.list_tasks(priority=Priority.HIGH)
        assert len(high_tasks) == 1
        assert high_tasks[0].title == "High"

    def test_list_tasks_filter_by_project(self, storage: JSONFileStorage):
        t1 = Task(title="In Project", project_id="proj-1")
        t2 = Task(title="No Project")
        storage.save_task(t1)
        storage.save_task(t2)
        proj_tasks = storage.list_tasks(project_id="proj-1")
        assert len(proj_tasks) == 1
        assert proj_tasks[0].title == "In Project"

    def test_delete_task(self, storage: JSONFileStorage):
        task = Task(title="To Delete")
        storage.save_task(task)
        result = storage.delete_task(task.id)
        assert result is True
        assert storage.get_task(task.id) is None

    def test_delete_nonexistent_task_returns_false(self, storage: JSONFileStorage):
        assert storage.delete_task("ghost-id") is False

    def test_update_task(self, storage: JSONFileStorage):
        task = Task(title="Original")
        storage.save_task(task)
        task.title = "Updated"
        storage.save_task(task)
        retrieved = storage.get_task(task.id)
        assert retrieved.title == "Updated"


class TestProjectStorage:
    def test_save_and_get_project(self, storage: JSONFileStorage):
        proj = Project(name="My Project")
        storage.save_project(proj)
        retrieved = storage.get_project(proj.id)
        assert retrieved is not None
        assert retrieved.name == "My Project"

    def test_get_nonexistent_project_returns_none(self, storage: JSONFileStorage):
        assert storage.get_project("ghost") is None

    def test_list_projects(self, storage: JSONFileStorage):
        p1 = Project(name="P1")
        p2 = Project(name="P2")
        storage.save_project(p1)
        storage.save_project(p2)
        projects = storage.list_projects()
        assert len(projects) == 2

    def test_delete_project(self, storage: JSONFileStorage):
        proj = Project(name="To Delete")
        storage.save_project(proj)
        result = storage.delete_project(proj.id)
        assert result is True
        assert storage.get_project(proj.id) is None

    def test_delete_nonexistent_project_returns_false(self, storage: JSONFileStorage):
        assert storage.delete_project("ghost") is False
```

---

### File: `team-test/tests/test_services.py`

```python
"""Tests for team_test.services — TaskService and ProjectService."""

import pytest

from team_test.models import Priority, Status, Task, Project
from team_test.services import ProjectService, TaskService


class TestTaskServiceCreate:
    def test_create_task_basic(self, task_service: TaskService):
        task = task_service.create_task(title="New Task")
        assert task.title == "New Task"
        assert task.status == Status.TODO
        assert task.priority == Priority.MEDIUM
        assert task.id is not None

    def test_create_task_with_all_fields(self, task_service: TaskService):
        task = task_service.create_task(
            title="Full Task",
            description="Desc",
            priority=Priority.CRITICAL,
            due_date="2026-12-31",
            project_id="proj-abc",
        )
        assert task.priority == Priority.CRITICAL
        assert task.due_date == "2026-12-31"
        assert task.project_id == "proj-abc"

    def test_create_task_empty_title_raises(self, task_service: TaskService):
        with pytest.raises(ValueError, match="empty"):
            task_service.create_task(title="   ")

    def test_create_task_strips_title(self, task_service: TaskService):
        task = task_service.create_task(title="  Padded  ")
        assert task.title == "Padded"


class TestTaskServiceChangeStatus:
    def test_change_status(self, task_service: TaskService, sample_task: Task):
        updated = task_service.change_status(sample_task.id, Status.IN_PROGRESS)
        assert updated.status == Status.IN_PROGRESS

    def test_change_status_to_done(self, task_service: TaskService, sample_task: Task):
        updated = task_service.change_status(sample_task.id, Status.DONE)
        assert updated.status == Status.DONE

    def test_change_status_nonexistent_raises(self, task_service: TaskService):
        with pytest.raises(KeyError):
            task_service.change_status("ghost-id", Status.DONE)


class TestTaskServiceUpdate:
    def test_update_title(self, task_service: TaskService, sample_task: Task):
        updated = task_service.update_task(sample_task.id, title="New Title")
        assert updated.title == "New Title"

    def test_update_priority(self, task_service: TaskService, sample_task: Task):
        updated = task_service.update_task(sample_task.id, priority=Priority.LOW)
        assert updated.priority == Priority.LOW

    def test_update_due_date(self, task_service: TaskService, sample_task: Task):
        updated = task_service.update_task(sample_task.id, due_date="2030-01-01")
        assert updated.due_date == "2030-01-01"

    def test_update_nonexistent_raises(self, task_service: TaskService):
        with pytest.raises(KeyError):
            task_service.update_task("ghost-id", title="X")


class TestTaskServiceAssignToProject:
    def test_assign_to_project(self, task_service: TaskService, sample_task: Task):
        updated = task_service.assign_to_project(sample_task.id, "proj-xyz")
        assert updated.project_id == "proj-xyz"

    def test_assign_nonexistent_task_raises(self, task_service: TaskService):
        with pytest.raises(KeyError):
            task_service.assign_to_project("ghost-id", "proj-xyz")


class TestTaskServiceOverdue:
    def test_overdue_task_detected(self, task_service: TaskService):
        task = task_service.create_task(
            title="Overdue Task",
            due_date="2000-01-01",  # far in the past
        )
        overdue = task_service.get_overdue_tasks()
        assert any(t.id == task.id for t in overdue)

    def test_future_task_not_overdue(self, task_service: TaskService):
        task = task_service.create_task(
            title="Future Task",
            due_date="2099-12-31",
        )
        overdue = task_service.get_overdue_tasks()
        assert not any(t.id == task.id for t in overdue)

    def test_done_task_not_overdue(self, task_service: TaskService):
        task = task_service.create_task(
            title="Done Overdue",
            due_date="2000-01-01",
        )
        task_service.change_status(task.id, Status.DONE)
        overdue = task_service.get_overdue_tasks()
        assert not any(t.id == task.id for t in overdue)

    def test_no_due_date_not_overdue(self, task_service: TaskService):
        task = task_service.create_task(title="No Due Date")
        overdue = task_service.get_overdue_tasks()
        assert not any(t.id == task.id for t in overdue)


class TestTaskServiceGetByPriority:
    def test_get_tasks_by_priority(self, task_service: TaskService):
        task_service.create_task(title="Low", priority=Priority.LOW)
        task_service.create_task(title="High", priority=Priority.HIGH)
        low_tasks = task_service.get_tasks_by_priority(Priority.LOW)
        assert len(low_tasks) == 1
        assert low_tasks[0].title == "Low"


class TestTaskServiceStats:
    def test_stats_empty(self, task_service: TaskService):
        stats = task_service.get_stats()
        assert stats["total"] == 0
        assert stats["overdue"] == 0

    def test_stats_counts(self, task_service: TaskService):
        task_service.create_task(title="T1", priority=Priority.HIGH)
        task_service.create_task(title="T2", priority=Priority.LOW)
        stats = task_service.get_stats()
        assert stats["total"] == 2
        assert stats["by_priority"]["high"] == 1
        assert stats["by_priority"]["low"] == 1
        assert stats["by_status"]["todo"] == 2


class TestProjectService:
    def test_create_project(self, project_service: ProjectService):
        proj = project_service.create_project(name="My Project")
        assert proj.name == "My Project"
        assert proj.id is not None

    def test_create_project_empty_name_raises(self, project_service: ProjectService):
        with pytest.raises(ValueError, match="empty"):
            project_service.create_project(name="  ")

    def test_list_projects(self, project_service: ProjectService):
        project_service.create_project(name="P1")
        project_service.create_project(name="P2")
        projects = project_service.list_projects()
        assert len(projects) == 2

    def test_delete_project(self, project_service: ProjectService, sample_project: Project):
        result = project_service.delete_project(sample_project.id)
        assert result is True
        assert project_service.get_project(sample_project.id) is None

    def test_get_project_tasks(
        self,
        task_service: TaskService,
        project_service: ProjectService,
        sample_project: Project,
    ):
        task_service.create_task(title="T1", project_id=sample_project.id)
        task_service.create_task(title="T2", project_id=sample_project.id)
        task_service.create_task(title="T3")  # no project
        tasks = project_service.get_project_tasks(sample_project.id, task_service)
        assert len(tasks) == 2
```

---

## Execution Phases Summary

| Phase | Files | Parallelizable? | Dependencies |
|-------|-------|-----------------|--------------|
| 1 | `team-test/models.py`, `team-test/__init__.py` | ✅ Yes | None |
| 2 | `team-test/storage.py` | ❌ No | Phase 1 (models) |
| 3 | `team-test/services.py` | ❌ No | Phase 1 + 2 |
| 4 | `team-test/cli.py` | ❌ No | Phase 1 + 2 + 3 |
| 5 | `team-test/requirements.txt` | ✅ Yes (with Phase 1) | None |
| 6 | `team-test/tests/conftest.py`, `test_models.py`, `test_storage.py`, `test_services.py`, `tests/__init__.py` | ✅ Partially (test files parallel after conftest) | Phase 1–4 |

---

## Implementation Notes

### Import Path
Since `team-test/` uses a hyphen (not valid Python identifier), the package must be imported as `team_test` (with underscore). The `__init__.py` at `team-test/__init__.py` makes it a package. When running tests, add `team-test/` to `PYTHONPATH` or use a `pytest.ini` / `pyproject.toml` `pythonpath` setting:

```ini
# In team-test/pytest.ini (or add to root pyproject.toml)
[pytest]
pythonpath = .
testpaths = tests
```

Or run from the `team-test/` directory:
```bash
cd team-test
PYTHONPATH=.. pytest tests/
```

### JSON DB File Location
- Default: `~/.task_manager/db.json`
- Override with `--db /path/to/custom.json` flag
- Tests always use `tmp_path` fixture (pytest built-in) — no shared state between tests

---

## Verification Criteria

After implementation, verify the tool works correctly by running the following commands from the `team-test/` directory:

### 1. Install dependencies
```bash
cd /Users/pradeepsharma/sasva/projects/vllm/team-test
pip install pytest pytest-cov
```

### 2. Run the full test suite
```bash
cd /Users/pradeepsharma/sasva/projects/vllm/team-test
PYTHONPATH=/Users/pradeepsharma/sasva/projects/vllm pytest tests/ -v
```
**Expected:** All tests pass. Minimum 30 test cases across `test_models.py`, `test_storage.py`, `test_services.py`. Zero failures, zero errors.

### 3. Run with coverage
```bash
PYTHONPATH=/Users/pradeepsharma/sasva/projects/vllm pytest tests/ --cov=. --cov-report=term-missing -v
```
**Expected:** Coverage ≥ 85% across `models.py`, `storage.py`, `services.py`.

### 4. CLI smoke tests (using a temp DB)
```bash
cd /Users/pradeepsharma/sasva/projects/vllm/team-test
export PYTHONPATH=/Users/pradeepsharma/sasva/projects/vllm
DB=/tmp/test_task_db.json

# Add a task
python cli.py --db $DB add "Write unit tests" --priority high --due-date 2026-04-01
# Expected: "✓ Task created: <uuid>" with green color output

# List tasks
python cli.py --db $DB list
# Expected: Table with 1 row showing "Write unit tests", priority=high, status=todo

# Add another task
python cli.py --db $DB add "Deploy to production" --priority critical

# List with filter
python cli.py --db $DB list --priority high
# Expected: Table with 1 row (only the high-priority task)

# Update status
TASK_ID=$(python -c "
import json
with open('$DB') as f:
    d = json.load(f)
print(list(d['tasks'].keys())[0])
")
python cli.py --db $DB update $TASK_ID --status in_progress
# Expected: "✓ Task <short-id> updated."

# Stats
python cli.py --db $DB stats
# Expected: Colored stats table showing total=2, by_status.in_progress=1, by_status.todo=1

# Create a project
python cli.py --db $DB projects --add "Sprint 1" --description "First sprint"
# Expected: "✓ Project created: <uuid>"

# List projects
python cli.py --db $DB projects
# Expected: Table with 1 row showing "Sprint 1"

# Delete a task
python cli.py --db $DB delete $TASK_ID
# Expected: "✓ Task <short-id> deleted."

# Verify deletion
python cli.py --db $DB list
# Expected: Table with 1 row (the remaining task)
```

### 5. Error handling verification
```bash
# Delete non-existent task — should exit with code 1
python cli.py --db $DB delete "nonexistent-id-12345"
# Expected: "✗ Task 'nonexistent-id-12345' not found." on stderr, exit code 1

# Add task with empty title — should raise ValueError
python -c "
from team_test.services import TaskService
from team_test.storage import JSONFileStorage
svc = TaskService(JSONFileStorage('/tmp/err_test.json'))
svc.create_task('   ')
"
# Expected: ValueError: Task title cannot be empty.
```

### 6. JSON persistence verification
```bash
# Verify the DB file is valid JSON after operations
python -c "import json; print(json.dumps(json.load(open('/tmp/test_task_db.json')), indent=2))"
# Expected: Valid JSON with "tasks" and "projects" keys
```

---

## File Checklist

| File | Purpose | Status |
|------|---------|--------|
| `team-test/__init__.py` | Package marker + version | ⬜ To create |
| `team-test/requirements.txt` | pytest + pytest-cov | ⬜ To create |
| `team-test/models.py` | Task, Project, Status, Priority | ⬜ To create |
| `team-test/storage.py` | JSONFileStorage | ⬜ To create |
| `team-test/services.py` | TaskService, ProjectService | ⬜ To create |
| `team-test/cli.py` | argparse CLI with colors | ⬜ To create |
| `team-test/tests/__init__.py` | Test package marker | ⬜ To create |
| `team-test/tests/conftest.py` | Shared fixtures | ⬜ To create |
| `team-test/tests/test_models.py` | Model unit tests | ⬜ To create |
| `team-test/tests/test_storage.py` | Storage unit tests | ⬜ To create |
| `team-test/tests/test_services.py` | Service unit tests | ⬜ To create |
