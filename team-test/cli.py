#!/usr/bin/env python3
"""
Command-line interface for the Task Management CLI tool.

Usage
-----
    python cli.py --help
    python cli.py add "Buy groceries" --priority high --due 2026-04-01 --tags shopping,food
    python cli.py list
    python cli.py list --status todo --priority high --sort priority
    python cli.py show <id>
    python cli.py update <id> --title "New title" --status in_progress
    python cli.py done <id>
    python cli.py cancel <id>
    python cli.py start <id>
    python cli.py delete <id>
    python cli.py search "keyword"
    python cli.py stats
    python cli.py backup

The default storage file is ``~/.tasks/tasks.json``.  Override it with the
``--file`` / ``-f`` global option or the ``TASKS_FILE`` environment variable.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib
import sys
import textwrap
from datetime import date
from typing import List, Optional

# ---------------------------------------------------------------------------
# Lazy module loading (mirrors the pattern used in services.py / storage.py)
# ---------------------------------------------------------------------------

def _load_module(name: str, filepath: pathlib.Path):
    """Load *filepath* as a module named *name* and register it in sys.modules."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, filepath)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name!r} from {filepath}")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


_here = pathlib.Path(__file__).parent
_models_mod  = _load_module("task_models",   _here / "models.py")
_storage_mod = _load_module("task_storage",  _here / "storage.py")
_service_mod = _load_module("task_services", _here / "services.py")

Priority          = _models_mod.Priority
Status            = _models_mod.Status
Task              = _models_mod.Task
TaskStorage       = _storage_mod.TaskStorage
TaskService       = _service_mod.TaskService
TaskNotFoundError = _service_mod.TaskNotFoundError
ServiceError      = _service_mod.ServiceError

# ---------------------------------------------------------------------------
# ANSI colour helpers (disabled when stdout is not a TTY or NO_COLOR is set)
# ---------------------------------------------------------------------------

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR", "") == ""

_RESET  = "\033[0m"  if _USE_COLOR else ""
_BOLD   = "\033[1m"  if _USE_COLOR else ""
_DIM    = "\033[2m"  if _USE_COLOR else ""
_RED    = "\033[31m" if _USE_COLOR else ""
_GREEN  = "\033[32m" if _USE_COLOR else ""
_YELLOW = "\033[33m" if _USE_COLOR else ""
_BLUE   = "\033[34m" if _USE_COLOR else ""
_CYAN   = "\033[36m" if _USE_COLOR else ""
_WHITE  = "\033[37m" if _USE_COLOR else ""


def _c(text: str, *codes: str) -> str:
    """Wrap *text* in ANSI escape codes."""
    if not codes:
        return text
    return "".join(codes) + text + _RESET


# ---------------------------------------------------------------------------
# Priority / Status colour maps
# ---------------------------------------------------------------------------

_PRIORITY_COLOR = {
    "low":      _DIM,
    "medium":   _WHITE,
    "high":     _YELLOW,
    "critical": _RED + _BOLD,
}

_STATUS_COLOR = {
    "todo":        _CYAN,
    "in_progress": _BLUE + _BOLD,
    "done":        _GREEN,
    "cancelled":   _DIM,
}


def _priority_str(p: str) -> str:
    color = _PRIORITY_COLOR.get(p, "")
    return _c(p.upper(), color)


def _status_str(s: str) -> str:
    color = _STATUS_COLOR.get(s, "")
    label = s.replace("_", " ").upper()
    return _c(label, color)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_COL_WIDTH = 80  # default terminal width for wrapping


def _fmt_date(d: Optional[str], task: Optional[Task] = None) -> str:
    """Format a due-date string, highlighting overdue dates in red."""
    if d is None:
        return _c("—", _DIM)
    if task is not None and task.is_overdue:
        return _c(d + " ⚠ OVERDUE", _RED + _BOLD)
    return d


def _fmt_tags(tags: List[str]) -> str:
    if not tags:
        return _c("(none)", _DIM)
    return " ".join(_c(f"#{t}", _CYAN) for t in tags)


def _print_task_row(task: Task, *, index: Optional[int] = None) -> None:
    """Print a single-line summary row for a task."""
    idx_str = f"{index:>3}. " if index is not None else "     "
    overdue_marker = _c(" ⚠", _RED + _BOLD) if task.is_overdue else ""
    due = task.due_date or "—"
    print(
        f"{idx_str}"
        f"{_c(task.short_id, _DIM)}  "
        f"{_priority_str(task.priority.value):<20}  "
        f"{_status_str(task.status.value):<22}  "
        f"{_c(due, _YELLOW)}{overdue_marker:<12}  "
        f"{task.title}"
    )


def _print_task_detail(task: Task) -> None:
    """Print a detailed view of a single task."""
    sep = _c("─" * 60, _DIM)
    print(sep)
    print(f"  {_c('ID', _BOLD)}          {task.id}")
    print(f"  {_c('Title', _BOLD)}       {task.title}")
    if task.description:
        wrapped = textwrap.fill(
            task.description,
            width=_COL_WIDTH - 14,
            subsequent_indent=" " * 14,
        )
        print(f"  {_c('Description', _BOLD)} {wrapped}")
    print(f"  {_c('Priority', _BOLD)}    {_priority_str(task.priority.value)}")
    print(f"  {_c('Status', _BOLD)}      {_status_str(task.status.value)}")
    print(f"  {_c('Due Date', _BOLD)}    {_fmt_date(task.due_date, task)}")
    print(f"  {_c('Tags', _BOLD)}        {_fmt_tags(task.tags)}")
    print(f"  {_c('Created', _BOLD)}     {task.created_at}")
    print(f"  {_c('Updated', _BOLD)}     {task.updated_at}")
    print(sep)


def _print_header(columns: str) -> None:
    print(_c(columns, _BOLD))
    print(_c("─" * len(columns), _DIM))


# ---------------------------------------------------------------------------
# Argument parsing (stdlib argparse — no third-party deps)
# ---------------------------------------------------------------------------

import argparse


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""

    parser = argparse.ArgumentParser(
        prog="tasks",
        description="A simple command-line task manager.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              tasks add "Write report" --priority high --due 2026-04-15
              tasks list --status todo --sort priority
              tasks done abc12345
              tasks search "report"
              tasks stats
        """),
    )

    # Global option: storage file
    default_file = os.environ.get("TASKS_FILE", "~/.tasks/tasks.json")
    parser.add_argument(
        "--file", "-f",
        metavar="PATH",
        default=default_file,
        help=(
            f"Path to the tasks JSON file "
            f"(default: {default_file!r}, or $TASKS_FILE env var)."
        ),
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True

    # ------------------------------------------------------------------ add
    p_add = subparsers.add_parser(
        "add",
        help="Create a new task.",
        description="Create a new task and add it to the task list.",
    )
    p_add.add_argument("title", help="Short title for the task.")
    p_add.add_argument(
        "--description", "-d",
        default="",
        metavar="TEXT",
        help="Optional longer description.",
    )
    p_add.add_argument(
        "--priority", "-p",
        default="medium",
        choices=["low", "medium", "high", "critical"],
        metavar="LEVEL",
        help="Priority level: low, medium, high, critical (default: medium).",
    )
    p_add.add_argument(
        "--status", "-s",
        default="todo",
        choices=["todo", "in_progress", "done", "cancelled"],
        metavar="STATUS",
        help="Initial status: todo, in_progress, done, cancelled (default: todo).",
    )
    p_add.add_argument(
        "--due",
        metavar="YYYY-MM-DD",
        default=None,
        help="Due date in YYYY-MM-DD format.",
    )
    p_add.add_argument(
        "--tags", "-t",
        metavar="TAG[,TAG...]",
        default=None,
        help="Comma-separated list of tags.",
    )

    # ----------------------------------------------------------------- list
    p_list = subparsers.add_parser(
        "list",
        aliases=["ls"],
        help="List tasks with optional filters.",
        description="List tasks, optionally filtered and sorted.",
    )
    p_list.add_argument(
        "--status", "-s",
        metavar="STATUS",
        default=None,
        help="Filter by status: todo, in_progress, done, cancelled.",
    )
    p_list.add_argument(
        "--priority", "-p",
        metavar="LEVEL",
        default=None,
        help="Filter by priority: low, medium, high, critical.",
    )
    p_list.add_argument(
        "--tag", "-t",
        metavar="TAG",
        default=None,
        help="Filter by tag.",
    )
    p_list.add_argument(
        "--overdue",
        action="store_true",
        default=False,
        help="Show only overdue tasks.",
    )
    p_list.add_argument(
        "--sort",
        metavar="FIELD",
        default="created_at",
        choices=["priority", "due_date", "created_at", "title"],
        help="Sort by: priority, due_date, created_at, title (default: created_at).",
    )
    p_list.add_argument(
        "--desc",
        action="store_true",
        default=False,
        help="Sort in descending order (default: ascending).",
    )

    # ----------------------------------------------------------------- show
    p_show = subparsers.add_parser(
        "show",
        help="Show full details of a task.",
        description="Display all fields of a single task.",
    )
    p_show.add_argument("id", help="Task ID (full UUID or short 8-char prefix).")

    # --------------------------------------------------------------- update
    p_update = subparsers.add_parser(
        "update",
        help="Update one or more fields of a task.",
        description="Modify an existing task's fields.",
    )
    p_update.add_argument("id", help="Task ID (full UUID or short 8-char prefix).")
    p_update.add_argument("--title", metavar="TEXT", default=None, help="New title.")
    p_update.add_argument(
        "--description", "-d",
        metavar="TEXT",
        default=None,
        help="New description.",
    )
    p_update.add_argument(
        "--priority", "-p",
        metavar="LEVEL",
        default=None,
        choices=["low", "medium", "high", "critical"],
        help="New priority level.",
    )
    p_update.add_argument(
        "--status", "-s",
        metavar="STATUS",
        default=None,
        choices=["todo", "in_progress", "done", "cancelled"],
        help="New status.",
    )
    p_update.add_argument(
        "--due",
        metavar="YYYY-MM-DD",
        default=None,
        help="New due date (YYYY-MM-DD), or empty string to clear.",
    )
    p_update.add_argument(
        "--tags", "-t",
        metavar="TAG[,TAG...]",
        default=None,
        help="New comma-separated tags (replaces existing tags).",
    )
    p_update.add_argument(
        "--clear-due",
        action="store_true",
        default=False,
        help="Clear the due date.",
    )

    # ----------------------------------------------------------------- done
    p_done = subparsers.add_parser(
        "done",
        help="Mark a task as done.",
        description="Set a task's status to 'done'.",
    )
    p_done.add_argument("id", help="Task ID (full UUID or short 8-char prefix).")

    # --------------------------------------------------------------- cancel
    p_cancel = subparsers.add_parser(
        "cancel",
        help="Mark a task as cancelled.",
        description="Set a task's status to 'cancelled'.",
    )
    p_cancel.add_argument("id", help="Task ID (full UUID or short 8-char prefix).")

    # ---------------------------------------------------------------- start
    p_start = subparsers.add_parser(
        "start",
        help="Mark a task as in-progress.",
        description="Set a task's status to 'in_progress'.",
    )
    p_start.add_argument("id", help="Task ID (full UUID or short 8-char prefix).")

    # --------------------------------------------------------------- delete
    p_delete = subparsers.add_parser(
        "delete",
        aliases=["rm"],
        help="Delete a task.",
        description="Permanently remove a task from the list.",
    )
    p_delete.add_argument("id", help="Task ID (full UUID or short 8-char prefix).")
    p_delete.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="Skip confirmation prompt.",
    )

    # -------------------------------------------------------------- search
    p_search = subparsers.add_parser(
        "search",
        help="Search tasks by title or description.",
        description="Find tasks whose title or description contains the query string.",
    )
    p_search.add_argument("query", help="Search string (case-insensitive).")

    # --------------------------------------------------------------- stats
    subparsers.add_parser(
        "stats",
        help="Show task statistics.",
        description="Display a summary of task counts by status and priority.",
    )

    # -------------------------------------------------------------- backup
    p_backup = subparsers.add_parser(
        "backup",
        help="Create a backup of the tasks file.",
        description="Copy the tasks JSON file to a backup location.",
    )
    p_backup.add_argument(
        "--suffix",
        metavar="SUFFIX",
        default=".bak",
        help="Suffix to append to the backup filename (default: .bak).",
    )

    return parser


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def _parse_tags(tags_str: Optional[str]) -> Optional[List[str]]:
    """Parse a comma-separated tag string into a list, or return None."""
    if tags_str is None:
        return None
    return [t.strip() for t in tags_str.split(",") if t.strip()]


def cmd_add(service: TaskService, args: argparse.Namespace) -> int:
    """Handle the ``add`` command."""
    tags = _parse_tags(args.tags)
    try:
        task = service.create_task(
            args.title,
            description=args.description,
            priority=args.priority,
            status=args.status,
            due_date=args.due,
            tags=tags if tags is not None else [],
        )
    except ValueError as exc:
        print(_c(f"Error: {exc}", _RED), file=sys.stderr)
        return 1

    print(_c("✓ Task created:", _GREEN), _c(task.short_id, _BOLD), task.title)
    return 0


def cmd_list(service: TaskService, args: argparse.Namespace) -> int:
    """Handle the ``list`` / ``ls`` command."""
    try:
        tasks = service.list_tasks(
            status=args.status,
            priority=args.priority,
            tag=args.tag,
            overdue_only=args.overdue,
            sort_by=args.sort,
            ascending=not args.desc,
        )
    except ValueError as exc:
        print(_c(f"Error: {exc}", _RED), file=sys.stderr)
        return 1

    if not tasks:
        print(_c("No tasks found.", _DIM))
        return 0

    header = f"{'#':>4}  {'ID':8}  {'PRIORITY':<12}  {'STATUS':<14}  {'DUE':12}  TITLE"
    _print_header(header)
    for i, task in enumerate(tasks, start=1):
        _print_task_row(task, index=i)

    print()
    print(_c(f"  {len(tasks)} task(s) shown.", _DIM))
    return 0


def cmd_show(service: TaskService, args: argparse.Namespace) -> int:
    """Handle the ``show`` command."""
    try:
        task = service.get_task(args.id)
    except TaskNotFoundError as exc:
        print(_c(f"Error: {exc}", _RED), file=sys.stderr)
        return 1

    _print_task_detail(task)
    return 0


def cmd_update(service: TaskService, args: argparse.Namespace) -> int:
    """Handle the ``update`` command."""
    # Build kwargs — only pass fields that were explicitly supplied
    kwargs = {}
    if args.title is not None:
        kwargs["title"] = args.title
    if args.description is not None:
        kwargs["description"] = args.description
    if args.priority is not None:
        kwargs["priority"] = args.priority
    if args.status is not None:
        kwargs["status"] = args.status
    if args.clear_due:
        kwargs["due_date"] = ""
    elif args.due is not None:
        kwargs["due_date"] = args.due
    if args.tags is not None:
        kwargs["tags"] = _parse_tags(args.tags) or []

    if not kwargs:
        print(_c("Nothing to update — no fields specified.", _YELLOW), file=sys.stderr)
        return 1

    try:
        task = service.update_task(args.id, **kwargs)
    except TaskNotFoundError as exc:
        print(_c(f"Error: {exc}", _RED), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(_c(f"Error: {exc}", _RED), file=sys.stderr)
        return 1

    print(_c("✓ Task updated:", _GREEN), _c(task.short_id, _BOLD), task.title)
    return 0


def cmd_done(service: TaskService, args: argparse.Namespace) -> int:
    """Handle the ``done`` command."""
    try:
        task = service.complete_task(args.id)
    except TaskNotFoundError as exc:
        print(_c(f"Error: {exc}", _RED), file=sys.stderr)
        return 1

    print(_c("✓ Marked as done:", _GREEN), _c(task.short_id, _BOLD), task.title)
    return 0


def cmd_cancel(service: TaskService, args: argparse.Namespace) -> int:
    """Handle the ``cancel`` command."""
    try:
        task = service.cancel_task(args.id)
    except TaskNotFoundError as exc:
        print(_c(f"Error: {exc}", _RED), file=sys.stderr)
        return 1

    print(_c("✓ Marked as cancelled:", _YELLOW), _c(task.short_id, _BOLD), task.title)
    return 0


def cmd_start(service: TaskService, args: argparse.Namespace) -> int:
    """Handle the ``start`` command."""
    try:
        task = service.start_task(args.id)
    except TaskNotFoundError as exc:
        print(_c(f"Error: {exc}", _RED), file=sys.stderr)
        return 1

    print(_c("✓ Marked as in-progress:", _BLUE), _c(task.short_id, _BOLD), task.title)
    return 0


def cmd_delete(service: TaskService, args: argparse.Namespace) -> int:
    """Handle the ``delete`` / ``rm`` command."""
    # Resolve the task first so we can show its title in the prompt
    try:
        task = service.get_task(args.id)
    except TaskNotFoundError as exc:
        print(_c(f"Error: {exc}", _RED), file=sys.stderr)
        return 1

    if not args.yes:
        prompt = (
            f"Delete task {_c(task.short_id, _BOLD)} "
            f"'{task.title}'? [y/N] "
        )
        answer = input(prompt).strip().lower()
        if answer not in ("y", "yes"):
            print(_c("Aborted.", _DIM))
            return 0

    try:
        service.delete_task(task.id)
    except TaskNotFoundError as exc:
        print(_c(f"Error: {exc}", _RED), file=sys.stderr)
        return 1

    print(_c("✓ Task deleted:", _RED), _c(task.short_id, _BOLD), task.title)
    return 0


def cmd_search(service: TaskService, args: argparse.Namespace) -> int:
    """Handle the ``search`` command."""
    tasks = service.search_tasks(args.query)

    if not tasks:
        print(_c(f"No tasks match '{args.query}'.", _DIM))
        return 0

    header = f"{'#':>4}  {'ID':8}  {'PRIORITY':<12}  {'STATUS':<14}  {'DUE':12}  TITLE"
    _print_header(header)
    for i, task in enumerate(tasks, start=1):
        _print_task_row(task, index=i)

    print()
    print(_c(f"  {len(tasks)} task(s) found.", _DIM))
    return 0


def cmd_stats(service: TaskService, _args: argparse.Namespace) -> int:
    """Handle the ``stats`` command."""
    stats = service.get_statistics()

    total   = stats["total"]
    by_stat = stats["by_status"]
    by_pri  = stats["by_priority"]
    overdue = stats["overdue"]

    sep = _c("─" * 40, _DIM)
    print(sep)
    print(_c("  Task Statistics", _BOLD))
    print(sep)
    print(f"  {'Total tasks':<20} {_c(str(total), _BOLD)}")
    print(f"  {'Overdue':<20} {_c(str(overdue), _RED + _BOLD) if overdue else _c('0', _DIM)}")
    print()
    print(_c("  By Status", _BOLD))
    for status_val, count in by_stat.items():
        label = status_val.replace("_", " ").upper()
        color = _STATUS_COLOR.get(status_val, "")
        print(f"    {_c(label, color):<30} {count}")
    print()
    print(_c("  By Priority", _BOLD))
    for pri_val, count in by_pri.items():
        color = _PRIORITY_COLOR.get(pri_val, "")
        print(f"    {_c(pri_val.upper(), color):<30} {count}")
    print(sep)
    return 0


def cmd_backup(service: TaskService, args: argparse.Namespace) -> int:
    """Handle the ``backup`` command."""
    try:
        backup_path = service.backup(suffix=args.suffix)
    except Exception as exc:
        print(_c(f"Error: {exc}", _RED), file=sys.stderr)
        return 1

    print(_c("✓ Backup created:", _GREEN), str(backup_path))
    return 0


# ---------------------------------------------------------------------------
# Command dispatch table
# ---------------------------------------------------------------------------

_COMMAND_HANDLERS = {
    "add":    cmd_add,
    "list":   cmd_list,
    "ls":     cmd_list,
    "show":   cmd_show,
    "update": cmd_update,
    "done":   cmd_done,
    "cancel": cmd_cancel,
    "start":  cmd_start,
    "delete": cmd_delete,
    "rm":     cmd_delete,
    "search": cmd_search,
    "stats":  cmd_stats,
    "backup": cmd_backup,
}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Parse arguments, build the service, and dispatch to the right handler.

    Parameters
    ----------
    argv:
        Argument list (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code: 0 on success, non-zero on error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    # Build the service
    storage = TaskStorage(args.file)
    service = TaskService(storage)

    handler = _COMMAND_HANDLERS.get(args.command)
    if handler is None:
        parser.print_help(sys.stderr)
        return 2

    try:
        return handler(service, args)
    except KeyboardInterrupt:
        print(_c("\nInterrupted.", _DIM), file=sys.stderr)
        return 130
    except Exception as exc:  # pylint: disable=broad-except
        print(_c(f"Unexpected error: {exc}", _RED), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
