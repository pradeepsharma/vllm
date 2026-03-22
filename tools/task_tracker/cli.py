"""
cli.py — Command-line interface for the CLI Task Tracker.

Provides four sub-commands that delegate to TaskStore:

    add     Create a new task.
    list    Display tasks (with optional filters).
    update  Change the status of an existing task.
    delete  Remove a task permanently.

Usage examples
--------------
    python cli.py add "Write unit tests" --priority high
    python cli.py list
    python cli.py list --status todo --priority high
    python cli.py update <partial-id> done
    python cli.py delete <partial-id>

Exit codes
----------
    0   Success.
    1   User / validation error (bad arguments, task not found, …).
    2   Unexpected internal error.
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

from task_store import (
    TaskStore,
    VALID_PRIORITIES,
    VALID_STATUSES,
)

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

# ANSI colour codes — disabled automatically when stdout is not a TTY.
_COLOURS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    # Status colours
    "todo": "\033[36m",        # cyan
    "in_progress": "\033[33m", # yellow
    "done": "\033[32m",        # green
    # Priority colours
    "low": "\033[34m",         # blue
    "medium": "\033[35m",      # magenta
    "high": "\033[31m",        # red
}


def _colour(text: str, *keys: str) -> str:
    """Wrap *text* in ANSI escape codes for the given *keys*.

    Returns plain *text* when stdout is not a TTY (e.g. piped output).
    """
    if not sys.stdout.isatty():
        return text
    prefix = "".join(_COLOURS.get(k, "") for k in keys)
    return f"{prefix}{text}{_COLOURS['reset']}"


def _fmt_status(status: str) -> str:
    """Return a fixed-width, coloured status label."""
    label = status.replace("_", " ").upper().ljust(11)
    return _colour(label, status)


def _fmt_priority(priority: str) -> str:
    """Return a fixed-width, coloured priority label."""
    label = priority.upper().ljust(6)
    return _colour(label, priority)


def _short_id(full_id: str, length: int = 8) -> str:
    """Return the first *length* characters of a UUID for display."""
    return full_id[:length]


def _print_task_row(task: dict) -> None:
    """Print a single task as a formatted table row."""
    sid = _colour(_short_id(task["id"]), "dim")
    title = task["title"]
    status = _fmt_status(task["status"])
    priority = _fmt_priority(task["priority"])
    created = task.get("created_at", "")[:10]  # YYYY-MM-DD portion only

    print(f"  {sid}  {status}  {priority}  {created}  {title}")


def _print_task_detail(task: dict) -> None:
    """Print all fields of a single task in a labelled block."""
    print()
    print(f"  {'ID':<12}: {task['id']}")
    print(f"  {'Title':<12}: {task['title']}")
    print(f"  {'Status':<12}: {_fmt_status(task['status'])}")
    print(f"  {'Priority':<12}: {_fmt_priority(task['priority'])}")
    print(f"  {'Created':<12}: {task.get('created_at', '')}")
    print()


def _print_table_header() -> None:
    """Print the column header row for the task list."""
    header = (
        f"  {'ID':8}  {'STATUS':11}  {'PRIORITY':6}  {'CREATED':10}  TITLE"
    )
    print(_colour(header, "bold"))
    print(_colour("  " + "-" * (len(header) - 2), "dim"))


# ---------------------------------------------------------------------------
# Sub-command handlers
# ---------------------------------------------------------------------------


def cmd_add(args: argparse.Namespace, store: TaskStore) -> int:
    """Handle the ``add`` sub-command.

    Creates a new task with the given title and optional priority, then
    prints a confirmation message with the new task's short ID.

    Returns
    -------
    int
        Exit code (0 on success, 1 on validation error).
    """
    try:
        task = store.add_task(title=args.title, priority=args.priority)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    short = _short_id(task["id"])
    print(
        f"Task added  [{_colour(short, 'bold')}]  "
        f"{_fmt_priority(task['priority'])}  {task['title']}"
    )
    return 0


def cmd_list(args: argparse.Namespace, store: TaskStore) -> int:
    """Handle the ``list`` sub-command.

    Retrieves tasks (optionally filtered by status and/or priority) and
    renders them as a formatted table.  Prints a friendly message when
    no tasks match the current filters.

    Returns
    -------
    int
        Exit code (0 on success, 1 on validation error).
    """
    status: Optional[str] = args.status
    priority: Optional[str] = args.priority

    try:
        tasks = store.list_tasks(status=status, priority=priority)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not tasks:
        filters = []
        if status:
            filters.append(f"status={status}")
        if priority:
            filters.append(f"priority={priority}")
        filter_str = f" (filters: {', '.join(filters)})" if filters else ""
        print(f"No tasks found{filter_str}.")
        return 0

    _print_table_header()
    for task in tasks:
        _print_task_row(task)

    total = len(tasks)
    noun = "task" if total == 1 else "tasks"
    print(_colour(f"\n  {total} {noun} listed.", "dim"))
    return 0


def cmd_update(args: argparse.Namespace, store: TaskStore) -> int:
    """Handle the ``update`` sub-command.

    Finds the task matching *args.id* (partial UUID) and sets its status
    to *args.status*, then prints a confirmation.

    Returns
    -------
    int
        Exit code (0 on success, 1 on validation / lookup error).
    """
    try:
        task = store.update_status(
            partial_id=args.id, new_status=args.status
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    short = _short_id(task["id"])
    print(
        f"Task updated  [{_colour(short, 'bold')}]  "
        f"status → {_fmt_status(task['status'])}  {task['title']}"
    )
    return 0


def cmd_delete(args: argparse.Namespace, store: TaskStore) -> int:
    """Handle the ``delete`` sub-command.

    Finds the task matching *args.id* (partial UUID), removes it from
    the store, and prints a confirmation.

    Returns
    -------
    int
        Exit code (0 on success, 1 on lookup error).
    """
    try:
        task = store.delete_task(partial_id=args.id)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    short = _short_id(task["id"])
    print(
        f"Task deleted  [{_colour(short, 'bold')}]  {task['title']}"
    )
    return 0


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------


def _build_parser(store_path: str = "tasks.json") -> argparse.ArgumentParser:
    """Construct and return the top-level argument parser.

    Parameters
    ----------
    store_path:
        Path to the JSON task store file.  Exposed as a parameter so
        tests can inject a temporary path without monkey-patching.

    Returns
    -------
    argparse.ArgumentParser
        Fully configured parser with all sub-commands registered.
    """
    parser = argparse.ArgumentParser(
        prog="task-tracker",
        description=(
            "A simple command-line task tracker.\n\n"
            "Manage your to-do list from the terminal.  Tasks are stored\n"
            "in a local JSON file and can be filtered, updated, and deleted\n"
            "using short partial IDs."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  task-tracker add \"Write unit tests\" --priority high\n"
            "  task-tracker list\n"
            "  task-tracker list --status todo\n"
            "  task-tracker list --priority high\n"
            "  task-tracker update a1b2c3d4 in_progress\n"
            "  task-tracker delete a1b2c3d4\n"
        ),
    )

    # Global option: custom store file path (useful for scripting / testing).
    parser.add_argument(
        "--store",
        metavar="FILE",
        default=store_path,
        help=(
            "Path to the JSON task store file "
            "(default: %(default)s)."
        ),
    )

    subparsers = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        title="available commands",
    )

    # ------------------------------------------------------------------ add
    add_parser = subparsers.add_parser(
        "add",
        help="Create a new task.",
        description="Create a new task and save it to the store.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  task-tracker add \"Fix login bug\"\n"
            "  task-tracker add \"Deploy to production\" --priority high\n"
        ),
    )
    add_parser.add_argument(
        "title",
        help="Short description of the task.",
    )
    add_parser.add_argument(
        "--priority",
        choices=list(VALID_PRIORITIES),
        default="medium",
        metavar="PRIORITY",
        help=(
            f"Task priority: {', '.join(VALID_PRIORITIES)} "
            "(default: medium)."
        ),
    )

    # ----------------------------------------------------------------- list
    list_parser = subparsers.add_parser(
        "list",
        help="Display tasks (with optional filters).",
        description=(
            "List all tasks, optionally filtered by status and/or priority.\n"
            "Results are sorted oldest-first."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  task-tracker list\n"
            "  task-tracker list --status todo\n"
            "  task-tracker list --priority high\n"
            "  task-tracker list --status in_progress --priority medium\n"
        ),
    )
    list_parser.add_argument(
        "--status",
        choices=list(VALID_STATUSES),
        default=None,
        metavar="STATUS",
        help=(
            f"Filter by status: {', '.join(VALID_STATUSES)}."
        ),
    )
    list_parser.add_argument(
        "--priority",
        choices=list(VALID_PRIORITIES),
        default=None,
        metavar="PRIORITY",
        help=(
            f"Filter by priority: {', '.join(VALID_PRIORITIES)}."
        ),
    )

    # --------------------------------------------------------------- update
    update_parser = subparsers.add_parser(
        "update",
        help="Change the status of an existing task.",
        description=(
            "Update the status of a task identified by a partial ID.\n"
            "The partial ID must uniquely match exactly one task."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  task-tracker update a1b2c3d4 in_progress\n"
            "  task-tracker update a1b2c3d4 done\n"
        ),
    )
    update_parser.add_argument(
        "id",
        metavar="PARTIAL_ID",
        help="Partial (or full) UUID of the task to update.",
    )
    update_parser.add_argument(
        "status",
        choices=list(VALID_STATUSES),
        metavar="STATUS",
        help=(
            f"New status for the task: {', '.join(VALID_STATUSES)}."
        ),
    )

    # --------------------------------------------------------------- delete
    delete_parser = subparsers.add_parser(
        "delete",
        help="Remove a task permanently.",
        description=(
            "Delete a task identified by a partial ID.\n"
            "The partial ID must uniquely match exactly one task.\n"
            "This action cannot be undone."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  task-tracker delete a1b2c3d4\n"
        ),
    )
    delete_parser.add_argument(
        "id",
        metavar="PARTIAL_ID",
        help="Partial (or full) UUID of the task to delete.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

_COMMAND_HANDLERS = {
    "add": cmd_add,
    "list": cmd_list,
    "update": cmd_update,
    "delete": cmd_delete,
}


def main(argv: Optional[list[str]] = None) -> int:
    """Parse *argv* and dispatch to the appropriate sub-command handler.

    Parameters
    ----------
    argv:
        Argument list to parse.  Defaults to ``sys.argv[1:]`` when
        ``None`` (standard argparse behaviour).

    Returns
    -------
    int
        Exit code to pass to ``sys.exit``.
    """
    # Build the parser with the default store path; the --store flag may
    # override it after parsing.
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    handler = _COMMAND_HANDLERS.get(args.command)
    if handler is None:
        # Should never happen — argparse enforces valid sub-commands.
        print(f"Error: Unknown command '{args.command}'.", file=sys.stderr)
        return 2

    store = TaskStore(store_path=args.store)

    try:
        return handler(args, store)
    except Exception as exc:  # noqa: BLE001
        print(f"Unexpected error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
