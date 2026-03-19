"""Command-line interface for the Task Management CLI Tool.

Provides a rich ``taskman`` CLI with sub-commands for managing tasks and
projects backed by a JSON file (default: ``~/.taskman/data.json``).

Usage examples::

    # Task commands
    taskman task add "Write docs" --priority high --due 2026-12-31
    taskman task list
    taskman task list --status todo --priority high
    taskman task list --overdue
    taskman task show <TASK_ID>
    taskman task update <TASK_ID> --title "New title" --priority low
    taskman task complete <TASK_ID>
    taskman task start <TASK_ID>
    taskman task reopen <TASK_ID>
    taskman task delete <TASK_ID>
    taskman task search "keyword"
    taskman task purge

    # Project commands
    taskman project add "Sprint 1" --description "First sprint"
    taskman project list
    taskman project show <PROJECT_ID>
    taskman project rename <PROJECT_ID> "New Name"
    taskman project delete <PROJECT_ID>
    taskman project delete <PROJECT_ID> --cascade
    taskman project summary <PROJECT_ID>
    taskman project summaries

    # Global options
    taskman --data /path/to/data.json task list
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path setup: allow running ``python cli.py`` directly from the team-test dir
# or from the workspace root.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from models import Priority, Status  # noqa: E402
from services import NotFoundError, ProjectService, TaskService, ValidationError  # noqa: E402
from storage import Storage  # noqa: E402

# ---------------------------------------------------------------------------
# ANSI colour helpers (disabled when stdout is not a TTY)
# ---------------------------------------------------------------------------

_USE_COLOUR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    """Wrap *text* in an ANSI escape sequence if colours are enabled."""
    if not _USE_COLOUR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(text: str) -> str:
    return _c(text, "1")


def _green(text: str) -> str:
    return _c(text, "32")


def _yellow(text: str) -> str:
    return _c(text, "33")


def _red(text: str) -> str:
    return _c(text, "31")


def _cyan(text: str) -> str:
    return _c(text, "36")


def _dim(text: str) -> str:
    return _c(text, "2")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_task(task, *, verbose: bool = False) -> None:
    """Print a formatted task to stdout."""
    status_colours = {
        "todo": _yellow,
        "in_progress": _cyan,
        "done": _green,
    }
    priority_colours = {
        "low": _dim,
        "medium": lambda t: t,
        "high": _yellow,
        "critical": _red,
    }

    colour_fn = status_colours.get(task.status.value, lambda t: t)
    prio_fn = priority_colours.get(task.priority.value, lambda t: t)

    status_str = colour_fn(f"[{task.status.value.upper()}]")
    priority_str = prio_fn(f"(priority: {task.priority.value})")
    overdue_str = _red(" [OVERDUE!]") if task.is_overdue() else ""
    due_str = f" [due: {task.due_date}]" if task.due_date else ""

    print(f"{status_str} {_bold(task.title)} {priority_str}{due_str}{overdue_str}")

    if verbose:
        print(f"  ID:          {task.id}")
        if task.description:
            print(f"  Description: {task.description}")
        if task.project_id:
            print(f"  Project:     {task.project_id}")
        print(f"  Created:     {task.created_at}")
        print(f"  Updated:     {task.updated_at}")


def _print_project(project, *, verbose: bool = False) -> None:
    """Print a formatted project to stdout."""
    print(f"{_bold(project.name)} [id: {project.id}]")
    if verbose:
        if project.description:
            print(f"  Description: {project.description}")
        print(f"  Created:     {project.created_at}")


def _print_summary(summary: dict) -> None:
    """Print a project summary / health metrics."""
    pct = summary["completion_pct"]
    pct_str = _green(f"{pct:.1f}%") if pct >= 100 else (
        _yellow(f"{pct:.1f}%") if pct >= 50 else _red(f"{pct:.1f}%")
    )
    print(f"{_bold(summary['name'])} [id: {summary['project_id']}]")
    if summary["description"]:
        print(f"  Description:    {summary['description']}")
    print(f"  Total tasks:    {summary['total_tasks']}")
    print(f"  Done:           {_green(str(summary['done']))}")
    print(f"  In progress:    {_cyan(str(summary['in_progress']))}")
    print(f"  Todo:           {_yellow(str(summary['todo']))}")
    if summary["overdue"] > 0:
        print(f"  Overdue:        {_red(str(summary['overdue']))}")
    print(f"  Completion:     {pct_str}")


def _err(message: str) -> None:
    """Print an error message to stderr."""
    print(f"{_red('Error:')} {message}", file=sys.stderr)


def _ok(message: str) -> None:
    """Print a success message to stdout."""
    print(_green(f"✓ {message}"))


# ---------------------------------------------------------------------------
# Service factory
# ---------------------------------------------------------------------------

def _make_services(data_path: str):
    """Return (task_svc, proj_svc) backed by *data_path*."""
    store = Storage(data_path)
    return TaskService(store), ProjectService(store)


# ---------------------------------------------------------------------------
# Task sub-commands
# ---------------------------------------------------------------------------

def cmd_task_add(args: argparse.Namespace) -> int:
    """Handle ``taskman task add``."""
    task_svc, proj_svc = _make_services(args.data)
    try:
        task = task_svc.add(
            args.title,
            description=args.description or "",
            priority=args.priority or "medium",
            due_date=args.due,
            project_id=args.project,
        )
    except (ValidationError, NotFoundError) as exc:
        _err(str(exc))
        return 1

    _ok(f"Task created: {task.id}")
    _print_task(task, verbose=True)
    return 0


def cmd_task_list(args: argparse.Namespace) -> int:
    """Handle ``taskman task list``."""
    task_svc, _ = _make_services(args.data)
    try:
        tasks = task_svc.list_all(
            project_id=args.project,
            status=args.status,
            priority=args.priority,
            overdue_only=args.overdue,
        )
    except (ValidationError, NotFoundError) as exc:
        _err(str(exc))
        return 1

    if not tasks:
        print(_dim("No tasks found."))
        return 0

    print(f"{_bold(str(len(tasks)))} task(s):")
    for task in tasks:
        _print_task(task, verbose=args.verbose)
    return 0


def cmd_task_show(args: argparse.Namespace) -> int:
    """Handle ``taskman task show``."""
    task_svc, _ = _make_services(args.data)
    try:
        task = task_svc.get(args.task_id)
    except NotFoundError as exc:
        _err(str(exc))
        return 1

    _print_task(task, verbose=True)
    return 0


def cmd_task_update(args: argparse.Namespace) -> int:
    """Handle ``taskman task update``."""
    task_svc, _ = _make_services(args.data)

    # Build kwargs only for fields that were explicitly provided
    kwargs = {}
    if args.title is not None:
        kwargs["title"] = args.title
    if args.description is not None:
        kwargs["description"] = args.description
    if args.status is not None:
        kwargs["status"] = args.status
    if args.priority is not None:
        kwargs["priority"] = args.priority
    if args.due is not None:
        kwargs["due_date"] = args.due
    if args.clear_due:
        kwargs["due_date"] = None
    if args.project is not None:
        kwargs["project_id"] = args.project
    if args.unlink_project:
        kwargs["project_id"] = None

    if not kwargs:
        _err("No fields to update. Provide at least one option.")
        return 1

    try:
        task = task_svc.update(args.task_id, **kwargs)
    except (NotFoundError, ValidationError) as exc:
        _err(str(exc))
        return 1

    _ok(f"Task updated: {task.id}")
    _print_task(task, verbose=True)
    return 0


def cmd_task_complete(args: argparse.Namespace) -> int:
    """Handle ``taskman task complete``."""
    task_svc, _ = _make_services(args.data)
    try:
        task = task_svc.complete(args.task_id)
    except NotFoundError as exc:
        _err(str(exc))
        return 1

    _ok(f"Task marked as DONE: {task.title}")
    return 0


def cmd_task_start(args: argparse.Namespace) -> int:
    """Handle ``taskman task start``."""
    task_svc, _ = _make_services(args.data)
    try:
        task = task_svc.start(args.task_id)
    except NotFoundError as exc:
        _err(str(exc))
        return 1

    _ok(f"Task marked as IN_PROGRESS: {task.title}")
    return 0


def cmd_task_reopen(args: argparse.Namespace) -> int:
    """Handle ``taskman task reopen``."""
    task_svc, _ = _make_services(args.data)
    try:
        task = task_svc.reopen(args.task_id)
    except NotFoundError as exc:
        _err(str(exc))
        return 1

    _ok(f"Task reopened (TODO): {task.title}")
    return 0


def cmd_task_delete(args: argparse.Namespace) -> int:
    """Handle ``taskman task delete``."""
    task_svc, _ = _make_services(args.data)
    try:
        task = task_svc.get(args.task_id)
        title = task.title
        task_svc.remove(args.task_id)
    except NotFoundError as exc:
        _err(str(exc))
        return 1

    _ok(f"Task deleted: {title}")
    return 0


def cmd_task_search(args: argparse.Namespace) -> int:
    """Handle ``taskman task search``."""
    task_svc, _ = _make_services(args.data)
    try:
        tasks = task_svc.search(args.keyword, project_id=args.project)
    except (ValidationError, NotFoundError) as exc:
        _err(str(exc))
        return 1

    if not tasks:
        print(_dim(f"No tasks matching '{args.keyword}'."))
        return 0

    print(f"{_bold(str(len(tasks)))} task(s) matching {_cyan(repr(args.keyword))}:")
    for task in tasks:
        _print_task(task, verbose=args.verbose)
    return 0


def cmd_task_purge(args: argparse.Namespace) -> int:
    """Handle ``taskman task purge``."""
    task_svc, _ = _make_services(args.data)
    try:
        count = task_svc.purge_completed(project_id=args.project)
    except (ValidationError, NotFoundError) as exc:
        _err(str(exc))
        return 1

    if count == 0:
        print(_dim("No completed tasks to purge."))
    else:
        _ok(f"Purged {count} completed task(s).")
    return 0


def cmd_task_complete_all(args: argparse.Namespace) -> int:
    """Handle ``taskman task complete-all``."""
    task_svc, _ = _make_services(args.data)
    try:
        updated = task_svc.complete_all(project_id=args.project)
    except (ValidationError, NotFoundError) as exc:
        _err(str(exc))
        return 1

    if not updated:
        print(_dim("No pending tasks to complete."))
    else:
        _ok(f"Marked {len(updated)} task(s) as DONE.")
    return 0


def cmd_task_overdue(args: argparse.Namespace) -> int:
    """Handle ``taskman task overdue``."""
    task_svc, _ = _make_services(args.data)
    report = task_svc.overdue_report()

    if not report:
        print(_green("No overdue tasks. 🎉"))
        return 0

    print(f"{_red(_bold(str(len(report))))} overdue task(s):")
    for item in report:
        prio = item["priority"]
        proj = f" [project: {item['project_id']}]" if item["project_id"] else ""
        print(f"  {_red('!')} {_bold(item['title'])} — due {item['due_date']} "
              f"(priority: {prio}){proj}")
        print(f"    id: {item['id']}")
    return 0


# ---------------------------------------------------------------------------
# Project sub-commands
# ---------------------------------------------------------------------------

def cmd_project_add(args: argparse.Namespace) -> int:
    """Handle ``taskman project add``."""
    _, proj_svc = _make_services(args.data)
    try:
        project = proj_svc.create(args.name, description=args.description or "")
    except ValidationError as exc:
        _err(str(exc))
        return 1

    _ok(f"Project created: {project.id}")
    _print_project(project, verbose=True)
    return 0


def cmd_project_list(args: argparse.Namespace) -> int:
    """Handle ``taskman project list``."""
    _, proj_svc = _make_services(args.data)
    projects = proj_svc.list_all()

    if not projects:
        print(_dim("No projects found."))
        return 0

    print(f"{_bold(str(len(projects)))} project(s):")
    for project in projects:
        _print_project(project, verbose=args.verbose)
    return 0


def cmd_project_show(args: argparse.Namespace) -> int:
    """Handle ``taskman project show``."""
    _, proj_svc = _make_services(args.data)
    try:
        project = proj_svc.get(args.project_id)
    except NotFoundError as exc:
        _err(str(exc))
        return 1

    _print_project(project, verbose=True)
    return 0


def cmd_project_rename(args: argparse.Namespace) -> int:
    """Handle ``taskman project rename``."""
    _, proj_svc = _make_services(args.data)
    try:
        project = proj_svc.rename(args.project_id, args.new_name)
    except (NotFoundError, ValidationError) as exc:
        _err(str(exc))
        return 1

    _ok(f"Project renamed to: {project.name}")
    return 0


def cmd_project_delete(args: argparse.Namespace) -> int:
    """Handle ``taskman project delete``."""
    _, proj_svc = _make_services(args.data)
    try:
        project = proj_svc.get(args.project_id)
        name = project.name
        proj_svc.remove(args.project_id, cascade=args.cascade)
    except NotFoundError as exc:
        _err(str(exc))
        return 1

    cascade_note = " (and all its tasks)" if args.cascade else ""
    _ok(f"Project deleted{cascade_note}: {name}")
    return 0


def cmd_project_summary(args: argparse.Namespace) -> int:
    """Handle ``taskman project summary``."""
    _, proj_svc = _make_services(args.data)
    try:
        summary = proj_svc.summary(args.project_id)
    except NotFoundError as exc:
        _err(str(exc))
        return 1

    _print_summary(summary)
    return 0


def cmd_project_summaries(args: argparse.Namespace) -> int:
    """Handle ``taskman project summaries``."""
    _, proj_svc = _make_services(args.data)
    summaries = proj_svc.all_summaries()

    if not summaries:
        print(_dim("No projects found."))
        return 0

    print(f"{_bold(str(len(summaries)))} project(s):\n")
    for i, summary in enumerate(summaries):
        _print_summary(summary)
        if i < len(summaries) - 1:
            print()
    return 0


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="taskman",
        description="A simple task management CLI tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  taskman task add 'Write docs' --priority high --due 2026-12-31\n"
            "  taskman task list --status todo\n"
            "  taskman task complete <TASK_ID>\n"
            "  taskman project add 'Sprint 1'\n"
            "  taskman project summary <PROJECT_ID>\n"
        ),
    )

    parser.add_argument(
        "--data",
        metavar="PATH",
        default=os.path.join(os.path.expanduser("~"), ".taskman", "data.json"),
        help="Path to the JSON data file (default: ~/.taskman/data.json).",
    )

    subparsers = parser.add_subparsers(dest="entity", metavar="ENTITY")
    subparsers.required = True

    _add_task_subparser(subparsers)
    _add_project_subparser(subparsers)

    return parser


def _add_task_subparser(subparsers) -> None:
    """Register all ``task`` sub-commands."""
    task_parser = subparsers.add_parser(
        "task",
        help="Manage tasks.",
        description="Commands for creating, listing, and managing tasks.",
    )
    task_sub = task_parser.add_subparsers(dest="action", metavar="ACTION")
    task_sub.required = True

    # ---- task add ----
    p_add = task_sub.add_parser("add", help="Create a new task.")
    p_add.add_argument("title", help="Task title (required).")
    p_add.add_argument("-d", "--description", metavar="TEXT", help="Optional description.")
    p_add.add_argument(
        "-p", "--priority",
        choices=["low", "medium", "high", "critical"],
        default="medium",
        help="Priority level (default: medium).",
    )
    p_add.add_argument("--due", metavar="YYYY-MM-DD", help="Due date.")
    p_add.add_argument("--project", metavar="PROJECT_ID", help="Assign to a project.")
    p_add.set_defaults(func=cmd_task_add)

    # ---- task list ----
    p_list = task_sub.add_parser("list", help="List tasks.")
    p_list.add_argument("--project", metavar="PROJECT_ID", help="Filter by project.")
    p_list.add_argument(
        "--status",
        choices=["todo", "in_progress", "done"],
        help="Filter by status.",
    )
    p_list.add_argument(
        "--priority",
        choices=["low", "medium", "high", "critical"],
        help="Filter by priority.",
    )
    p_list.add_argument(
        "--overdue",
        action="store_true",
        default=False,
        help="Show only overdue tasks.",
    )
    p_list.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Show full task details.",
    )
    p_list.set_defaults(func=cmd_task_list)

    # ---- task show ----
    p_show = task_sub.add_parser("show", help="Show details of a task.")
    p_show.add_argument("task_id", help="Task ID.")
    p_show.set_defaults(func=cmd_task_show)

    # ---- task update ----
    p_update = task_sub.add_parser("update", help="Update a task's fields.")
    p_update.add_argument("task_id", help="Task ID.")
    p_update.add_argument("--title", help="New title.")
    p_update.add_argument("-d", "--description", metavar="TEXT", help="New description.")
    p_update.add_argument(
        "--status",
        choices=["todo", "in_progress", "done"],
        help="New status.",
    )
    p_update.add_argument(
        "--priority",
        choices=["low", "medium", "high", "critical"],
        help="New priority.",
    )
    p_update.add_argument("--due", metavar="YYYY-MM-DD", help="New due date.")
    p_update.add_argument(
        "--clear-due",
        dest="clear_due",
        action="store_true",
        default=False,
        help="Clear the due date.",
    )
    p_update.add_argument("--project", metavar="PROJECT_ID", help="Move to project.")
    p_update.add_argument(
        "--unlink-project",
        dest="unlink_project",
        action="store_true",
        default=False,
        help="Unlink from current project.",
    )
    p_update.set_defaults(func=cmd_task_update)

    # ---- task complete ----
    p_complete = task_sub.add_parser("complete", help="Mark a task as DONE.")
    p_complete.add_argument("task_id", help="Task ID.")
    p_complete.set_defaults(func=cmd_task_complete)

    # ---- task start ----
    p_start = task_sub.add_parser("start", help="Mark a task as IN_PROGRESS.")
    p_start.add_argument("task_id", help="Task ID.")
    p_start.set_defaults(func=cmd_task_start)

    # ---- task reopen ----
    p_reopen = task_sub.add_parser("reopen", help="Reset a task to TODO.")
    p_reopen.add_argument("task_id", help="Task ID.")
    p_reopen.set_defaults(func=cmd_task_reopen)

    # ---- task delete ----
    p_delete = task_sub.add_parser("delete", help="Delete a task.")
    p_delete.add_argument("task_id", help="Task ID.")
    p_delete.set_defaults(func=cmd_task_delete)

    # ---- task search ----
    p_search = task_sub.add_parser("search", help="Search tasks by keyword.")
    p_search.add_argument("keyword", help="Keyword to search for.")
    p_search.add_argument("--project", metavar="PROJECT_ID", help="Restrict to project.")
    p_search.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Show full task details.",
    )
    p_search.set_defaults(func=cmd_task_search)

    # ---- task purge ----
    p_purge = task_sub.add_parser("purge", help="Delete all completed (DONE) tasks.")
    p_purge.add_argument("--project", metavar="PROJECT_ID", help="Restrict to project.")
    p_purge.set_defaults(func=cmd_task_purge)

    # ---- task complete-all ----
    p_complete_all = task_sub.add_parser(
        "complete-all", help="Mark all pending tasks as DONE."
    )
    p_complete_all.add_argument(
        "--project", metavar="PROJECT_ID", help="Restrict to project."
    )
    p_complete_all.set_defaults(func=cmd_task_complete_all)

    # ---- task overdue ----
    p_overdue = task_sub.add_parser("overdue", help="List all overdue tasks.")
    p_overdue.set_defaults(func=cmd_task_overdue)


def _add_project_subparser(subparsers) -> None:
    """Register all ``project`` sub-commands."""
    project_parser = subparsers.add_parser(
        "project",
        help="Manage projects.",
        description="Commands for creating, listing, and managing projects.",
    )
    proj_sub = project_parser.add_subparsers(dest="action", metavar="ACTION")
    proj_sub.required = True

    # ---- project add ----
    p_add = proj_sub.add_parser("add", help="Create a new project.")
    p_add.add_argument("name", help="Project name (required).")
    p_add.add_argument("-d", "--description", metavar="TEXT", help="Optional description.")
    p_add.set_defaults(func=cmd_project_add)

    # ---- project list ----
    p_list = proj_sub.add_parser("list", help="List all projects.")
    p_list.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Show full project details.",
    )
    p_list.set_defaults(func=cmd_project_list)

    # ---- project show ----
    p_show = proj_sub.add_parser("show", help="Show details of a project.")
    p_show.add_argument("project_id", help="Project ID.")
    p_show.set_defaults(func=cmd_project_show)

    # ---- project rename ----
    p_rename = proj_sub.add_parser("rename", help="Rename a project.")
    p_rename.add_argument("project_id", help="Project ID.")
    p_rename.add_argument("new_name", help="New project name.")
    p_rename.set_defaults(func=cmd_project_rename)

    # ---- project delete ----
    p_delete = proj_sub.add_parser("delete", help="Delete a project.")
    p_delete.add_argument("project_id", help="Project ID.")
    p_delete.add_argument(
        "--cascade",
        action="store_true",
        default=False,
        help="Also delete all tasks belonging to this project.",
    )
    p_delete.set_defaults(func=cmd_project_delete)

    # ---- project summary ----
    p_summary = proj_sub.add_parser(
        "summary", help="Show health metrics for a project."
    )
    p_summary.add_argument("project_id", help="Project ID.")
    p_summary.set_defaults(func=cmd_project_summary)

    # ---- project summaries ----
    p_summaries = proj_sub.add_parser(
        "summaries", help="Show health metrics for all projects."
    )
    p_summaries.set_defaults(func=cmd_project_summaries)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """Parse *argv* and dispatch to the appropriate command handler.

    Parameters
    ----------
    argv:
        Argument list to parse.  Defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    int
        Exit code: ``0`` on success, ``1`` on error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
