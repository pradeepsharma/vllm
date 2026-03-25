"""
cli.py — Click-based command-line interface for the task tracker.

Commands
--------
add     Add a new task.
list    List tasks (optionally filtered by status or priority).
done    Mark a task as done by partial ID.
delete  Delete a task by partial ID (with confirmation).
stats   Show aggregate task statistics.
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from task_tracker.task_store import (
    AmbiguousIDError,
    TaskNotFoundError,
    TaskStore,
)

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_store() -> TaskStore:
    """Return a TaskStore using the default (or env-overridden) path."""
    return TaskStore()


def _error(msg: str) -> None:
    """Print an error message in red and exit with code 1."""
    console.print(f"[bold red]Error:[/bold red] {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(package_name="task-tracker")
def task_tracker_cli() -> None:
    """A simple CLI task tracker with persistent JSON storage."""


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


@task_tracker_cli.command("add")
@click.option("--title", required=True, help="Task title / description.")
@click.option(
    "--priority",
    default="medium",
    show_default=True,
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    help="Task priority.",
)
def add_task(title: str, priority: str) -> None:
    """Add a new task."""
    store = _get_store()
    try:
        task = store.add_task(title=title, priority=priority.lower())
    except ValueError as exc:
        _error(str(exc))
        return
    console.print(f"✅ Added task [bold cyan]{task['id']}[/bold cyan]")


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@task_tracker_cli.command("list")
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
def list_tasks(status: str | None, priority: str | None) -> None:
    """List tasks, optionally filtered by status and/or priority."""
    store = _get_store()
    try:
        tasks = store.list_tasks(
            status=status.lower() if status else None,
            priority=priority.lower() if priority else None,
        )
    except ValueError as exc:
        _error(str(exc))
        return

    if not tasks:
        console.print("No tasks found.")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="cyan", no_wrap=True, min_width=8)
    table.add_column("Title", style="white", min_width=20)
    table.add_column("Status", style="green", min_width=10)
    table.add_column("Priority", style="yellow", min_width=8)
    table.add_column("Created", style="dim", min_width=20)

    for task in tasks:
        table.add_row(
            task["id"][:8],
            task["title"],
            task["status"],
            task["priority"],
            task["created_at"][:19].replace("T", " "),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# done
# ---------------------------------------------------------------------------


@task_tracker_cli.command("done")
@click.argument("partial_id")
def mark_done(partial_id: str) -> None:
    """Mark a task as done by partial ID."""
    store = _get_store()
    try:
        task = store.update_status(partial_id, "done")
    except TaskNotFoundError:
        console.print(
            f"[bold red]Error:[/bold red] No task matching [bold]{partial_id!r}[/bold]."
        )
        sys.exit(1)
    except AmbiguousIDError as exc:
        _error(str(exc))
        return
    except ValueError as exc:
        _error(str(exc))
        return
    console.print(
        f"✅ Marked done: [bold cyan]{task['id'][:8]}[/bold cyan] — {task['title']}"
    )


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@task_tracker_cli.command("delete")
@click.argument("partial_id")
def delete_task(partial_id: str) -> None:
    """Delete a task by partial ID (prompts for confirmation)."""
    store = _get_store()
    try:
        task = store._match(partial_id)  # peek without mutating
    except TaskNotFoundError:
        console.print(
            f"[bold red]Error:[/bold red] No task matching [bold]{partial_id!r}[/bold]."
        )
        sys.exit(1)
    except AmbiguousIDError as exc:
        _error(str(exc))
        return

    click.confirm(
        f"Delete task '{task['title']}' ({task['id'][:8]})? This cannot be undone.",
        abort=True,
    )
    store.delete_task(partial_id)
    console.print(
        f"🗑 Deleted: [bold cyan]{task['id'][:8]}[/bold cyan] — {task['title']}"
    )


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@task_tracker_cli.command("stats")
def show_stats() -> None:
    """Show aggregate task statistics."""
    store = _get_store()
    stats = store.get_stats()

    lines: list[str] = [
        f"[bold]Total tasks:[/bold] {stats['total']}",
        "",
        "[bold]By status:[/bold]",
    ]
    for status, count in sorted(stats["by_status"].items()):
        lines.append(f"  {status:<12} {count}")

    lines.append("")
    lines.append("[bold]By priority:[/bold]")
    for priority, count in sorted(stats["by_priority"].items()):
        lines.append(f"  {priority:<12} {count}")

    panel_content = "\n".join(lines)
    console.print(Panel(panel_content, title="Task Statistics", expand=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    task_tracker_cli()
