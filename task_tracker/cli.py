"""
cli.py — Click-based CLI for the task tracker.

Commands:
  add     Add a new task
  list    List tasks (with optional filters)
  done    Mark a task as done
  delete  Delete a task (with confirmation)
  stats   Show task statistics
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from task_tracker.task_store import (
    AmbiguousTaskIDError,
    InvalidPriorityError,
    InvalidStatusError,
    TaskNotFoundError,
    TaskStore,
)

console = Console()
err_console = Console(stderr=True)


def _get_store() -> TaskStore:
    """Return a TaskStore using the default tasks.json path."""
    return TaskStore()


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """Task Tracker — a simple local task management CLI."""


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--title", required=True, help="Task title / description.")
@click.option(
    "--priority",
    default="medium",
    show_default=True,
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    help="Task priority.",
)
def add(title: str, priority: str) -> None:
    """Add a new task."""
    store = _get_store()
    try:
        task = store.add_task(title=title, priority=priority.lower())
    except (ValueError, InvalidPriorityError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    short_id = task["id"][:8]
    # Use click.echo to avoid Rich treating [high] as markup
    click.echo(
        f"\u2705 Task added: {short_id} \u2014 {task['title']} [{task['priority']}]"
    )


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
def list_tasks(status: str | None, priority: str | None) -> None:
    """List tasks, optionally filtered by status or priority."""
    store = _get_store()
    try:
        tasks = store.list_tasks(
            status=status.lower() if status else None,
            priority=priority.lower() if priority else None,
        )
    except (InvalidStatusError, InvalidPriorityError) as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    if not tasks:
        click.echo("No tasks found.")
        return

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("ID", style="dim", width=10)
    table.add_column("Title", min_width=20)
    table.add_column("Status", width=12)
    table.add_column("Priority", width=10)
    table.add_column("Created", width=22)

    status_styles = {
        "todo": "white",
        "in_progress": "yellow",
        "done": "green",
    }
    priority_styles = {
        "low": "blue",
        "medium": "white",
        "high": "red",
    }

    for task in tasks:
        s = task["status"]
        p = task["priority"]
        created = task.get("created_at", "")[:19].replace("T", " ")
        table.add_row(
            task["id"][:8],
            escape(task["title"]),
            f"[{status_styles.get(s, 'white')}]{s}[/]",
            f"[{priority_styles.get(p, 'white')}]{p}[/]",
            created,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# done
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("partial_id")
def done(partial_id: str) -> None:
    """Mark a task as done (PARTIAL_ID is a prefix of the task ID)."""
    store = _get_store()
    try:
        task = store.update_status(partial_id, "done")
    except TaskNotFoundError:
        click.echo(f"Error: No task found matching '{partial_id}'", err=True)
        sys.exit(1)
    except AmbiguousTaskIDError as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"\u2705 Task marked as done: {task['title']}")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("partial_id")
def delete(partial_id: str) -> None:
    """Delete a task (PARTIAL_ID is a prefix of the task ID)."""
    store = _get_store()

    # Resolve the task first so we can show its title in the confirmation prompt.
    try:
        tasks = store.list_tasks()
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    matches = [t for t in tasks if t["id"].startswith(partial_id)]
    if len(matches) == 0:
        click.echo(f"Error: No task found matching '{partial_id}'", err=True)
        sys.exit(1)
    if len(matches) > 1:
        click.echo(
            f"Error: Ambiguous ID prefix '{partial_id}' matches multiple tasks.",
            err=True,
        )
        sys.exit(1)

    task = matches[0]
    confirmed = click.confirm(f"Delete task '{task['title']}'?", default=False)
    if not confirmed:
        click.echo("Aborted.")
        return

    try:
        store.delete_task(partial_id)
    except (TaskNotFoundError, AmbiguousTaskIDError) as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)

    click.echo("\u2705 Task deleted")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@cli.command()
def stats() -> None:
    """Show task statistics."""
    store = _get_store()
    s = store.get_stats()

    content = (
        f"[bold]Total:[/bold]       {s['total']}\n\n"
        f"[bold]By Status[/bold]\n"
        f"  todo:        {s['todo']}\n"
        f"  in_progress: {s['in_progress']}\n"
        f"  done:        {s['done']}\n\n"
        f"[bold]By Priority[/bold]\n"
        f"  high:        {s['high']}\n"
        f"  medium:      {s['medium']}\n"
        f"  low:         {s['low']}"
    )

    console.print(Panel(content, title="[bold cyan]Task Stats[/bold cyan]", expand=False))


if __name__ == "__main__":
    cli()
