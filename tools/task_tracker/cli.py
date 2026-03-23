"""
cli.py — Click-based CLI interface for the task tracker.

Subcommands
-----------
add     Create a new task.
list    List tasks, optionally filtered by status and/or priority.
done    Mark a task as done (shortcut for update_status → "done").
delete  Remove a task permanently.
stats   Show a summary panel with task counts by status and priority.

Global option
-------------
--store PATH   Path to the JSON store file.
               Also readable from the TASK_STORE_PATH environment variable.
               Defaults to "tasks.json" in the current working directory.

Examples
--------
    task add --title "Fix login bug" --priority high
    task list --status todo
    task list --priority high
    task done a3f2504e
    task delete a3f2504e --yes
    task stats
"""

from __future__ import annotations

import os
import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from task_store import VALID_PRIORITIES, VALID_STATUSES, TaskStore

# ---------------------------------------------------------------------------
# Module-level singletons and constants
# ---------------------------------------------------------------------------

console = Console()

DEFAULT_STORE_PATH: str = os.environ.get("TASK_STORE_PATH", "tasks.json")

# Colour mappings for Rich rendering
PRIORITY_STYLE: dict[str, str] = {
    "low": "green",
    "medium": "yellow",
    "high": "red bold",
}

STATUS_STYLE: dict[str, str] = {
    "todo": "cyan",
    "in_progress": "magenta",
    "done": "dim green",
}

# Human-friendly labels used in the stats panel
STATUS_LABEL: dict[str, str] = {
    "todo": "To Do",
    "in_progress": "In Progress",
    "done": "Done",
}

PRIORITY_LABEL: dict[str, str] = {
    "low": "Low",
    "medium": "Medium",
    "high": "High",
}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def get_store(ctx: click.Context) -> TaskStore:
    """Instantiate and return a :class:`TaskStore` from the context object.

    Parameters
    ----------
    ctx:
        The active Click context.  ``ctx.obj["store_path"]`` must be set
        by the ``cli`` group callback before any subcommand runs.

    Returns
    -------
    TaskStore
        A freshly loaded store backed by the configured JSON file.
    """
    return TaskStore(ctx.obj["store_path"])


def _styled(text: str, style_map: dict[str, str], key: str) -> Text:
    """Return a Rich :class:`~rich.text.Text` object with the appropriate style.

    Falls back to plain text if *key* is not found in *style_map*.
    """
    style = style_map.get(key, "")
    return Text(text, style=style)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.option(
    "--store",
    default=DEFAULT_STORE_PATH,
    envvar="TASK_STORE_PATH",
    show_default=True,
    help="Path to the tasks JSON store file.",
    metavar="PATH",
)
@click.pass_context
def cli(ctx: click.Context, store: str) -> None:
    """A simple command-line task tracker.

    Tasks are persisted to a JSON file (default: tasks.json).
    Use --store or the TASK_STORE_PATH environment variable to
    specify a custom file location.
    """
    # Ensure ctx.obj is always a dict so subcommands can safely read from it.
    ctx.ensure_object(dict)
    ctx.obj["store_path"] = store


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "--title",
    required=True,
    help="Short description of the task.",
    metavar="TEXT",
)
@click.option(
    "--priority",
    default="medium",
    show_default=True,
    type=click.Choice(sorted(VALID_PRIORITIES), case_sensitive=False),
    help="Task priority level.",
)
@click.pass_context
def add(ctx: click.Context, title: str, priority: str) -> None:
    """Add a new task to the store.

    \b
    Examples:
        task add --title "Fix login bug" --priority high
        task add --title "Write docs"
    """
    store = get_store(ctx)
    try:
        task = store.add_task(title=title, priority=priority.lower())
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    short_id = task["id"][:8]
    console.print(
        f"[bold green]✓[/bold green] Added task "
        f"[bold]{short_id}[/bold]: {task['title']} "
        f"([{PRIORITY_STYLE.get(task['priority'], '')}]{task['priority']}[/])"
    )


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@cli.command(name="list")
@click.option(
    "--status",
    default=None,
    type=click.Choice(sorted(VALID_STATUSES), case_sensitive=False),
    help="Filter by task status.",
)
@click.option(
    "--priority",
    default=None,
    type=click.Choice(sorted(VALID_PRIORITIES), case_sensitive=False),
    help="Filter by task priority.",
)
@click.pass_context
def list_tasks(ctx: click.Context, status: str | None, priority: str | None) -> None:
    """List tasks, optionally filtered by status and/or priority.

    \b
    Examples:
        task list
        task list --status todo
        task list --priority high
        task list --status in_progress --priority medium
    """
    store = get_store(ctx)
    try:
        tasks = store.list_tasks(
            status=status.lower() if status else None,
            priority=priority.lower() if priority else None,
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    if not tasks:
        console.print("[dim]No tasks found.[/dim]")
        return

    table = Table(
        title="Tasks",
        show_header=True,
        header_style="bold blue",
        border_style="bright_black",
        expand=False,
    )
    table.add_column("ID", style="dim", no_wrap=True, min_width=8)
    table.add_column("Title", min_width=20)
    table.add_column("Status", no_wrap=True, min_width=11)
    table.add_column("Priority", no_wrap=True, min_width=8)
    table.add_column("Created At", no_wrap=True, min_width=20)

    for task in tasks:
        short_id = task["id"][:8]
        status_text = _styled(
            STATUS_LABEL.get(task["status"], task["status"]),
            STATUS_STYLE,
            task["status"],
        )
        priority_text = _styled(
            PRIORITY_LABEL.get(task["priority"], task["priority"]),
            PRIORITY_STYLE,
            task["priority"],
        )
        table.add_row(
            short_id,
            task["title"],
            status_text,
            priority_text,
            task["created_at"],
        )

    console.print(table)


# ---------------------------------------------------------------------------
# done
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("task_id")
@click.pass_context
def done(ctx: click.Context, task_id: str) -> None:
    """Mark a task as done.

    TASK_ID is a full UUID or a unique prefix of one (e.g. a3f2504e).

    \b
    Examples:
        task done a3f2504e
        task done a3f2504e-1234-5678-abcd-ef0123456789
    """
    store = get_store(ctx)
    try:
        task = store.update_status(task_id, "done")
    except KeyError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    short_id = task["id"][:8]
    console.print(
        f"[bold green]✓[/bold green] Task [bold]{short_id}[/bold] marked as "
        f"[{STATUS_STYLE['done']}]done[/]."
    )


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("task_id")
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip the confirmation prompt.",
)
@click.pass_context
def delete(ctx: click.Context, task_id: str, yes: bool) -> None:
    """Permanently delete a task.

    TASK_ID is a full UUID or a unique prefix of one (e.g. a3f2504e).

    \b
    Examples:
        task delete a3f2504e
        task delete a3f2504e --yes
    """
    if not yes:
        confirmed = click.confirm(
            f"Are you sure you want to delete task '{task_id}'?",
            default=False,
        )
        if not confirmed:
            console.print("[dim]Aborted.[/dim]")
            return

    store = get_store(ctx)
    try:
        task = store.delete_task(task_id)
    except KeyError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    short_id = task["id"][:8]
    console.print(
        f"[bold red]✗[/bold red] Task [bold]{short_id}[/bold] "
        f"([dim]{task['title']}[/dim]) deleted."
    )


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show a summary of tasks grouped by status and priority.

    \b
    Examples:
        task stats
    """
    store = get_store(ctx)
    all_tasks = store.list_tasks()
    total = len(all_tasks)

    # ── Status breakdown ────────────────────────────────────────────────────
    status_counts: dict[str, int] = {s: 0 for s in ("todo", "in_progress", "done")}
    for task in all_tasks:
        s = task.get("status", "todo")
        if s in status_counts:
            status_counts[s] += 1

    # ── Priority breakdown ──────────────────────────────────────────────────
    priority_counts: dict[str, int] = {p: 0 for p in ("low", "medium", "high")}
    for task in all_tasks:
        p = task.get("priority", "medium")
        if p in priority_counts:
            priority_counts[p] += 1

    # ── Build the stats table ───────────────────────────────────────────────
    stats_table = Table(
        show_header=True,
        header_style="bold blue",
        border_style="bright_black",
        expand=True,
        padding=(0, 1),
    )
    stats_table.add_column("Category", style="bold", min_width=12)
    stats_table.add_column("Value", min_width=12)
    stats_table.add_column("Count", justify="right", min_width=6)

    # Status rows
    for status_key in ("todo", "in_progress", "done"):
        count = status_counts[status_key]
        label = STATUS_LABEL[status_key]
        stats_table.add_row(
            "Status",
            _styled(label, STATUS_STYLE, status_key),
            str(count),
        )

    # Separator row (empty)
    stats_table.add_row("", "", "")

    # Priority rows
    for priority_key in ("high", "medium", "low"):
        count = priority_counts[priority_key]
        label = PRIORITY_LABEL[priority_key]
        stats_table.add_row(
            "Priority",
            _styled(label, PRIORITY_STYLE, priority_key),
            str(count),
        )

    # ── Wrap in a Rich Panel ────────────────────────────────────────────────
    panel_title = Text("Task Statistics", style="bold white")
    subtitle = Text(f"Total tasks: {total}", style="dim")

    panel = Panel(
        stats_table,
        title=panel_title,
        subtitle=subtitle,
        border_style="blue",
        padding=(1, 2),
    )
    console.print(panel)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
