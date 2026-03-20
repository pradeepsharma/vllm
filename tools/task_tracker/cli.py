"""
cli.py — Click-based CLI entry point for the Task Tracker.

Commands
--------
  add     Add a new task.
  list    List tasks (with optional status/priority filters).
  done    Mark a task as done (by partial ID).
  delete  Delete a task (by partial ID, with confirmation).
  stats   Show a summary of task counts by status and priority.

Usage examples
--------------
  task add --title "Fix login bug" --priority high
  task list
  task list --status todo --priority high
  task done abc123
  task delete abc123
  task stats
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

from task_store import TaskStore

# ---------------------------------------------------------------------------
# Shared console — used by all commands for consistent output
# ---------------------------------------------------------------------------

console = Console()

# ---------------------------------------------------------------------------
# Colour / style maps
# ---------------------------------------------------------------------------

STATUS_STYLES: dict[str, str] = {
    "todo": "cyan",
    "in_progress": "yellow",
    "done": "green",
}

PRIORITY_STYLES: dict[str, str] = {
    "low": "dim white",
    "medium": "white",
    "high": "bold red",
}

STATUS_LABELS: dict[str, str] = {
    "todo": "todo",
    "in_progress": "in progress",
    "done": "done",
}

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _get_store(ctx: click.Context) -> TaskStore:
    """Return the TaskStore instance stored in the Click context object."""
    return ctx.obj["store"]


def _fmt_created_at(iso: str) -> str:
    """Format an ISO-8601 UTC timestamp into a human-friendly local string."""
    try:
        dt = datetime.fromisoformat(iso)
        # Convert to local time for display
        local_dt = dt.astimezone()
        return local_dt.strftime("%Y-%m-%d %H:%M")
    except (ValueError, TypeError):
        return iso  # Fall back to raw string if parsing fails


def _build_task_table(tasks: list[dict], title: str = "Tasks") -> Table:
    """Build and return a Rich Table populated with *tasks*."""
    table = Table(
        title=title,
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )

    table.add_column("ID (short)", style="dim", no_wrap=True, min_width=8)
    table.add_column("Title", min_width=20)
    table.add_column("Status", justify="center", min_width=11)
    table.add_column("Priority", justify="center", min_width=8)
    table.add_column("Created", justify="right", min_width=16)

    for task in tasks:
        short_id = task["id"][:8]
        status = task.get("status", "todo")
        priority = task.get("priority", "medium")

        status_style = STATUS_STYLES.get(status, "white")
        priority_style = PRIORITY_STYLES.get(priority, "white")

        table.add_row(
            short_id,
            task.get("title", ""),
            f"[{status_style}]{STATUS_LABELS.get(status, status)}[/{status_style}]",
            f"[{priority_style}]{priority}[/{priority_style}]",
            _fmt_created_at(task.get("created_at", "")),
        )

    return table


# ---------------------------------------------------------------------------
# Click group
# ---------------------------------------------------------------------------


@click.group()
@click.option(
    "--store",
    "store_path",
    default=None,
    envvar="TASK_STORE_PATH",
    help="Path to the tasks JSON file (default: tasks.json in current directory).",
    metavar="PATH",
)
@click.pass_context
def cli(ctx: click.Context, store_path: str | None) -> None:
    """Task Tracker — a lightweight CLI task manager.

    Manage your tasks from the terminal with full CRUD support and
    rich, colourful output.
    """
    ctx.ensure_object(dict)
    # Resolve store path: CLI flag > env var > default
    resolved_path = store_path or os.environ.get("TASK_STORE_PATH", "tasks.json")
    ctx.obj["store"] = TaskStore(store_path=resolved_path)


# ---------------------------------------------------------------------------
# add command
# ---------------------------------------------------------------------------


@cli.command("add")
@click.option(
    "--title",
    "-t",
    required=True,
    help="Title of the new task.",
    metavar="TEXT",
)
@click.option(
    "--priority",
    "-p",
    default="medium",
    show_default=True,
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    help="Priority level of the task.",
)
@click.pass_context
def add(ctx: click.Context, title: str, priority: str) -> None:
    """Add a new task.

    \b
    Examples:
      task add --title "Fix login bug"
      task add --title "Write docs" --priority low
      task add -t "Deploy to prod" -p high
    """
    store = _get_store(ctx)
    try:
        task = store.add_task(title=title, priority=priority.lower())
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    short_id = task["id"][:8]
    panel = Panel(
        f"[bold green]✅ Task added successfully![/bold green]\n\n"
        f"  [dim]ID:[/dim]       [cyan]{task['id']}[/cyan]\n"
        f"  [dim]Title:[/dim]    {task['title']}\n"
        f"  [dim]Status:[/dim]   [cyan]{STATUS_LABELS['todo']}[/cyan]\n"
        f"  [dim]Priority:[/dim] [{PRIORITY_STYLES[task['priority']]}]{task['priority']}[/{PRIORITY_STYLES[task['priority']]}]\n"
        f"  [dim]Created:[/dim]  {_fmt_created_at(task['created_at'])}",
        title=f"[bold]Task Added[/bold] · [dim]{short_id}[/dim]",
        border_style="green",
        expand=False,
    )
    console.print(panel)


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


@cli.command("list")
@click.option(
    "--status",
    "-s",
    default=None,
    type=click.Choice(["todo", "in_progress", "done"], case_sensitive=False),
    help="Filter by status.",
)
@click.option(
    "--priority",
    "-p",
    default=None,
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    help="Filter by priority.",
)
@click.pass_context
def list_tasks(ctx: click.Context, status: str | None, priority: str | None) -> None:
    """List tasks, with optional filters.

    \b
    Examples:
      task list
      task list --status todo
      task list --priority high
      task list --status in_progress --priority medium
    """
    store = _get_store(ctx)
    try:
        tasks = store.list_tasks(
            status=status.lower() if status else None,
            priority=priority.lower() if priority else None,
        )
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    if not tasks:
        # Build a friendly "no tasks" message that reflects active filters.
        filter_parts: list[str] = []
        if status:
            filter_parts.append(f"status=[cyan]{status}[/cyan]")
        if priority:
            filter_parts.append(f"priority=[cyan]{priority}[/cyan]")
        filter_str = " and ".join(filter_parts)
        msg = (
            f"No tasks found matching {filter_str}."
            if filter_str
            else "No tasks yet. Use [bold]task add[/bold] to create one."
        )
        console.print(f"[dim]{msg}[/dim]")
        return

    # Build title that reflects active filters
    title_parts: list[str] = []
    if status:
        title_parts.append(f"status={status}")
    if priority:
        title_parts.append(f"priority={priority}")
    title = "Tasks" + (f" [{', '.join(title_parts)}]" if title_parts else "")

    table = _build_task_table(tasks, title=title)
    console.print(table)
    console.print(f"  [dim]{len(tasks)} task(s) shown.[/dim]")


# ---------------------------------------------------------------------------
# done command
# ---------------------------------------------------------------------------


@cli.command("done")
@click.argument("partial_id")
@click.pass_context
def done(ctx: click.Context, partial_id: str) -> None:
    """Mark a task as done.

    PARTIAL_ID is a prefix or substring of the task's UUID.

    \b
    Examples:
      task done abc123
      task done 3f2a
    """
    store = _get_store(ctx)
    try:
        task = store.update_status(partial_id=partial_id, new_status="done")
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    short_id = task["id"][:8]
    panel = Panel(
        f"[bold green]✅ Task marked as done![/bold green]\n\n"
        f"  [dim]ID:[/dim]    [cyan]{task['id']}[/cyan]\n"
        f"  [dim]Title:[/dim] {task['title']}",
        title=f"[bold]Done[/bold] · [dim]{short_id}[/dim]",
        border_style="green",
        expand=False,
    )
    console.print(panel)


# ---------------------------------------------------------------------------
# delete command
# ---------------------------------------------------------------------------


@cli.command("delete")
@click.argument("partial_id")
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
@click.pass_context
def delete(ctx: click.Context, partial_id: str, yes: bool) -> None:
    """Delete a task (with confirmation).

    PARTIAL_ID is a prefix or substring of the task's UUID.

    \b
    Examples:
      task delete abc123
      task delete abc123 --yes
    """
    store = _get_store(ctx)

    # Load and resolve the task first so we can show its title in the prompt.
    from task_store import TaskStore as _TS  # noqa: F401 — already imported

    tasks = store._load()  # type: ignore[attr-defined]
    try:
        # Peek at the task without deleting yet
        task = store._find_by_partial_id(partial_id, tasks)  # type: ignore[attr-defined]
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    short_id = task["id"][:8]

    if not yes:
        confirmed = click.confirm(
            f"Delete task [{short_id}] \"{task['title']}\"?",
            default=False,
        )
        if not confirmed:
            console.print("[dim]Deletion cancelled.[/dim]")
            return

    try:
        deleted = store.delete_task(partial_id=partial_id)
    except ValueError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    panel = Panel(
        f"[bold red]🗑  Task deleted.[/bold red]\n\n"
        f"  [dim]ID:[/dim]    [cyan]{deleted['id']}[/cyan]\n"
        f"  [dim]Title:[/dim] {deleted['title']}",
        title=f"[bold]Deleted[/bold] · [dim]{short_id}[/dim]",
        border_style="red",
        expand=False,
    )
    console.print(panel)


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------


@cli.command("stats")
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show a summary of task counts by status and priority.

    \b
    Example:
      task stats
    """
    store = _get_store(ctx)
    tasks = store.list_tasks()

    total = len(tasks)

    # ── Status breakdown ────────────────────────────────────────────────────
    status_counts: dict[str, int] = {s: 0 for s in ("todo", "in_progress", "done")}
    for task in tasks:
        s = task.get("status", "todo")
        if s in status_counts:
            status_counts[s] += 1

    # ── Priority breakdown ──────────────────────────────────────────────────
    priority_counts: dict[str, int] = {p: 0 for p in ("low", "medium", "high")}
    for task in tasks:
        p = task.get("priority", "medium")
        if p in priority_counts:
            priority_counts[p] += 1

    # ── Build breakdown table ───────────────────────────────────────────────
    breakdown = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    breakdown.add_column("Category", style="dim")
    breakdown.add_column("Value", style="bold")
    breakdown.add_column("Count", justify="right")

    # Status rows
    for status_key, label in STATUS_LABELS.items():
        style = STATUS_STYLES.get(status_key, "white")
        count = status_counts[status_key]
        breakdown.add_row(
            "Status",
            f"[{style}]{label}[/{style}]",
            str(count),
        )

    breakdown.add_section()  # visual separator

    # Priority rows
    for prio in ("low", "medium", "high"):
        style = PRIORITY_STYLES.get(prio, "white")
        count = priority_counts[prio]
        breakdown.add_row(
            "Priority",
            f"[{style}]{prio}[/{style}]",
            str(count),
        )

    # ── Wrap in a Panel ─────────────────────────────────────────────────────
    panel = Panel(
        breakdown,
        title="[bold magenta]📊 Task Statistics[/bold magenta]",
        subtitle=f"[dim]Total tasks: {total}[/dim]",
        border_style="magenta",
        expand=False,
    )
    console.print(panel)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
