"""
cli — Click-based command-line interface for the task tracker.

Subcommands:
    add     — Create a new task
    list    — List tasks (with optional filters)
    done    — Mark a task as done
    delete  — Remove a task permanently
    stats   — Show a summary panel with task counts
"""

from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from task_tracker.task_store import TaskStore, VALID_STATUSES, VALID_PRIORITIES

console = Console()


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.option(
    "--store-path",
    default="tasks.json",
    show_default=True,
    help="Path to the JSON file used for persistence.",
    envvar="TASK_STORE_PATH",
)
@click.pass_context
def cli(ctx: click.Context, store_path: str) -> None:
    """A simple task tracker CLI backed by a JSON flat file."""
    ctx.ensure_object(dict)
    ctx.obj["store"] = TaskStore(store_path)


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


@cli.command()
@click.option("--title", required=True, help="Task title / description.")
@click.option(
    "--priority",
    default="medium",
    show_default=True,
    type=click.Choice(VALID_PRIORITIES, case_sensitive=False),
    help="Task priority.",
)
@click.pass_context
def add(ctx: click.Context, title: str, priority: str) -> None:
    """Add a new task."""
    store: TaskStore = ctx.obj["store"]
    try:
        task = store.add_task(title, priority=priority)  # type: ignore[arg-type]
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    short_id = task.id[:8]
    console.print(
        f"[green]✅ Added task[/green] [bold]{short_id}[/bold] "
        f'"{task.title}" (priority: {task.priority})'
    )


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@cli.command(name="list")
@click.option(
    "--status",
    default=None,
    type=click.Choice(VALID_STATUSES, case_sensitive=False),
    help="Filter by status.",
)
@click.option(
    "--priority",
    default=None,
    type=click.Choice(VALID_PRIORITIES, case_sensitive=False),
    help="Filter by priority.",
)
@click.pass_context
def list_tasks(ctx: click.Context, status: str | None, priority: str | None) -> None:
    """List tasks, optionally filtered by status and/or priority."""
    store: TaskStore = ctx.obj["store"]
    try:
        tasks = store.list_tasks(
            status=status,  # type: ignore[arg-type]
            priority=priority,  # type: ignore[arg-type]
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    if not tasks:
        console.print("[yellow]No tasks found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID", style="dim", width=10)
    table.add_column("Title", no_wrap=True)
    table.add_column("Status", width=12)
    table.add_column("Priority", width=10)
    table.add_column("Created", width=22)

    status_styles = {
        "todo": "white",
        "in_progress": "yellow",
        "done": "green",
    }
    priority_styles = {
        "low": "dim",
        "medium": "blue",
        "high": "red",
    }

    for task in tasks:
        s_style = status_styles.get(task.status, "white")
        p_style = priority_styles.get(task.priority, "white")
        table.add_row(
            task.id[:8],
            task.title,
            f"[{s_style}]{task.status}[/{s_style}]",
            f"[{p_style}]{task.priority}[/{p_style}]",
            task.created_at[:19].replace("T", " "),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# done
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("partial_id")
@click.pass_context
def done(ctx: click.Context, partial_id: str) -> None:
    """Mark a task as done (identified by a partial ID prefix)."""
    store: TaskStore = ctx.obj["store"]
    try:
        task = store.update_status(partial_id, "done")
    except ValueError as exc:
        console.print(f"[red]Error:[/red] No task found with id starting with {partial_id!r}")
        sys.exit(1)

    console.print(
        f"[green]✅ Marked task[/green] [bold]{task.id[:8]}[/bold] as done"
    )


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@cli.command()
@click.argument("partial_id")
@click.option("--yes", is_flag=True, default=False, help="Skip confirmation prompt.")
@click.pass_context
def delete(ctx: click.Context, partial_id: str, yes: bool) -> None:
    """Delete a task permanently (identified by a partial ID prefix)."""
    store: TaskStore = ctx.obj["store"]

    # We need to resolve the task first to show the title in the prompt.
    try:
        tasks = store.list_tasks()
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    # Find matching task for confirmation message
    matches = [t for t in tasks if t.id.startswith(partial_id)]
    if len(matches) == 0:
        console.print(
            f"[red]Error:[/red] No task found with id starting with {partial_id!r}"
        )
        sys.exit(1)
    if len(matches) > 1:
        ids = ", ".join(t.id[:8] for t in matches)
        console.print(
            f"[red]Error:[/red] Ambiguous ID prefix {partial_id!r} matches: {ids}. "
            "Please provide more characters."
        )
        sys.exit(1)

    target = matches[0]

    if not yes:
        click.confirm(
            f'Delete task "{target.title}" ({target.id[:8]})?', abort=True
        )

    try:
        deleted = store.delete_task(partial_id)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    console.print(
        f'[yellow]🗑️  Deleted task[/yellow] [bold]{deleted.id[:8]}[/bold] "{deleted.title}"'
    )


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@cli.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show a summary panel with task counts by status and priority."""
    store: TaskStore = ctx.obj["store"]
    try:
        tasks = store.list_tasks()
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    total = len(tasks)
    counts_by_status = {s: 0 for s in VALID_STATUSES}
    counts_by_priority = {p: 0 for p in VALID_PRIORITIES}

    for task in tasks:
        counts_by_status[task.status] += 1
        counts_by_priority[task.priority] += 1

    lines = [
        f"[bold]Total:[/bold] {total}",
        "",
        "[bold]By Status:[/bold]",
        f"  todo:        {counts_by_status['todo']}",
        f"  in_progress: {counts_by_status['in_progress']}",
        f"  done:        {counts_by_status['done']}",
        "",
        "[bold]By Priority:[/bold]",
        f"  high:   {counts_by_priority['high']}",
        f"  medium: {counts_by_priority['medium']}",
        f"  low:    {counts_by_priority['low']}",
    ]

    panel = Panel(
        "\n".join(lines),
        title="[bold cyan]Task Statistics[/bold cyan]",
        expand=False,
    )
    console.print(panel)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
