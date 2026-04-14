"""
cli.py — Click-based CLI for the task tracker.

Commands:
  task add    --title TEXT [--priority low|medium|high]
  task list   [--status todo|in_progress|done] [--priority low|medium|high]
  task done   <partial-id>
  task delete <partial-id>
  task stats
"""

from __future__ import annotations

import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from task_tracker.task_store import TaskStore, VALID_PRIORITIES, VALID_STATUSES

console = Console()
err_console = Console(stderr=True)

# Default data file lives next to the user's current working directory
DEFAULT_DATA_FILE = Path.home() / ".task_tracker" / "tasks.json"


def _get_store(data_file: str | None = None) -> TaskStore:
    """Return a TaskStore, using the default path if none is given."""
    path = data_file or str(DEFAULT_DATA_FILE)
    return TaskStore(filepath=path)


# ---------------------------------------------------------------------------
# Priority / status colour helpers
# ---------------------------------------------------------------------------

_PRIORITY_COLOURS = {"high": "red", "medium": "yellow", "low": "green"}
_STATUS_COLOURS = {"todo": "cyan", "in_progress": "blue", "done": "green"}


def _priority_markup(p: str) -> str:
    colour = _PRIORITY_COLOURS.get(p, "white")
    return f"[{colour}]{p}[/{colour}]"


def _status_markup(s: str) -> str:
    colour = _STATUS_COLOURS.get(s, "white")
    return f"[{colour}]{s}[/{colour}]"


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
@click.version_option(version="0.1.0", prog_name="task")
@click.option(
    "--data-file",
    envvar="TASK_DATA_FILE",
    default=None,
    hidden=True,
    help="Override the path to the tasks JSON file (for testing).",
)
@click.pass_context
def cli(ctx: click.Context, data_file: str | None) -> None:
    """A simple CLI task tracker.

    Manage your tasks from the command line with priorities and statuses.
    """
    ctx.ensure_object(dict)
    ctx.obj["data_file"] = data_file


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


@cli.command("add")
@click.option("--title", required=True, help="Task title.")
@click.option(
    "--priority",
    default="medium",
    show_default=True,
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    help="Task priority.",
)
@click.pass_context
def add_cmd(ctx: click.Context, title: str, priority: str) -> None:
    """Add a new task."""
    store = _get_store(ctx.obj.get("data_file"))
    try:
        task = store.add_task(title=title, priority=priority.lower())
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    console.print(
        f"[green]Task added[/green] \u2014 id: [bold]{task['id'][:8]}[/bold]  "
        f"title: {task['title']}  priority: {_priority_markup(task['priority'])}"
    )


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@cli.command("list")
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
@click.pass_context
def list_cmd(ctx: click.Context, status: str | None, priority: str | None) -> None:
    """List tasks, optionally filtered by status and/or priority."""
    store = _get_store(ctx.obj.get("data_file"))
    try:
        tasks = store.list_tasks(
            status=status.lower() if status else None,
            priority=priority.lower() if priority else None,
        )
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    if not tasks:
        console.print("[yellow]No tasks found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("ID (short)", style="dim", width=10)
    table.add_column("Title")
    table.add_column("Status", width=12)
    table.add_column("Priority", width=10)
    table.add_column("Created", width=22)

    for t in tasks:
        table.add_row(
            t["id"][:8],
            t["title"],
            _status_markup(t["status"]),
            _priority_markup(t["priority"]),
            t["created_at"][:19].replace("T", " "),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# done
# ---------------------------------------------------------------------------


@cli.command("done")
@click.argument("partial_id")
@click.pass_context
def done_cmd(ctx: click.Context, partial_id: str) -> None:
    """Mark a task as done (identified by partial ID)."""
    store = _get_store(ctx.obj.get("data_file"))
    try:
        task = store.update_status(partial_id=partial_id, new_status="done")
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    console.print(
        f"[green]Task marked as done[/green] \u2014 id: [bold]{task['id'][:8]}[/bold]  "
        f"title: {task['title']}"
    )


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@cli.command("delete")
@click.argument("partial_id")
@click.option("--yes", "-y", is_flag=True, default=False, help="Skip confirmation.")
@click.pass_context
def delete_cmd(ctx: click.Context, partial_id: str, yes: bool) -> None:
    """Delete a task (identified by partial ID)."""
    store = _get_store(ctx.obj.get("data_file"))

    # We need to find the task first so we can show its title in the prompt
    try:
        tasks = store.list_tasks()
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    # Use the store's private helper via a quick lookup
    needle = partial_id.strip().lower()
    matches = [t for t in tasks if t["id"].lower().startswith(needle)]

    if len(matches) == 0:
        err_console.print(
            f"[red]Error:[/red] No task found matching id prefix '{partial_id}'"
        )
        sys.exit(1)
    if len(matches) > 1:
        ids = ", ".join(t["id"][:8] for t in matches)
        err_console.print(
            f"[red]Error:[/red] Ambiguous id prefix '{partial_id}' matches {len(matches)} tasks: {ids}"
        )
        sys.exit(1)

    task = matches[0]

    if not yes:
        confirmed = click.confirm(
            f"Delete task '{task['title']}' ({task['id'][:8]})?"
        )
        if not confirmed:
            console.print("[yellow]Aborted.[/yellow]")
            return

    try:
        store.delete_task(partial_id=partial_id)
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    console.print(
        f"[green]Task deleted[/green] \u2014 id: [bold]{task['id'][:8]}[/bold]  "
        f"title: {task['title']}"
    )


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@cli.command("stats")
@click.pass_context
def stats_cmd(ctx: click.Context) -> None:
    """Show task statistics."""
    store = _get_store(ctx.obj.get("data_file"))
    try:
        all_tasks = store.list_tasks()
    except ValueError as exc:
        err_console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    total = len(all_tasks)

    status_counts: dict[str, int] = {s: 0 for s in sorted(VALID_STATUSES)}
    priority_counts: dict[str, int] = {p: 0 for p in sorted(VALID_PRIORITIES)}

    for t in all_tasks:
        status_counts[t.get("status", "todo")] += 1
        priority_counts[t.get("priority", "medium")] += 1

    lines = [f"[bold]Total tasks:[/bold] {total}", ""]
    lines.append("[bold]By status:[/bold]")
    for s, count in sorted(status_counts.items()):
        lines.append(f"  {_status_markup(s)}: {count}")
    lines.append("")
    lines.append("[bold]By priority:[/bold]")
    for p, count in sorted(priority_counts.items()):
        lines.append(f"  {_priority_markup(p)}: {count}")

    panel = Panel("\n".join(lines), title="Task Stats", border_style="blue")
    console.print(panel)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cli()
