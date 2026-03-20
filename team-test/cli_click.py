"""Click-based command-line interface for the Task Management CLI Tool.

Provides a ``taskman`` CLI built with `Click <https://click.palletsprojects.com/>`_
and `Rich <https://rich.readthedocs.io/>`_ for beautiful terminal output.

The data layer is :class:`~task_store.TaskStore`, which persists tasks to a
local JSON file (default: ``~/.taskman/tasks.json``).

Usage examples::

    # Add tasks
    taskman add "Write docs"
    taskman add "Fix bug" --priority high
    taskman add "Deploy release" --priority critical

    # List tasks
    taskman list
    taskman list --status todo
    taskman list --status in_progress
    taskman list --priority high

    # Transition task status
    taskman start <TASK_ID>
    taskman done <TASK_ID>

    # Delete a task
    taskman delete <TASK_ID>

    # Show statistics
    taskman stats

    # Global options
    taskman --data /path/to/tasks.json list
"""
from __future__ import annotations

import os
import sys
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup: allow running ``python cli_click.py`` directly from the
# team-test directory or from the workspace root.
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import click  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich import box  # noqa: E402
from rich.text import Text  # noqa: E402

from task_store import NotFoundError, TaskStore, ValidationError  # noqa: E402
from models import Priority, Status  # noqa: E402

# ---------------------------------------------------------------------------
# Rich console instances
# ---------------------------------------------------------------------------

# stdout console for normal output
console = Console()

# stderr console for error messages
err_console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Shared Click context object
# ---------------------------------------------------------------------------

_DEFAULT_TASKS_JSON = os.path.join(
    os.path.expanduser("~"), ".taskman", "tasks.json"
)


class _Context:
    """Holds shared state passed through Click's context object."""

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path

    def store(self) -> TaskStore:
        """Return a :class:`~task_store.TaskStore` backed by :attr:`data_path`."""
        return TaskStore(path=self.data_path)


pass_ctx = click.make_pass_decorator(_Context)

# ---------------------------------------------------------------------------
# Colour / style helpers
# ---------------------------------------------------------------------------

_STATUS_STYLES: dict[str, str] = {
    "todo": "yellow",
    "in_progress": "cyan",
    "done": "green",
}

_PRIORITY_STYLES: dict[str, str] = {
    "low": "dim",
    "medium": "white",
    "high": "yellow",
    "critical": "bold red",
}


def _status_text(status_value: str) -> Text:
    """Return a styled :class:`~rich.text.Text` for a status value."""
    style = _STATUS_STYLES.get(status_value, "white")
    label = status_value.upper().replace("_", " ")
    return Text(f"[{label}]", style=style)


def _priority_text(priority_value: str) -> Text:
    """Return a styled :class:`~rich.text.Text` for a priority value."""
    style = _PRIORITY_STYLES.get(priority_value, "white")
    return Text(priority_value, style=style)


def _overdue_badge() -> Text:
    """Return a red OVERDUE badge."""
    return Text(" ⚠ OVERDUE", style="bold red")


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _render_task_table(tasks, *, verbose: bool = False) -> Table:
    """Build a Rich :class:`~rich.table.Table` for a list of tasks.

    Parameters
    ----------
    tasks:
        Iterable of :class:`~models.Task` objects to display.
    verbose:
        When ``True``, include the full task ID, created/updated timestamps,
        and description columns.

    Returns
    -------
    Table
        A fully populated Rich table ready to be printed.
    """
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=False,
        highlight=True,
    )

    if verbose:
        table.add_column("ID", style="dim", no_wrap=True, min_width=8)
    else:
        table.add_column("ID (short)", style="dim", no_wrap=True, min_width=8)

    table.add_column("Title", min_width=20)
    table.add_column("Status", justify="center", min_width=12)
    table.add_column("Priority", justify="center", min_width=10)

    if verbose:
        table.add_column("Created", style="dim", min_width=20)
        table.add_column("Updated", style="dim", min_width=20)

    for task in tasks:
        status_cell = _status_text(task.status.value)
        priority_cell = _priority_text(task.priority.value)

        title_text = Text(task.title)
        if task.is_overdue():
            title_text.append_text(_overdue_badge())

        if verbose:
            table.add_row(
                task.id,
                title_text,
                status_cell,
                priority_cell,
                task.created_at,
                task.updated_at,
            )
        else:
            # Show only the first 8 characters of the UUID for brevity
            table.add_row(
                task.id[:8] + "…",
                title_text,
                status_cell,
                priority_cell,
            )

    return table


def _render_task_panel(task) -> Panel:
    """Build a Rich :class:`~rich.panel.Panel` showing full task details.

    Parameters
    ----------
    task:
        A :class:`~models.Task` instance to display.

    Returns
    -------
    Panel
        A Rich panel with all task fields.
    """
    status_text = _status_text(task.status.value)
    priority_text = _priority_text(task.priority.value)

    lines = Text()
    lines.append("ID:       ", style="bold")
    lines.append(task.id + "\n", style="dim")

    lines.append("Status:   ", style="bold")
    lines.append_text(status_text)
    lines.append("\n")

    lines.append("Priority: ", style="bold")
    lines.append_text(priority_text)
    lines.append("\n")

    lines.append("Created:  ", style="bold")
    lines.append(task.created_at + "\n", style="dim")

    lines.append("Updated:  ", style="bold")
    lines.append(task.updated_at + "\n", style="dim")

    if task.is_overdue():
        lines.append_text(_overdue_badge())
        lines.append("\n")

    title_str = f"📋 {task.title}"
    return Panel(lines, title=title_str, border_style="blue", expand=False)


def _render_stats_panel(stats: dict) -> Panel:
    """Build a Rich :class:`~rich.panel.Panel` for the stats summary.

    Parameters
    ----------
    stats:
        The dict returned by :meth:`~task_store.TaskStore.stats`.

    Returns
    -------
    Panel
        A Rich panel with task statistics.
    """
    by_status = stats.get("tasks_by_status", {})
    by_priority = stats.get("tasks_by_priority", {})
    overdue = stats.get("overdue_tasks", 0)

    lines = Text()

    # Totals
    lines.append("Total tasks:    ", style="bold")
    lines.append(str(stats.get("total_tasks", 0)) + "\n")

    lines.append("Total projects: ", style="bold")
    lines.append(str(stats.get("total_projects", 0)) + "\n\n")

    # By status
    lines.append("By status:\n", style="bold underline")
    for status_val, count in by_status.items():
        style = _STATUS_STYLES.get(status_val, "white")
        label = status_val.upper().replace("_", " ")
        lines.append(f"  {label:<14}", style=style)
        lines.append(f"{count}\n")

    lines.append("\n")

    # By priority
    lines.append("By priority:\n", style="bold underline")
    for prio_val, count in by_priority.items():
        style = _PRIORITY_STYLES.get(prio_val, "white")
        lines.append(f"  {prio_val.capitalize():<14}", style=style)
        lines.append(f"{count}\n")

    # Overdue
    if overdue > 0:
        lines.append("\n")
        lines.append("Overdue tasks:  ", style="bold")
        lines.append(str(overdue), style="bold red")

    return Panel(lines, title="📊 Task Statistics", border_style="magenta", expand=False)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.option(
    "--data",
    "data_path",
    default=_DEFAULT_TASKS_JSON,
    show_default=True,
    metavar="PATH",
    envvar="TASKMAN_DATA",
    help=(
        "Path to the JSON tasks file.  "
        "Can also be set via the TASKMAN_DATA environment variable."
    ),
)
@click.version_option(version="1.0.0", prog_name="taskman")
@click.pass_context
def cli(ctx: click.Context, data_path: str) -> None:
    """taskman — a simple task management CLI.

    Manage your tasks from the command line.  Tasks are persisted to a local
    JSON file (default: ~/.taskman/tasks.json).

    \b
    Quick start:
      taskman add "Write docs" --priority high
      taskman list
      taskman done <TASK_ID>
    """
    ctx.ensure_object(dict)
    ctx.obj = _Context(data_path=data_path)


# ---------------------------------------------------------------------------
# ``add`` command
# ---------------------------------------------------------------------------


@cli.command("add")
@click.argument("title")
@click.option(
    "-p",
    "--priority",
    type=click.Choice(
        ["low", "medium", "high", "critical"],
        case_sensitive=False,
    ),
    default="medium",
    show_default=True,
    help="Task priority level.",
)
@pass_ctx
def cmd_add(ctx: _Context, title: str, priority: str) -> None:
    """Add a new task.

    TITLE is the short human-readable summary of the task.

    \b
    Examples:
      taskman add "Write unit tests"
      taskman add "Deploy to production" --priority critical
    """
    store = ctx.store()
    try:
        task = store.add_task(title, priority=priority.lower())
    except ValidationError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc
    except ValueError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc

    console.print(f"[green]✓ Task created:[/green] [dim]{task.id}[/dim]")
    console.print(_render_task_panel(task))


# ---------------------------------------------------------------------------
# ``list`` command
# ---------------------------------------------------------------------------


@cli.command("list")
@click.option(
    "--status",
    type=click.Choice(
        ["todo", "in_progress", "done"],
        case_sensitive=False,
    ),
    default=None,
    help="Filter tasks by status.",
)
@click.option(
    "--priority",
    type=click.Choice(
        ["low", "medium", "high", "critical"],
        case_sensitive=False,
    ),
    default=None,
    help="Filter tasks by priority.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Show full task IDs and timestamps.",
)
@pass_ctx
def cmd_list(
    ctx: _Context,
    status: Optional[str],
    priority: Optional[str],
    verbose: bool,
) -> None:
    """List tasks, optionally filtered by status or priority.

    \b
    Examples:
      taskman list
      taskman list --status todo
      taskman list --priority high
      taskman list --status in_progress --priority critical
      taskman list --verbose
    """
    store = ctx.store()
    try:
        tasks = store.list_tasks(
            status=status,
            priority=priority,
        )
    except ValueError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc

    if not tasks:
        console.print("[dim]No tasks found.[/dim]")
        return

    # Build a descriptive heading using Rich markup so colour tags render.
    filters: list[str] = []
    if status:
        filters.append(f"status=[cyan]{status}[/cyan]")
    if priority:
        filters.append(f"priority=[yellow]{priority}[/yellow]")
    filter_suffix = "  (" + ", ".join(filters) + ")" if filters else ""

    console.print(f"[bold]{len(tasks)}[/bold] task(s){filter_suffix}")
    console.print(_render_task_table(tasks, verbose=verbose))


# ---------------------------------------------------------------------------
# ``done`` command
# ---------------------------------------------------------------------------


@cli.command("done")
@click.argument("task_id")
@pass_ctx
def cmd_done(ctx: _Context, task_id: str) -> None:
    """Mark a task as DONE.

    TASK_ID can be the full UUID or a unique prefix (at least 1 character).

    \b
    Examples:
      taskman done abc12345
      taskman done abc12345-6789-...
    """
    store = ctx.store()
    try:
        task = store.update_status(task_id, "done")
    except NotFoundError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc
    except ValueError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc

    console.print(
        f"[green]✓ Task marked as DONE:[/green] [bold]{task.title}[/bold]"
    )


# ---------------------------------------------------------------------------
# ``start`` command
# ---------------------------------------------------------------------------


@cli.command("start")
@click.argument("task_id")
@pass_ctx
def cmd_start(ctx: _Context, task_id: str) -> None:
    """Mark a task as IN PROGRESS.

    TASK_ID can be the full UUID or a unique prefix (at least 1 character).

    \b
    Examples:
      taskman start abc12345
    """
    store = ctx.store()
    try:
        task = store.update_status(task_id, "in_progress")
    except NotFoundError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc
    except ValueError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc

    console.print(
        f"[cyan]✓ Task marked as IN PROGRESS:[/cyan] [bold]{task.title}[/bold]"
    )


# ---------------------------------------------------------------------------
# ``reopen`` command
# ---------------------------------------------------------------------------


@cli.command("reopen")
@click.argument("task_id")
@pass_ctx
def cmd_reopen(ctx: _Context, task_id: str) -> None:
    """Reset a task back to TODO.

    TASK_ID can be the full UUID or a unique prefix (at least 1 character).

    \b
    Examples:
      taskman reopen abc12345
    """
    store = ctx.store()
    try:
        task = store.update_status(task_id, "todo")
    except NotFoundError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc
    except ValueError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc

    console.print(
        f"[yellow]✓ Task reopened (TODO):[/yellow] [bold]{task.title}[/bold]"
    )


# ---------------------------------------------------------------------------
# ``delete`` command
# ---------------------------------------------------------------------------


@cli.command("delete")
@click.argument("task_id")
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    default=False,
    help="Skip the confirmation prompt.",
)
@pass_ctx
def cmd_delete(ctx: _Context, task_id: str, yes: bool) -> None:
    """Delete a task permanently.

    TASK_ID can be the full UUID or a unique prefix (at least 1 character).

    \b
    Examples:
      taskman delete abc12345
      taskman delete abc12345 --yes
    """
    store = ctx.store()

    # Resolve the task first so we can show its title in the confirmation
    # prompt and in the success message.
    try:
        tasks = store.list_tasks()
        matches = [t for t in tasks if t.id.startswith(task_id)]
    except Exception as exc:  # pragma: no cover
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc

    if len(matches) == 0:
        err_console.print(
            f"[bold red]Error:[/bold red] No task found matching id prefix '{task_id}'."
        )
        raise SystemExit(1)
    if len(matches) > 1:
        err_console.print(
            f"[bold red]Error:[/bold red] Ambiguous id prefix '{task_id}' "
            f"matches {len(matches)} tasks. Please provide more characters."
        )
        raise SystemExit(1)

    task = matches[0]

    if not yes:
        click.confirm(
            f"Delete task '{task.title}' ({task.id[:8]}…)?",
            abort=True,
        )

    try:
        store.delete_task(task_id)
    except NotFoundError as exc:
        err_console.print(f"[bold red]Error:[/bold red] {exc}")
        raise SystemExit(1) from exc

    console.print(f"[green]✓ Task deleted:[/green] [bold]{task.title}[/bold]")


# ---------------------------------------------------------------------------
# ``stats`` command
# ---------------------------------------------------------------------------


@cli.command("stats")
@pass_ctx
def cmd_stats(ctx: _Context) -> None:
    """Show task statistics.

    Displays a summary of all tasks broken down by status and priority,
    plus a count of overdue tasks.

    \b
    Examples:
      taskman stats
    """
    store = ctx.store()
    stats = store.stats()
    console.print(_render_stats_panel(stats))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ``taskman`` CLI.

    Invokes the Click application.  Click handles argument parsing, help
    generation, and error reporting.  The function does not return a numeric
    exit code — Click calls :func:`sys.exit` internally when needed.
    """
    cli(prog_name="taskman")


if __name__ == "__main__":
    main()
