"""Command-line interface for the Task Tracker application.

This module defines the Click command group and all sub-commands that make up
the ``task-tracker`` CLI.  Rich is used throughout for polished terminal output
(tables, panels, styled text, progress spinners, etc.).

Commands
--------
add       – Create a new task.
list      – List tasks with optional filters.
show      – Display full details of a single task.
update    – Edit a task's title, priority, tags, or notes.
status    – Change the status of a task (todo / in_progress / done).
delete    – Permanently remove a task.
stats     – Show a summary of task counts by status and priority.
tags      – List all unique tags used across tasks.

Usage example::

    task-tracker add "Write unit tests" --priority high --tags dev,testing
    task-tracker list --status todo --priority high
    task-tracker status <id> done
    task-tracker delete <id>
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from task_tracker.task_store import (
    Task,
    TaskNotFoundError,
    TaskPriority,
    TaskStatus,
    TaskStore,
    TaskStoreError,
)

# ---------------------------------------------------------------------------
# Shared console instances
# ---------------------------------------------------------------------------

#: Standard output console (stdout).
console = Console()

#: Error console (stderr) — used for all error / warning messages.
err_console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------

_STATUS_STYLE: dict[TaskStatus, str] = {
    TaskStatus.TODO: "yellow",
    TaskStatus.IN_PROGRESS: "cyan",
    TaskStatus.DONE: "green",
}

_PRIORITY_STYLE: dict[TaskPriority, str] = {
    TaskPriority.HIGH: "bold red",
    TaskPriority.MEDIUM: "bold yellow",
    TaskPriority.LOW: "dim",
}

_STATUS_EMOJI: dict[TaskStatus, str] = {
    TaskStatus.TODO: "⏳",
    TaskStatus.IN_PROGRESS: "🔄",
    TaskStatus.DONE: "✅",
}

_PRIORITY_EMOJI: dict[TaskPriority, str] = {
    TaskPriority.HIGH: "🔴",
    TaskPriority.MEDIUM: "🟡",
    TaskPriority.LOW: "🟢",
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _short_id(task_id: str, length: int = 8) -> str:
    """Return the first *length* characters of a UUID for compact display."""
    return task_id[:length]


def _format_tags(tags: list[str]) -> str:
    """Return a comma-separated tag string, or a dash if there are no tags."""
    return ", ".join(tags) if tags else "—"


def _styled_status(status: TaskStatus) -> Text:
    """Return a Rich :class:`~rich.text.Text` object for *status*."""
    emoji = _STATUS_EMOJI[status]
    style = _STATUS_STYLE[status]
    return Text(f"{emoji} {status.label()}", style=style)


def _styled_priority(priority: TaskPriority) -> Text:
    """Return a Rich :class:`~rich.text.Text` object for *priority*."""
    emoji = _PRIORITY_EMOJI[priority]
    style = _PRIORITY_STYLE[priority]
    return Text(f"{emoji} {priority.label()}", style=style)


def _get_store(ctx: click.Context) -> TaskStore:
    """Retrieve the :class:`TaskStore` instance from the Click context."""
    return ctx.obj["store"]


def _abort(message: str, exit_code: int = 1) -> None:
    """Print *message* to stderr and exit with *exit_code*."""
    err_console.print(f"[bold red]Error:[/bold red] {message}")
    sys.exit(exit_code)


def _resolve_task_id(store: TaskStore, id_prefix: str) -> str:
    """Resolve a full task UUID from a prefix (or full ID).

    If *id_prefix* is already a full UUID that exists in the store it is
    returned as-is.  Otherwise all task IDs are searched for a unique prefix
    match.

    Args:
        store:     The :class:`TaskStore` to search.
        id_prefix: A full UUID or a unique prefix thereof.

    Returns:
        The full UUID string of the matching task.

    Raises:
        SystemExit: If no match or multiple matches are found.
    """
    # Fast path: exact match.
    if id_prefix in store:
        return id_prefix

    # Prefix search.
    matches = [tid for tid in (t.id for t in store) if tid.startswith(id_prefix)]

    if len(matches) == 1:
        return matches[0]

    if len(matches) == 0:
        _abort(
            f"No task found matching ID prefix [bold]{id_prefix!r}[/bold]. "
            "Use [bold]task-tracker list[/bold] to see available IDs."
        )

    # Multiple matches — show them so the user can be more specific.
    err_console.print(
        f"[bold red]Error:[/bold red] Ambiguous ID prefix [bold]{id_prefix!r}[/bold] "
        f"matches {len(matches)} tasks:"
    )
    for mid in matches:
        err_console.print(f"  • {mid}")
    sys.exit(1)

    # Unreachable — satisfies type checkers.
    return ""  # pragma: no cover


def _build_task_table(tasks: list[Task], *, show_notes: bool = False) -> Table:
    """Build a Rich :class:`~rich.table.Table` for a list of tasks.

    Args:
        tasks:      The tasks to display.
        show_notes: If ``True`` an extra *Notes* column is included.

    Returns:
        A configured :class:`~rich.table.Table` ready to be printed.
    """
    table = Table(
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        expand=False,
        highlight=True,
    )

    table.add_column("ID", style="dim", no_wrap=True, min_width=8)
    table.add_column("Title", min_width=20)
    table.add_column("Status", no_wrap=True)
    table.add_column("Priority", no_wrap=True)
    table.add_column("Tags")
    table.add_column("Updated", no_wrap=True, style="dim")

    if show_notes:
        table.add_column("Notes")

    for task in tasks:
        updated = task.updated_at.strftime("%Y-%m-%d %H:%M")
        row: list = [
            _short_id(task.id),
            task.title,
            _styled_status(task.status),
            _styled_priority(task.priority),
            _format_tags(task.tags),
            updated,
        ]
        if show_notes:
            row.append(task.notes or "—")
        table.add_row(*row)

    return table


# ---------------------------------------------------------------------------
# Click parameter types / callbacks
# ---------------------------------------------------------------------------


class _CommaSeparatedList(click.ParamType):
    """A Click parameter type that splits a comma-separated string into a list."""

    name = "TEXT"

    def convert(
        self,
        value: str,
        param: Optional[click.Parameter],
        ctx: Optional[click.Context],
    ) -> list[str]:
        if not value:
            return []
        return [v.strip() for v in value.split(",") if v.strip()]


_TAGS_TYPE = _CommaSeparatedList()

# ---------------------------------------------------------------------------
# Root command group
# ---------------------------------------------------------------------------


@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    invoke_without_command=True,
)
@click.option(
    "--store",
    "store_path",
    default=None,
    metavar="PATH",
    envvar="TASK_TRACKER_STORE",
    help=(
        "Path to the JSON task store file.  "
        "Defaults to ~/.task_tracker/tasks.json.  "
        "Can also be set via the TASK_TRACKER_STORE environment variable."
    ),
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.version_option(package_name="task-tracker", prog_name="task-tracker")
@click.pass_context
def cli(ctx: click.Context, store_path: Optional[Path]) -> None:
    """📋 Task Tracker — a simple CLI task manager powered by Click and Rich.

    Run [bold]task-tracker COMMAND --help[/bold] for help on a specific command.
    """
    ctx.ensure_object(dict)
    ctx.obj["store"] = TaskStore(store_path)

    # If invoked with no sub-command, show help.
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


@cli.command("add")
@click.argument("title")
@click.option(
    "-p",
    "--priority",
    default="medium",
    show_default=True,
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    help="Priority level of the task.",
)
@click.option(
    "-t",
    "--tags",
    default="",
    metavar="TAG[,TAG…]",
    type=_TAGS_TYPE,
    help="Comma-separated list of tags to attach to the task.",
)
@click.option(
    "-n",
    "--notes",
    default="",
    metavar="TEXT",
    help="Optional free-form notes or description for the task.",
)
@click.pass_context
def cmd_add(
    ctx: click.Context,
    title: str,
    priority: str,
    tags: list[str],
    notes: str,
) -> None:
    """Add a new task.

    TITLE is the short description of the task.  Wrap it in quotes if it
    contains spaces.

    \b
    Examples:
      task-tracker add "Buy groceries"
      task-tracker add "Write unit tests" --priority high --tags dev,testing
      task-tracker add "Read book" -p low -n "Start with chapter 3"
    """
    store = _get_store(ctx)
    try:
        task = store.add(title, priority=priority, tags=tags or [], notes=notes)
    except ValueError as exc:
        _abort(str(exc))
    except TaskStoreError as exc:
        _abort(str(exc))

    console.print(
        Panel(
            f"[bold green]✅ Task created![/bold green]\n\n"
            f"  [bold]ID:[/bold]       {_short_id(task.id)} ([dim]{task.id}[/dim])\n"
            f"  [bold]Title:[/bold]    {task.title}\n"
            f"  [bold]Priority:[/bold] {_styled_priority(task.priority)}\n"
            f"  [bold]Status:[/bold]   {_styled_status(task.status)}\n"
            f"  [bold]Tags:[/bold]     {_format_tags(task.tags)}\n"
            + (f"  [bold]Notes:[/bold]    {task.notes}\n" if task.notes else ""),
            title="[bold]New Task[/bold]",
            border_style="green",
            expand=False,
        )
    )


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


@cli.command("list")
@click.option(
    "-s",
    "--status",
    default=None,
    type=click.Choice(["todo", "in_progress", "done"], case_sensitive=False),
    help="Filter by task status.",
)
@click.option(
    "-p",
    "--priority",
    default=None,
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    help="Filter by priority level.",
)
@click.option(
    "-t",
    "--tags",
    default="",
    metavar="TAG[,TAG…]",
    type=_TAGS_TYPE,
    help="Filter to tasks that have ALL of the given tags.",
)
@click.option(
    "-q",
    "--search",
    default=None,
    metavar="TEXT",
    help="Case-insensitive substring search across title and notes.",
)
@click.option(
    "--notes",
    "show_notes",
    is_flag=True,
    default=False,
    help="Include a Notes column in the output.",
)
@click.pass_context
def cmd_list(
    ctx: click.Context,
    status: Optional[str],
    priority: Optional[str],
    tags: list[str],
    search: Optional[str],
    show_notes: bool,
) -> None:
    """List tasks, with optional filters.

    \b
    Examples:
      task-tracker list
      task-tracker list --status todo
      task-tracker list --priority high --status in_progress
      task-tracker list --tags dev,testing
      task-tracker list --search "unit test"
    """
    store = _get_store(ctx)
    try:
        tasks = store.list_tasks(
            status=status,
            priority=priority,
            tags=tags or None,
            search=search,
        )
    except (ValueError, TaskStoreError) as exc:
        _abort(str(exc))

    if not tasks:
        console.print("[dim]No tasks found matching the given filters.[/dim]")
        return

    # Build filter description for the panel title.
    filters: list[str] = []
    if status:
        filters.append(f"status={status}")
    if priority:
        filters.append(f"priority={priority}")
    if tags:
        filters.append(f"tags={','.join(tags)}")
    if search:
        filters.append(f"search={search!r}")
    filter_str = "  [dim](" + ", ".join(filters) + ")[/dim]" if filters else ""

    table = _build_task_table(tasks, show_notes=show_notes)
    console.print()
    console.print(
        f"[bold magenta]Tasks[/bold magenta]{filter_str}  "
        f"[dim]({len(tasks)} task{'s' if len(tasks) != 1 else ''})[/dim]"
    )
    console.print(table)


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


@cli.command("show")
@click.argument("task_id")
@click.pass_context
def cmd_show(ctx: click.Context, task_id: str) -> None:
    """Show full details of a single task.

    TASK_ID may be a full UUID or a unique prefix (at least 4 characters).

    \b
    Examples:
      task-tracker show a1b2c3d4
      task-tracker show a1b2c3d4-e5f6-7890-abcd-ef1234567890
    """
    store = _get_store(ctx)
    try:
        full_id = _resolve_task_id(store, task_id)
        task = store.get(full_id)
    except TaskNotFoundError:
        _abort(f"No task found with ID [bold]{task_id!r}[/bold].")
    except TaskStoreError as exc:
        _abort(str(exc))

    created = task.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    updated = task.updated_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    detail = (
        f"  [bold]ID:[/bold]         {task.id}\n"
        f"  [bold]Title:[/bold]      {task.title}\n"
        f"  [bold]Status:[/bold]     {_styled_status(task.status)}\n"
        f"  [bold]Priority:[/bold]   {_styled_priority(task.priority)}\n"
        f"  [bold]Tags:[/bold]       {_format_tags(task.tags)}\n"
        f"  [bold]Created:[/bold]    {created}\n"
        f"  [bold]Updated:[/bold]    {updated}\n"
    )
    if task.notes:
        detail += f"\n  [bold]Notes:[/bold]\n    {task.notes}\n"

    console.print(
        Panel(
            detail,
            title=f"[bold]Task Details[/bold]",
            border_style="blue",
            expand=False,
        )
    )


# ---------------------------------------------------------------------------
# update
# ---------------------------------------------------------------------------


@cli.command("update")
@click.argument("task_id")
@click.option(
    "--title",
    default=None,
    metavar="TEXT",
    help="New title for the task.",
)
@click.option(
    "-p",
    "--priority",
    default=None,
    type=click.Choice(["low", "medium", "high"], case_sensitive=False),
    help="New priority level.",
)
@click.option(
    "-t",
    "--tags",
    default=None,
    metavar="TAG[,TAG…]",
    type=_TAGS_TYPE,
    help="New comma-separated list of tags (replaces existing tags).",
)
@click.option(
    "-n",
    "--notes",
    default=None,
    metavar="TEXT",
    help="New notes / description (replaces existing notes).",
)
@click.pass_context
def cmd_update(
    ctx: click.Context,
    task_id: str,
    title: Optional[str],
    priority: Optional[str],
    tags: Optional[list[str]],
    notes: Optional[str],
) -> None:
    """Update one or more fields of an existing task.

    TASK_ID may be a full UUID or a unique prefix.  At least one option must
    be provided.

    \b
    Examples:
      task-tracker update a1b2c3d4 --title "New title"
      task-tracker update a1b2c3d4 --priority low
      task-tracker update a1b2c3d4 --tags backend,api --notes "See ticket #42"
    """
    if title is None and priority is None and tags is None and notes is None:
        _abort(
            "Nothing to update.  Provide at least one of: "
            "--title, --priority, --tags, --notes."
        )

    store = _get_store(ctx)
    try:
        full_id = _resolve_task_id(store, task_id)
        task = store.update(
            full_id,
            title=title,
            priority=priority,
            tags=tags,
            notes=notes,
        )
    except TaskNotFoundError:
        _abort(f"No task found with ID [bold]{task_id!r}[/bold].")
    except (ValueError, TaskStoreError) as exc:
        _abort(str(exc))

    console.print(
        Panel(
            f"[bold cyan]✏️  Task updated![/bold cyan]\n\n"
            f"  [bold]ID:[/bold]       {_short_id(task.id)} ([dim]{task.id}[/dim])\n"
            f"  [bold]Title:[/bold]    {task.title}\n"
            f"  [bold]Priority:[/bold] {_styled_priority(task.priority)}\n"
            f"  [bold]Status:[/bold]   {_styled_status(task.status)}\n"
            f"  [bold]Tags:[/bold]     {_format_tags(task.tags)}\n"
            + (f"  [bold]Notes:[/bold]    {task.notes}\n" if task.notes else ""),
            title="[bold]Updated Task[/bold]",
            border_style="cyan",
            expand=False,
        )
    )


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@cli.command("status")
@click.argument("task_id")
@click.argument(
    "new_status",
    type=click.Choice(["todo", "in_progress", "done"], case_sensitive=False),
)
@click.pass_context
def cmd_status(ctx: click.Context, task_id: str, new_status: str) -> None:
    """Change the status of a task.

    TASK_ID may be a full UUID or a unique prefix.
    NEW_STATUS must be one of: todo, in_progress, done.

    \b
    Examples:
      task-tracker status a1b2c3d4 in_progress
      task-tracker status a1b2c3d4 done
    """
    store = _get_store(ctx)
    try:
        full_id = _resolve_task_id(store, task_id)
        task = store.update_status(full_id, new_status)
    except TaskNotFoundError:
        _abort(f"No task found with ID [bold]{task_id!r}[/bold].")
    except (ValueError, TaskStoreError) as exc:
        _abort(str(exc))

    console.print(
        f"[bold green]✅ Status updated:[/bold green] "
        f"[bold]{task.title}[/bold] → {_styled_status(task.status)}"
    )


# ---------------------------------------------------------------------------
# delete
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
@click.pass_context
def cmd_delete(ctx: click.Context, task_id: str, yes: bool) -> None:
    """Permanently delete a task.

    TASK_ID may be a full UUID or a unique prefix.  You will be asked to
    confirm unless --yes is supplied.

    \b
    Examples:
      task-tracker delete a1b2c3d4
      task-tracker delete a1b2c3d4 --yes
    """
    store = _get_store(ctx)
    try:
        full_id = _resolve_task_id(store, task_id)
        task = store.get(full_id)
    except TaskNotFoundError:
        _abort(f"No task found with ID [bold]{task_id!r}[/bold].")
    except TaskStoreError as exc:
        _abort(str(exc))

    if not yes:
        console.print(
            f"[bold yellow]⚠️  About to delete:[/bold yellow] "
            f"[bold]{task.title}[/bold] ([dim]{task.id}[/dim])"
        )
        confirmed = click.confirm("Are you sure?", default=False)
        if not confirmed:
            console.print("[dim]Deletion cancelled.[/dim]")
            return

    try:
        store.delete(task.id)
    except (TaskNotFoundError, TaskStoreError) as exc:
        _abort(str(exc))

    console.print(
        f"[bold red]🗑️  Task deleted:[/bold red] [bold]{task.title}[/bold] "
        f"([dim]{task.id}[/dim])"
    )


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@cli.command("stats")
@click.pass_context
def cmd_stats(ctx: click.Context) -> None:
    """Show a summary of task counts grouped by status and priority.

    \b
    Example:
      task-tracker stats
    """
    store = _get_store(ctx)
    try:
        all_tasks = store.list_tasks()
    except TaskStoreError as exc:
        _abort(str(exc))

    total = len(all_tasks)

    if total == 0:
        console.print("[dim]No tasks in the store yet.[/dim]")
        return

    # --- Status breakdown ---
    status_table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )
    status_table.add_column("Status", no_wrap=True)
    status_table.add_column("Count", justify="right")
    status_table.add_column("Bar", min_width=20)

    for status in TaskStatus:
        count = sum(1 for t in all_tasks if t.status == status)
        bar_len = int((count / total) * 20) if total else 0
        bar = "█" * bar_len + "░" * (20 - bar_len)
        status_table.add_row(
            _styled_status(status),
            str(count),
            Text(bar, style=_STATUS_STYLE[status]),
        )

    # --- Priority breakdown ---
    priority_table = Table(
        box=box.SIMPLE_HEAVY,
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )
    priority_table.add_column("Priority", no_wrap=True)
    priority_table.add_column("Count", justify="right")
    priority_table.add_column("Bar", min_width=20)

    for priority in (TaskPriority.HIGH, TaskPriority.MEDIUM, TaskPriority.LOW):
        count = sum(1 for t in all_tasks if t.priority == priority)
        bar_len = int((count / total) * 20) if total else 0
        bar = "█" * bar_len + "░" * (20 - bar_len)
        priority_table.add_row(
            _styled_priority(priority),
            str(count),
            Text(bar, style=_PRIORITY_STYLE[priority]),
        )

    console.print()
    console.print(
        Panel(
            f"[bold]Total tasks:[/bold] {total}",
            title="[bold]📊 Task Statistics[/bold]",
            border_style="magenta",
            expand=False,
        )
    )
    console.print()
    console.print("[bold]By Status[/bold]")
    console.print(status_table)
    console.print("[bold]By Priority[/bold]")
    console.print(priority_table)


# ---------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------


@cli.command("tags")
@click.pass_context
def cmd_tags(ctx: click.Context) -> None:
    """List all unique tags used across all tasks.

    \b
    Example:
      task-tracker tags
    """
    store = _get_store(ctx)
    try:
        all_tags = store.all_tags()
    except TaskStoreError as exc:
        _abort(str(exc))

    if not all_tags:
        console.print("[dim]No tags found.[/dim]")
        return

    table = Table(
        box=box.SIMPLE,
        show_header=True,
        header_style="bold magenta",
        expand=False,
    )
    table.add_column("Tag")
    table.add_column("Task Count", justify="right")

    try:
        all_tasks = store.list_tasks()
    except TaskStoreError as exc:
        _abort(str(exc))

    for tag in all_tags:
        count = sum(1 for t in all_tasks if tag in t.tags)
        table.add_row(f"[bold cyan]#{tag}[/bold cyan]", str(count))

    console.print()
    console.print(
        f"[bold magenta]Tags[/bold magenta]  "
        f"[dim]({len(all_tags)} unique tag{'s' if len(all_tags) != 1 else ''})[/dim]"
    )
    console.print(table)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for the ``task-tracker`` console script."""
    cli(obj={})


if __name__ == "__main__":
    main()
