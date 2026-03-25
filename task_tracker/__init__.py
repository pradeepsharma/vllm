"""
task_tracker — A self-contained Python CLI task tracker built with Click and Rich.

This package provides a simple command-line interface for managing tasks with
persistent JSON storage. It exposes five subcommands via the ``task`` entry point:

- ``task add``    — Create a new task with a title and optional priority
- ``task list``   — List tasks, optionally filtered by status or priority
- ``task done``   — Mark a task as done (or any other status)
- ``task delete`` — Remove a task permanently
- ``task stats``  — Display a summary panel with task counts by status

Modules:
    task_store: Data layer — ``Task`` dataclass, enums, and ``TaskStore`` class.
    cli:        Presentation layer — Click command group and Rich-rendered output.
"""
