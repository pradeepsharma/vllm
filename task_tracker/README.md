# Task Tracker CLI

A lightweight, persistent command-line task tracker built with **Click** and **Rich**.  
Tasks are stored in a plain JSON file — no database, no server, no setup beyond `pip install`.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [add](#add)
  - [list](#list)
  - [done](#done)
  - [delete](#delete)
  - [stats](#stats)
- [Partial ID Matching](#partial-id-matching)
- [Task Schema](#task-schema)
- [Development](#development)
  - [Running Tests](#running-tests)
  - [Coverage Report](#coverage-report)
- [Project Structure](#project-structure)

---

## Features

- **Add tasks** with a title and optional priority (`low` / `medium` / `high`)
- **List tasks** in a Rich-formatted table, with optional filters by status or priority
- **Mark tasks done** using a short prefix of the task UUID — no need to type the full ID
- **Delete tasks** with a confirmation prompt to prevent accidental removal
- **View statistics** — total count, breakdown by status, and breakdown by priority
- **Persistent storage** — tasks survive between sessions via a local JSON file
- **Configurable storage path** via the `TASK_TRACKER_FILE` environment variable

---

## Requirements

- Python **3.10** or later
- [Click](https://click.palletsprojects.com/) >= 8.0
- [Rich](https://rich.readthedocs.io/) >= 13.0

---

## Installation

### From source (recommended for development)

```bash
# Clone or download the project, then from the task_tracker/ directory:
pip install -e .
```

This registers the `task-tracker` command on your `PATH` via the entry point defined in `pyproject.toml`.

### Runtime dependencies only

```bash
pip install -r requirements.txt
```

---

## Configuration

By default, tasks are stored at:

```
~/.task_tracker/tasks.json
```

The parent directory is created automatically on first use.

To use a custom location, set the `TASK_TRACKER_FILE` environment variable:

```bash
export TASK_TRACKER_FILE=/path/to/my/tasks.json
task-tracker list
```

You can also place this in a `.env` file and load it with a tool such as
[`direnv`](https://direnv.net/) or [`python-dotenv`](https://pypi.org/project/python-dotenv/).
A ready-made template is provided at `.env.example`.

---

## Usage

Run `task-tracker --help` at any time to see available commands:

```
Usage: task-tracker [OPTIONS] COMMAND [ARGS]...

  A simple CLI task tracker with persistent JSON storage.

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  add     Add a new task.
  delete  Delete a task by partial ID (prompts for confirmation).
  done    Mark a task as done by partial ID.
  list    List tasks, optionally filtered by status and/or priority.
  stats   Show aggregate task statistics.
```

---

### add

Add a new task.

```bash
task-tracker add --title "Task description" [--priority low|medium|high]
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--title` | ✅ Yes | — | Task title / description |
| `--priority` | No | `medium` | Priority level: `low`, `medium`, or `high` |

**Examples:**

```bash
# Add a high-priority task
task-tracker add --title "Fix critical bug" --priority high

# Add a task with default (medium) priority
task-tracker add --title "Review pull request"

# Add a low-priority task
task-tracker add --title "Update README" --priority low
```

**Output:**

```
✅ Added task 3f2a1b8c...
```

---

### list

List tasks in a formatted table.

```bash
task-tracker list [--status todo|in_progress|done] [--priority low|medium|high]
```

| Option | Required | Default | Description |
|--------|----------|---------|-------------|
| `--status` | No | *(all)* | Filter by status: `todo`, `in_progress`, or `done` |
| `--priority` | No | *(all)* | Filter by priority: `low`, `medium`, or `high` |

**Examples:**

```bash
# List all tasks
task-tracker list

# List only tasks that are still to do
task-tracker list --status todo

# List only completed tasks
task-tracker list --status done

# List all high-priority tasks
task-tracker list --priority high

# Combine filters: high-priority tasks not yet started
task-tracker list --status todo --priority high
```

**Output:**

```
 ID         Title                  Status    Priority   Created
 ─────────  ─────────────────────  ────────  ─────────  ───────────────────
 3f2a1b8c   Fix critical bug       todo      high       2026-03-24 12:00:00
 a9c4e7f1   Review pull request    todo      medium     2026-03-24 12:01:00
```

---

### done

Mark a task as **done** using a partial task ID.

```bash
task-tracker done <partial-id>
```

| Argument | Description |
|----------|-------------|
| `partial-id` | A prefix or substring of the task's UUID (see [Partial ID Matching](#partial-id-matching)) |

**Examples:**

```bash
# Mark a task done using the first 8 characters of its ID
task-tracker done 3f2a1b8c

# Use more characters if needed to disambiguate
task-tracker done 3f2a1b8cd4e5
```

**Output:**

```
✅ Marked done: 3f2a1b8c — Fix critical bug
```

---

### delete

Delete a task permanently. A confirmation prompt is shown before deletion.

```bash
task-tracker delete <partial-id>
```

| Argument | Description |
|----------|-------------|
| `partial-id` | A prefix or substring of the task's UUID (see [Partial ID Matching](#partial-id-matching)) |

**Examples:**

```bash
# Delete a task (will prompt for confirmation)
task-tracker delete 3f2a1b8c
```

**Output:**

```
Delete task 'Fix critical bug' (3f2a1b8c)? This cannot be undone. [y/N]: y
🗑 Deleted: 3f2a1b8c — Fix critical bug
```

Press **Enter** or type **N** to cancel without deleting.

---

### stats

Display aggregate statistics about all tasks.

```bash
task-tracker stats
```

No options or arguments.

**Output:**

```
╭─ Task Statistics ──────────────────╮
│ Total tasks: 5                     │
│                                    │
│ By status:                         │
│   done         2                   │
│   in_progress  1                   │
│   todo         2                   │
│                                    │
│ By priority:                       │
│   high         1                   │
│   low          2                   │
│   medium       2                   │
╰────────────────────────────────────╯
```

---

## Partial ID Matching

To avoid typing full 32-character UUIDs, `done` and `delete` accept a **partial ID** — a prefix or substring of the task's UUID hex string.

Matching is performed in three stages:

1. **Exact match** — the input equals the full UUID.
2. **Prefix match** — the UUID starts with the input string.
3. **Substring match** — the input string appears anywhere in the UUID.

If exactly one task matches at any stage, it is selected.  
If **multiple** tasks match, an `AmbiguousIDError` is raised and the candidates are listed — use a longer prefix to disambiguate.  
If **no** task matches, a `TaskNotFoundError` is raised.

**Tip:** The first 8 characters of a UUID are almost always unique within a personal task list.

---

## Task Schema

Each task is stored as a JSON object with five fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | UUID4 hex string (32 lowercase hex characters, no hyphens) |
| `title` | `string` | Free-form task description |
| `status` | `string` | One of `todo`, `in_progress`, or `done` |
| `priority` | `string` | One of `low`, `medium`, or `high` |
| `created_at` | `string` | ISO-8601 UTC timestamp (e.g. `2026-03-24T12:00:00.000000+00:00`) |

**Example `tasks.json`:**

```json
[
  {
    "id": "3f2a1b8cd4e5f6a7b8c9d0e1f2a3b4c5",
    "title": "Fix critical bug",
    "status": "done",
    "priority": "high",
    "created_at": "2026-03-24T12:00:00.000000+00:00"
  },
  {
    "id": "a9c4e7f1b2d3e4f5a6b7c8d9e0f1a2b3",
    "title": "Review pull request",
    "status": "todo",
    "priority": "medium",
    "created_at": "2026-03-24T12:01:00.000000+00:00"
  }
]
```

The file is written atomically (via a temporary file + rename) so a crash during a write cannot corrupt existing data.

---

## Development

### Running Tests

Install development dependencies first:

```bash
pip install -r requirements-dev.txt
```

Run the full test suite:

```bash
pytest tests/ -v
```

Run a specific test file:

```bash
pytest tests/test_store.py -v   # unit tests for TaskStore
pytest tests/test_cli.py  -v   # integration tests via CliRunner
```

### Coverage Report

```bash
pytest tests/ --cov=task_tracker --cov-report=term-missing
```

---

## Project Structure

```
task_tracker/
├── task_tracker/
│   ├── __init__.py       # Package marker and version string
│   ├── task_store.py     # TaskStore class — all JSON persistence logic
│   └── cli.py            # Click CLI — commands and Rich output
├── tests/
│   ├── __init__.py
│   ├── test_store.py     # Unit tests for TaskStore
│   └── test_cli.py       # Integration tests via Click's CliRunner
├── pyproject.toml        # Project metadata, dependencies, entry point
├── requirements.txt      # Runtime dependencies (click, rich)
├── requirements-dev.txt  # Development dependencies (pytest, pytest-cov)
├── .env.example          # Template for environment variable configuration
└── README.md             # This file
```

---

## License

MIT
