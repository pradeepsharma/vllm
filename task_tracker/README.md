# Task Tracker CLI

A simple command-line task tracker with persistent JSON storage, built with Click and Rich.

## Installation

```bash
pip install -e .
```

## Usage

```bash
# Add tasks
task-tracker add --title "Write tests" --priority high
task-tracker add --title "Fix bug" --priority low
task-tracker add --title "Review PR"          # default priority=medium

# List all tasks
task-tracker list

# Filter by status
task-tracker list --status todo
task-tracker list --status done

# Mark a task done (use first 8+ chars of UUID)
task-tracker done <partial-id>

# Show statistics
task-tracker stats

# Delete a task (prompts for confirmation)
task-tracker delete <partial-id>
```

## Running Tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```
