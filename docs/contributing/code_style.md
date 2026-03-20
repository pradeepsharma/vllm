# Code Style

vLLM enforces consistent code style through automated tools: **Ruff** for linting and formatting, **mypy** for type checking, and **pre-commit** to run these checks automatically before every commit.

---

## Overview

| Tool | Purpose | Config Location |
|---|---|---|
| **Ruff** | Linting + formatting (replaces flake8, isort, black) | `pyproject.toml` |
| **mypy** | Static type checking | `pyproject.toml` |
| **pre-commit** | Runs all checks before commits | `requirements/lint.txt` |
| **typos** | Spell checking | `pyproject.toml` |
| **markdownlint** | Markdown linting | `.github/workflows/matchers/` |
| **actionlint** | GitHub Actions workflow linting | `.github/workflows/matchers/` |

---

## Pre-commit

[pre-commit](https://pre-commit.com/) is the primary mechanism for enforcing code quality. It runs a suite of hooks automatically on every `git commit`.

### Installation

```bash
# Install pre-commit (included in requirements/lint.txt)
pip install pre-commit==4.0.1

# Install the hooks into your local git repository
pre-commit install
```

### Running Manually

```bash
# Run all hooks on all files (same as CI)
pre-commit run --all-files --hook-stage manual

# Run on staged files only (default behavior on commit)
pre-commit run

# Run a specific hook
pre-commit run ruff
pre-commit run mypy
```

### CI Integration

The pre-commit workflow runs on every pull request and push to `main` via GitHub Actions (`.github/workflows/pre-commit.yml`). It uses Python 3.12 and runs all hooks with `--all-files --hook-stage manual`.

If pre-commit fails in CI, fix the issues locally and push again:

```bash
pre-commit run --all-files --hook-stage manual
git add -A
git commit -m "fix: address pre-commit failures"
```

---

## Ruff

[Ruff](https://docs.astral.sh/ruff/) is an extremely fast Python linter and formatter written in Rust. It replaces flake8, isort, pyupgrade, and black in a single tool.

### Configuration

Ruff is configured in `pyproject.toml`:

```toml
[tool.ruff.lint]
select = [
    "E",    # pycodestyle errors
    "F",    # Pyflakes (undefined names, unused imports, etc.)
    "UP",   # pyupgrade (modernize Python syntax)
    "B",    # flake8-bugbear (likely bugs and design issues)
    "ISC",  # flake8-implicit-str-concat
    "SIM",  # flake8-simplify
    "I",    # isort (import ordering)
    "G",    # flake8-logging-format
]
ignore = [
    "F405", "F403",  # star imports
    "E731",          # lambda expression assignment
    "B905",          # zip without strict=
    "B007",          # loop control variable not used
    "UP032",         # f-string format
]

[tool.ruff.format]
docstring-code-format = true
```

### Per-File Ignores

Some files have specific rules disabled:

```toml
[tool.ruff.lint.per-file-ignores]
"vllm/third_party/**" = ["ALL"]       # Third-party code
"vllm/version.py" = ["F401"]          # Version file
"vllm/grpc/*_pb2.py" = ["ALL"]        # Generated protobuf files
```

### Running Ruff

```bash
# Check for linting issues
ruff check vllm/

# Auto-fix fixable issues
ruff check --fix vllm/

# Format code (like black)
ruff format vllm/

# Check formatting without applying
ruff format --check vllm/
```

### Common Ruff Rules

**Import ordering (I):**
```python
# ❌ Wrong order
import torch
import os
from vllm import LLM

# ✅ Correct order (stdlib → third-party → local)
import os

import torch

from vllm import LLM
```

**Unused imports (F401):**
```python
# ❌ Unused import
import json  # never used

# ✅ Remove or use it
```

**pyupgrade (UP):**
```python
# ❌ Old-style type hints
from typing import List, Dict, Optional

def foo(x: Optional[List[Dict[str, int]]]) -> None: ...

# ✅ Modern type hints (Python 3.10+)
def foo(x: list[dict[str, int]] | None) -> None: ...
```

**Logging format (G):**
```python
# ❌ f-string in logging (evaluated even if log level is disabled)
logger.debug(f"Processing {len(requests)} requests")

# ✅ Lazy formatting
logger.debug("Processing %d requests", len(requests))
```

---

## mypy

[mypy](https://mypy.readthedocs.io/) performs static type checking to catch type errors before runtime.

### Configuration

```toml
[tool.mypy]
plugins = ['pydantic.mypy']
ignore_missing_imports = true
check_untyped_defs = true
follow_imports = "silent"
```

### Type Annotation Guidelines

vLLM uses type annotations throughout the codebase. Follow these conventions:

**Function signatures:**
```python
def process_requests(
    requests: list[str],
    max_tokens: int = 100,
    temperature: float = 1.0,
) -> list[str]:
    ...
```

**Class attributes:**
```python
class MyConfig:
    model_name: str
    max_batch_size: int = 32
    dtype: torch.dtype = torch.float16
```

**Optional values:**
```python
# Use X | None instead of Optional[X]
def get_model(name: str | None = None) -> Model | None:
    ...
```

**TYPE_CHECKING guard for circular imports:**
```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
```

### Running mypy

```bash
# Type-check the vllm package
mypy vllm/

# Type-check a specific file
mypy vllm/engine/llm_engine.py

# Show error codes (useful for suppressing specific errors)
mypy --show-error-codes vllm/
```

### Suppressing mypy Errors

Use `# type: ignore[error-code]` sparingly and only when necessary:

```python
# Acceptable: third-party library without stubs
import some_library  # type: ignore[import]

# Acceptable: known false positive
result = complex_operation()  # type: ignore[assignment]
```

---

## Code Style Guidelines

Beyond automated checks, follow these conventions:

### Imports

```python
# Standard library imports first
import os
import sys
from typing import TYPE_CHECKING

# Third-party imports second
import torch
import numpy as np
from transformers import AutoConfig

# Local imports last
from vllm.config import VllmConfig
from vllm.logger import init_logger

# TYPE_CHECKING imports at the end
if TYPE_CHECKING:
    from vllm.engine import LLMEngine
```

### Logging

Always use vLLM's logger, not `print()`:

```python
from vllm.logger import init_logger

logger = init_logger(__name__)

# Use appropriate log levels
logger.debug("Detailed debug info: %s", detail)
logger.info("Processing %d requests", count)
logger.warning("Deprecated parameter: %s", param_name)
logger.error("Failed to load model: %s", error)
```

### Docstrings

Use Google-style docstrings for public APIs:

```python
def generate(
    self,
    prompts: list[str],
    sampling_params: SamplingParams,
) -> list[RequestOutput]:
    """Generate completions for the given prompts.

    Args:
        prompts: List of input prompts to generate completions for.
        sampling_params: Sampling parameters controlling generation
            behavior (temperature, top_p, max_tokens, etc.).

    Returns:
        List of RequestOutput objects, one per prompt, containing
        the generated text and metadata.

    Raises:
        ValueError: If prompts is empty or sampling_params is invalid.
    """
    ...
```

### Constants and Configuration

```python
# Use UPPER_SNAKE_CASE for module-level constants
MAX_BATCH_SIZE = 256
DEFAULT_DTYPE = torch.float16

# Use dataclasses or Pydantic models for configuration
from dataclasses import dataclass

@dataclass
class EngineConfig:
    model: str
    max_batch_size: int = 256
    dtype: str = "auto"
```

### Error Messages

Write clear, actionable error messages:

```python
# ❌ Vague
raise ValueError("Invalid input")

# ✅ Specific and actionable
raise ValueError(
    f"max_tokens must be a positive integer, got {max_tokens}. "
    "Set max_tokens to a value greater than 0."
)
```

---

## Spell Checking

vLLM uses [typos](https://github.com/crate-ci/typos) for spell checking. It's configured in `pyproject.toml`:

```toml
[tool.typos.files]
extend-exclude = [
    "tests/models/fixtures/*",
    "tests/prompts/*",
    "benchmarks/sonnet.txt",
    # ... other excluded paths
]
```

Custom word allowances are defined under `[tool.typos.default.extend-words]` for domain-specific terms like `arange`, `HSA`, `thr`, etc.

---

## Markdown Style

Documentation files are linted with [markdownlint](https://github.com/DavidAnson/markdownlint). Key rules:

- Use ATX-style headings (`#`, `##`, `###`)
- Fenced code blocks with language specifiers (` ```python `)
- No trailing whitespace
- Blank line before and after headings and code blocks
- Maximum line length of 120 characters (soft limit)

---

## Editor Integration

### VS Code

Install the [Ruff extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) and [mypy extension](https://marketplace.visualstudio.com/items?itemName=ms-python.mypy-type-checker):

```json
// .vscode/settings.json
{
  "[python]": {
    "editor.defaultFormatter": "charliermarsh.ruff",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll.ruff": "explicit",
      "source.organizeImports.ruff": "explicit"
    }
  },
  "mypy-type-checker.args": ["--config-file=pyproject.toml"]
}
```

### PyCharm

1. Install the [Ruff plugin](https://plugins.jetbrains.com/plugin/20574-ruff)
2. Go to **Settings → Tools → Ruff** and enable "Run on save"
3. For mypy: **Settings → Tools → External Tools** → add mypy

---

## Summary Checklist

Before submitting a PR, verify:

- [ ] `pre-commit run --all-files --hook-stage manual` passes
- [ ] No new mypy errors introduced
- [ ] All new public functions/classes have type annotations
- [ ] All new public APIs have docstrings
- [ ] Imports are properly ordered
- [ ] No `print()` statements (use `logger` instead)
- [ ] Error messages are clear and actionable
