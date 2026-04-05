# Contributing to vLLM

Welcome to the vLLM contributor guide. This section covers everything you need to know to contribute effectively to the project — from setting up your development environment to understanding the CI/CD pipeline and release process.

## What's in This Section

| Page | Description |
|------|-------------|
| [Development Environment](dev-environment.md) | Clone, virtualenv, editable install, pre-commit hooks |
| [Code Style & Linting](code-style.md) | Ruff rules, mypy type checking, custom lint hooks |
| [CI/CD Pipeline](ci-cd.md) | GitHub Actions workflows, Buildkite hardware CI |
| [Buildkite Test Areas](buildkite-test-areas.md) | Test area definitions and coverage |
| [Adding a New Model](adding-new-model.md) | Step-by-step guide to registering a new model |
| [Pull Request Process](pr-process.md) | CODEOWNERS, mergify, autolabeling, stale bot |
| [Release Process](release-process.md) | Versioning, release branches, PyPI publishing |
| [Security Policy](security.md) | Reporting vulnerabilities, severity levels, prenotification |

## Quick Start for Contributors

```bash
# 1. Fork and clone
git clone https://github.com/<your-username>/vllm.git
cd vllm

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install in editable mode (CUDA build)
pip install -e ".[dev]"

# 4. Install pre-commit hooks
pip install pre-commit==4.0.1
pre-commit install

# 5. Run the linter
pre-commit run --all-files
```

## Community Resources

- **GitHub Issues**: Bug reports and feature requests at [github.com/vllm-project/vllm/issues](https://github.com/vllm-project/vllm/issues)
- **Slack**: Join the community at [slack.vllm.ai](https://slack.vllm.ai/)
- **Documentation**: Full docs at [docs.vllm.ai](https://docs.vllm.ai/)
- **Releases**: Release history at [vllm.ai/releases](https://vllm.ai/releases)

## Code of Conduct

All contributors are expected to follow the [Code of Conduct](https://github.com/vllm-project/vllm/blob/main/CODE_OF_CONDUCT.md). Be respectful, inclusive, and constructive in all interactions.

## License

vLLM is licensed under the **Apache 2.0** license. All contributions must include the SPDX header:

```python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
```

This header is enforced automatically by the `check-spdx-header` pre-commit hook.
