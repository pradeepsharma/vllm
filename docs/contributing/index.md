# Contributing to vLLM

Welcome to the vLLM contributor community! vLLM is an open-source, high-throughput inference engine for large language models, and it thrives because of contributions from engineers, researchers, and practitioners around the world.

This section covers everything you need to know to contribute effectively — from setting up your development environment to submitting pull requests, adding new models, and understanding the CI/CD pipeline.

---

## Why Contribute?

vLLM is at the frontier of LLM serving technology. Contributing gives you the opportunity to:

- **Shape the future of LLM inference** — work on cutting-edge features like speculative decoding, quantization, and distributed serving
- **Collaborate with experts** — engage with a global community of ML engineers and researchers
- **Build your portfolio** — your contributions are visible to the entire open-source community
- **Learn deeply** — understand how production-grade inference systems are built

---

## Ways to Contribute

There are many ways to contribute to vLLM beyond writing code:

| Contribution Type | Description |
|---|---|
| 🐛 **Bug Reports** | Report issues you encounter with detailed reproduction steps |
| 🚀 **Feature Requests** | Propose new capabilities or improvements |
| 🤗 **New Models** | Add support for new model architectures |
| ⚡ **Quantization Backends** | Implement new quantization methods |
| 🖥️ **Hardware Platforms** | Add support for new accelerators |
| 📝 **Documentation** | Improve guides, fix typos, add examples |
| 🧪 **Tests** | Add test coverage for existing or new features |
| 💬 **Community Support** | Help others in GitHub Discussions and Slack |

---

## Before You Start

### Read the Code of Conduct

All contributors must follow the [vLLM Code of Conduct](https://github.com/vllm-project/vllm/blob/main/CODE_OF_CONDUCT.md). We are committed to maintaining a welcoming, inclusive, and harassment-free community for everyone.

Key principles:
- Demonstrate empathy and kindness
- Be respectful of differing opinions and experiences
- Give and gracefully accept constructive feedback
- Focus on what is best for the community

Violations can be reported in the `#code-of-conduct` channel on [vLLM Slack](https://slack.vllm.ai).

### Join the Community

- **Slack**: [slack.vllm.ai](https://slack.vllm.ai) — real-time discussion, help, and announcements
- **GitHub Discussions**: For longer-form technical discussions
- **GitHub Issues**: For bug reports, feature requests, and model requests

---

## Contribution Workflow

The standard contribution workflow for vLLM is:

```
Fork → Branch → Develop → Test → PR → Review → Merge
```

### Step 1: Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/<your-username>/vllm.git
cd vllm

# Add the upstream remote
git remote add upstream https://github.com/vllm-project/vllm.git
```

### Step 2: Create a Branch

```bash
# Sync with upstream main
git fetch upstream
git checkout -b my-feature upstream/main
```

Use descriptive branch names like `fix/scheduler-deadlock`, `feat/add-llama4-model`, or `docs/update-quantization-guide`.

### Step 3: Develop and Test

Set up your development environment (see [Development Setup](development_setup.md)), make your changes, and run the relevant tests.

### Step 4: Submit a Pull Request

When your changes are ready:

1. Push your branch to your fork
2. Open a PR against `vllm-project/vllm:main`
3. Fill in the PR template completely:
   - **Purpose**: What does this PR do? Link any related issues
   - **Test Plan**: How did you test the changes?
   - **Test Results**: Paste relevant output or benchmarks
4. Update documentation if your change is user-facing
5. Update release notes in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit) if applicable

### Step 5: Respond to Review

The vLLM team will review your PR. Be prepared to:
- Answer questions about your design choices
- Make requested changes
- Rebase on main if there are conflicts

---

## Issue Templates

When filing issues, use the appropriate template:

| Template | Use For |
|---|---|
| 🐛 **Bug Report** | Unexpected behavior, crashes, incorrect output |
| 🚀 **Feature Request** | New capabilities you'd like to see |
| 🤗 **New Model** | Request support for a HuggingFace model |
| 🧪 **CI Failure** | Failing tests in the CI pipeline |
| ⚡ **Performance Discussion** | Performance regressions or optimization ideas |
| 💬 **RFC** | Major architectural changes requiring community feedback |
| 📝 **Documentation** | Documentation improvements or corrections |

For bug reports, always run `python vllm/collect_env.py` and include the output.

---

## Documentation in This Section

| Guide | Description |
|---|---|
| [Development Setup](development_setup.md) | Install dependencies, build from source, configure your IDE |
| [Code Style](code_style.md) | Ruff linting, mypy type checking, pre-commit hooks |
| [Testing](testing.md) | Running tests, writing new tests, test organization |
| [Adding a Model](adding_model.md) | Step-by-step guide to adding a new model architecture |
| [Adding Quantization](adding_quantization.md) | Implementing a new quantization backend |
| [Adding a Platform](adding_platform.md) | Supporting a new hardware accelerator |
| [CI/CD Pipeline](ci_cd.md) | Buildkite and GitHub Actions workflows |
| [Release Process](release_process.md) | How vLLM releases are managed and versioned |

---

## Getting Help

If you're stuck, don't hesitate to ask:

- **Slack `#contributing` channel**: Best for quick questions
- **GitHub Discussions**: For longer technical discussions
- **PR comments**: Tag a maintainer if you need guidance on a specific PR

We appreciate every contribution, no matter how small. Thank you for helping make vLLM better!
