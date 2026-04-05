# Development Environment Setup

This page walks through setting up a complete local development environment for contributing to vLLM, including cloning the repository, creating a virtual environment, installing the package in editable mode, and configuring pre-commit hooks.

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10–3.13** (vLLM supports all four versions)
- **Git** (with tags available — shallow clones will break version detection)
- **CUDA toolkit** (for GPU builds; see [Hardware Support](../09-hardware/overview.md))
- **CMake ≥ 3.26.1** and **Ninja** (required for C++/CUDA extension builds)
- **GCC / Clang** compatible with your CUDA version

> **macOS note**: On macOS, `VLLM_TARGET_DEVICE` is automatically set to `cpu`. GPU builds are only supported on Linux.

## Step 1: Fork and Clone

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-username>/vllm.git
cd vllm

# Add the upstream remote so you can pull in changes
git remote add upstream https://github.com/vllm-project/vllm.git
```

> **Important**: The build system uses `setuptools-scm` to derive the package version from Git tags. If you clone with `--depth 1` (a shallow clone), the version will be incorrect. Run `git fetch --unshallow --tags` to fix this, or use `tools/check_repo.sh` to verify.

```bash
# Verify the repo is in a good state
bash tools/check_repo.sh
```

## Step 2: Create a Virtual Environment

Using a virtual environment keeps your system Python clean:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows (WSL recommended)
```

Alternatively, use `uv` for faster dependency resolution:

```bash
pip install uv
uv venv .venv
source .venv/bin/activate
```

> **Note**: The `pyproject.toml` includes a `[tool.ty.environment]` section that points to `.venv`, so type-checking tools will automatically use this environment.

## Step 3: Install in Editable Mode

vLLM uses a `setup.py` that compiles CUDA/C++ extensions. The editable install (`-e`) lets you modify Python source files without reinstalling.

### CUDA Build (default on Linux with CUDA)

```bash
pip install -e .
```

The build system auto-detects your hardware:
- If `torch.version.cuda` is set → builds CUDA extensions
- If `torch.version.hip` is set → builds ROCm extensions
- Otherwise → CPU-only build

### CPU-Only Build

```bash
VLLM_TARGET_DEVICE=cpu pip install -e .
```

### Build Dependencies

The build requires these packages (listed in `requirements/build.txt`):

```
cmake>=3.26.1
ninja
packaging>=24.2
setuptools>=77.0.3,<81.0.0
setuptools-scm>=8.0
torch==2.10.0
wheel
jinja2
grpcio-tools==1.78.0
```

These are automatically installed when you run `pip install -e .` because they are declared in `pyproject.toml`'s `[build-system]` section.

### Build Acceleration

The build system automatically uses `sccache` or `ccache` if available, which dramatically speeds up incremental rebuilds:

```bash
# Install sccache (recommended)
cargo install sccache
# or
pip install sccache

# Disable sccache if needed
VLLM_DISABLE_SCCACHE=1 pip install -e .
```

## Step 4: Install Development Dependencies

Install the full development dependency set (linting + testing):

```bash
pip install -r requirements/dev.txt
```

This installs:
- `requirements/lint.txt` — `pre-commit==4.0.1`
- `requirements/test.txt` — pytest, test utilities, model-specific test dependencies

For a lighter install (linting only):

```bash
pip install -r requirements/lint.txt
```

## Step 5: Install Pre-Commit Hooks

vLLM uses [pre-commit](https://pre-commit.com/) to enforce code quality checks before every commit:

```bash
pre-commit install
```

This installs Git hooks that run automatically on `git commit`. To run all hooks manually against all files:

```bash
pre-commit run --all-files
```

To run a specific hook:

```bash
pre-commit run ruff --all-files
pre-commit run mypy --all-files
```

See [Code Style & Linting](code-style.md) for a full description of all hooks.

## Step 6: Verify the Installation

Run a quick sanity check to confirm everything is working:

```bash
# Check vLLM version (should show a semver string, not "0.0.0")
python -c "import vllm; print(vllm.__version__)"

# Run a small subset of unit tests
pytest tests/test_config.py -x -q

# Check that the repo is clean and tagged
bash tools/check_repo.sh
```

## Environment Variables for Development

Several environment variables control build and test behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_TARGET_DEVICE` | auto-detected | Target device: `cuda`, `rocm`, `cpu`, `xpu`, `tpu`, `empty` |
| `VLLM_DISABLE_SCCACHE` | `0` | Set to `1` to disable sccache |
| `VLLM_CI_NO_SKIP` | `0` | Set to `1` to run all model variants in tests |
| `VLLM_CI_DTYPE` | `None` | Override dtype used in tests |
| `VLLM_CI_ENFORCE_EAGER` | `None` | Force eager mode in tests |

These CI-specific variables are defined in `tests/ci_envs.py`.

## Directory Structure Overview

```
vllm/
├── vllm/                    # Main Python package
│   ├── model_executor/      # Model implementations
│   ├── v1/                  # V1 engine (scheduler, executor)
│   ├── entrypoints/         # CLI, OpenAI server, LLM class
│   └── ...
├── tests/                   # Test suite
├── tools/                   # Developer utilities
│   └── pre_commit/          # Custom pre-commit hook scripts
├── requirements/            # Pinned dependency files
├── docker/                  # Dockerfiles for each platform
├── pyproject.toml           # Build config, ruff, mypy, pytest settings
└── setup.py                 # C++/CUDA extension build logic
```

## Troubleshooting

**Version shows `0.0.0`**: The repo is a shallow clone. Run:
```bash
git fetch --unshallow --tags
pip install -e .
```

**CUDA extension build fails**: Ensure `CUDA_HOME` is set and your CUDA version matches the PyTorch build:
```bash
echo $CUDA_HOME
python -c "import torch; print(torch.version.cuda)"
```

**`ninja` not found**: Install it with `pip install ninja` or your system package manager.

**Pre-commit hook fails on first run**: Some hooks (like `mypy`) may need additional stubs. Run `pre-commit run --all-files` once to let hooks self-install their environments.
