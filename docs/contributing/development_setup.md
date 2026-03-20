# Development Environment Setup

This guide walks you through setting up a complete vLLM development environment, including all dependencies, build tools, and IDE configuration.

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10–3.13** (Python 3.12 recommended for development)
- **Git** 2.x or later
- **CUDA 12.x** (for GPU development; see [installation guides](../getting_started/installation/index.md) for other platforms)
- **CMake 3.26.1+** and **Ninja** (for building C++/CUDA extensions)
- At least **50 GB of disk space** (for model weights, build artifacts, and test data)

---

## Step 1: Fork and Clone

```bash
# Fork the repository on GitHub first, then:
git clone https://github.com/<your-username>/vllm.git
cd vllm

# Add the upstream remote to stay in sync
git remote add upstream https://github.com/vllm-project/vllm.git
```

---

## Step 2: Create a Virtual Environment

Using a virtual environment is strongly recommended to isolate your development dependencies.

### Using `venv` (standard)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### Using `uv` (faster, recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager that significantly speeds up dependency installation.

```bash
pip install uv
uv venv .venv
source .venv/bin/activate
```

---

## Step 3: Install Build Dependencies

```bash
pip install cmake>=3.26.1 ninja packaging>=24.2 setuptools>=77.0.3 wheel
```

Or install from the build requirements file:

```bash
pip install -r requirements/build.txt
```

---

## Step 4: Install vLLM in Editable Mode

For GPU (CUDA) development:

```bash
# Install PyTorch first (required before building vLLM)
pip install torch==2.10.0

# Install vLLM in editable mode with all development dependencies
pip install -e ".[dev]"
```

The `-e` flag installs vLLM in "editable" mode, meaning changes to Python source files take effect immediately without reinstalling. C++/CUDA extensions still require a rebuild when modified.

### Platform-Specific Installation

For non-CUDA platforms, use the appropriate requirements file:

```bash
# CPU-only development
pip install -r requirements/cpu.txt
pip install -e .

# ROCm (AMD GPU)
pip install -r requirements/rocm.txt
pip install -e .

# Intel XPU
pip install -r requirements/xpu.txt
pip install -e .
```

---

## Step 5: Install Development Dependencies

```bash
# Install linting and testing tools
pip install -r requirements/dev.txt
```

This installs:
- `pre-commit==4.0.1` — for running linters before commits
- `pytest` and related plugins — for running tests
- Various testing utilities (see `requirements/test.in` for the full list)

---

## Step 6: Set Up Pre-commit Hooks

Pre-commit hooks automatically run linters and formatters before each commit, catching issues early.

```bash
# Install the pre-commit hooks
pre-commit install

# (Optional) Run all hooks on all files to verify setup
pre-commit run --all-files --hook-stage manual
```

After installation, hooks run automatically on `git commit`. See [Code Style](code_style.md) for details on what each hook checks.

---

## Step 7: Verify Your Setup

Run a quick sanity check to confirm everything is working:

```python
# test_setup.py
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(["Hello, world!"], SamplingParams(max_tokens=20))
print(outputs[0].outputs[0].text)
```

```bash
python test_setup.py
```

You can also run the basic correctness tests:

```bash
cd tests
pytest -v -s basic_correctness/test_basic_correctness.py -k "test_vllm_gc_ed"
```

---

## Building C++/CUDA Extensions

When you modify C++ or CUDA source files in `csrc/`, you need to rebuild the extensions:

```bash
# Rebuild extensions (from the repo root)
pip install -e . --no-build-isolation
```

For faster incremental builds, use `ninja` directly:

```bash
# Build only the changed files
python setup.py build_ext --inplace
```

### Compilation Environment Variables

| Variable | Description |
|---|---|
| `MAX_JOBS` | Number of parallel compilation jobs (default: CPU count) |
| `NVCC_THREADS` | Threads per NVCC compilation job |
| `VLLM_TARGET_DEVICE` | Target device: `cuda`, `rocm`, `cpu`, `tpu`, `xpu` |
| `CMAKE_BUILD_TYPE` | `Release` (default) or `Debug` |

Example:

```bash
MAX_JOBS=8 pip install -e . --no-build-isolation
```

---

## IDE Configuration

### VS Code

Install the recommended extensions:

```json
// .vscode/extensions.json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.mypy-type-checker",
    "charliermarsh.ruff",
    "ms-vscode.cmake-tools"
  ]
}
```

Configure the Python interpreter to use your virtual environment:

1. Press `Ctrl+Shift+P` → "Python: Select Interpreter"
2. Choose `.venv/bin/python`

### PyCharm

1. Go to **Settings → Project → Python Interpreter**
2. Click **Add Interpreter → Existing Environment**
3. Select `.venv/bin/python`

Enable Ruff as an external tool for linting (see [Code Style](code_style.md)).

---

## Environment Variables for Development

Set these environment variables to improve the development experience:

```bash
# Enable verbose logging for debugging
export VLLM_LOGGING_LEVEL=DEBUG

# Trace all function calls (useful for debugging hangs/crashes)
export VLLM_TRACE_FUNCTION=1

# Use spawn method for multiprocessing (more reliable in development)
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Disable CUDA graph capture for faster startup during development
export VLLM_SKIP_WARMUP=true

# Set visible GPUs (e.g., use only GPU 0 and 1)
export CUDA_VISIBLE_DEVICES=0,1
```

---

## Working with the Codebase

### Key Directories

```
vllm/
├── entrypoints/          # CLI and server entry points
│   ├── cli/              # vllm CLI commands
│   └── openai/           # OpenAI-compatible API server
├── engine/               # LLM engine (v0 legacy)
├── v1/                   # V1 engine (current)
│   ├── engine/           # Core engine logic
│   ├── scheduler/        # Request scheduling
│   └── worker/           # GPU worker processes
├── model_executor/       # Model execution
│   ├── models/           # Model implementations
│   └── layers/           # Custom layers (attention, quantization, etc.)
├── distributed/          # Distributed communication utilities
├── platforms/            # Hardware platform abstractions
└── config.py             # Configuration dataclasses

tests/
├── basic_correctness/    # Core correctness tests
├── distributed/          # Multi-GPU tests
├── models/               # Per-model tests
├── quantization/         # Quantization tests
└── v1/                   # V1 engine tests
```

### Keeping Your Fork in Sync

```bash
# Fetch upstream changes
git fetch upstream

# Rebase your branch on the latest main
git rebase upstream/main

# Or merge (less preferred)
git merge upstream/main
```

---

## Troubleshooting

### Build Failures

**CUDA version mismatch:**
```bash
# Check your CUDA version
nvcc --version
python -c "import torch; print(torch.version.cuda)"
# These should match
```

**CMake not found:**
```bash
pip install cmake>=3.26.1
# Or install system CMake:
sudo apt-get install cmake  # Ubuntu/Debian
```

**Out of memory during build:**
```bash
# Reduce parallel jobs
MAX_JOBS=2 pip install -e . --no-build-isolation
```

### Import Errors

If you see `ImportError` for vLLM modules after editing:

```bash
# Reinstall in editable mode
pip install -e . --no-build-isolation
```

### Pre-commit Hook Failures

```bash
# Run hooks manually to see what's failing
pre-commit run --all-files

# Skip hooks for a specific commit (use sparingly)
git commit --no-verify -m "WIP: ..."
```

---

## Next Steps

- [Code Style](code_style.md) — learn about Ruff, mypy, and pre-commit configuration
- [Testing](testing.md) — understand how to run and write tests
- [Adding a Model](adding_model.md) — contribute a new model architecture
