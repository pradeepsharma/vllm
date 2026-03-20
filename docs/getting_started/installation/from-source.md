---
description: >
  Build vLLM from source using CMake — full compilation guide for CUDA, ROCm,
  CPU, and other backends, including ccache, sccache, and incremental builds.
toc_depth: 3
---

# Building vLLM from Source

Building vLLM from source is required when you need to:

- Modify C++ or CUDA/HIP kernel code
- Target a non-default CUDA version
- Use a custom PyTorch build
- Build for CPU, TPU, XPU, or other non-CUDA backends
- Create a portable wheel for distribution

For most users, [pre-built wheels](gpu-cuda.md#option-1--install-via-pip) are
the recommended installation method.

---

## Prerequisites

### System dependencies

=== "CUDA (NVIDIA)"

    ```bash
    # Ubuntu / Debian
    sudo apt-get update -y
    sudo apt-get install -y \
        git curl wget \
        gcc-10 g++-10 \
        ccache \
        libibverbs-dev

    # Set GCC 10 as default (avoids CUTLASS compilation issues)
    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 110 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-10
    ```

=== "CPU (x86)"

    ```bash
    sudo apt-get update -y
    sudo apt-get install -y \
        git curl wget \
        gcc-12 g++-12 \
        libnuma-dev libtcmalloc-minimal4 \
        ccache

    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-12
    ```

=== "ROCm (AMD)"

    Follow the [ROCm installation guide](gpu-rocm.md#option-3--build-from-source)
    for ROCm-specific prerequisites.

=== "macOS"

    ```bash
    # Install Xcode Command Line Tools
    xcode-select --install
    ```

### Python environment

We recommend [uv](https://docs.astral.sh/uv/) for environment management:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment
uv venv --python 3.12 --seed
source .venv/bin/activate
```

### Clone the repository

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
```

---

## Build Methods

### Method 1 — Python-only build (no recompilation)

If you only need to modify Python code, use the precompiled build mode.
This downloads pre-built compiled libraries and uses them with your local
Python changes:

```bash
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

Changes to `.py` files are reflected immediately without recompiling.

**Environment variables:**

| Variable | Description |
|---|---|
| `VLLM_USE_PRECOMPILED` | Set to `1` to enable precompiled mode |
| `VLLM_PRECOMPILED_WHEEL_COMMIT` | Override commit hash (or `nightly`) |
| `VLLM_PRECOMPILED_WHEEL_LOCATION` | Exact URL or local path to a wheel |
| `VLLM_PRECOMPILED_WHEEL_VARIANT` | Variant, e.g. `cu129`, `cu130`, `cpu` |

!!! note
    If you rebase your branch, re-run the install command to refresh the
    compiled libraries.

### Method 2 — Full build (with C++/CUDA compilation)

This compiles all C++ and CUDA kernels. It takes **10–30 minutes** on the
first build, but subsequent builds are faster with caching.

```bash
# Install build dependencies
uv pip install -r requirements/build.txt \
    --extra-index-url https://download.pytorch.org/whl/cu129

# Build and install in editable mode
uv pip install -e .
```

!!! tip "Editable install"
    The `-e` flag installs vLLM in editable mode, so Python changes are
    reflected immediately. C++/CUDA changes still require rebuilding.

### Method 3 — Build a portable wheel

Build a standalone `.whl` file that can be installed on other machines:

```bash
# Install build dependencies
uv pip install -r requirements/build.txt \
    --extra-index-url https://download.pytorch.org/whl/cu129

# Build the wheel
uv build --wheel

# Install the wheel
pip install dist/vllm-*.whl
```

---

## Backend-Specific Build Instructions

### CUDA (NVIDIA GPU)

```bash
# Install CUDA dependencies
uv pip install -r requirements/cuda.txt \
    --extra-index-url https://download.pytorch.org/whl/cu129

# Build vLLM (CUDA is the default target)
uv pip install -e .
```

**Control which GPU architectures to compile for:**

```bash
# Compile for specific architectures (faster build)
export TORCH_CUDA_ARCH_LIST="8.0 9.0"   # Ampere + Hopper only

# Compile for all supported architectures (default)
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.9 9.0 10.0 12.0"

uv pip install -e .
```

**GPU architecture reference:**

| Architecture | Compute Capability | GPU Examples |
|---|---|---|
| Volta | 7.0 | V100 |
| Turing | 7.5 | T4, RTX 20xx |
| Ampere | 8.0, 8.6 | A100, A10, RTX 30xx |
| Ada Lovelace | 8.9 | L4, L40, RTX 40xx |
| Hopper | 9.0 | H100, H200 |
| Blackwell | 10.0, 12.0 | B200, GB200 |

### CPU (x86, ARM, macOS)

```bash
# Install CPU-specific dependencies
uv pip install -r requirements/cpu-build.txt --torch-backend cpu
uv pip install -r requirements/cpu.txt --torch-backend cpu

# Build for CPU
VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation
```

For cross-compilation (building for a different CPU than the build host):

```bash
# Cross-compile for AVX-512 (from a non-AVX-512 host)
VLLM_CPU_AVX512=1 VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation

# Cross-compile for ARM BF16
VLLM_CPU_ARM_BF16=1 VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation
```

### ROCm (AMD GPU)

```bash
# Install ROCm dependencies
pip install -r requirements/rocm.txt

# Set target GPU architecture
export PYTORCH_ROCM_ARCH="gfx942"   # MI300X

# Build vLLM for ROCm
python3 setup.py develop
```

!!! note
    `pip install .` does not work for ROCm. Use `setup.py develop`.

### TPU (Google Cloud)

```bash
# Install TPU dependencies
export VLLM_TARGET_DEVICE=tpu
pip install -r requirements/tpu.txt

# Build vLLM for TPU
pip install -e .
```

### XPU (Intel GPU)

```bash
# Source oneAPI environment first
source /opt/intel/oneapi/setvars.sh --force
source /opt/intel/oneapi/ccl/2021.15/env/vars.sh --force

# Install XPU dependencies
uv pip install -r requirements/xpu.txt

# Build vLLM for XPU
export VLLM_TARGET_DEVICE=xpu
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export CMAKE_PREFIX_PATH="$(python -c 'import site; print(site.getsitepackages()[0])'):${CMAKE_PREFIX_PATH}"

uv pip install --no-build-isolation .
```

---

## Speeding Up Builds

### ccache (local compilation cache)

[ccache](https://ccache.dev/) caches compilation results locally, making
subsequent builds much faster:

```bash
# Install ccache
sudo apt install ccache        # Ubuntu/Debian
conda install ccache           # conda

# Verify ccache is on PATH
which ccache

# Build with ccache (it's used automatically when on PATH)
uv pip install -e .
```

For editable installs with pip, use:

```bash
CCACHE_NOHASHDIR="true" pip install --no-build-isolation -e .
```

!!! tip
    After the first build, subsequent builds with ccache are typically
    **5–10× faster**.

### sccache (remote compilation cache)

[sccache](https://github.com/mozilla/sccache) supports S3-backed remote
caches, useful for CI/CD pipelines:

```bash
# Configure S3 cache
export SCCACHE_BUCKET=vllm-build-sccache
export SCCACHE_REGION=us-west-2
export SCCACHE_S3_NO_CREDENTIALS=1
export SCCACHE_IDLE_TIMEOUT=0

# Build with sccache
uv pip install -e .
```

### Parallel compilation

Control the number of parallel compilation jobs:

```bash
# Set maximum parallel jobs
export MAX_JOBS=8

# Set NVCC threads per job (for CUDA builds)
export NVCC_THREADS=2

uv pip install -e .
```

### Incremental kernel builds

For frequent C++/CUDA kernel changes, use the incremental compilation
workflow to rebuild only modified files:

```bash
# After initial uv pip install -e . setup:
# See docs/contributing/incremental_build.md for details
```

---

## Using an Existing PyTorch Installation

If you have a custom PyTorch build (nightly, ROCm, etc.) and want to build
vLLM against it without reinstalling PyTorch:

```bash
# Tell vLLM to use the existing PyTorch
python use_existing_torch.py --prefix

# Build without reinstalling PyTorch
uv pip install -e . --no-build-isolation
```

The `use_existing_torch.py` script modifies the build requirements to skip
PyTorch installation.

---

## Build Configuration Reference

### Key environment variables

| Variable | Description | Default |
|---|---|---|
| `VLLM_TARGET_DEVICE` | Target device: `cuda`, `cpu`, `tpu`, `xpu`, `empty` | `cuda` |
| `TORCH_CUDA_ARCH_LIST` | CUDA architectures to compile for | `7.0 7.5 8.0 8.9 9.0 10.0 12.0` |
| `PYTORCH_ROCM_ARCH` | ROCm GPU architectures (e.g., `gfx942`) | (auto) |
| `MAX_JOBS` | Maximum parallel compilation jobs | (auto) |
| `NVCC_THREADS` | NVCC threads per job | `2` |
| `VLLM_USE_PRECOMPILED` | Use pre-built compiled libraries | `0` |
| `VLLM_CPU_DISABLE_AVX512` | Disable AVX-512 for CPU builds | `0` |
| `VLLM_CPU_AVX512` | Force-enable AVX-512 (cross-compilation) | `0` |
| `VLLM_CPU_ARM_BF16` | Force-enable ARM BF16 (cross-compilation) | `0` |
| `CCACHE_NOHASHDIR` | Required for ccache with editable pip installs | (unset) |

### CMake build arguments

vLLM uses CMake internally. You can pass CMake arguments via environment
variables or the `cmake` command directly:

```bash
# Disable CUDA detection during CPU builds
CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON VLLM_TARGET_DEVICE=cpu uv pip install -e .
```

---

## Verifying the Build

```bash
# Check the installed version
python -c "import vllm; print(f'vLLM {vllm.__version__}')"

# Run a basic inference test
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='facebook/opt-125m')
outputs = llm.generate(['Hello'], SamplingParams(max_tokens=10))
print(outputs[0].outputs[0].text)
"

# Collect environment information for bug reports
python vllm/collect_env.py
```

---

## Troubleshooting

??? question "NumPy ≥ 2.0 error"
    Downgrade NumPy: `pip install "numpy<2.0"`

??? question "CMake picks up CUDA during CPU build"
    ```bash
    CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON VLLM_TARGET_DEVICE=cpu pip install -e .
    ```

??? question "Wheel not found for precompiled build"
    The wheel for your base commit may still be building. Wait ~1 hour and
    retry, or pin to an earlier commit:
    ```bash
    export VLLM_PRECOMPILED_WHEEL_COMMIT=$(git rev-parse HEAD~1)
    VLLM_USE_PRECOMPILED=1 uv pip install --editable .
    ```

??? question "CUDA out of memory during compilation"
    Reduce parallel jobs: `export MAX_JOBS=4`

??? question "Undefined symbol errors after build"
    Ensure your CUDA version matches the PyTorch CUDA version:
    ```bash
    python -c "import torch; print(torch.version.cuda)"
    nvcc --version
    ```

??? question "Build fails on ARM64 with x86 ISA flags"
    Do not mix x86 ISA flags (AVX2, AVX512) with ARM64 builds. Use
    `VLLM_CPU_ARM_BF16=1` for ARM BF16 instead.

---

## Next Steps

- **[CUDA Installation](gpu-cuda.md)** — NVIDIA GPU installation details
- **[CPU Installation](cpu.md)** — CPU-only installation details
- **[First Inference](../first-inference.md)** — Run your first generation
- **[Contributing Guide](../../contributing/index.md)** — How to contribute to vLLM
