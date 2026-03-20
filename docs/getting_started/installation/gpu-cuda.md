---
description: >
  Install vLLM for NVIDIA GPU (CUDA) — pip, Docker, and from-source instructions
  for all supported CUDA versions and GPU architectures.
toc_depth: 3
---

# NVIDIA GPU (CUDA) Installation

vLLM provides first-class support for NVIDIA GPUs via CUDA. Pre-built wheels are
available for the most common CUDA versions, and Docker images are published to
Docker Hub for zero-configuration deployments.

---

## Requirements

| Requirement | Value |
|---|---|
| **Operating System** | Linux (x86\_64 or aarch64) |
| **Python** | 3.10 – 3.13 |
| **CUDA** | 12.1 or later (12.9 recommended) |
| **GPU Compute Capability** | 7.0 or higher (Volta, Turing, Ampere, Ada, Hopper, Blackwell) |
| **NVIDIA Driver** | Compatible with your CUDA version |

!!! tip "Supported GPU families"
    V100, T4, RTX 20xx/30xx/40xx, A10, A100, L4, L40, H100, H200, B200, GB200, and more.
    Use `nvidia-smi` to verify your driver and GPU model.

!!! note "Windows"
    vLLM does not support Windows natively. Use
    [WSL2](https://learn.microsoft.com/en-us/windows/wsl/) with Ubuntu, or refer to
    community-maintained forks such as
    [vllm-windows](https://github.com/SystemPanic/vllm-windows).

---

## Option 1 — Install via pip (Recommended)

### Create a Python environment

We recommend [uv](https://docs.astral.sh/uv/) for fast, reproducible environment
management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment with Python 3.12
uv venv --python 3.12 --seed
source .venv/bin/activate
```

??? console "conda alternative"
    ```bash
    conda create -n vllm python=3.12 -y
    conda activate vllm
    pip install --upgrade uv
    ```

### Install vLLM

```bash
# Recommended: let uv auto-detect your CUDA version
uv pip install vllm --torch-backend=auto
```

`--torch-backend=auto` inspects your installed CUDA driver via `nvidia-smi` and
automatically selects the matching PyTorch CUDA index (e.g., `cu129`, `cu128`).
If this fails, run `uv self update` first.

??? console "pip alternative"
    ```bash
    # Install vLLM with CUDA 12.9 (default)
    pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129
    ```

#### Install for a specific CUDA version

vLLM ships pre-built wheels for **CUDA 12.8**, **12.9**, and **13.0**:

```bash
# Example: install for CUDA 13.0
export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest \
    | jq -r .tag_name | sed 's/^v//')
export CUDA_VERSION=130   # 128, 129, or 130
export CPU_ARCH=$(uname -m)   # x86_64 or aarch64

uv pip install \
    "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu${CUDA_VERSION}-cp38-abi3-manylinux_2_35_${CPU_ARCH}.whl" \
    --extra-index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION}"
```

!!! note "NVIDIA Blackwell (B200, GB200)"
    Blackwell GPUs require **CUDA 12.8 or later**. Use `--torch-backend=cu128` or
    `--torch-backend=cu129` when installing with `uv`.

#### Install the latest nightly build

vLLM publishes wheels for every commit on the `main` branch:

```bash
uv pip install -U vllm \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/nightly
```

Available nightly variants:

| Index URL | Description |
|---|---|
| `https://wheels.vllm.ai/nightly` | Default (CUDA 12.9) |
| `https://wheels.vllm.ai/nightly/cu130` | CUDA 13.0 |
| `https://wheels.vllm.ai/nightly/cpu` | CPU-only |

!!! warning "pip caveat with nightly"
    `pip` cannot reliably install nightly wheels because it merges all indexes and
    picks the highest version number, which may resolve to the stable release.
    Use `uv` for nightly installs.

#### Install a specific commit

```bash
export VLLM_COMMIT=72d9c316d3f6ede485146fe5aabd4e61dbc59069  # full commit SHA
uv pip install vllm \
    --torch-backend=auto \
    --extra-index-url "https://wheels.vllm.ai/${VLLM_COMMIT}"
```

### Verify the installation

```bash
python -c "import vllm; print(f'vLLM {vllm.__version__} installed successfully')"
```

---

## Option 2 — Docker (Zero Local Install)

### Pull the official image

```bash
docker pull vllm/vllm-openai:latest
```

All published tags are available at
[hub.docker.com/r/vllm/vllm-openai](https://hub.docker.com/r/vllm/vllm-openai/tags).

### Run the OpenAI-compatible server

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    --env "HF_TOKEN=${HF_TOKEN}" \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-Instruct-v0.1
```

| Flag | Purpose |
|---|---|
| `--runtime nvidia --gpus all` | Expose all NVIDIA GPUs to the container |
| `-v ~/.cache/huggingface:...` | Persist the model cache across runs |
| `--ipc=host` | Required for tensor-parallel multi-GPU setups |
| `--env HF_TOKEN=...` | Authenticate with Hugging Face for gated models |

### Build a custom Docker image

The main `docker/Dockerfile` supports a rich set of build arguments:

```bash
DOCKER_BUILDKIT=1 docker build \
    --build-arg CUDA_VERSION=12.9.1 \
    --build-arg PYTHON_VERSION=3.12 \
    --build-arg torch_cuda_arch_list="7.0 7.5 8.0 8.9 9.0 10.0 12.0" \
    --build-arg max_jobs=8 \
    --build-arg nvcc_threads=2 \
    --tag vllm/vllm-openai:custom \
    --target vllm-openai \
    -f docker/Dockerfile \
    .
```

**Key build arguments:**

| Argument | Default | Description |
|---|---|---|
| `CUDA_VERSION` | `12.9.1` | CUDA version for the build and runtime images |
| `PYTHON_VERSION` | `3.12` | Python version inside the container |
| `torch_cuda_arch_list` | `7.0 7.5 8.0 8.9 9.0 10.0 12.0` | GPU architectures to compile kernels for |
| `max_jobs` | (auto) | Parallel compilation jobs |
| `nvcc_threads` | `2` | NVCC threads per job |
| `BUILD_BASE_IMAGE` | `nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04` | Base image for compilation |
| `FINAL_BASE_IMAGE` | `nvidia/cuda:${CUDA_VERSION}-base-ubuntu22.04` | Slim runtime base image |
| `INSTALL_KV_CONNECTORS` | `false` | Bundle KV-connector libraries (LMCache, NIXL) |

!!! tip "Targeting specific GPU architectures"
    Compiling only for your GPU architecture significantly reduces build time and
    image size. For example, for H100 only: `--build-arg torch_cuda_arch_list="9.0"`.

#### Build for NVIDIA Hopper / Blackwell (GH200, GB200)

```bash
DOCKER_BUILDKIT=1 docker build \
    --build-arg CUDA_VERSION=12.9.1 \
    --build-arg max_jobs=66 \
    --build-arg nvcc_threads=2 \
    --build-arg torch_cuda_arch_list="9.0 10.0+PTX" \
    --build-arg RUN_WHEEL_CHECK=false \
    --tag vllm/vllm-gh200-openai:latest \
    --target vllm-openai \
    -f docker/Dockerfile \
    .
```

For GB300 (Blackwell Ultra), use CUDA 13:

```bash
DOCKER_BUILDKIT=1 docker build \
    --build-arg CUDA_VERSION=13.0.1 \
    --build-arg BUILD_BASE_IMAGE=nvidia/cuda:13.0.1-devel-ubuntu22.04 \
    --build-arg max_jobs=256 \
    --build-arg nvcc_threads=2 \
    --build-arg RUN_WHEEL_CHECK=false \
    --build-arg torch_cuda_arch_list="9.0 10.0+PTX" \
    --platform "linux/arm64" \
    --tag vllm/vllm-gb300-openai:latest \
    --target vllm-openai \
    -f docker/Dockerfile \
    .
```

!!! note "Cross-compiling for ARM64 on x86"
    If building a `linux/arm64` image on an x86 host, register QEMU first:
    ```bash
    docker run --rm --privileged multiarch/qemu-user-static --reset -p yes
    ```

#### Run the custom image

```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --env "HF_TOKEN=${HF_TOKEN}" \
    vllm/vllm-openai:custom \
    --model meta-llama/Llama-3.1-8B-Instruct
```

---

## Option 3 — Build from Source

### Python-only build (no recompilation)

If you only need to modify Python code, use the precompiled build mode. Changes
to `.py` files are reflected immediately without recompiling C++/CUDA:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

This downloads the pre-built wheel for the current branch's base commit and
reuses its compiled libraries.

**Environment variables for precompiled builds:**

| Variable | Description |
|---|---|
| `VLLM_USE_PRECOMPILED` | Set to `1` to enable precompiled mode |
| `VLLM_PRECOMPILED_WHEEL_COMMIT` | Override the commit hash (or `nightly`) |
| `VLLM_PRECOMPILED_WHEEL_LOCATION` | Exact URL or local path to a wheel file |
| `VLLM_PRECOMPILED_WHEEL_VARIANT` | Variant subdirectory, e.g. `cu129`, `cpu` |

!!! note
    If you rebase your branch, re-run the install command to refresh the compiled
    libraries.

### Full build (with C++/CUDA compilation)

Use this when modifying C++ or CUDA kernel code:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv pip install -e .
```

This compiles all CUDA kernels and can take **10–30 minutes** depending on your
hardware and the number of target GPU architectures.

!!! tip "Speed up with ccache"
    Install [ccache](https://ccache.dev/) to cache compilation results:
    ```bash
    # Ubuntu / Debian
    sudo apt install ccache

    # conda
    conda install ccache
    ```
    Once `ccache` is on your `PATH`, the build system uses it automatically.
    For editable installs, use:
    ```bash
    CCACHE_NOHASHDIR="true" pip install --no-build-isolation -e .
    ```

!!! tip "Speed up with sccache (remote cache)"
    [sccache](https://github.com/mozilla/sccache) supports S3-backed remote caches:
    ```bash
    SCCACHE_BUCKET=vllm-build-sccache \
    SCCACHE_REGION=us-west-2 \
    SCCACHE_S3_NO_CREDENTIALS=1 \
    SCCACHE_IDLE_TIMEOUT=0 \
    uv pip install -e .
    ```

#### Use an existing PyTorch installation

If you have a custom PyTorch build (nightly, ROCm, etc.) and want to build vLLM
against it without reinstalling PyTorch:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
python use_existing_torch.py --prefix
uv pip install -e . --no-build-isolation
```

#### Build a portable wheel

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
uv build --wheel
pip install dist/vllm-*.whl
```

---

## Supported Features

See the [Feature × Hardware compatibility matrix](../../features/README.md#feature-x-hardware)
for a full list of features supported on NVIDIA CUDA.

---

## Troubleshooting

??? question "NCCL issues when using conda"
    PyTorch installed via `conda` statically links NCCL, which can conflict with
    vLLM's dynamic NCCL usage. Use `uv` or `pip` (not `conda install`) to install
    PyTorch. See [GitHub #8420](https://github.com/vllm-project/vllm/issues/8420).

??? question "Binary incompatibility errors"
    vLLM wheels are compiled for a specific CUDA version. If you see errors like
    `undefined symbol` or `CUDA error: no kernel image is available`, ensure your
    CUDA driver version matches the wheel's CUDA version. Use
    `--torch-backend=auto` to let `uv` select the right wheel automatically.

??? question "CUDA forward compatibility"
    On datacenter GPUs with older drivers, enable CUDA forward compatibility:
    ```bash
    export VLLM_ENABLE_CUDA_COMPATIBILITY=1
    ```

??? question "Out-of-memory errors"
    Reduce `--gpu-memory-utilization` (default `0.90`) or use
    `--max-model-len` to limit the context window. See the
    [memory conservation guide](../../configuration/conserving_memory.md).

---

## Next Steps

- **[First Inference](../first-inference.md)** — Run your first generation
- **[OpenAI-Compatible Server](../../serving/openai_compatible_server.md)** — Serve models via HTTP
- **[Configuration](../../configuration/engine_args.md)** — Tune engine arguments
- **[Parallelism & Scaling](../../serving/parallelism_scaling.md)** — Multi-GPU and multi-node
