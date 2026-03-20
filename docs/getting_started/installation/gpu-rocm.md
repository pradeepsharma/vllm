---
description: >
  Install vLLM for AMD GPUs using ROCm — pip, Docker, and from-source instructions
  for MI200, MI300, MI350, and Radeon RX series.
toc_depth: 3
---

# AMD GPU (ROCm) Installation

vLLM supports AMD GPUs through the ROCm open-source GPU compute stack. Pre-built
wheels are available for ROCm 7.0, and official Docker images are published to
Docker Hub.

---

## Requirements

| Requirement | Value |
|---|---|
| **Operating System** | Linux (x86\_64) |
| **Python** | 3.10 – 3.13 (3.12 recommended for pre-built wheels) |
| **ROCm** | 6.3 or later (7.0 recommended) |
| **glibc** | ≥ 2.35 (for pre-built wheels) |

**Supported GPU architectures:**

| GPU Family | Architecture | Minimum ROCm |
|---|---|---|
| MI200 (MI210, MI250, MI250X) | gfx90a | 6.3 |
| MI300 (MI300X, MI300A) | gfx942 | 6.3 |
| MI350 | gfx950 | 7.0 |
| Radeon RX 7900 series | gfx1100 / gfx1101 | 6.3 |
| Radeon RX 9000 series | gfx1200 / gfx1201 | 7.0 |
| Ryzen AI MAX / AI 300 | gfx1151 / gfx1150 | 7.0.2 |

!!! tip "Check your GPU architecture"
    Run `rocminfo | grep gfx` to find your GPU's gfx architecture string.

---

## Option 1 — Install via pip

### Create a Python environment

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment
uv venv --python 3.12 --seed
source .venv/bin/activate
```

### Install vLLM

```bash
# Install the latest vLLM for ROCm 7.0
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/
```

!!! tip "Finding available ROCm versions"
    Browse [https://wheels.vllm.ai/rocm/](https://wheels.vllm.ai/rocm/) to see
    all available vLLM versions and their ROCm variants.

#### Install a specific version

```bash
# Install vLLM 0.15.0 for ROCm 7.0
uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/0.15.0/rocm700
```

!!! warning "Using pip instead of uv"
    `pip` merges all package indexes and picks the highest version, making it
    difficult to install from a custom index when exact versions are pinned.
    Use `uv` for ROCm installs. If you must use `pip`:
    ```bash
    pip install vllm==0.15.0+rocm700 \
        --extra-index-url https://wheels.vllm.ai/rocm/0.15.0/rocm700
    ```

### Verify the installation

```bash
python -c "import vllm; print(f'vLLM {vllm.__version__} installed successfully')"
```

---

## Option 2 — Docker

### Pull the official ROCm image

```bash
docker pull vllm/vllm-openai-rocm:latest
```

All tags are available at
[hub.docker.com/r/vllm/vllm-openai-rocm](https://hub.docker.com/r/vllm/vllm-openai-rocm/tags).

### Run the OpenAI-compatible server

```bash
docker run --rm \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=${HF_TOKEN}" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai-rocm:latest \
    --model Qwen/Qwen3-0.6B
```

| Flag | Purpose |
|---|---|
| `--group-add=video` | Access GPU render nodes |
| `--cap-add=SYS_PTRACE` | Required for ROCm profiling |
| `--security-opt seccomp=unconfined` | Required for ROCm system calls |
| `--device /dev/kfd` | Kernel Fusion Driver (GPU compute) |
| `--device /dev/dri` | Direct Rendering Infrastructure |
| `--ipc=host` | Shared memory for multi-GPU tensor parallel |

### AMD nightly images

AMD publishes nightly pre-built images optimized for MI300X:

```bash
docker pull rocm/vllm-dev:nightly

docker run -it --rm \
    --network=host \
    --group-add=video \
    --ipc=host \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v /path/to/your/models:/app/models \
    -e HF_HOME="/app/models" \
    rocm/vllm-dev:nightly
```

!!! tip
    See [LLM inference performance validation on AMD Instinct MI300X](https://rocm.docs.amd.com/en/latest/how-to/performance-validation/mi300x/vllm-benchmark.html)
    for benchmarking instructions with this image.

---

## Option 3 — Build from Source

### Prerequisites

Install ROCm and PyTorch before building vLLM. You can start from an AMD
PyTorch Docker image to skip this step:

```bash
# Start from AMD's PyTorch image (recommended)
docker pull rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0
docker run -it --rm \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    rocm/pytorch:rocm7.0_ubuntu22.04_py3.10_pytorch_release_2.8.0
```

Or install PyTorch manually:

```bash
pip uninstall torch -y
pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/rocm7.0
```

### Step 1 — Install Triton for ROCm

```bash
python3 -m pip install ninja cmake wheel pybind11
pip uninstall -y triton

git clone https://github.com/ROCm/triton.git
cd triton
git checkout f9e5bf54   # validated commit (see docker/Dockerfile.rocm_base)
if [ ! -f setup.py ]; then cd python; fi
python3 setup.py install
cd ../..
```

!!! note
    The validated Triton commit is tracked in
    [docker/Dockerfile.rocm_base](https://github.com/vllm-project/vllm/blob/main/docker/Dockerfile.rocm_base).

### Step 2 — (Optional) Install Flash Attention for ROCm

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git checkout 0e60e394   # validated commit (see docker/Dockerfile.rocm_base)
git submodule update --init

# Replace gfx942 with your GPU architecture
GPU_ARCHS="gfx942" python3 setup.py install
cd ..
```

### Step 3 — (Optional) Build AITER

AITER provides optimized AMD GPU kernels:

```bash
python3 -m pip uninstall -y aiter
git clone --recursive https://github.com/ROCm/aiter.git
cd aiter
git checkout $AITER_BRANCH_OR_COMMIT   # see docker/Dockerfile.rocm_base
git submodule sync
git submodule update --init --recursive
python3 setup.py develop
```

### Step 4 — Build vLLM

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm

pip install --upgrade pip
pip install /opt/rocm/share/amd_smi   # AMD SMI Python bindings
pip install --upgrade numba scipy huggingface-hub[cli,hf_transfer] setuptools_scm
pip install -r requirements/rocm.txt

# Build for a single architecture (faster, recommended)
export PYTORCH_ROCM_ARCH="gfx942"

# Or build for multiple architectures
# export PYTORCH_ROCM_ARCH="gfx90a;gfx942"

python3 setup.py develop
```

!!! note
    `pip install .` does not work for ROCm source builds. Use `setup.py develop`.
    This step can take **5–10 minutes**.

### Build a Docker image from source

```bash
# Enable BuildKit
export DOCKER_BUILDKIT=1

# Build the ROCm base image (optional — pre-built at rocm/vllm-dev:base)
docker build \
    -f docker/Dockerfile.rocm_base \
    -t rocm/vllm-dev:base .

# Build the vLLM ROCm image
docker build \
    -f docker/Dockerfile.rocm \
    -t vllm/vllm-openai-rocm .
```

**Build arguments for `docker/Dockerfile.rocm`:**

| Argument | Default | Description |
|---|---|---|
| `BASE_IMAGE` | `rocm/vllm-dev:base` | ROCm base image |
| `ARG_PYTORCH_ROCM_ARCH` | (from base) | Override GPU architecture list |
| `REMOTE_VLLM` | `0` | `1` = clone vLLM from GitHub instead of local copy |
| `VLLM_REPO` | vllm-project/vllm | Repository URL when `REMOTE_VLLM=1` |
| `VLLM_BRANCH` | `main` | Branch when `REMOTE_VLLM=1` |

#### Run the custom image

```bash
docker run --rm \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=${HF_TOKEN}" \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai-rocm \
    --model meta-llama/Llama-3.1-8B-Instruct
```

#### Interactive development session

```bash
docker run --rm -it \
    --group-add=video \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --device /dev/kfd \
    --device /dev/dri \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "HF_TOKEN=${HF_TOKEN}" \
    --network=host \
    --ipc=host \
    --entrypoint bash \
    vllm/vllm-openai-rocm
```

---

## Performance Tuning

!!! tip "MI300X optimization guide"
    For MI300X (gfx942) users, refer to the
    [MI300x tuning guide](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html)
    and the
    [vLLM performance optimization guide](https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/vllm-optimization.html).

**Key environment variables:**

| Variable | Description |
|---|---|
| `PYTORCH_ROCM_ARCH` | GPU architecture(s) to compile for (e.g., `gfx942`) |
| `HIP_FORCE_DEV_KERNARG` | Set to `1` for performance (enabled in Docker images) |
| `SAFETENSORS_FAST_GPU` | Set to `1` for faster tensor loading |
| `TOKENIZERS_PARALLELISM` | Set to `false` to suppress tokenizer warnings |

---

## Supported Features

See the [Feature × Hardware compatibility matrix](../../features/README.md#feature-x-hardware)
for a full list of features supported on AMD ROCm.

---

## Troubleshooting

??? question "ROCm version mismatch"
    The ROCm version of PyTorch should match your ROCm driver version. Run
    `rocm-smi --showversion` to check your driver version.

??? question "Kernel compilation errors"
    If you see errors during `python3 setup.py develop`, ensure your
    `PYTORCH_ROCM_ARCH` matches your GPU. Run `rocminfo | grep gfx` to find
    the correct architecture string.

??? question "Docker: /dev/kfd permission denied"
    Add your user to the `render` and `video` groups:
    ```bash
    sudo usermod -aG render,video $USER
    ```
    Then log out and back in.

---

## Community & Support

- **Slack**: `#sig-amd` channel at [slack.vllm.ai](https://slack.vllm.ai/)
- **GitHub Issues**: Add `[ROCm]` to the issue title for faster triage

---

## Next Steps

- **[First Inference](../first-inference.md)** — Run your first generation
- **[OpenAI-Compatible Server](../../serving/openai_compatible_server.md)** — Serve models via HTTP
- **[Parallelism & Scaling](../../serving/parallelism_scaling.md)** — Multi-GPU tensor parallel
