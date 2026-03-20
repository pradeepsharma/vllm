---
description: >
  Install vLLM for CPU-only inference — x86 (AVX-512), ARM AArch64, ppc64le,
  IBM Z (s390x), and Apple Silicon. Includes pip, Docker, and from-source guides.
toc_depth: 3
---

# CPU-Only Installation

vLLM supports CPU-only inference across multiple architectures. While CPU
performance is lower than GPU, it enables deployment on any Linux server or
macOS machine without specialized hardware.

Select your CPU architecture:

=== "Intel / AMD x86"

    vLLM supports x86 CPUs with FP32, FP16, and BF16 data types. AVX-512 is
    strongly recommended for best performance.

=== "ARM AArch64"

    vLLM supports ARM CPUs with NEON SIMD, FP32, FP16, and BF16 data types.

=== "Apple Silicon"

    vLLM has experimental support for macOS Apple Silicon (M-series). See the
    dedicated [macOS guide](macos.md) for full instructions.

=== "IBM Z (s390x)"

    vLLM has experimental support for IBM Z (s390x). FP32 only.

=== "IBM Power (ppc64le)"

    vLLM has experimental support for IBM Power (ppc64le).

---

## Requirements

| Requirement | Value |
|---|---|
| **Operating System** | Linux (all architectures); macOS 13+ (Apple Silicon) |
| **Python** | 3.10 – 3.13 |
| **Compiler** | GCC/G++ ≥ 12.3.0 (recommended) |

**Architecture-specific requirements:**

=== "Intel / AMD x86"

    | Requirement | Notes |
    |---|---|
    | **CPU flags** | `avx512f` recommended; `avx512_bf16`, `avx512_vnni` optional |
    | **Memory** | Sufficient RAM for model weights + KV cache |

    !!! tip
        Run `lscpu | grep -i avx` to check your CPU's instruction set support.
        AMD requires at least 4th-gen Zen 4 (Genoa) for AVX-512.

=== "ARM AArch64"

    | Requirement | Notes |
    |---|---|
    | **ISA** | NEON support required |
    | **Compiler** | GCC/G++ ≥ 12.3.0 recommended |

=== "Apple Silicon"

    | Requirement | Notes |
    |---|---|
    | **OS** | macOS Sonoma (14) or later |
    | **SDK** | Xcode 15.4+ with Command Line Tools |
    | **Compiler** | Apple Clang ≥ 15.0.0 |

    See the [macOS guide](macos.md) for full instructions.

=== "IBM Z (s390x)"

    | Requirement | Notes |
    |---|---|
    | **ISA** | VXE support required (IBM Z14 or later) |
    | **Compiler** | GCC/G++ ≥ 12.3.0 |
    | **Data types** | FP32 only |

=== "IBM Power (ppc64le)"

    | Requirement | Notes |
    |---|---|
    | **OS** | Linux |
    | **Compiler** | GCC/G++ ≥ 12.3.0 |

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

### Install pre-built wheels

=== "Intel / AMD x86"

    Pre-built wheels with AVX-512 are available since vLLM 0.13.0:

    ```bash
    export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest \
        | jq -r .tag_name | sed 's/^v//')

    uv pip install \
        "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_x86_64.whl" \
        --torch-backend cpu
    ```

    ??? console "pip alternative"
        ```bash
        pip install \
            "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_x86_64.whl" \
            --extra-index-url https://download.pytorch.org/whl/cpu
        ```

    !!! warning "Set `LD_PRELOAD` after wheel install"
        TCMalloc and Intel OpenMP must be on `LD_PRELOAD` for optimal performance:
        ```bash
        sudo apt-get install -y libtcmalloc-minimal4

        TC_PATH=$(find / -name "libtcmalloc_minimal.so.4" 2>/dev/null | head -1)
        IOMP_PATH=$(find / -name "libiomp5.so" 2>/dev/null | head -1)
        export LD_PRELOAD="${TC_PATH}:${IOMP_PATH}:${LD_PRELOAD}"
        ```

    **Install the latest nightly:**

    ```bash
    uv pip install vllm \
        --extra-index-url https://wheels.vllm.ai/nightly/cpu \
        --index-strategy first-index \
        --torch-backend cpu
    ```

    **Install a specific commit:**

    ```bash
    export VLLM_COMMIT=730bd35378bf2a5b56b6d3a45be28b3092d26519
    uv pip install vllm \
        --extra-index-url "https://wheels.vllm.ai/${VLLM_COMMIT}/cpu" \
        --index-strategy first-index \
        --torch-backend cpu
    ```

=== "ARM AArch64"

    Pre-built ARM wheels are available since vLLM 0.11.2:

    ```bash
    export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest \
        | jq -r .tag_name | sed 's/^v//')

    uv pip install \
        "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cpu-cp38-abi3-manylinux_2_35_aarch64.whl"
    ```

    !!! warning "Set `LD_PRELOAD` after wheel install"
        ```bash
        sudo apt-get install -y libtcmalloc-minimal4
        TC_PATH=$(find / -name "libtcmalloc_minimal.so.4" 2>/dev/null | head -1)
        export LD_PRELOAD="${TC_PATH}:${LD_PRELOAD}"
        ```

    **Install the latest nightly:**

    ```bash
    uv pip install vllm \
        --extra-index-url https://wheels.vllm.ai/nightly/cpu \
        --index-strategy first-index
    ```

=== "Apple Silicon"

    No pre-built wheels are available for Apple Silicon. See the
    [macOS guide](macos.md) for build-from-source instructions.

=== "IBM Z (s390x)"

    No pre-built wheels are available for IBM Z. See the
    [build from source](#option-2--build-from-source) section below.

=== "IBM Power (ppc64le)"

    No pre-built wheels are available for IBM Power. See the
    [build from source](#option-2--build-from-source) section below.

---

## Option 2 — Build from Source

### Install system dependencies

=== "Intel / AMD x86"

    ```bash
    sudo apt-get update -y
    sudo apt-get install -y \
        gcc-12 g++-12 libnuma-dev libtcmalloc-minimal4 \
        ccache git curl wget ca-certificates

    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-12
    ```

=== "ARM AArch64"

    ```bash
    sudo apt-get update -y
    sudo apt-get install -y \
        gcc-12 g++-12 libnuma-dev libtcmalloc-minimal4 \
        ccache git curl wget ca-certificates

    sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 10 \
        --slave /usr/bin/g++ g++ /usr/bin/g++-12
    ```

=== "IBM Z (s390x)"

    On RHEL 9.4:
    ```bash
    dnf install -y \
        which procps findutils tar vim git gcc g++ make patch cython \
        zlib-devel libjpeg-turbo-devel libtiff-devel libpng-devel \
        libwebp-devel freetype-devel harfbuzz-devel openssl-devel \
        openblas openblas-devel wget autoconf automake libtool cmake numactl-devel

    # Install Rust (required for outlines-core and uvloop)
    curl https://sh.rustup.rs -sSf | sh -s -- -y
    source "$HOME/.cargo/env"
    ```

### Create a Python environment

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 --seed
source .venv/bin/activate
```

### Clone and install vLLM

=== "Intel / AMD x86"

    ```bash
    git clone https://github.com/vllm-project/vllm.git
    cd vllm

    # Install build and runtime dependencies
    uv pip install -r requirements/cpu-build.txt --torch-backend cpu
    uv pip install -r requirements/cpu.txt --torch-backend cpu

    # Build and install (editable for development)
    VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation
    ```

    ??? console "pip alternative"
        ```bash
        pip install --upgrade pip
        pip install -r requirements/cpu-build.txt \
            --extra-index-url https://download.pytorch.org/whl/cpu
        pip install -r requirements/cpu.txt \
            --extra-index-url https://download.pytorch.org/whl/cpu
        VLLM_TARGET_DEVICE=cpu pip install -e . --no-build-isolation
        ```

    **Build a portable wheel:**

    ```bash
    VLLM_TARGET_DEVICE=cpu uv build --wheel
    uv pip install dist/*.whl
    ```

=== "ARM AArch64"

    ```bash
    git clone https://github.com/vllm-project/vllm.git
    cd vllm

    uv pip install -r requirements/cpu-build.txt
    uv pip install -r requirements/cpu.txt

    VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation
    ```

=== "IBM Z (s390x)"

    ```bash
    git clone https://github.com/vllm-project/vllm.git
    cd vllm

    # Remove torch from build requirements (use nightly builds)
    sed -i '/^torch/d' requirements/build.txt

    uv pip install -v \
        --torch-backend auto \
        -r requirements/build.txt \
        -r requirements/cpu.txt

    VLLM_TARGET_DEVICE=cpu python setup.py bdist_wheel
    uv pip install dist/*.whl
    ```

=== "IBM Power (ppc64le)"

    ```bash
    git clone https://github.com/vllm-project/vllm.git
    cd vllm

    uv pip install -r requirements/cpu-build.txt
    uv pip install -r requirements/cpu.txt

    VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation
    ```

### Python-only build (no recompilation)

If you only need to modify Python code, use the precompiled build mode:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 VLLM_PRECOMPILED_WHEEL_VARIANT=cpu VLLM_TARGET_DEVICE=cpu \
    uv pip install --editable .
```

---

## Option 3 — Docker

### Pull pre-built images

=== "Intel / AMD x86"

    ```bash
    # Latest image
    docker pull vllm/vllm-openai-cpu:latest-x86_64

    # Specific version
    export VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest \
        | jq -r .tag_name | sed 's/^v//')
    docker pull "vllm/vllm-openai-cpu:v${VLLM_VERSION}-x86_64"
    ```

    All tags: [hub.docker.com/r/vllm/vllm-openai-cpu/tags](https://hub.docker.com/r/vllm/vllm-openai-cpu/tags)

    ```bash
    docker run \
        -v ~/.cache/huggingface:/root/.cache/huggingface \
        -p 8000:8000 \
        --env "HF_TOKEN=${HF_TOKEN}" \
        vllm/vllm-openai-cpu:latest-x86_64 \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --dtype bfloat16
    ```

    !!! warning
        Pre-built images require AVX-512. On CPUs without AVX-512, build a
        custom image (see below).

=== "ARM AArch64"

    No pre-built ARM Docker images are available. Build from source below.

=== "IBM Z (s390x)"

    No pre-built IBM Z Docker images are available. Build from source below.

### Build a custom Docker image

=== "Intel / AMD x86"

    ```bash
    # Auto-detect CPU capabilities (default)
    docker build -f docker/Dockerfile.cpu \
        --tag vllm-cpu-env \
        --target vllm-openai \
        .

    # Cross-compile for AVX-512
    docker build -f docker/Dockerfile.cpu \
        --build-arg VLLM_CPU_AVX512=true \
        --build-arg VLLM_CPU_AVX512BF16=true \
        --build-arg VLLM_CPU_AVX512VNNI=true \
        --tag vllm-cpu-avx512 \
        --target vllm-openai \
        .

    # Build without AVX-512 (for older CPUs)
    docker build -f docker/Dockerfile.cpu \
        --build-arg VLLM_CPU_DISABLE_AVX512=true \
        --tag vllm-cpu-avx2 \
        --target vllm-openai \
        .
    ```

    **Available build arguments:**

    | Argument | Default | Description |
    |---|---|---|
    | `PYTHON_VERSION` | `3.12` | Python version |
    | `VLLM_CPU_DISABLE_AVX512` | `false` | Disable AVX-512 (use AVX2 fallback) |
    | `VLLM_CPU_AVX2` | `false` | Force-enable AVX2 (cross-compilation) |
    | `VLLM_CPU_AVX512` | `false` | Force-enable AVX-512 (cross-compilation) |
    | `VLLM_CPU_AVX512BF16` | `false` | Force-enable AVX-512 BF16 |
    | `VLLM_CPU_AVX512VNNI` | `false` | Force-enable AVX-512 VNNI |
    | `VLLM_CPU_AMXBF16` | `true` | Enable AMX BF16 (Intel 4th gen+) |

    **Run the image:**

    ```bash
    docker run --rm \
        --security-opt seccomp=unconfined \
        --cap-add SYS_NICE \
        --shm-size=4g \
        -p 8000:8000 \
        -e VLLM_CPU_KVCACHE_SPACE=40 \
        vllm-cpu-env \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --dtype bfloat16
    ```

=== "ARM AArch64"

    ```bash
    docker buildx build \
        --platform=linux/arm64 \
        -f docker/Dockerfile.cpu \
        --tag vllm-cpu-arm64 \
        --target vllm-openai \
        .
    ```

=== "IBM Z (s390x)"

    ```bash
    docker build -f docker/Dockerfile.s390x \
        --tag vllm-cpu-s390x \
        .

    docker run --rm \
        --privileged \
        --shm-size=4g \
        -p 8000:8000 \
        -e VLLM_CPU_KVCACHE_SPACE=40 \
        -e VLLM_CPU_OMP_THREADS_BIND=0-31 \
        vllm-cpu-s390x \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --dtype float
    ```

---

## Runtime Configuration

### Key environment variables

| Variable | Description | Default |
|---|---|---|
| `VLLM_CPU_KVCACHE_SPACE` | KV cache size in GiB (e.g., `40` = 40 GiB) | `0` |
| `VLLM_CPU_OMP_THREADS_BIND` | CPU cores for OpenMP threads (e.g., `0-31`) | `auto` |
| `VLLM_CPU_NUM_OF_RESERVED_CPU` | CPU cores reserved for the vLLM frontend | `None` |
| `CPU_VISIBLE_MEMORY_NODES` | NUMA memory nodes visible to vLLM workers | (all) |
| `VLLM_CPU_SGL_KERNEL` | Enable small-batch optimized kernels (x86, AMX) | `0` |

### Recommended serving configuration

```bash
# Reserve 1 CPU core for the frontend, use the rest for inference
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_NUM_OF_RESERVED_CPU=1

vllm serve meta-llama/Llama-3.2-1B-Instruct --dtype bfloat16
```

Or with explicit core binding (32-core system, reserve core 31):

```bash
export VLLM_CPU_KVCACHE_SPACE=40
export VLLM_CPU_OMP_THREADS_BIND=0-30

vllm serve meta-llama/Llama-3.2-1B-Instruct --dtype bfloat16
```

### Tensor parallel across NUMA nodes

```bash
# Determine the number of NUMA nodes
lscpu | grep "NUMA node(s):"

# Use tensor-parallel-size equal to the number of NUMA nodes
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --dtype bfloat16 \
    --tensor-parallel-size 2
```

---

## Supported Features

See the [Feature × Hardware compatibility matrix](../../features/README.md#feature-x-hardware)
for CPU-specific feature support.

**Supported quantization on CPU:**

| Quantization | x86 | ARM | s390x |
|---|---|---|---|
| AWQ | ✅ | ❌ | ❌ |
| GPTQ | ✅ | ❌ | ❌ |
| compressed-tensors INT8 W8A8 | ✅ | ❌ | ✅ |

---

## Troubleshooting

??? question "Which dtype should I use?"
    Use `--dtype bfloat16` for best performance on modern CPUs. FP16 has
    unstable support on CPU; BF16 is recommended if you see accuracy issues.

??? question "`get_mempolicy: Operation not permitted` in Docker"
    Add `--cap-add SYS_NICE --security-opt seccomp=unconfined` to your
    `docker run` command. In Kubernetes, add:
    ```yaml
    securityContext:
      seccompProfile:
        type: Unconfined
      capabilities:
        add: [SYS_NICE]
    ```

??? question "NumPy ≥ 2.0 error"
    Downgrade NumPy: `pip install "numpy<2.0"`

??? question "CMake picks up CUDA during CPU build"
    Add `CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON` to prevent CUDA detection:
    ```bash
    CMAKE_DISABLE_FIND_PACKAGE_CUDA=ON VLLM_TARGET_DEVICE=cpu pip install -e .
    ```

??? question "Illegal instruction error with pre-built images"
    The pre-built x86 images require AVX-512. Build a custom image with
    `--build-arg VLLM_CPU_DISABLE_AVX512=true` for older CPUs.

---

## Community & Support

- **Slack**: `#sig-cpu` channel at [slack.vllm.ai](https://slack.vllm.ai/)
- **GitHub Issues**: Add `[CPU Backend]` to the issue title

---

## Next Steps

- **[First Inference](../first-inference.md)** — Run your first generation
- **[macOS (Apple Silicon)](macos.md)** — macOS-specific instructions
- **[Configuration](../../configuration/engine_args.md)** — Tune engine arguments
