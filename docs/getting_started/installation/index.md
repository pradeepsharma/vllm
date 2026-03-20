---
description: >
  vLLM installation overview — choose the right guide for your hardware platform.
  Supports NVIDIA CUDA, AMD ROCm, Intel XPU/HPU, Google TPU, Huawei Ascend NPU,
  CPU-only (x86, ARM, Apple Silicon, IBM Z), and more.
---

# Installation Overview

vLLM supports a wide range of hardware platforms. Use this page to find the
right installation guide for your hardware.

---

## Platform Comparison

| Platform | Method | Pre-built Wheels | Docker Image | From Source |
|---|---|---|---|---|
| **NVIDIA GPU (CUDA)** | pip / Docker | ✅ | ✅ | ✅ |
| **AMD GPU (ROCm)** | pip / Docker | ✅ (ROCm 7.0) | ✅ | ✅ |
| **Intel GPU (XPU)** | Docker / source | ❌ | ✅ | ✅ |
| **Google TPU** | pip / Docker | ✅ (`vllm-tpu`) | ✅ | ✅ |
| **Intel Gaudi HPU** | Docker / source | ❌ | ✅ (plugin) | ✅ |
| **Huawei Ascend NPU** | Docker / source | ❌ | ✅ (plugin) | ✅ |
| **CPU — Intel/AMD x86** | pip / Docker | ✅ (AVX-512) | ✅ | ✅ |
| **CPU — ARM AArch64** | pip / source | ✅ | ❌ | ✅ |
| **CPU — Apple Silicon** | source | ❌ | ❌ | ✅ |
| **CPU — IBM Z (s390x)** | source | ❌ | ❌ | ✅ |
| **CPU — IBM Power (ppc64le)** | source | ❌ | ❌ | ✅ |

---

## Quick Install (NVIDIA CUDA)

If you have an NVIDIA GPU, get started in seconds:

```bash
# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install vLLM
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

`--torch-backend=auto` automatically selects the right PyTorch CUDA index
for your driver version.

---

## Installation Guides by Platform

### GPU Platforms

<div class="grid cards" markdown>

-   :simple-nvidia: **NVIDIA GPU (CUDA)**

    ---

    Pre-built wheels for CUDA 12.8, 12.9, and 13.0. Supports Volta through
    Blackwell (compute capability 7.0+).

    [:octicons-arrow-right-24: CUDA Installation](gpu-cuda.md)

-   :simple-amd: **AMD GPU (ROCm)**

    ---

    Pre-built wheels for ROCm 7.0. Supports MI200, MI300, MI350, and
    Radeon RX 7000/9000 series.

    [:octicons-arrow-right-24: ROCm Installation](gpu-rocm.md)

-   :simple-intel: **Intel GPU (XPU)**

    ---

    Docker and source builds for Intel Data Center GPU and Intel Arc.
    Requires oneAPI 2025.3.

    [:octicons-arrow-right-24: XPU Installation](xpu.md)

</div>

### Accelerator Platforms

<div class="grid cards" markdown>

-   :simple-google: **Google TPU**

    ---

    Install via `pip install vllm-tpu` or Docker. Supports TPU v4, v5e,
    v5p, and v6e (Trillium).

    [:octicons-arrow-right-24: TPU Installation](tpu.md)

-   :simple-intel: **Intel Gaudi HPU**

    ---

    Plugin-based installation via `vllm-gaudi`. Supports Gaudi 2 and
    Gaudi 3 accelerators.

    [:octicons-arrow-right-24: HPU Installation](hpu.md)

-   **Huawei Ascend NPU**

    ---

    Plugin-based installation via `vllm-ascend`. Supports Ascend 910B
    and 910C series.

    [:octicons-arrow-right-24: NPU Installation](npu.md)

</div>

### CPU Platforms

<div class="grid cards" markdown>

-   :material-cpu-64-bit: **CPU — All Architectures**

    ---

    x86 (AVX-512), ARM AArch64, IBM Z (s390x), and IBM Power (ppc64le).
    Pre-built wheels for x86 and ARM.

    [:octicons-arrow-right-24: CPU Installation](cpu.md)

-   :simple-apple: **macOS Apple Silicon**

    ---

    Experimental support for M1/M2/M3/M4 chips. Build from source.
    Optional GPU acceleration via `vllm-metal`.

    [:octicons-arrow-right-24: macOS Installation](macos.md)

</div>

### Build from Source

<div class="grid cards" markdown>

-   :material-hammer-wrench: **Build from Source**

    ---

    Full CMake build guide for all backends. Includes ccache, sccache,
    incremental builds, and cross-compilation.

    [:octicons-arrow-right-24: From-Source Guide](from-source.md)

</div>

---

## System Requirements

### Common Requirements

| Requirement | Value |
|---|---|
| **Python** | 3.10 – 3.13 (3.12 recommended) |
| **Operating System** | Linux (all platforms); macOS 14+ (Apple Silicon CPU only) |

!!! warning "Windows"
    vLLM does not support Windows natively. Use
    [WSL2](https://learn.microsoft.com/en-us/windows/wsl/) with Ubuntu, or
    refer to community-maintained forks such as
    [vllm-windows](https://github.com/SystemPanic/vllm-windows).

### GPU Requirements

| Platform | Driver / Runtime | Minimum Compute |
|---|---|---|
| **NVIDIA CUDA** | CUDA 12.1+, NVIDIA driver ≥ 525 | Compute capability 7.0 (Volta) |
| **AMD ROCm** | ROCm 6.3+, glibc ≥ 2.35 | gfx90a (MI200) or newer |
| **Intel XPU** | Intel GPU driver, oneAPI 2025.3 | Intel Data Center GPU Max |
| **Google TPU** | TPU VM with SynapseAI | TPU v4 or newer |
| **Intel Gaudi** | SynapseAI 1.19+ | Gaudi 2 or newer |
| **Huawei Ascend** | CANN toolkit | Ascend 910B or newer |

### CPU Requirements

| Platform | Notes |
|---|---|
| **Intel / AMD x86** | AVX-512 recommended; AVX2 supported |
| **ARM AArch64** | NEON SIMD required |
| **Apple Silicon** | macOS 14+, Xcode 15.4+, Apple Clang 15+ |
| **IBM Z (s390x)** | VXE required (Z14 or later), FP32 only |
| **IBM Power (ppc64le)** | GCC 12.3+ recommended |

---

## Hardware Plugins

vLLM supports third-party hardware accelerators through its plugin system.
These plugins live outside the main repository and follow the
[Hardware-Pluggable RFC](../../design/plugin_system.md).

| Plugin | Hardware | Repository |
|---|---|---|
| `vllm-gaudi` | Intel Gaudi 2/3 | [vllm-project/vllm-gaudi](https://github.com/vllm-project/vllm-gaudi) |
| `vllm-ascend` | Huawei Ascend 910B/C | [vllm-project/vllm-ascend](https://github.com/vllm-project/vllm-ascend) |
| `vllm-metal` | Apple Silicon (Metal/MLX) | [vllm-project/vllm-metal](https://github.com/vllm-project/vllm-metal) |
| `vllm-tpu` | Google Cloud TPU | [PyPI: vllm-tpu](https://pypi.org/project/vllm-tpu/) |

For the full list of supported hardware, see [vllm.ai/#hardware](https://vllm.ai/#hardware).

---

## After Installation

Once vLLM is installed, run your first inference:

```bash
# Quick test — offline inference
python3 -c "
from vllm import LLM, SamplingParams
llm = LLM(model='facebook/opt-125m')
outputs = llm.generate(['Hello, vLLM!'], SamplingParams(max_tokens=50))
print(outputs[0].outputs[0].text)
"
```

Or start the OpenAI-compatible server:

```bash
vllm serve facebook/opt-125m --port 8000
```

Then query it:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "facebook/opt-125m", "prompt": "Hello", "max_tokens": 20}'
```

---

## Next Steps

- **[First Inference](../first-inference.md)** — Step-by-step first inference walkthrough
- **[Quickstart](../quickstart.md)** — Offline and online inference in 5 minutes
- **[OpenAI-Compatible Server](../../serving/openai_compatible_server.md)** — Production serving
- **[Configuration](../../configuration/engine_args.md)** — Engine arguments reference
