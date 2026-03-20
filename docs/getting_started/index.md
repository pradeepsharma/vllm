---
description: >
  Choose your vLLM installation path — GPU, CPU, Docker, or source — and get
  up and running in minutes with the right guide for your hardware and use case.
---

# Getting Started with vLLM

Welcome to vLLM! This page helps you find the right installation path and points you to the resources you need to go from zero to a running inference server as quickly as possible.

---

## :compass: Which Path Is Right for You?

Use the decision tree below to find your installation guide. Answer each question in order and follow the branch that matches your situation.

```
Do you have a GPU?
│
├── YES ──► What GPU vendor?
│           │
│           ├── NVIDIA (CUDA) ──► Install via pip (recommended)
│           │                     See: Installation › GPU › NVIDIA CUDA
│           │
│           ├── AMD (ROCm)    ──► Install via pip with ROCm index
│           │                     See: Installation › GPU › AMD ROCm
│           │
│           └── Intel (XPU)   ──► Install via pip with Intel index
│                                 See: Installation › GPU › Intel XPU
│
└── NO ───► What CPU architecture?
            │
            ├── Intel / AMD x86_64 ──► Build from source (optimized)
            │                          See: Installation › CPU › Intel/AMD x86
            │
            ├── ARM AArch64        ──► Build from source
            │                          See: Installation › CPU › ARM AArch64
            │
            ├── Apple Silicon (M-series) ──► Build from source (macOS)
            │                               See: Installation › CPU › Apple Silicon
            │
            ├── IBM Z (S390X)      ──► Build from source
            │                          See: Installation › CPU › IBM Z
            │
            └── Google TPU         ──► Install vllm-tpu package
                                       See: vLLM on TPU docs
```

---

## :package: Installation Guides

### GPU Installation

| Platform | Recommended Method | Guide |
|---|---|---|
| **NVIDIA CUDA** | `pip install vllm` | [GPU › NVIDIA CUDA](installation/gpu.md#nvidia-cuda) |
| **AMD ROCm** | `pip install vllm` + ROCm index | [GPU › AMD ROCm](installation/gpu.md#amd-rocm) |
| **Intel XPU** | `pip install vllm` + Intel index | [GPU › Intel XPU](installation/gpu.md#intel-xpu) |

### CPU Installation

| Platform | Recommended Method | Guide |
|---|---|---|
| **Intel / AMD x86** | Build from source | [CPU › Intel/AMD x86](installation/cpu.md#intelamd-x86) |
| **ARM AArch64** | Build from source | [CPU › ARM AArch64](installation/cpu.md#arm-aarch64) |
| **Apple Silicon** | Build from source | [CPU › Apple Silicon](installation/cpu.md#apple-silicon) |
| **IBM Z (S390X)** | Build from source | [CPU › IBM Z](installation/cpu.md#ibm-z-s390x) |

### Hardware Plugins

vLLM supports third-party hardware accelerators through its plugin system. These live outside the main repository and follow the [Hardware-Pluggable RFC](../design/plugin_system.md).

Supported hardware plugins include Intel Gaudi, IBM Spyre, Huawei Ascend, and more. See the full list at [vllm.ai/#hardware](https://vllm.ai/#hardware).

---

## :zap: Quick Install (NVIDIA CUDA)

If you have an NVIDIA GPU and just want to get started immediately:

```bash
# Recommended: use uv for fast environment management
uv venv --python 3.12 --seed
source .venv/bin/activate
uv pip install vllm --torch-backend=auto
```

Or with pip directly:

```bash
pip install vllm
```

!!! note "Python version"
    vLLM requires **Python 3.10 – 3.13**. We recommend Python 3.12 for best compatibility.

!!! tip "Using uv"
    [uv](https://docs.astral.sh/uv/) is a very fast Python package manager that automatically selects the right PyTorch CUDA index for your driver version via `--torch-backend=auto`. It is the recommended way to install vLLM.

---

## :whale: Docker (No Local Install Required)

If you prefer not to install vLLM locally, you can run it directly from a pre-built Docker image:

```bash
# NVIDIA CUDA
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 \
    --ipc=host \
    vllm/vllm-openai:latest \
    --model mistralai/Mistral-7B-Instruct-v0.1
```

See the [Docker deployment guide](../deployment/docker.md) for full details, including AMD ROCm images.

---

## :books: After Installation

Once vLLM is installed, choose your next step:

<div class="grid cards" markdown>

-   :material-rocket-launch: **Quickstart**

    ---

    Run your first offline inference job and start an OpenAI-compatible server in under 5 minutes.

    [:octicons-arrow-right-24: Quickstart guide](quickstart.md)

-   :material-server: **OpenAI-Compatible Server**

    ---

    Serve any supported model with a production-ready HTTP API compatible with OpenAI clients.

    [:octicons-arrow-right-24: Server guide](../serving/openai_compatible_server.md)

-   :material-code-braces: **Offline Batch Inference**

    ---

    Use the `LLM` Python class to run high-throughput batch inference without a server.

    [:octicons-arrow-right-24: Offline inference](../usage/offline_inference.md)

-   :material-tune: **Configuration**

    ---

    Tune engine arguments, environment variables, and memory settings for your workload.

    [:octicons-arrow-right-24: Configuration](../configuration/index.md)

</div>

---

## :white_check_mark: System Requirements

### Common Requirements

| Requirement | Value |
|---|---|
| **Python** | 3.10 – 3.13 |
| **Operating System** | Linux (recommended); macOS for Apple Silicon CPU builds |

!!! warning "Windows"
    vLLM does not support Windows natively. Use [WSL2](https://learn.microsoft.com/en-us/windows/wsl/) with a compatible Linux distribution, or refer to community-maintained forks such as [vllm-windows](https://github.com/SystemPanic/vllm-windows).

### GPU-Specific Requirements

| Platform | Driver / Runtime | Minimum Compute |
|---|---|---|
| **NVIDIA CUDA** | CUDA 12.1+ | Volta (sm_70) or newer |
| **AMD ROCm** | ROCm 7.0+, glibc ≥ 2.35 | — |
| **Intel XPU** | Intel GPU driver | — |

### CPU-Specific Requirements

| Platform | Notes |
|---|---|
| **Intel / AMD x86** | AVX-512 recommended for best performance |
| **ARM AArch64** | NEON SIMD support required |
| **Apple Silicon** | macOS 13 (Ventura) or later |
| **IBM Z (S390X)** | z14 or later |

---

## :question: Frequently Asked Questions

??? question "Can I run vLLM without a GPU?"
    Yes! vLLM supports CPU-only inference on x86, ARM, Apple Silicon, and IBM Z platforms. Performance will be lower than GPU inference, but it is fully functional. See the [CPU installation guide](installation/cpu.md) for details.

??? question "Which Python version should I use?"
    Python **3.12** is recommended for the best compatibility with PyTorch and vLLM's dependencies. Python 3.10, 3.11, and 3.13 are also supported.

??? question "Do I need to build from source?"
    Only for CPU backends and some advanced GPU configurations. For NVIDIA CUDA and AMD ROCm, pre-built wheels are available via `pip`. See [Pre-built wheels](installation/gpu.md#pre-built-wheels) for details.

??? question "How do I install a specific version of vLLM?"
    ```bash
    pip install vllm==0.x.y
    ```
    Check [GitHub Releases](https://github.com/vllm-project/vllm/releases) for available versions.

??? question "Can I use vLLM with multiple GPUs?"
    Yes. vLLM supports tensor parallelism, pipeline parallelism, data parallelism, and expert parallelism across multiple GPUs and nodes. See the [Parallelism & Scaling guide](../serving/parallelism_scaling.md).

??? question "Where can I get help?"
    - **GitHub Issues**: [github.com/vllm-project/vllm/issues](https://github.com/vllm-project/vllm/issues)
    - **User Forum**: [discuss.vllm.ai](https://discuss.vllm.ai)
    - **Developer Slack**: [slack.vllm.ai](https://slack.vllm.ai)

---

## :link: Related Resources

- [vLLM Roadmap](https://roadmap.vllm.ai) — See what's coming next
- [GitHub Releases](https://github.com/vllm-project/vllm/releases) — Changelog and release notes
- [vLLM Blog](https://blog.vllm.ai) — Deep dives and announcements
- [vLLM Paper](https://arxiv.org/abs/2309.06180) — Original PagedAttention research (SOSP 2023)
