---
description: >
  Complete reference for `vllm collect-env` — collecting system, GPU, Python,
  and vLLM environment information for debugging and bug reports.
---

# `vllm collect-env` — Environment Diagnostics Reference

`vllm collect-env` gathers comprehensive information about your system environment and prints it to stdout. It is the first tool to run when debugging installation issues, performance problems, or unexpected behavior — and the output should always be included when filing a bug report.

```
vllm collect-env
```

This command takes no arguments.

---

## :zap: Usage

```bash
vllm collect-env
```

To save the output to a file:

```bash
vllm collect-env > env_info.txt
```

To save and also view it:

```bash
vllm collect-env | tee env_info.txt
```

---

## :page_facing_up: Output Sections

`vllm collect-env` prints a structured report divided into the following sections:

### System Info

| Field | Description |
|---|---|
| **OS** | Operating system name and version (e.g., `Ubuntu 22.04.3 LTS (x86_64)`). |
| **GCC version** | Version of the GCC C compiler. |
| **Clang version** | Version of the Clang C compiler. |
| **CMake version** | Version of CMake. |
| **Libc version** | Version of the C standard library (Linux only). |

### PyTorch Info

| Field | Description |
|---|---|
| **PyTorch version** | Installed PyTorch version string. |
| **Is debug build** | Whether PyTorch was built in debug mode. |
| **CUDA used to build PyTorch** | CUDA version PyTorch was compiled against. |
| **ROCm used to build PyTorch** | HIP/ROCm version PyTorch was compiled against (AMD GPUs). |

### Python Environment

| Field | Description |
|---|---|
| **Python version** | Python version and runtime bitness (e.g., `3.12.3 (64-bit runtime)`). |
| **Python platform** | Detailed platform string from `platform.platform()`. |

### CUDA / GPU Info

| Field | Description |
|---|---|
| **Is CUDA available** | Whether `torch.cuda.is_available()` returns `True`. |
| **CUDA runtime version** | Installed CUDA runtime version. |
| **CUDA_MODULE_LOADING** | Value of the `CUDA_MODULE_LOADING` environment variable. |
| **GPU models and configuration** | GPU model names and memory (from `nvidia-smi`). |
| **Nvidia driver version** | NVIDIA driver version. |
| **cuDNN version** | cuDNN library version. |
| **HIP runtime version** | AMD ROCm HIP runtime version. |
| **MIOpen runtime version** | AMD MIOpen version. |
| **Is XNNPACK available** | Whether PyTorch's XNNPACK backend is enabled. |

### CPU Info

Detailed CPU information collected from the OS:

- **Linux**: `/proc/cpuinfo` (model name, cores, flags)
- **macOS**: `sysctl -n machdep.cpu.brand_string`
- **Windows**: `wmic cpu get` (name, manufacturer, clock speed, cache)

### Versions of Relevant Libraries

Filtered output of `pip list` (or `uv pip list`) showing packages relevant to vLLM:

- `torch`, `numpy`, `triton`, `optree`
- `transformers`, `tokenizers`
- `nvidia-*` packages (CUDA libraries)
- `flashinfer-python`, `helion`
- `nccl`, `pynvml`, `zmq`
- `onnx`, `mypy`, `flake8`

Also shows conda packages if a conda environment is active.

### vLLM Info

| Field | Description |
|---|---|
| **ROCm Version** | Installed ROCm version (AMD GPUs). |
| **vLLM Version** | Installed vLLM version. |
| **vLLM Build Flags** | Compile-time flags used when building vLLM (e.g., CUDA arch list, custom ops). |
| **GPU Topology** | Output of `nvidia-smi topo -m` showing NVLink and PCIe connectivity between GPUs. |

### Environment Variables

All active `VLLM_*` environment variables and their current values. This includes settings for:

- Logging (`VLLM_LOGGING_LEVEL`, `VLLM_LOGGING_CONFIG_PATH`)
- Memory and caching (`VLLM_CPU_KVCACHE_SPACE`, `VLLM_GPU_MEMORY_UTILIZATION`)
- Attention backends (`VLLM_ATTENTION_BACKEND`)
- Worker and executor settings (`VLLM_WORKER_MULTIPROC_METHOD`)
- Feature flags and developer options (`VLLM_SERVER_DEV_MODE`, `VLLM_ALLOW_RUNTIME_LORA_UPDATING`)
- Tracing and observability (`VLLM_TRACE_FUNCTION`)

---

## :clipboard: Example Output

```
Collecting environment information...

==============================
        System Info
==============================
OS                           : Ubuntu 22.04.3 LTS (x86_64)
GCC version                  : (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Clang version                : Could not collect
CMake version                : version 3.26.4
Libc version                 : glibc-2.35

==============================
       PyTorch Info
==============================
PyTorch version              : 2.6.0+cu124
Is debug build               : No
CUDA used to build PyTorch   : 12.4
ROCm used to build PyTorch   : N/A

==============================
      Python Environment
==============================
Python version               : 3.12.3 (64-bit runtime)
Python platform              : Linux-5.15.0-91-generic-x86_64-with-glibc2.35

==============================
       CUDA / GPU Info
==============================
Is CUDA available            : Yes
CUDA runtime version         : 12.4.131
CUDA_MODULE_LOADING set to   :
GPU models and configuration :
  GPU 0: NVIDIA A100-SXM4-80GB
  GPU 1: NVIDIA A100-SXM4-80GB
  GPU 2: NVIDIA A100-SXM4-80GB
  GPU 3: NVIDIA A100-SXM4-80GB
Nvidia driver version        : 550.54.15
cuDNN version                : 90100
HIP runtime version          : N/A
MIOpen runtime version       : N/A
Is XNNPACK available         : Yes

==============================
          CPU Info
==============================
Architecture:                    x86_64
CPU(s):                          128
Model name:                      Intel(R) Xeon(R) Platinum 8380 CPU @ 2.30GHz
...

==============================
Versions of relevant libraries
==============================
[pip3] flashinfer-python==0.2.3+cu124torch2.6
[pip3] numpy==2.2.3
[pip3] nvidia-cublas-cu12==12.4.5.8
[pip3] nvidia-cuda-runtime-cu12==12.4.127
[pip3] nvidia-cudnn-cu12==9.1.0.70
[pip3] torch==2.6.0+cu124
[pip3] transformers==4.50.0
[pip3] triton==3.2.0

==============================
         vLLM Info
==============================
ROCm Version                 : Could not collect
vLLM Version                 : 0.8.3
vLLM Build Flags:
  CUDA Arch: 8.0+PTX;8.6+PTX;9.0+PTX
  ...
GPU Topology:
        GPU0    GPU1    GPU2    GPU3    ...
  GPU0   X      NV12    NV12    NV12   ...
  GPU1  NV12     X      NV12    NV12   ...
  ...

==============================
     Environment Variables
==============================
VLLM_LOGGING_LEVEL = INFO
VLLM_ATTENTION_BACKEND = FLASH_ATTN
```

---

## :bug: Using Output for Bug Reports

When filing a GitHub issue, always include the full output of `vllm collect-env`. This helps maintainers quickly identify:

1. **Version mismatches** — e.g., PyTorch built against CUDA 12.1 but runtime is 12.4
2. **Missing dependencies** — e.g., FlashInfer not installed
3. **Driver issues** — e.g., NVIDIA driver too old for the installed CUDA version
4. **Configuration problems** — e.g., unexpected `VLLM_*` environment variables set
5. **Hardware topology** — e.g., GPUs connected via PCIe instead of NVLink

### Recommended bug report template

```markdown
## Environment

<paste output of `vllm collect-env` here>

## Problem Description

...

## Steps to Reproduce

...

## Expected Behavior

...

## Actual Behavior

...
```

---

## :lock: Privacy Considerations

The output of `vllm collect-env` may include:

- **Hostnames and paths** embedded in package versions or environment variables
- **API keys or tokens** if stored in `VLLM_*` environment variables (e.g., `VLLM_HF_TOKEN`)
- **Internal network addresses** if set in distributed configuration variables

Before sharing the output publicly, review it and redact any sensitive values. In particular, check the **Environment Variables** section for any secrets.

---

## :information_source: Implementation Notes

- The command is implemented in `vllm/collect_env.py`, adapted from PyTorch's `torch.utils.collect_env`.
- Package versions are collected by running `pip list --format=freeze` (or `uv pip list` in uv environments) and filtering for relevant package name patterns.
- GPU topology is collected via `nvidia-smi topo -m` (NVIDIA) or equivalent ROCm tools.
- If PyTorch is not installed, CUDA/GPU fields will show `N/A` or `Could not collect`.
- On Linux, if a PyTorch minidump file is detected in the crash handler directory, a warning is printed to stderr pointing to the dump file and its creation time.

---

## :link: Related

- [CLI Overview](index.md) — All vLLM CLI commands
- [GitHub Issues](https://github.com/vllm-project/vllm/issues) — File a bug report
- [Configuration reference](../configuration/index.md) — `VLLM_*` environment variables
