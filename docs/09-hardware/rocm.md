# AMD ROCm Platform

vLLM's AMD ROCm platform (`RocmPlatform`) provides GPU acceleration for AMD Instinct and Radeon GPUs using the HIP runtime. It shares the CUDA-like programming model and supports NCCL (via ROCm's HIP-NCCL), FlashAttention, and the AITER (AMD Inference Toolkit for Efficient Runtimes) kernel library.

## Source File

`vllm/platforms/rocm.py`

## Supported GPU Architectures

### CDNA (Compute DNA) — Data Center GPUs

| GFX Arch | GPU | Notes |
|----------|-----|-------|
| `gfx90a` | AMD Instinct MI200 series | CDNA2 |
| `gfx942` | AMD Instinct MI300X, MI300A, MI308X | CDNA3, primary target |
| `gfx950` | AMD Instinct MI325X | CDNA3+ |

### RDNA (Radeon DNA) — Consumer/Workstation GPUs

| GFX Arch | GPU | Notes |
|----------|-----|-------|
| `gfx1100` | Radeon RX 7900 XTX | RDNA3 |
| `gfx1101` | Radeon RX 7800 XT | RDNA3 |
| `gfx1200` | Radeon RX 9070 XT | RDNA4 |

### Architecture Detection

vLLM detects the GFX architecture at module load time using `amdsmi` (without initializing the CUDA/HIP context):

```python
_GCN_ARCH = _get_gcn_arch()  # e.g., "gfx942"

_ON_GFX1X = any(arch in _GCN_ARCH for arch in ["gfx11", "gfx12"])  # RDNA3/4
_ON_MI3XX = any(arch in _GCN_ARCH for arch in ["gfx942", "gfx950"])  # MI300/MI325
_ON_GFX9  = any(arch in _GCN_ARCH for arch in ["gfx90a", "gfx942", "gfx950"])
_ON_GFX942 = "gfx942" in _GCN_ARCH
_ON_GFX950 = "gfx950" in _GCN_ARCH
```

Helper functions are available for conditional logic:

```python
from vllm.platforms.rocm import on_gfx9, on_mi3xx, on_gfx1x, on_gfx942, on_gfx950
```

### GFX Capability Mapping

The `_capability_from_gcn_arch()` function maps GFX strings to `(major, minor)` tuples following HIP's `hipDeviceProp_t` convention:

| GFX String | major | minor |
|-----------|-------|-------|
| `gfx90a` | 9 | 0 |
| `gfx942` | 9 | 4 |
| `gfx950` | 9 | 5 |
| `gfx1100` | 11 | 0 |
| `gfx1200` | 12 | 0 |

## Key Platform Properties

```python
class RocmPlatform(Platform):
    _enum = PlatformEnum.ROCM
    device_name: str = "rocm"
    device_type: str = "cuda"   # ROCm uses "cuda" device type in PyTorch
    dispatch_key: str = "CUDA"
    ray_device_key: str = "GPU"
    dist_backend: str = "nccl"  # HIP-NCCL
    device_control_env_var: str = "CUDA_VISIBLE_DEVICES"
    ray_noset_device_env_vars: list[str] = [
        "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES",
        "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES",
    ]
```

> **Note**: ROCm uses `"cuda"` as the PyTorch device type because PyTorch's ROCm backend is a HIP-based CUDA compatibility layer. The `CUDA_VISIBLE_DEVICES` and `HIP_VISIBLE_DEVICES` environment variables are kept in sync automatically.

## Supported Quantization Methods

```python
supported_quantization: list[str] = [
    "awq",
    "awq_marlin",
    "gptq",
    "gptq_marlin",
    "fp8",
    "compressed-tensors",
    "fbgemm_fp8",
    "gguf",
    "quark",
    "ptpc_fp8",
    "mxfp4",
    "petit_nvfp4",
    "torchao",
    "bitsandbytes",
]
```

## Attention Backends

### Standard Attention (non-MLA)

Backend priority on ROCm depends on the GPU family:

**CDNA (gfx9 family — MI200/MI300/MI325)**:
1. `ROCM_AITER_UNIFIED_ATTN` (if AITER unified attention enabled)
2. `ROCM_AITER_FA` (AITER Flash Attention)
3. `FLASH_ATTN` (upstream FlashAttention)
4. `ROCM_CUSTOM_ATTN` (custom paged attention)
5. `TRITON_ATTN`
6. `TORCH_SDPA`

**RDNA (gfx11xx/gfx12xx — RX 7900/9070)**:
1. `FLASH_ATTN` (Triton-AMD backend, requires `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`)
2. `ROCM_CUSTOM_ATTN`
3. `TRITON_ATTN`
4. `TORCH_SDPA`

### MLA Attention (DeepSeek-style)

```python
# Sparse MLA (DSv3.2-style)
[AttentionBackendEnum.ROCM_AITER_MLA_SPARSE]

# Dense MLA priority
[
    AttentionBackendEnum.ROCM_AITER_MLA,
    AttentionBackendEnum.ROCM_AITER_TRITON_MLA,
    AttentionBackendEnum.TRITON_MLA,
]
```

### ViT Attention

For vision transformer models:
1. `ROCM_AITER_FA` (if AITER enabled and on gfx9)
2. `FLASH_ATTN` (on gfx9 with flash_attn installed)
3. `FLASH_ATTN` (on RDNA with Triton backend)
4. `TORCH_SDPA` (fallback)

## AITER Kernels

AITER (AMD Inference Toolkit for Efficient Runtimes) provides highly optimized kernels for AMD GPUs. It is enabled via environment variables:

```bash
VLLM_ROCM_USE_AITER=1                    # Enable AITER
VLLM_ROCM_USE_AITER_MHA=1               # AITER Flash Attention
VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 # AITER Unified Attention
```

AITER provides:
- **Flash Attention** (`ROCM_AITER_FA`): Optimized attention for gfx9 GPUs
- **Unified Attention** (`ROCM_AITER_UNIFIED_ATTN`): Combined prefill/decode attention
- **MLA** (`ROCM_AITER_MLA`): Multi-head Latent Attention for DeepSeek models
- **Fused MoE**: Optimized Mixture-of-Experts routing and computation
- **RMS Norm**: Fused RMS normalization
- **FP8 Linear**: FP8 matrix multiplication

```python
# From rocm.py check_and_update_config
from vllm._aiter_ops import rocm_aiter_ops

use_aiter_fused_moe    = rocm_aiter_ops.is_fused_moe_enabled()
use_aiter_rms_norm     = rocm_aiter_ops.is_rmsnorm_enabled()
use_aiter_fp8_linear   = rocm_aiter_ops.is_linear_fp8_enabled()
```

## Custom Paged Attention

The ROCm platform includes a custom paged attention kernel (`ROCM_CUSTOM_ATTN`) optimized for AMD hardware. It is enabled via `VLLM_ROCM_CUSTOM_PAGED_ATTN=1` and supports:

**CDNA (gfx9)**:
- dtypes: `float16`, `bfloat16`
- head sizes: 64, 128
- block sizes: 16, 32
- GQA ratios: 1–16
- max sequence length: 128K tokens

**RDNA (gfx11xx/gfx12xx)**:
- dtypes: `float16`, `bfloat16`
- head size: 128 only
- block size: 16 only
- GQA ratios: 3–16
- max sequence length: 128K tokens

## Flash Attention on RDNA

RDNA GPUs (gfx11xx/gfx12xx) use a Triton-based FlashAttention backend:

```python
def flash_attn_triton_available() -> bool:
    if not on_gfx1x():
        return False
    # Requires flash_attn with flash_attn_triton_amd module
    # and FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
    ...
```

To enable:
```bash
FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE vllm serve ...
```

## HIP/CUDA Environment Variable Sync

ROCm automatically synchronizes `HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` at import time:

```python
def _sync_hip_cuda_env_vars():
    hip_val  = os.environ.get("HIP_VISIBLE_DEVICES") or None
    cuda_val = os.environ.get("CUDA_VISIBLE_DEVICES") or None
    if hip_val and cuda_val and hip_val != cuda_val:
        raise ValueError("Inconsistent GPU visibility env vars")
    elif hip_val:
        os.environ["CUDA_VISIBLE_DEVICES"] = hip_val
    elif cuda_val:
        os.environ["HIP_VISIBLE_DEVICES"] = cuda_val
```

## Device ID Map

Known AMD GPU device IDs:

| PCI Device ID | GPU Model |
|--------------|-----------|
| `0x74a0` | AMD Instinct MI300A |
| `0x74a1` | AMD Instinct MI300X |
| `0x74b5` | AMD Instinct MI300X (VF) |
| `0x74a2` | AMD Instinct MI308X |
| `0x74a5` | AMD Instinct MI325X |
| `0x74b9` | AMD Instinct MI325X (VF) |
| `0x744c` | AMD Radeon RX 7900 XTX |

## Building the ROCm Docker Image

The ROCm Dockerfile (`docker/Dockerfile.rocm`) accepts the target GFX architecture as a build argument:

```bash
# Build for MI300X and MI325X
docker build \
  --build-arg ARG_PYTORCH_ROCM_ARCH='gfx942;gfx950' \
  --build-arg max_jobs=16 \
  -f docker/Dockerfile.rocm \
  --target test \
  -t rocm/vllm-ci:latest .
```

The `PYTORCH_ROCM_ARCH` environment variable controls which GFX architectures are compiled.

## ROCm-Specific C Extensions

The ROCm platform loads additional C extensions beyond the standard `vllm._C`:

```python
@classmethod
def import_kernels(cls) -> None:
    super().import_kernels()
    import vllm._rocm_C  # ROCm-specific kernels
```

## Distributed Communication

ROCm uses NCCL (via HIP-NCCL) for distributed operations, with the same `ProcessGroupNCCL` interface as CUDA:

```python
dist_backend: str = "nccl"
```

The `stateless_init_device_torch_dist_pg()` method creates NCCL process groups compatible with ROCm's HIP backend.

## Sleep Mode

The ROCm platform supports GPU sleep mode (power gating) on MI3xx GPUs:

```python
def is_sleep_mode_available(self) -> bool:
    return self._enum in (PlatformEnum.CUDA, PlatformEnum.ROCM)
```

## Related Pages

- [Platform Overview](overview.md)
- [NVIDIA CUDA Platform](cuda.md)
- [Hardware CI](hardware-ci.md)
