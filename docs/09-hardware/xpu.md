# Intel XPU Platform

vLLM's Intel XPU platform (`XPUPlatform`) provides GPU acceleration for Intel Data Center GPU Max series and Arc consumer GPUs using Intel's oneAPI/SYCL stack. It integrates with Intel Extension for PyTorch (IPEX) and uses XCCL for distributed communication.

## Source File

`vllm/platforms/xpu.py`

## Supported Hardware

| GPU Family | Examples | Notes |
|-----------|---------|-------|
| Intel Data Center GPU Max | Max 1100, Max 1550 | Primary target for inference |
| Intel Arc A-series | Arc A770 | Consumer GPU; BF16 has known accuracy issues |

> **Note**: Intel Arc A770 has a known BF16 accuracy issue. Use `--dtype=half` (float16) instead.

## Key Platform Properties

```python
class XPUPlatform(Platform):
    _enum = PlatformEnum.XPU
    device_name: str = "xpu"
    device_type: str = "xpu"
    dispatch_key: str = "XPU"
    ray_device_key: str = "GPU"   # Ray uses "GPU" for Intel XPU
    dist_backend: str = "xccl"   # Intel Collective Communications Library
    device_control_env_var: str = "ZE_AFFINITY_MASK"
```

## IPEX and Kernel Extensions

The XPU platform loads three kernel extension modules at import time:

```python
import vllm_xpu_kernels._C      # Core XPU kernels
import vllm_xpu_kernels._moe_C  # MoE (Mixture of Experts) kernels
import vllm_xpu_kernels._xpu_C  # XPU-specific ops
```

These are provided by the `vllm-xpu-kernels` package, which wraps Intel's SYCL/oneAPI kernels.

## Attention Backends

The XPU platform forces NHD (N=batch, H=heads, D=dim) KV cache layout, which is the only layout supported by XPU attention kernels:

```python
@classmethod
def get_attn_backend_cls(cls, selected_backend, attn_selector_config, num_heads=None):
    from vllm.v1.attention.backends.utils import set_kv_cache_layout
    set_kv_cache_layout("NHD")
    ...
```

### Backend Selection Logic

| Condition | Backend Selected |
|-----------|----------------|
| MLA enabled | `TRITON_MLA` |
| `TRITON_ATTN` explicitly selected | `TRITON_ATTN` |
| `dtype == float32` | `TRITON_ATTN` (Flash Attention doesn't support FP32) |
| `FLASH_ATTN` explicitly selected | `FLASH_ATTN` |
| Default | `FLASH_ATTN` |

### ViT Attention Backends

Supported backends for vision transformer models:
1. `FLASH_ATTN` (default)
2. `TRITON_ATTN`
3. `TORCH_SDPA`

## XCCL Distributed Communication

Intel XPU uses XCCL (Intel's Collective Communications Library) for multi-GPU communication:

```python
dist_backend: str = "xccl"
```

XCCL support is detected at runtime:

```python
from vllm.utils.torch_utils import supports_xccl

if supports_xccl():
    XPUPlatform.dist_backend = "xccl"
```

The device communicator class:

```python
@classmethod
def get_device_communicator_cls(cls) -> str:
    return "vllm.distributed.device_communicators.xpu_communicator.XpuCommunicator"
```

## XPU Graph Support

CUDA-style graph capture is supported on XPU when the PyTorch version supports it:

```python
@classmethod
def get_static_graph_wrapper_cls(cls) -> str:
    return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

@classmethod
def support_static_graph_mode(cls) -> bool:
    return True
```

However, there are constraints:
- XPU graphs are disabled when `world_size_across_dp > 1` (data parallel)
- Flash Attention SYCL-TLA kernels cannot be captured with full XPU graphs, so `PIECEWISE` graph mode is used instead

```python
if (attention_config.backend == AttentionBackendEnum.FLASH_ATTN
        and compilation_config.cudagraph_mode not in {CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE}):
    compilation_config.cudagraph_mode = CUDAGraphMode.PIECEWISE
```

## Device Control

Use `ZE_AFFINITY_MASK` to control which Intel GPUs are visible:

```bash
# Use only the first Intel GPU
ZE_AFFINITY_MASK=0 vllm serve meta-llama/Llama-3.1-8B

# Use GPUs 0 and 1 for tensor parallelism
ZE_AFFINITY_MASK=0,1 vllm serve meta-llama/Llama-3.1-8B --tensor-parallel-size 2
```

## Supported Data Types

XPU supports BF16, FP16, and FP32. However:
- Arc A770 has BF16 accuracy issues — use FP16 instead
- Flash Attention on XPU does not support FP32 — falls back to Triton

```python
@classmethod
def check_if_supports_dtype(cls, dtype: torch.dtype):
    if dtype == torch.bfloat16:
        device_name = cls.get_device_name().lower()
        if device_name.count("a770") > 0:
            raise ValueError(
                "Intel Arc A770 have bfloat16 accuracy known issue. "
                "You can use float16 instead..."
            )
```

## FP8 Support

FP8 is supported on Intel XPU using the `torch.float8_e4m3fn` format:

```python
@classmethod
def fp8_dtype(cls) -> torch.dtype:
    return torch.float8_e4m3fn
```

## Memory Management

Total device memory is queried via PyTorch's XPU API:

```python
@classmethod
def get_device_total_memory(cls, device_id: int = 0) -> int:
    device_props = torch.xpu.get_device_properties(device_id)
    return device_props.total_memory
```

Memory usage tracking:

```python
@classmethod
def get_current_memory_usage(cls, device=None) -> float:
    torch.xpu.reset_peak_memory_stats(device)
    return torch.xpu.max_memory_allocated(device)
```

## Block Swapping

XPU implements custom block swap operations for KV cache management:

```python
@classmethod
def insert_blocks_to_device(cls, src_cache, dst_cache,
                             src_block_indices, dst_block_indices):
    """Copy blocks from src_cache to dst_cache on XPU."""
    _src_cache = src_cache[:, src_block_indices]
    dst_cache[:, dst_block_indices] = _src_cache.to(dst_cache.device)

@classmethod
def swap_out_blocks_to_host(cls, src_cache, dst_cache,
                             src_block_indices, dst_block_indices):
    """Copy blocks from XPU to host (CPU)."""
    _src_cache = src_cache[:, src_block_indices]
    dst_cache[:, dst_block_indices] = _src_cache.cpu()
```

## LoRA Support

XPU supports LoRA with a dedicated punica wrapper. The Triton kernel variant can be enabled via environment variable:

```python
@classmethod
def get_punica_wrapper(cls) -> str:
    xpu_use_triton_kernel = os.getenv("XPU_USE_TRITON_KERNEL", "0") == "1"
    if not xpu_use_triton_kernel:
        return "vllm.lora.punica_wrapper.punica_xpu.PunicaWrapperXPU"
    else:
        return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"
```

## Worker Configuration

```python
# Default worker class
parallel_config.worker_cls = "vllm.v1.worker.xpu_worker.XPUWorker"

# Default KV cache block size
cache_config.block_size = 64  # NHD layout requires 64
```

## KV Transfer (P/D Disaggregation)

XPU supports KV cache transfer for prefill/decode disaggregation:

```python
if vllm_config.kv_transfer_config is not None:
    vllm_config.kv_transfer_config.enable_permute_local_kv = True
```

## UCX Memory Cache

To prevent memory type misdetection with UCX:

```python
os.environ["UCX_MEMTYPE_CACHE"] = "n"
```

## Building the XPU Docker Image

```bash
docker build -t vllm-xpu -f docker/Dockerfile.xpu .
```

## Running XPU Tests

```bash
# Basic offline inference
python3 examples/offline_inference/basic/generate.py \
  --model facebook/opt-125m \
  --block-size 64 \
  --enforce-eager

# With tensor parallelism
python3 examples/offline_inference/basic/generate.py \
  --model facebook/opt-125m \
  --block-size 64 \
  --enforce-eager \
  -tp 2 \
  --distributed-executor-backend mp

# With Triton attention backend
python3 examples/offline_inference/basic/generate.py \
  --model facebook/opt-125m \
  --block-size 64 \
  --enforce-eager \
  --attention-backend=TRITON_ATTN
```

## Related Pages

- [Platform Overview](overview.md)
- [NVIDIA CUDA Platform](cuda.md)
- [Hardware CI](hardware-ci.md)
