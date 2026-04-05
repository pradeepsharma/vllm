# CompilationConfig Reference

`CompilationConfig` is the central configuration object for vLLM's compilation system. It is defined in `vllm/config/compilation.py` and controls every aspect of the compilation pipeline: mode selection, graph splitting, CUDA graph capture, Inductor passes, and caching.

## Quick Start

```python
from vllm import LLM
from vllm.config import CompilationConfig

# Minimal: use defaults (mode 3, FULL_AND_PIECEWISE cudagraphs)
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")

# Custom configuration
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    compilation_config=CompilationConfig(
        mode=3,
        cudagraph_mode="FULL_AND_PIECEWISE",
        cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        custom_ops=["all"],
    )
)

# Or via dict (JSON-compatible)
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    compilation_config={
        "mode": 3,
        "cudagraph_mode": "FULL_AND_PIECEWISE",
    }
)
```

## Top-Level Compilation Control

### `mode`

**Type:** `CompilationMode | int | str`  
**Default:** `None` (auto-selects mode 3 for V1 engine)

The compilation approach used for `torch.compile`-based compilation.

| Value | Name | Description |
|-------|------|-------------|
| `0` | `NONE` | No compilation. Pure eager PyTorch. |
| `1` | `STOCK_TORCH_COMPILE` | Standard `torch.compile` pipeline. |
| `2` | `DYNAMO_TRACE_ONCE` | Single Dynamo trace, guards dropped. |
| `3` | `VLLM_COMPILE` | Full vLLM custom backend (default). |

```python
# Accept integer, string, or enum
compilation_config={"mode": 3}
compilation_config={"mode": "VLLM_COMPILE"}
compilation_config={"mode": CompilationMode.VLLM_COMPILE}
```

### `backend`

**Type:** `str`  
**Default:** `""` (auto-selects `"inductor"` on CUDA-like platforms)

The Inductor backend to use for compilation.

- `""` — use platform default (usually `"inductor"`)
- `"eager"` — compile to eager PyTorch (no Triton kernels)
- `"inductor"` — use TorchInductor (recommended)
- `"full.module.name"` — qualified name of a custom backend function

### `custom_ops`

**Type:** `list[str]`  
**Default:** `[]` (all custom ops enabled without Inductor, disabled with Inductor)

Fine-grained control over which custom ops to enable or disable. Custom ops are vLLM's hand-written CUDA kernels (e.g., `rms_norm`, `rotary_embedding`). When Inductor is active, custom ops are disabled by default so Inductor can generate fused Triton kernels instead.

```python
custom_ops=["all"]           # Enable all custom ops
custom_ops=["none"]          # Disable all custom ops
custom_ops=["all", "-rms_norm"]   # Enable all except rms_norm
custom_ops=["none", "+rms_norm", "+rotary_embedding"]  # Enable only these two
```

### `splitting_ops`

**Type:** `list[str] | None`  
**Default:** `None` (auto-selects attention ops for piecewise cudagraphs)

A list of ops at which to split the FX graph for piecewise compilation. These ops are excluded from CUDA graph capture.

When `None`, defaults to the attention ops list:
```python
_attention_ops = [
    "vllm::unified_attention",
    "vllm::unified_attention_with_output",
    "vllm::unified_mla_attention",
    "vllm::mamba_mixer2",
    "vllm::mamba_mixer",
    # ... more attention/Mamba ops
]
```

Set to `[]` (empty list) to disable piecewise splitting (use with `cudagraph_mode="FULL"`).

The behavior depends on `use_inductor_graph_partition`:
- `False` (default): ops are used for Dynamo FX-level graph splitting before Inductor
- `True`: ops are registered as Inductor partition rules, splitting happens at codegen time

### `compile_mm_encoder`

**Type:** `bool`  
**Default:** `False`

Whether to compile the multimodal encoder (e.g., vision encoder for Qwen2.5-VL, mLLaMA4). Disabled by default until more models are tested.

### `debug_dump_path`

**Type:** `Path | None`  
**Default:** `None`

Path to dump debug information (depyf output, computation graphs). When set, vLLM dumps the compiled FX graph and Triton kernels for inspection.

### `cache_dir`

**Type:** `str`  
**Default:** `""` (auto-generated from hash of compilation factors)

Directory to store compiled artifacts. When empty, vLLM generates a cache directory under `VLLM_CACHE_ROOT/torch_compile_cache/<hash>/`.

### `compile_cache_save_format`

**Type:** `Literal["binary", "unpacked"]`  
**Default:** `"binary"` (from `VLLM_COMPILE_CACHE_SAVE_FORMAT`)

Format for saving the compilation cache:
- `"binary"` — single binary file, multiprocess safe (production)
- `"unpacked"` — directory structure, human-readable (debugging only, NOT multiprocess safe)

## CUDA Graph Configuration

### `cudagraph_mode`

**Type:** `CUDAGraphMode | str`  
**Default:** `None` (auto-selects `FULL_AND_PIECEWISE` for V1)

Controls how CUDA graphs are captured and used.

| Mode | Description |
|------|-------------|
| `NONE` | No CUDA graph capture |
| `PIECEWISE` | Capture each compiled subgraph separately |
| `FULL` | Capture entire model forward pass as one graph |
| `FULL_DECODE_ONLY` | Full graphs for decode batches only |
| `FULL_AND_PIECEWISE` | Full for decode, piecewise for prefill (default) |

```python
compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"}
compilation_config={"cudagraph_mode": "NONE"}  # Disable CUDA graphs
```

### `cudagraph_capture_sizes`

**Type:** `list[int] | None`  
**Default:** `None` (auto-generated)

Batch sizes for which to capture CUDA graphs. When `None`, sizes are generated automatically:

```python
[1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_size + 1, 16))
```

Where `max_size = min(max_num_seqs * 2, 512)`.

```python
# Custom capture sizes
compilation_config={
    "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    "max_cudagraph_capture_size": 512,
}
```

> **Note:** More capture sizes = longer startup time but less padding waste at runtime.

### `max_cudagraph_capture_size`

**Type:** `int`  
**Default:** `None` (auto-set to `min(max_num_seqs * 2, 512)`)

The maximum batch size for CUDA graph capture. Batches larger than this run without CUDA graphs.

### `cudagraph_num_of_warmups`

**Type:** `int`  
**Default:** `0`

Number of warmup runs before CUDA graph capture begins. Warmup runs allow lazy initializations (cuBLAS workspace, etc.) to complete before recording.

### `cudagraph_copy_inputs`

**Type:** `bool`  
**Default:** `False`

Whether to copy input tensors into static buffers before CUDA graph capture. Required when the caller cannot guarantee that the same input buffers are reused across calls.

> **Note:** Only effective when `cudagraph_mode` is `PIECEWISE`.

### `cudagraph_specialize_lora`

**Type:** `bool`  
**Default:** `True`

Whether to create separate CUDA graphs for cases with and without active LoRA adapters. When `True`, eliminates LoRA overhead when no adapters are active.

### `use_inductor_graph_partition`

**Type:** `bool`  
**Default:** `None` (auto-selected)

Use Inductor's graph partitioning instead of FX-level splitting. Requires PyTorch >= 2.9.0.

When `True`:
- Graph splitting happens at Inductor codegen time (after all passes and fusions)
- Custom passes can operate on the full graph
- Enables additional cross-layer optimizations

## Inductor Compilation

### `compile_sizes`

**Type:** `list[int | str] | None`  
**Default:** `None`

Specific batch sizes to compile with Inductor using concrete (non-symbolic) shapes. Concrete-shape compilation allows Inductor to apply more aggressive optimizations.

Also accepts `"cudagraph_capture_sizes"` as a special value to compile for all CUDA graph capture sizes:

```python
compilation_config={
    "compile_sizes": [1, 4, 16, "cudagraph_capture_sizes"],
}
```

### `compile_ranges_split_points`

**Type:** `list[int] | None`  
**Default:** `None`

Split points that define compile ranges for Inductor. A range `[start, end]` means Inductor compiles a single kernel that handles all batch sizes in that range.

For split points `[8, 64, 512]`, the ranges are:
- `[1, 8]`
- `[9, 64]`
- `[65, 512]`

```python
compilation_config={
    "compile_ranges_split_points": [8, 64, 512],
}
```

### `inductor_compile_config`

**Type:** `dict`  
**Default:** `{}`

Additional configuration passed directly to TorchInductor. See [Inductor documentation](https://pytorch.org/docs/stable/torch.compiler_inductor_profiling.html) for available options.

```python
compilation_config={
    "inductor_compile_config": {
        "max_autotune": True,
        "coordinate_descent_tuning": True,
    }
}
```

### `inductor_passes`

**Type:** `dict[str, str]`  
**Default:** `{}`

Additional custom passes for Inductor. Maps pass name to qualified function name:

```python
compilation_config={
    "inductor_passes": {
        "my_custom_pass": "my_module.my_pass_function",
    }
}
```

## Pass Configuration

The `pass_config` field (type `PassConfig`) controls vLLM's custom Inductor fusion passes:

### Fusion Passes

| Field | Default | Description |
|-------|---------|-------------|
| `fuse_norm_quant` | `None` (auto) | Fuse RMSNorm + quantize ops |
| `fuse_act_quant` | `None` (auto) | Fuse SiluMul + quantize ops |
| `fuse_attn_quant` | `None` (auto) | Fuse attention + quantize ops |
| `fuse_gemm_comms` | `None` (auto) | Enable async TP (fuse GEMM + communications) |
| `fuse_allreduce_rms` | `None` (auto) | Fuse FlashInfer allreduce + RMSNorm |
| `enable_qk_norm_rope_fusion` | `False` | Fuse Q/K RMSNorm + RoPE |
| `eliminate_noops` | `True` | Eliminate no-op operations |
| `enable_sp` | `None` (auto) | Enable sequence parallelism (requires TP > 1) |

### ROCm-Specific Passes

| Field | Default | Description |
|-------|---------|-------------|
| `fuse_act_padding` | `None` (auto) | Fuse RMSNorm + padding ops (ROCm only) |
| `fuse_rope_kvcache` | `None` (auto) | Fuse QK RoPE + KV cache ops (ROCm only) |

### Example: Enable Fusion Passes

```python
from vllm.config import CompilationConfig, PassConfig

compilation_config = CompilationConfig(
    mode=3,
    pass_config=PassConfig(
        fuse_norm_quant=True,
        fuse_act_quant=True,
        eliminate_noops=True,
    )
)
```

## Dynamic Shapes Configuration

The `dynamic_shapes_config` field (type `DynamicShapesConfig`) controls how Dynamo handles dynamic batch sizes:

```python
class DynamicShapesType(str, enum.Enum):
    BACKED = "backed"              # Default: guards may be added but are dropped
    UNBACKED = "unbacked"          # No guards guaranteed (most sound)
    BACKED_SIZE_OBLIVIOUS = "backed_size_oblivious"  # Experimental
```

```python
compilation_config={
    "dynamic_shapes_config": {
        "type": "BACKED",
        "evaluate_guards": False,
        "assume_32_bit_indexing": False,
    }
}
```

## Complete Example

```python
from vllm import LLM
from vllm.config import CompilationConfig, PassConfig

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    compilation_config=CompilationConfig(
        # Compilation mode
        mode=3,
        backend="inductor",

        # Custom ops: disable all, let Inductor fuse
        custom_ops=["none"],

        # Graph splitting at attention ops (default)
        splitting_ops=None,

        # CUDA graphs
        cudagraph_mode="FULL_AND_PIECEWISE",
        cudagraph_capture_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256],
        max_cudagraph_capture_size=256,
        cudagraph_copy_inputs=False,

        # Inductor compilation
        compile_sizes=[1, 4, 16],
        inductor_compile_config={
            "max_autotune": True,
        },

        # Fusion passes
        pass_config=PassConfig(
            fuse_norm_quant=True,
            fuse_act_quant=True,
            eliminate_noops=True,
        ),

        # Caching
        cache_dir="",  # auto-generated
        compile_cache_save_format="binary",
    )
)
```

## Environment Variable Overrides

| Variable | Affects |
|----------|---------|
| `VLLM_CACHE_ROOT` | Root directory for all caches |
| `VLLM_COMPILE_CACHE_SAVE_FORMAT` | Default for `compile_cache_save_format` |
| `VLLM_USE_STANDALONE_COMPILE` | Enable standalone Inductor compile |
| `VLLM_USE_MEGA_AOT_ARTIFACT` | Bundle all artifacts into one file |
| `VLLM_USE_AOT_COMPILE` | Enable AOT compilation mode |
| `VLLM_DISABLE_COMPILE_CACHE` | Disable compilation cache |

## See Also

- [Compilation Overview](overview.md) — mode descriptions and tradeoffs
- [Piecewise Compilation](piecewise.md) — `splitting_ops` in depth
- [CUDA Graph Capture](cuda-graphs.md) — `cudagraph_*` options in depth
- [Compilation Caching](caching.md) — `cache_dir` and caching behavior
- [`@support_torch_compile` Decorator](decorators.md) — annotating model classes
