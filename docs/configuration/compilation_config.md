# Compilation Configuration

`CompilationConfig` controls how vLLM uses `torch.compile` and CUDA graph capture to optimize model execution. Proper compilation configuration can dramatically improve throughput and reduce latency.

**Source:** `vllm/config/compilation.py`  
**CLI flag:** `--compilation-config` / `-cc` (JSON dict)

---

## Overview

vLLM's compilation pipeline has two main components:

1. **`torch.compile` (Inductor)** — Compiles the model graph using PyTorch's Inductor backend, enabling kernel fusion, operator optimization, and shape specialization.

2. **CUDA Graphs** — Captures the GPU execution trace for fixed batch sizes, eliminating CPU overhead for repeated calls.

These are controlled by `CompilationMode` and `CUDAGraphMode` respectively.

---

## Compilation Mode

### `mode`

```
Type:    CompilationMode (int)
Default: None (auto-select; V1 engine uses mode 3)
CLI:     --compilation-config '{"mode": 3}'
         -cc.mode=3
```

The compilation approach:

| Mode | Name | Description |
|---|---|---|
| `0` | `NONE` | No compilation. Pure eager PyTorch. Fastest startup, lowest throughput. |
| `1` | `STOCK_TORCH_COMPILE` | Standard `torch.compile` pipeline. |
| `2` | `DYNAMO_TRACE_ONCE` | Single Dynamo trace, no recompilation. Requires no dynamic-shape-dependent control flow. |
| `3` | `VLLM_COMPILE` | **Default for V1.** Custom vLLM Inductor backend with caching, piecewise compilation, shape specialization, and custom passes. |

```bash
# Disable compilation (fastest startup, debugging)
vllm serve mymodel -cc.mode=0

# Full vLLM compilation (best throughput)
vllm serve mymodel -cc.mode=3
```

---

## CUDA Graph Mode

### `cudagraph_mode`

```
Type:    CUDAGraphMode
Default: None (auto-select based on optimization_level)
CLI:     --compilation-config '{"cudagraph_mode": "FULL_AND_PIECEWISE"}'
```

Controls CUDA graph capture strategy:

| Mode | Description |
|---|---|
| `NONE` | No CUDA graph capture. |
| `PIECEWISE` | Piecewise CUDA graphs — attention ops excluded from graphs. |
| `FULL` | Full CUDA graphs for all batches. |
| `FULL_DECODE_ONLY` | Full CUDA graphs for decode-only batches; prefill runs without graphs. |
| `FULL_AND_PIECEWISE` | **V1 default.** Full graphs for decode, piecewise for prefill/mixed batches. |

**Optimization level mapping:**
- `O0` → `NONE`
- `O1` → `PIECEWISE`
- `O2`/`O3` → `FULL_AND_PIECEWISE`

### `cudagraph_capture_sizes`

```
Type:    list[int] | None
Default: None (inferred from vLLM config)
CLI:     --cudagraph-capture-sizes [1, 2, 4, 8, 16, 32, 64, 128, 256]
         -cc.cudagraph_capture_sizes='[1,2,4,8,16,32,64,128,256]'
```

Explicit list of batch sizes for CUDA graph capture. A CUDA graph captured for size `N` can only be used for exactly `N` tokens.

```bash
# Capture graphs for specific sizes
vllm serve mymodel \
  --cudagraph-capture-sizes 1 2 4 8 16 32 64 128 256
```

### `max_cudagraph_capture_size`

```
Type:    int | None
Default: None (inferred)
CLI:     --max-cudagraph-capture-size
```

Maximum batch size for CUDA graph capture. Batches larger than this run without CUDA graphs.

### `cudagraph_num_of_warmups`

```
Type:    int
Default: 0
CLI:     -cc.cudagraph_num_of_warmups=1
```

Number of warmup runs before CUDA graph capture. The first `N` runs are treated as warmup; the graph is captured on run `N+1`.

### `cudagraph_copy_inputs`

```
Type:    bool
Default: False
CLI:     -cc.cudagraph_copy_inputs=true
```

Copy input tensors before CUDA graph replay. Required when input buffers may change between calls. Only effective in `PIECEWISE` mode.

### `cudagraph_specialize_lora`

```
Type:    bool
Default: True
CLI:     -cc.cudagraph_specialize_lora=false
```

Create separate CUDA graphs for cases with and without active LoRA adapters. When `False`, the LoRA-enabled graph is always used (even without adapters), incurring LoRA overhead. When `True`, eliminates this overhead at the cost of increased startup time.

---

## Inductor Compilation

### `compile_sizes`

```
Type:    list[int | str] | None
Default: None
CLI:     -cc.compile_sizes='[1,4,16,64]'
```

Batch sizes to compile with Inductor for shape specialization. Also accepts `"cudagraph_capture_sizes"` as a special value to use the CUDA graph capture sizes.

Shape-specialized compilation allows Inductor to generate more optimized kernels for specific sizes.

### `compile_ranges_split_points`

```
Type:    list[int] | None
Default: None
```

Split points defining compile ranges for Inductor. The ranges are:
- `[1, split_points[0]]`
- `[split_points[0]+1, split_points[1]]`
- ...
- `[split_points[-1]+1, max_num_batched_tokens]`

### `inductor_compile_config`

```
Type:    dict
Default: {}
CLI:     -cc.inductor_compile_config='{"max_autotune": true}'
```

Additional configuration passed directly to the Inductor backend. See [PyTorch Inductor docs](https://pytorch.org/docs/stable/torch.compiler_inductor_profiling.html) for available options.

### `inductor_passes`

```
Type:    dict[str, str]
Default: {}
CLI:     -cc.inductor_passes='{"my_pass": "mymodule.my_pass_fn"}'
```

Custom Inductor passes as a dict from pass name to fully-qualified function name. Can also be passed as Python functions when constructing `CompilationConfig` directly.

---

## Custom Operations

### `custom_ops`

```
Type:    list[str]
Default: []
CLI:     -cc.custom_ops='["all", "-flash_attn"]'
```

Fine-grained control over which custom ops to enable or disable:

- `"all"` — Enable all custom ops
- `"none"` — Disable all custom ops
- `"+op_name"` — Enable specific op
- `"-op_name"` — Disable specific op

**Default behavior:**
- Without Inductor: all custom ops enabled
- With Inductor (mode > 0, backend="inductor"): all custom ops disabled (Inductor generates fused Triton kernels instead)

```bash
# Enable all except flash_attn
vllm serve mymodel -cc.custom_ops='["all", "-flash_attn"]'

# Enable only specific ops
vllm serve mymodel -cc.custom_ops='["none", "+rms_norm", "+silu_and_mul"]'
```

### `splitting_ops`

```
Type:    list[str] | None
Default: None (attention ops)
CLI:     -cc.splitting_ops='["vllm.unified_attention", "vllm.unified_attention_with_output"]'
```

Ops to exclude from CUDA graphs in piecewise compilation. These ops are used as split points for graph partitioning.

- `None` — Default attention ops
- `[]` — No splitting (suitable for full CUDA graphs)

### `compile_mm_encoder`

```
Type:    bool
Default: False
CLI:     -cc.compile_mm_encoder=true
```

Compile the multimodal encoder with `torch.compile`. Currently only works for `Qwen2_5_vl` and `mLLaMa4` models on selected platforms.

---

## Graph Partitioning

### `use_inductor_graph_partition`

```
Type:    bool | None
Default: None (auto-select)
CLI:     -cc.use_inductor_graph_partition=true
```

Use Inductor graph partitioning to split the graph at CUDA-graph-unsafe ops. This happens at Inductor codegen time after all passes and fusions, allowing the full graph to be optimized before splitting.

Benefits:
- Supports both full and piecewise CUDA graphs without compiling twice
- Fusions can operate on the complete graph

---

## Caching

### `cache_dir`

```
Type:    str
Default: "" (auto-generated from model info)
CLI:     -cc.cache_dir=/path/to/cache
```

Directory to store compiled graph artifacts. Accelerates subsequent startups by reusing cached compilations.

### `compile_cache_save_format`

```
Type:    Literal["binary", "unpacked"]
Default: "binary" (from VLLM_COMPILE_CACHE_SAVE_FORMAT)
CLI:     -cc.compile_cache_save_format=unpacked
```

Format for saving the compile cache:
- `"binary"` — Single binary file (multiprocess safe, default)
- `"unpacked"` — Directory structure (human-readable, NOT multiprocess safe)

### `debug_dump_path`

```
Type:    Path | None
Default: None
CLI:     -cc.debug_dump_path=/tmp/vllm_debug
```

Path to dump debug information (FX graphs, Triton kernels, etc.). Overridden by `VLLM_DEBUG_DUMP_PATH` environment variable.

---

## Backend

### `backend`

```
Type:    str
Default: "" (inductor on CUDA-alike platforms)
CLI:     -cc.backend=inductor
```

The `torch.compile` backend:

- `""` — Default (Inductor on CUDA/ROCm)
- `"eager"` — Eager mode (no compilation)
- `"inductor"` — PyTorch Inductor
- `"openxla"` — OpenXLA (TPU)
- `"full.module.name"` — Custom backend function

---

## Custom Fusion Passes (PassConfig)

`PassConfig` controls fine-grained kernel fusion optimizations. These are nested inside `CompilationConfig`:

```bash
vllm serve mymodel \
  --compilation-config '{
    "pass_config": {
      "fuse_norm_quant": true,
      "fuse_act_quant": true,
      "enable_sp": false
    }
  }'
```

| Pass | Default | Description |
|---|---|---|
| `fuse_norm_quant` | auto | Fuse RMSNorm + FP8 quantization ops |
| `fuse_act_quant` | auto | Fuse SiluMul + FP8 quantization ops |
| `fuse_attn_quant` | auto | Fuse attention + FP8 quantization ops |
| `eliminate_noops` | `True` | Eliminate no-op operations |
| `enable_sp` | auto | Enable sequence parallelism (requires TP>1) |
| `fuse_gemm_comms` | auto | Enable async TP (fuse GEMM + communication) |
| `fuse_allreduce_rms` | auto | FlashInfer allreduce + RMSNorm fusion (H100/B100) |
| `enable_qk_norm_rope_fusion` | `False` | Fused Q/K RMSNorm + RoPE pass |
| `fuse_act_padding` | auto | Fuse RMSNorm + padding ops (ROCm/AITER) |
| `fuse_rope_kvcache` | auto | Fuse QK RoPE + KV cache ops (ROCm/AITER) |

### `fi_allreduce_fusion_max_size_mb`

```
Type:    float | None
Default: None (device/world-size dependent)
```

Maximum tensor size (in MB) for FlashInfer fused allreduce. Larger tensors use standard NCCL allreduce.

### `sp_min_token_num`

```
Type:    int | None
Default: None (device/world-size dependent)
```

Minimum token count above which sequence parallelism is used.

---

## Dynamic Shapes (DynamicShapesConfig)

Controls how `torch.compile` handles dynamic tensor shapes:

```bash
vllm serve mymodel \
  --compilation-config '{
    "dynamic_shapes_config": {
      "type": "backed",
      "evaluate_guards": false
    }
  }'
```

| Field | Default | Description |
|---|---|---|
| `type` | `"backed"` | Dynamic shapes type: `backed`, `unbacked`, `backed_size_oblivious` |
| `evaluate_guards` | `False` | Debug mode: fail on dynamic shape specialization |
| `assume_32_bit_indexing` | `False` | Assume all tensor sizes fit in 32-bit indexing (PyTorch 2.10+) |

---

## Optimization Level Presets

The `--optimization-level` flag sets a bundle of compilation settings:

| Level | Mode | CUDA Graph Mode | Fusions |
|---|---|---|---|
| `O0` | `NONE` | `NONE` | None |
| `O1` | `VLLM_COMPILE` | `PIECEWISE` | norm+quant, act+quant |
| `O2` | `VLLM_COMPILE` | `FULL_AND_PIECEWISE` | All applicable |
| `O3` | `VLLM_COMPILE` | `FULL_AND_PIECEWISE` | All applicable |

---

## Common Configurations

### Debug mode (no compilation)

```bash
vllm serve mymodel --optimization-level O0
# or
vllm serve mymodel -cc.mode=0
```

### Fast startup (piecewise graphs only)

```bash
vllm serve mymodel --optimization-level O1
```

### Maximum performance (default)

```bash
vllm serve mymodel --optimization-level O2
```

### Custom CUDA graph sizes

```bash
vllm serve mymodel \
  --optimization-level O2 \
  --cudagraph-capture-sizes 1 2 4 8 16 32 64 128 256 512
```

### Persistent compile cache

```bash
vllm serve mymodel \
  --compilation-config '{"cache_dir": "/shared/vllm_cache"}'
```

### Disable specific fusions

```bash
vllm serve mymodel \
  --compilation-config '{
    "pass_config": {
      "fuse_allreduce_rms": false,
      "enable_sp": false
    }
  }'
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VLLM_DISABLE_COMPILE_CACHE` | `0` | Disable compilation cache |
| `VLLM_USE_AOT_COMPILE` | auto | Enable AOT compilation |
| `VLLM_USE_BYTECODE_HOOK` | `1` | Enable bytecode hook in TorchCompile |
| `VLLM_FORCE_AOT_LOAD` | `0` | Force loading AOT compiled models |
| `VLLM_USE_MEGA_AOT_ARTIFACT` | `0` | Load from mega AOT artifact |
| `VLLM_USE_STANDALONE_COMPILE` | `1` | Use Inductor standalone compile |
| `VLLM_ENABLE_PREGRAD_PASSES` | `0` | Enable Inductor pre-grad passes |
| `VLLM_COMPILE_CACHE_SAVE_FORMAT` | `"binary"` | Cache save format |
| `VLLM_DEBUG_DUMP_PATH` | `None` | Path for debug graph dumps |
| `VLLM_PATTERN_MATCH_DEBUG` | `None` | Debug pattern matching (FX node name) |
| `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE` | `1` | Enable Inductor max autotune |
| `VLLM_ENABLE_INDUCTOR_COORDINATE_DESCENT_TUNING` | `1` | Enable coordinate descent tuning |
| `VLLM_ENABLE_CUDAGRAPH_GC` | `0` | Enable CUDA graph garbage collection |
