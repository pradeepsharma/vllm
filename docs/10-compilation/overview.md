# Compilation Overview

vLLM's compilation system transforms PyTorch model code into highly optimized GPU kernels using `torch.compile`. This page explains the motivation, the four compilation levels, their tradeoffs, and how to choose the right mode for your workload.

## Why Compile LLMs?

Large language model inference is dominated by two bottlenecks:

1. **Memory bandwidth** — loading model weights from HBM (High Bandwidth Memory) for each token
2. **Kernel launch overhead** — dispatching hundreds of individual CUDA kernels per forward pass

`torch.compile` addresses the second bottleneck by:

- **Graph capture** — tracing the entire forward pass into a single computation graph
- **Operator fusion** — merging adjacent elementwise operations (e.g., RMSNorm + quantize) into a single Triton kernel
- **Memory planning** — reusing intermediate buffers to reduce allocation overhead
- **Constant folding** — pre-computing static expressions at compile time

For LLM serving, the gains are most pronounced during **decode** (single-token generation), where the batch is small and kernel launch overhead is a significant fraction of total time.

## The Four Compilation Modes

vLLM defines compilation modes in `vllm/config/compilation.py` as `CompilationMode`:

```python
class CompilationMode(enum.IntEnum):
    NONE = 0
    """No torch.compile compilation is applied, model runs in fully eager pytorch mode."""
    STOCK_TORCH_COMPILE = 1
    """The standard `torch.compile` compilation pipeline."""
    DYNAMO_TRACE_ONCE = 2
    """Single Dynamo trace through the model, avoiding recompilation."""
    VLLM_COMPILE = 3
    """Custom vLLM Inductor-based backend with caching, piecewise compilation,
    shape specialization, and custom passes."""
```

### Mode 0 — NONE (Eager)

The model runs in standard PyTorch eager mode. Every operation dispatches a separate CUDA kernel.

**When to use:**
- Debugging model correctness
- Platforms where compilation is not supported
- Rapid prototyping

**Tradeoffs:**
- ✅ Fastest startup (no compilation overhead)
- ✅ Easiest to debug (standard Python stack traces)
- ❌ Slowest inference throughput
- ❌ No kernel fusion

```python
from vllm import LLM
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    compilation_config={"mode": 0}
)
```

### Mode 1 — STOCK_TORCH_COMPILE

Uses the standard `torch.compile` API with the Inductor backend. Dynamo traces the entire model forward pass and Inductor compiles it as a single graph.

**When to use:**
- When you want standard PyTorch compilation behavior
- Models that don't support vLLM's piecewise splitting

**Tradeoffs:**
- ✅ Standard PyTorch behavior, well-tested
- ✅ Full graph optimization
- ❌ No vLLM-specific fusion passes
- ❌ No CUDA graph management by vLLM
- ❌ May recompile on shape changes

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    compilation_config={"mode": 1}
)
```

### Mode 2 — DYNAMO_TRACE_ONCE

Dynamo traces the model once and drops all guards. This prevents recompilation when input shapes change between calls.

**How it works:** The `TorchCompileWithNoGuardsWrapper` class sets `guard_filter_fn = lambda x: [False for _ in x]`, which causes Dynamo to discard all shape guards after the first trace. The compiled graph is reused for all subsequent calls regardless of input shape.

**When to use:**
- Models with no shape-dependent control flow
- When you want to avoid recompilation overhead but don't need full vLLM features

**Tradeoffs:**
- ✅ No recompilation after first trace
- ✅ Simpler than Mode 3
- ❌ Requires no dynamic-shape-dependent control flow
- ❌ No piecewise compilation or CUDA graph management

### Mode 3 — VLLM_COMPILE (Default for V1)

The full vLLM compilation pipeline. This is the default mode for the V1 engine and provides the best performance for production serving.

**What it does:**
1. Dynamo traces the model into an FX graph (guards dropped after first trace)
2. `VllmBackend.__call__` receives the graph
3. The graph is split at attention ops into piecewise subgraphs
4. Each subgraph is compiled by Inductor with custom fusion passes
5. CUDA graphs are captured for each compiled subgraph
6. Compiled artifacts are cached to disk for fast restart

**When to use:**
- Production serving (always)
- When you need maximum throughput
- When startup time can be amortized over many requests

**Tradeoffs:**
- ✅ Maximum inference throughput
- ✅ Custom fusion passes (RMSNorm+quant, attention+quant, etc.)
- ✅ CUDA graph capture eliminates CPU overhead
- ✅ Compilation cache avoids recompilation on restart
- ❌ Longer first-start compilation time (mitigated by caching)
- ❌ More complex debugging

```python
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    compilation_config={
        "mode": 3,
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64, 128, 256],
    }
)
```

## Mode Comparison

| Feature | Mode 0 | Mode 1 | Mode 2 | Mode 3 |
|---------|--------|--------|--------|--------|
| Compilation | None | Full graph | Full graph (no guards) | Piecewise |
| CUDA Graphs | No | No | No | Yes |
| Custom Passes | No | No | No | Yes |
| Caching | No | No | No | Yes |
| Startup Time | Instant | Slow | Slow | Slow (cached: fast) |
| Throughput | Lowest | Medium | Medium | Highest |
| Debug Ease | Easiest | Medium | Medium | Hardest |

## Compilation Startup Time

The first time vLLM starts with Mode 3, it must compile the model. This can take **30–300 seconds** depending on model size and hardware. Subsequent starts reuse the cached artifacts and start in seconds.

The compilation time is logged:
```
INFO: Dynamo bytecode transform time: 12.34 s
INFO: torch.compile took 87.23 s in total
```

To monitor compilation progress, set `VLLM_LOGGING_LEVEL=DEBUG`.

## Dynamic Shapes

vLLM handles dynamic batch sizes (number of tokens) through `DynamicShapesConfig`:

```python
class DynamicShapesType(str, enum.Enum):
    BACKED = "backed"
    """Default PyTorch behavior with potential guards ignored."""
    UNBACKED = "unbacked"
    """No guards guaranteed (most sound) but may throw data dependent errors."""
    BACKED_SIZE_OBLIVIOUS = "backed_size_oblivious"
    """Experimental safer alternative to backed/unbacked."""
```

The `@support_torch_compile` decorator marks specific tensor dimensions as dynamic (see [Decorator Reference](decorators.md)), so Dynamo knows to treat the batch dimension symbolically rather than specializing on a concrete value.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_CACHE_ROOT` | `~/.cache/vllm` | Root directory for compilation cache |
| `VLLM_USE_STANDALONE_COMPILE` | `0` | Use standalone Inductor compile for serializable artifacts |
| `VLLM_USE_MEGA_AOT_ARTIFACT` | `0` | Bundle all subgraph artifacts into a single file |
| `VLLM_USE_AOT_COMPILE` | `0` | Enable AOT (ahead-of-time) compilation mode |
| `VLLM_USE_BYTECODE_HOOK` | `1` | Use bytecode hook for compilation dispatch |
| `VLLM_COMPILE_CACHE_SAVE_FORMAT` | `binary` | Cache format: `binary` or `unpacked` |

## See Also

- [Piecewise Compilation](piecewise.md) — how the graph is split at attention ops
- [CUDA Graph Capture](cuda-graphs.md) — static shape capture and replay
- [Compilation Caching](caching.md) — persisting compiled artifacts
- [CompilationConfig Reference](config-reference.md) — all configuration options
- [`@support_torch_compile` Decorator](decorators.md) — annotating model classes
