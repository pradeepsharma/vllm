# Configuration Overview

vLLM is configured through a layered system of Python dataclasses, CLI arguments, and environment variables. Understanding how these layers interact is key to tuning vLLM for your workload.

---

## Configuration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        VllmConfig                           │
│  (master config — assembles all sub-configs at startup)     │
├──────────────┬──────────────┬──────────────┬────────────────┤
│ ModelConfig  │ CacheConfig  │ParallelConfig│SchedulerConfig │
├──────────────┼──────────────┼──────────────┼────────────────┤
│  LoadConfig  │  LoRAConfig  │SpeculativeC. │CompilationConf.│
├──────────────┴──────────────┴──────────────┴────────────────┤
│              ObservabilityConfig  +  others …               │
└─────────────────────────────────────────────────────────────┘
                          ▲
              EngineArgs (CLI / Python API)
                          ▲
              VLLM_* Environment Variables
```

### How the layers work

| Layer | Where it lives | Who sets it |
|---|---|---|
| **Environment variables** | `vllm/envs.py` | Shell / container env |
| **EngineArgs** | `vllm/engine/arg_utils.py` | CLI flags or Python constructor |
| **Sub-configs** | `vllm/config/*.py` | Derived from EngineArgs |
| **VllmConfig** | `vllm/config/vllm.py` | Assembled at engine startup |

---

## Quick-Start: Most Important Settings

### Minimal server launch

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192
```

### High-throughput batch inference

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --max-num-batched-tokens 32768 \
  --max-num-seqs 512 \
  --enable-chunked-prefill
```

### Low-latency interactive serving

```bash
vllm serve Qwen/Qwen3-7B \
  --performance-mode interactivity \
  --optimization-level O2 \
  --max-num-seqs 64
```

---

## Configuration Reference Pages

| Page | What it covers |
|---|---|
| [Engine Args](engine_args.md) | Complete `EngineArgs` reference — every CLI flag |
| [Model Config](model_config.md) | Model loading, dtype, quantization, tokenizer |
| [Parallel Config](parallel_config.md) | Tensor/pipeline/data parallelism, expert parallelism |
| [Scheduler Config](scheduler_config.md) | Batching, chunked prefill, scheduling policy |
| [Cache Config](cache_config.md) | KV cache sizing, prefix caching, FP8 KV cache |
| [Quantization Config](quantization_config.md) | Quantization methods and options |
| [LoRA Config](lora_config.md) | LoRA adapter serving |
| [Speculative Config](speculative_config.md) | Speculative decoding (draft models, ngram, EAGLE) |
| [Compilation Config](compilation_config.md) | `torch.compile`, CUDA graphs, custom passes |
| [Observability Config](observability_config.md) | Prometheus metrics, OpenTelemetry tracing |
| [Environment Variables](environment_variables.md) | All `VLLM_*` environment variables |

---

## Optimization Levels

vLLM provides a shorthand `--optimization-level` (`-O`) flag that sets a bundle of compilation and kernel settings:

| Level | Flag | Description |
|---|---|---|
| **O0** | `--optimization-level O0` | No compilation, no CUDA graphs. Fastest startup, lowest throughput. |
| **O1** | `--optimization-level O1` | Dynamo+Inductor compilation + piecewise CUDA graphs. |
| **O2** | `--optimization-level O2` | Full + piecewise CUDA graphs. **Default.** |
| **O3** | `--optimization-level O3` | Same as O2 (reserved for future use). |

---

## Performance Modes

The `--performance-mode` flag adjusts runtime behavior:

| Mode | Description |
|---|---|
| `balanced` | Default. Balances latency and throughput. |
| `interactivity` | Favors low per-request latency at small batch sizes. |
| `throughput` | Favors aggregate tokens/sec at high concurrency. |

---

## Configuration via Python API

All CLI flags map directly to `EngineArgs` fields when using vLLM programmatically:

```python
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs

args = EngineArgs(
    model="meta-llama/Llama-3.1-8B-Instruct",
    tensor_parallel_size=2,
    gpu_memory_utilization=0.90,
    max_model_len=8192,
    enable_prefix_caching=True,
)

llm = LLM(**vars(args))
```

Or pass a `VllmConfig` directly for full control:

```python
from vllm.config import VllmConfig, ModelConfig, CacheConfig, ParallelConfig

config = VllmConfig(
    model_config=ModelConfig(model="Qwen/Qwen3-7B"),
    cache_config=CacheConfig(gpu_memory_utilization=0.85),
    parallel_config=ParallelConfig(tensor_parallel_size=2),
)
```

---

## Configuration File (JSON)

Complex sub-configs (like `CompilationConfig`) can be passed as JSON strings on the CLI:

```bash
# Pass compilation config as JSON
vllm serve mymodel \
  --compilation-config '{"mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]}'

# Shorthand using -cc prefix
vllm serve mymodel -cc.mode=3 -cc.cudagraph_capture_sizes='[1,2,4,8]'
```

---

## Environment Variable Precedence

Environment variables are read at import time and provide defaults that can be overridden by explicit CLI flags or Python constructor arguments. The precedence order (highest to lowest) is:

1. Explicit Python constructor argument
2. CLI flag
3. Environment variable
4. Compiled-in default

See [Environment Variables](environment_variables.md) for the complete list.
