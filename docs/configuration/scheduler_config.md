# Scheduler Configuration

`SchedulerConfig` controls how vLLM batches and schedules requests. Proper tuning of these parameters is critical for achieving optimal throughput and latency for your workload.

**Source:** `vllm/config/scheduler.py`  
**CLI flags:** See [EngineArgs](engine_args.md) — Scheduler section.

---

## Core Batching Parameters

### `max_num_batched_tokens`

```
Type:    int
Default: auto (set by engine based on model and hardware)
CLI:     --max-num-batched-tokens
```

Maximum number of tokens processed in a single forward pass. This is the primary knob for controlling throughput vs. latency:

- **Higher values** → more tokens per iteration → higher throughput, higher latency per request
- **Lower values** → fewer tokens per iteration → lower throughput, lower latency per request

Supports human-readable suffixes on the CLI: `32k`, `64K`, `128k`.

**Constraints:**
- Must be ≥ `max_model_len` when chunked prefill is disabled
- Must be ≥ `max_num_seqs`

```bash
# High-throughput configuration
vllm serve mymodel --max-num-batched-tokens 65536

# Low-latency configuration
vllm serve mymodel --max-num-batched-tokens 4096
```

### `max_num_seqs`

```
Type:    int
Default: auto (set by engine)
CLI:     --max-num-seqs
```

Maximum number of sequences (requests) processed in a single iteration. Controls the maximum batch size in terms of request count.

**Constraints:**
- Must be ≤ `max_num_batched_tokens`

```bash
vllm serve mymodel --max-num-seqs 256
```

---

## Chunked Prefill

Chunked prefill splits long prompts across multiple iterations, allowing decode requests to be interleaved with prefill work. This significantly improves time-to-first-token (TTFT) for concurrent requests.

### `enable_chunked_prefill`

```
Type:    bool | None
Default: None (auto-enabled for most models)
CLI:     --enable-chunked-prefill / --no-enable-chunked-prefill
```

Enable chunked prefill. When enabled, long prompts are split into chunks of at most `max_num_batched_tokens` tokens, and decode requests can be batched alongside prefill chunks.

**Automatically disabled for:**
- Encoder-decoder models
- Some attention backends that don't support mixed batches

```bash
# Explicitly enable
vllm serve mymodel --enable-chunked-prefill

# Explicitly disable
vllm serve mymodel --no-enable-chunked-prefill
```

### `max_num_partial_prefills`

```
Type:    int
Default: 1
CLI:     --max-num-partial-prefills
```

Maximum number of sequences that can be partially prefilled concurrently. Setting this > 1 enables concurrent partial prefills, which can improve GPU utilization when there are multiple long prompts.

**Requires:** `enable_chunked_prefill=True`

```bash
vllm serve mymodel \
  --enable-chunked-prefill \
  --max-num-partial-prefills 4
```

### `max_long_partial_prefills`

```
Type:    int
Default: 1
CLI:     --max-long-partial-prefills
```

Maximum number of "long" prompts (longer than `long_prefill_token_threshold`) that can be partially prefilled concurrently. Setting this < `max_num_partial_prefills` allows shorter prompts to jump ahead of longer ones, improving latency for short requests.

**Constraints:** Must be ≤ `max_num_partial_prefills`

### `long_prefill_token_threshold`

```
Type:    int
Default: 0 (auto: 4% of max_model_len when max_num_partial_prefills > 1)
CLI:     --long-prefill-token-threshold
```

Token count above which a prompt is considered "long" for the purposes of `max_long_partial_prefills`. When `max_num_partial_prefills > 1` and this is 0, it defaults to `max_model_len × 0.04`.

**Constraints:** Must be ≤ `max_model_len`

### `disable_chunked_mm_input`

```
Type:    bool
Default: False
CLI:     --disable-chunked-mm-input
```

When chunked prefill is enabled, prevent partial scheduling of multimodal items. Ensures that multimodal tokens (e.g., image tokens) are always scheduled as a complete unit, not split across iterations.

**Example:** A prompt with text tokens `TTTT` followed by image tokens `IIIIIIIIII` will be scheduled as `TTTT` in one step and `IIIIIIIIII` in the next, rather than `TTTTIIIIII` + `IIII`.

---

## Scheduling Policy

### `policy`

```
Type:    Literal["fcfs", "priority"]
Default: "fcfs"
CLI:     --scheduling-policy
```

Request scheduling policy:

| Policy | Description |
|---|---|
| `fcfs` | First-Come-First-Served. Requests handled in arrival order. |
| `priority` | Priority-based. Lower priority value = earlier handling. Ties broken by arrival time. |

```bash
# Priority-based scheduling
vllm serve mymodel --scheduling-policy priority
```

When using `priority` scheduling, set the `priority` field in `SamplingParams`:

```python
from vllm import SamplingParams
params = SamplingParams(priority=1)  # Lower = higher priority
```

### `scheduler_cls`

```
Type:    str | type | None
Default: None (uses built-in scheduler)
CLI:     --scheduler-cls
```

Custom scheduler class. Can be a fully-qualified class path string or a class object.

```bash
vllm serve mymodel --scheduler-cls mypackage.schedulers.CustomScheduler
```

!!! warning
    The scheduler interface is not public and compatibility may not be maintained across versions.

---

## Async Scheduling

### `async_scheduling`

```
Type:    bool | None
Default: None (auto-enabled when beneficial)
CLI:     --async-scheduling / --no-async-scheduling
```

Enable asynchronous scheduling. Async scheduling overlaps the scheduler's CPU work with GPU execution, reducing gaps in GPU utilization and improving both latency and throughput.

When enabled, `disable_nccl_for_dp_synchronization` defaults to `True`.

```bash
# Explicitly enable
vllm serve mymodel --async-scheduling

# Explicitly disable
vllm serve mymodel --no-async-scheduling
```

---

## Streaming

### `stream_interval`

```
Type:    int
Default: 1
CLI:     --stream-interval
```

Token buffer size for streaming responses. Controls the trade-off between streaming smoothness and host overhead:

- `1` — Send each token immediately (smoothest streaming, highest overhead)
- `10` — Buffer 10 tokens before sending (less smooth, lower overhead, higher throughput)

```bash
# Smooth streaming (default)
vllm serve mymodel --stream-interval 1

# Batched streaming for throughput
vllm serve mymodel --stream-interval 8
```

---

## Hybrid KV Cache Manager

### `disable_hybrid_kv_cache_manager`

```
Type:    bool | None
Default: None (auto-determined)
CLI:     --disable-hybrid-kv-cache-manager
```

When `True`, the KV cache manager allocates the same block size for all attention layers, even when the model has mixed attention types (e.g., full attention + sliding window attention). When `None`, the default is determined based on the environment and model configuration.

---

## Multimodal Encoder Budget

### `max_num_encoder_input_tokens` (internal)

```
Type:    int
Default: max_num_batched_tokens (set internally)
```

Compute budget for the multimodal encoder. Not directly configurable; derived from `max_num_batched_tokens`.

### `encoder_cache_size` (internal)

```
Type:    int
Default: max_num_batched_tokens (set internally)
```

Multimodal encoder cache size. Not directly configurable.

---

## Tuning Guide

### Throughput-optimized configuration

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --max-num-batched-tokens 65536 \
  --max-num-seqs 512 \
  --enable-chunked-prefill \
  --max-num-partial-prefills 4 \
  --async-scheduling
```

### Latency-optimized configuration

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 32 \
  --stream-interval 1 \
  --performance-mode interactivity
```

### Mixed workload (long prompts + short decode)

```bash
vllm serve mymodel \
  --enable-chunked-prefill \
  --max-num-batched-tokens 32768 \
  --max-num-partial-prefills 4 \
  --max-long-partial-prefills 2 \
  --long-prefill-token-threshold 4096
```

### Priority-based serving

```bash
vllm serve mymodel \
  --scheduling-policy priority \
  --max-num-seqs 128
```

---

## Understanding Chunked Prefill

Without chunked prefill, a long prompt monopolizes the GPU for many iterations, causing high TTFT for other requests:

```
Iteration 1: [PREFILL: long prompt (8192 tokens)]
Iteration 2: [DECODE: long prompt] [DECODE: other requests]
```

With chunked prefill, the long prompt is split and decode requests are interleaved:

```
Iteration 1: [PREFILL chunk: 2048 tokens] [DECODE: other requests]
Iteration 2: [PREFILL chunk: 2048 tokens] [DECODE: all requests]
Iteration 3: [PREFILL chunk: 2048 tokens] [DECODE: all requests]
Iteration 4: [PREFILL chunk: 2048 tokens] [DECODE: all requests]
```

This dramatically reduces TTFT for concurrent requests at a small throughput cost.
