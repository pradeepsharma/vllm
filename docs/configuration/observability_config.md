# Observability Configuration

`ObservabilityConfig` controls metrics collection, distributed tracing, and logging in vLLM. It covers Prometheus metrics, OpenTelemetry tracing, CUDA graph metrics, and per-layer profiling.

**Source:** `vllm/config/observability.py`  
**CLI flags:** See [EngineArgs](engine_args.md) — Observability section.

---

## Prometheus Metrics

vLLM exposes Prometheus metrics at `/metrics` on the API server. Metrics are logged periodically (controlled by `VLLM_LOG_STATS_INTERVAL`) and exposed via the HTTP endpoint.

### `show_hidden_metrics_for_version`

```
Type:    str | None
Default: None
CLI:     --show-hidden-metrics-for-version
```

Re-enable deprecated Prometheus metrics that were hidden in the specified version. Use this as a temporary escape hatch while migrating to new metrics.

```bash
# Re-enable metrics hidden since v0.7.0
vllm serve mymodel --show-hidden-metrics-for-version 0.7
```

The metric will likely be removed completely in an upcoming release.

---

## OpenTelemetry Tracing

vLLM supports distributed tracing via OpenTelemetry (OTLP protocol). Traces capture request lifecycle events including scheduling, prefill, decode, and output processing.

### `otlp_traces_endpoint`

```
Type:    str | None
Default: None (tracing disabled)
CLI:     --otlp-traces-endpoint
```

Target URL for OpenTelemetry traces (OTLP/gRPC or OTLP/HTTP endpoint).

```bash
# Send traces to a local Jaeger instance
vllm serve mymodel \
  --otlp-traces-endpoint http://localhost:4317

# Send to an OTLP collector
vllm serve mymodel \
  --otlp-traces-endpoint http://otel-collector:4317
```

**Requires:** `opentelemetry-sdk` and `opentelemetry-exporter-otlp` packages.

### `collect_detailed_traces`

```
Type:    list[str] | None
Default: None
CLI:     --collect-detailed-traces
```

Collect detailed timing traces for specific modules. Only meaningful when `--otlp-traces-endpoint` is set.

**Warning:** Detailed traces involve potentially costly or blocking operations and may impact performance.

| Value | Description |
|---|---|
| `model` | Trace model forward pass timing |
| `worker` | Trace model execute timing |
| `all` | Trace all modules |

```bash
vllm serve mymodel \
  --otlp-traces-endpoint http://localhost:4317 \
  --collect-detailed-traces model worker
```

---

## KV Cache Metrics

### `kv_cache_metrics`

```
Type:    bool
Default: False
CLI:     --kv-cache-metrics
```

Enable KV cache residency metrics including:
- Block lifetime distribution
- Block idle time
- Block reuse gaps

Uses sampling to minimize overhead. Requires stats logging to be enabled (i.e., `--disable-log-stats` must NOT be set).

```bash
vllm serve mymodel --kv-cache-metrics
```

### `kv_cache_metrics_sample`

```
Type:    float (0 < value ≤ 1)
Default: 0.01
CLI:     --kv-cache-metrics-sample
```

Sampling rate for KV cache metrics. `0.01` = 1% of blocks are sampled. Lower values reduce overhead; higher values give more accurate statistics.

```bash
# Sample 5% of blocks for more accurate metrics
vllm serve mymodel \
  --kv-cache-metrics \
  --kv-cache-metrics-sample 0.05
```

---

## CUDA Graph Metrics

### `cudagraph_metrics`

```
Type:    bool
Default: False
CLI:     --cudagraph-metrics
```

Enable CUDA graph dispatch metrics, including:
- Number of padded/unpadded tokens
- Runtime CUDA graph dispatch modes
- Observed frequencies at each logging interval

```bash
vllm serve mymodel --cudagraph-metrics
```

---

## NVTX Tracing

### `enable_layerwise_nvtx_tracing`

```
Type:    bool
Default: False
CLI:     --enable-layerwise-nvtx-tracing
```

Enable per-layer NVTX range markers for profiling with Nsight Systems. Each layer/module execution is annotated with input/output shapes.

**Incompatible with CUDA graphs.** Use with `--optimization-level O0` or `--enforce-eager`.

```bash
# Profile with Nsight Systems
nsys profile --trace=cuda,nvtx \
  vllm serve mymodel \
    --enable-layerwise-nvtx-tracing \
    --optimization-level O0
```

---

## MFU Metrics

### `enable_mfu_metrics`

```
Type:    bool
Default: False
CLI:     --enable-mfu-metrics
```

Enable Model FLOPs Utilization (MFU) metrics. MFU measures how efficiently the GPU's theoretical peak FLOP/s are being utilized.

```bash
vllm serve mymodel --enable-mfu-metrics
```

---

## Iteration Logging

### `enable_logging_iteration_details`

```
Type:    bool
Default: False
CLI:     --enable-logging-iteration-details
```

Enable detailed per-iteration logging in the EngineCore. Logs include:
- Number of context/generation requests
- Number of context/generation tokens
- Elapsed CPU time per iteration

```bash
vllm serve mymodel --enable-logging-iteration-details
```

---

## Stats Logging

### `disable_log_stats` (EngineArgs)

```
Type:    bool
Default: False
CLI:     --disable-log-stats
```

Disable periodic stats logging. When disabled, vLLM will not log throughput, latency, and queue statistics at regular intervals.

```bash
# Disable stats logging (e.g., for benchmarking)
vllm serve mymodel --disable-log-stats
```

### `VLLM_LOG_STATS_INTERVAL` (environment variable)

```
Type:    float
Default: 10.0 (seconds)
```

Interval in seconds between stats log entries. Set to a higher value to reduce log verbosity.

```bash
export VLLM_LOG_STATS_INTERVAL=30
vllm serve mymodel
```

---

## Logging Configuration

vLLM's logging is configured via environment variables:

| Variable | Default | Description |
|---|---|---|
| `VLLM_CONFIGURE_LOGGING` | `1` | Enable vLLM logging configuration |
| `VLLM_LOGGING_LEVEL` | `"INFO"` | Log level: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `VLLM_LOGGING_PREFIX` | `""` | Prefix prepended to all log messages |
| `VLLM_LOGGING_STREAM` | `"ext://sys.stdout"` | Log output stream |
| `VLLM_LOGGING_CONFIG_PATH` | `None` | Path to custom logging config file |
| `VLLM_LOGGING_COLOR` | `"auto"` | Color output: `auto`, `1` (always), `0` (never) |
| `NO_COLOR` | `0` | Standard Unix flag to disable ANSI colors |
| `VLLM_TRACE_FUNCTION` | `0` | Enable function call tracing (debugging) |

```bash
# Debug logging
export VLLM_LOGGING_LEVEL=DEBUG
vllm serve mymodel

# Custom log prefix
export VLLM_LOGGING_PREFIX="[instance-1] "
vllm serve mymodel
```

---

## Multimodal Processor Stats

### `enable_mm_processor_stats`

```
Type:    bool
Default: False
```

Enable collection of timing statistics for multimodal processor operations. This is for internal use (benchmarks) and is not exposed as a CLI argument.

---

## Complete Observability Setup

### Production monitoring with Prometheus + Grafana

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-metrics \
  --enable-mfu-metrics \
  --cudagraph-metrics \
  --enable-logging-iteration-details
```

Then scrape `/metrics` with Prometheus and visualize in Grafana.

### Distributed tracing with Jaeger

```bash
# Start Jaeger
docker run -d --name jaeger \
  -p 16686:16686 \
  -p 4317:4317 \
  jaegertracing/all-in-one:latest

# Start vLLM with tracing
vllm serve mymodel \
  --otlp-traces-endpoint http://localhost:4317 \
  --collect-detailed-traces model worker
```

View traces at `http://localhost:16686`.

### GPU profiling with Nsight Systems

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --output=/tmp/vllm_profile \
  vllm serve mymodel \
    --enable-layerwise-nvtx-tracing \
    --optimization-level O0 \
    --max-num-seqs 1
```

### Benchmark mode (minimal overhead)

```bash
vllm serve mymodel \
  --disable-log-stats \
  --optimization-level O2
```

---

## Key Metrics Reference

When stats logging is enabled, vLLM logs these metrics periodically:

| Metric | Description |
|---|---|
| `avg_prompt_throughput` | Average prompt tokens/second |
| `avg_generation_throughput` | Average generation tokens/second |
| `scheduler_running` | Number of sequences currently running |
| `scheduler_waiting` | Number of sequences waiting in queue |
| `scheduler_swapped` | Number of sequences swapped to CPU |
| `gpu_cache_usage` | GPU KV cache utilization (0–1) |
| `cpu_cache_usage` | CPU KV cache utilization (0–1) |
| `num_preemptions` | Number of preemption events |

These are also exposed as Prometheus metrics at `/metrics`.
