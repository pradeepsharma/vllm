# ObservabilityConfig Reference

`ObservabilityConfig` is the central configuration object for all vLLM observability features. It controls OpenTelemetry tracing, Prometheus metric collection, KV cache residency metrics, CUDA graph metrics, and MFU (Model FLOPs Utilization) reporting.

**Source:** `vllm/config/observability.py`

## Configuration Fields

### `otlp_traces_endpoint`

```python
otlp_traces_endpoint: str | None = None
```

The OTLP endpoint URL to which OpenTelemetry traces are exported. When set, vLLM initializes an OTel `TracerProvider` with a `BatchSpanProcessor` and exports spans to this endpoint.

**CLI flag:** `--otlp-traces-endpoint`

**Examples:**
```bash
# gRPC (default protocol)
--otlp-traces-endpoint http://localhost:4317

# HTTP/protobuf (set OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf)
--otlp-traces-endpoint http://localhost:4318/v1/traces
```

**Validation:** If set, the OpenTelemetry SDK packages must be installed. If they are not, vLLM raises a `ValueError` with the import error traceback.

---

### `collect_detailed_traces`

```python
collect_detailed_traces: list[DetailedTraceModules] | None = None
```

Controls which modules emit detailed timing spans. Only meaningful when `otlp_traces_endpoint` is also set.

**Type:** `list[Literal["model", "worker", "all"]]`

**CLI flag:** `--collect-detailed-traces`

| Value | Effect |
|-------|--------|
| `model` | Enables `gen_ai.latency.time_in_model_forward` span attribute — traces the model forward pass timing |
| `worker` | Enables `gen_ai.latency.time_in_model_execute` span attribute — traces the worker execute call |
| `all` | Enables both `model` and `worker` tracing |

**Derived properties:**

```python
@cached_property
def collect_model_forward_time(self) -> bool:
    """Whether to collect model forward time for the request."""
    return self.collect_detailed_traces is not None and (
        "model" in self.collect_detailed_traces
        or "all" in self.collect_detailed_traces
    )

@cached_property
def collect_model_execute_time(self) -> bool:
    """Whether to collect model execute time for the request."""
    return self.collect_detailed_traces is not None and (
        "worker" in self.collect_detailed_traces
        or "all" in self.collect_detailed_traces
    )
```

> **Performance Warning:** Collecting detailed traces involves blocking operations and can impact throughput. Use only for debugging or profiling.

**Validation:** Raises `ValueError` if `collect_detailed_traces` is set without `otlp_traces_endpoint`.

---

### `show_hidden_metrics_for_version`

```python
show_hidden_metrics_for_version: str | None = None
```

Re-enables deprecated Prometheus metrics that were hidden in a specific vLLM release. This is a temporary escape hatch while migrating to new metric names.

**CLI flag:** `--show-hidden-metrics-for-version`

**Example:**
```bash
# Re-enable metrics deprecated since v0.7.0
--show-hidden-metrics-for-version 0.7
```

**Derived property:**

```python
@cached_property
def show_hidden_metrics(self) -> bool:
    if self.show_hidden_metrics_for_version is None:
        return False
    return version._prev_minor_version_was(self.show_hidden_metrics_for_version)
```

**Validation:** The version string must be a valid PEP 440 version (validated with `packaging.version.parse`).

---

### `kv_cache_metrics`

```python
kv_cache_metrics: bool = False
```

Enables KV cache residency metrics: block lifetime, idle time before eviction, and reuse gap histograms. Uses sampling to minimize overhead.

**CLI flag:** `--kv-cache-metrics`

**Requires:** Log stats must be enabled (i.e., `--disable-log-stats` must NOT be set).

When enabled, registers three additional Prometheus histograms:
- `vllm:kv_block_lifetime_seconds`
- `vllm:kv_block_idle_before_evict_seconds`
- `vllm:kv_block_reuse_gap_seconds`

---

### `kv_cache_metrics_sample`

```python
kv_cache_metrics_sample: float = Field(default=0.01, gt=0, le=1)
```

Sampling rate for KV cache metrics. Controls what fraction of KV cache blocks are tracked for residency metrics.

**CLI flag:** `--kv-cache-metrics-sample`

**Range:** `(0.0, 1.0]` — must be greater than 0 and at most 1.

**Default:** `0.01` (1% of blocks)

**Example:**
```bash
# Track 5% of blocks for higher accuracy
--kv-cache-metrics --kv-cache-metrics-sample 0.05
```

---

### `cudagraph_metrics`

```python
cudagraph_metrics: bool = False
```

Enables CUDA graph metrics: number of padded/unpadded tokens, runtime CUDA graph dispatch modes, and their observed frequencies at every logging interval.

**CLI flag:** `--cudagraph-metrics`

---

### `enable_mfu_metrics`

```python
enable_mfu_metrics: bool = False
```

Enables Model FLOPs Utilization (MFU) metrics. When enabled, registers three Prometheus counters:
- `vllm:estimated_flops_per_gpu_total`
- `vllm:estimated_read_bytes_per_gpu_total`
- `vllm:estimated_write_bytes_per_gpu_total`

**CLI flag:** `--enable-mfu-metrics`

These counters allow computing GPU utilization efficiency:

```promql
# Average TFLOPS per GPU
rate(vllm:estimated_flops_per_gpu_total[1m]) / 1e12

# Memory bandwidth in GB/s
(rate(vllm:estimated_read_bytes_per_gpu_total[1m]) +
 rate(vllm:estimated_write_bytes_per_gpu_total[1m])) / 1e9
```

---

### `enable_layerwise_nvtx_tracing`

```python
enable_layerwise_nvtx_tracing: bool = False
```

Enables layerwise NVTX tracing. Traces the execution of each layer or module in the model and attaches information such as input/output shapes to NVTX range markers.

> **Note:** Does not work with CUDA graphs enabled.

---

### `enable_mm_processor_stats`

```python
enable_mm_processor_stats: bool = False
```

Enables collection of timing statistics for multimodal processor operations. This is for internal use only (e.g., benchmarks) and is not exposed as a CLI argument.

---

### `enable_logging_iteration_details`

```python
enable_logging_iteration_details: bool = False
```

Enables detailed logging of iteration details. When set, the vLLM EngineCore logs per-iteration details including:
- Number of context/generation requests and tokens
- Elapsed CPU time for the iteration

**CLI flag:** `--enable-logging-iteration-details`

---

## Complete Configuration Example

### Via CLI

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --otlp-traces-endpoint http://localhost:4317 \
  --collect-detailed-traces model,worker \
  --kv-cache-metrics \
  --kv-cache-metrics-sample 0.05 \
  --enable-mfu-metrics \
  --cudagraph-metrics
```

### Via Python API

```python
from vllm import LLM
from vllm.config import ObservabilityConfig

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    observability_config=ObservabilityConfig(
        otlp_traces_endpoint="http://localhost:4317",
        collect_detailed_traces=["model", "worker"],
        kv_cache_metrics=True,
        kv_cache_metrics_sample=0.05,
        enable_mfu_metrics=True,
    ),
)
```

### Via `VllmConfig`

```python
from vllm.config import VllmConfig, ObservabilityConfig

config = VllmConfig(
    model="meta-llama/Llama-3.1-8B-Instruct",
    observability_config=ObservabilityConfig(
        otlp_traces_endpoint="http://jaeger:4317",
        collect_detailed_traces=["all"],
        enable_mfu_metrics=True,
    ),
)
```

## Field Summary Table

| Field | Type | Default | CLI Flag | Description |
|-------|------|---------|----------|-------------|
| `otlp_traces_endpoint` | `str \| None` | `None` | `--otlp-traces-endpoint` | OTLP endpoint for trace export |
| `collect_detailed_traces` | `list[str] \| None` | `None` | `--collect-detailed-traces` | Modules for detailed tracing |
| `show_hidden_metrics_for_version` | `str \| None` | `None` | `--show-hidden-metrics-for-version` | Re-enable deprecated metrics |
| `kv_cache_metrics` | `bool` | `False` | `--kv-cache-metrics` | Enable KV cache residency metrics |
| `kv_cache_metrics_sample` | `float` | `0.01` | `--kv-cache-metrics-sample` | Sampling rate for KV metrics |
| `cudagraph_metrics` | `bool` | `False` | `--cudagraph-metrics` | Enable CUDA graph metrics |
| `enable_mfu_metrics` | `bool` | `False` | `--enable-mfu-metrics` | Enable MFU Prometheus counters |
| `enable_layerwise_nvtx_tracing` | `bool` | `False` | — | Enable NVTX layer tracing |
| `enable_mm_processor_stats` | `bool` | `False` | — | Enable multimodal processor timing |
| `enable_logging_iteration_details` | `bool` | `False` | `--enable-logging-iteration-details` | Log per-iteration details |

## Validation Rules

1. `show_hidden_metrics_for_version` must be a valid PEP 440 version string
2. `otlp_traces_endpoint` requires OpenTelemetry SDK packages to be installed
3. `collect_detailed_traces` requires `otlp_traces_endpoint` to be set
4. `kv_cache_metrics_sample` must be in the range `(0.0, 1.0]`

## Related Pages

- [OpenTelemetry Tracing](tracing.md)
- [Prometheus Metrics Reference](metrics.md)
- [Structured Logging](logging.md)
- [API Endpoints](endpoints.md)
