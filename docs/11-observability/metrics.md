# Prometheus Metrics Reference

vLLM exposes a rich set of Prometheus metrics covering scheduler state, token throughput, request latency, KV cache utilization, speculative decoding, and more. All metrics use the `vllm:` namespace prefix and are labeled with `model_name` and `engine` to support multi-model and data-parallel deployments.

## Metric Labels

Every vLLM metric carries at minimum two labels:

| Label | Description | Example |
|-------|-------------|---------|
| `model_name` | The served model name (from `--served-model-name` or model path) | `meta-llama/Llama-3.1-8B-Instruct` |
| `engine` | Engine index for data-parallel deployments | `0`, `1`, `2` |

Some metrics carry additional labels (e.g., `finished_reason`, `source`, `position`, `sleep_state`).

## Scheduler State Metrics

These gauges reflect the real-time state of the vLLM scheduler.

| Metric Name | Type | Description |
|-------------|------|-------------|
| `vllm:num_requests_running` | Gauge | Number of requests currently in model execution batches |
| `vllm:num_requests_waiting` | Gauge | Number of requests waiting to be processed (queue depth) |
| `vllm:kv_cache_usage_perc` | Gauge | KV cache utilization as a fraction (0.0 = empty, 1.0 = full) |
| `vllm:engine_sleep_state` | Gauge | Engine sleep state; labeled by `sleep_state` (`awake`, `weights_offloaded`, `discard_all`) |

### Engine Sleep State

The `vllm:engine_sleep_state` metric has an additional `sleep_state` label with three possible values:

- `awake` — 1 when the engine is active, 0 when sleeping
- `weights_offloaded` — 1 when sleep level 1 (weights offloaded to CPU)
- `discard_all` — 1 when sleep level 2 (all state discarded)

## Token Throughput Counters

These counters track cumulative token processing volumes.

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `vllm:prompt_tokens_total` | Counter | `model_name`, `engine` | Total prefill tokens processed |
| `vllm:generation_tokens_total` | Counter | `model_name`, `engine` | Total generation tokens produced |
| `vllm:prompt_tokens_by_source_total` | Counter | `model_name`, `engine`, `source` | Prompt tokens broken down by source (e.g., `computed`, `cached`, `transferred`) |
| `vllm:prompt_tokens_cached_total` | Counter | `model_name`, `engine` | Cached prompt tokens (local + external) |
| `vllm:prompt_tokens_recomputed_total` | Counter | `model_name`, `engine` | Cached tokens recomputed for forward pass |
| `vllm:num_preemptions_total` | Counter | `model_name`, `engine` | Cumulative preemptions from the engine |
| `vllm:corrupted_requests_total` | Counter | `model_name`, `engine` | Requests with NaN values in logits (only when `VLLM_COMPUTE_NANS_IN_LOGITS=1`) |

### Prompt Token Sources

The `source` label on `vllm:prompt_tokens_by_source_total` can take values defined in `PromptTokenStats.ALL_SOURCES`, which includes categories such as:
- `computed` — tokens that required a full forward pass
- `cached` — tokens served from the prefix cache
- `transferred` — tokens transferred via KV connector

## Request Outcome Counters

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `vllm:request_success_total` | Counter | `model_name`, `engine`, `finished_reason` | Successfully completed requests, labeled by finish reason |

The `finished_reason` label takes values from the `FinishReason` enum: `stop`, `length`, `abort`, `error`, etc.

## Prefix Cache Metrics

| Metric Name | Type | Description |
|-------------|------|-------------|
| `vllm:prefix_cache_queries_total` | Counter | Prefix cache queries (number of queried tokens) |
| `vllm:prefix_cache_hits_total` | Counter | Prefix cache hits (number of cached tokens served) |
| `vllm:external_prefix_cache_queries_total` | Counter | External prefix cache queries via KV connector |
| `vllm:external_prefix_cache_hits_total` | Counter | External prefix cache hits via KV connector |

## Multi-Modal Cache Metrics

| Metric Name | Type | Description |
|-------------|------|-------------|
| `vllm:mm_cache_queries_total` | Counter | Multi-modal cache queries (number of queried items) |
| `vllm:mm_cache_hits_total` | Counter | Multi-modal cache hits (number of cached items) |

## Request Latency Histograms

These histograms capture per-request timing distributions. All use the same bucket set:
`[0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0, 7680.0]` seconds.

| Metric Name | Type | Description |
|-------------|------|-------------|
| `vllm:e2e_request_latency_seconds` | Histogram | End-to-end request latency (arrival to last token) |
| `vllm:request_queue_time_seconds` | Histogram | Time spent in WAITING phase (queue time) |
| `vllm:request_inference_time_seconds` | Histogram | Time spent in RUNNING phase (inference time) |
| `vllm:request_prefill_time_seconds` | Histogram | Time spent in PREFILL phase |
| `vllm:request_decode_time_seconds` | Histogram | Time spent in DECODE phase |

## Token Latency Histograms

| Metric Name | Type | Buckets | Description |
|-------------|------|---------|-------------|
| `vllm:time_to_first_token_seconds` | Histogram | `[0.001, 0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0, 160.0, 640.0, 2560.0]` | Time from request arrival to first generated token |
| `vllm:inter_token_latency_seconds` | Histogram | `[0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, 20.0, 40.0, 80.0]` | Time between consecutive generated tokens |
| `vllm:request_time_per_output_token_seconds` | Histogram | Same as ITL | Mean time per output token for each request |

## Request Parameter Histograms

| Metric Name | Type | Buckets | Description |
|-------------|------|---------|-------------|
| `vllm:request_prompt_tokens` | Histogram | 1-2-5 series up to `max_model_len` | Number of prefill tokens per request |
| `vllm:request_generation_tokens` | Histogram | 1-2-5 series up to `max_model_len` | Number of generation tokens per request |
| `vllm:request_max_num_generation_tokens` | Histogram | 1-2-5 series up to `max_model_len` | Maximum requested generation tokens per request |
| `vllm:request_params_n` | Histogram | `[1, 2, 5, 10, 20]` | Distribution of the `n` request parameter |
| `vllm:request_params_max_tokens` | Histogram | 1-2-5 series up to `max_model_len` | Distribution of the `max_tokens` request parameter |
| `vllm:request_prefill_kv_computed_tokens` | Histogram | 1-2-5 series up to `max_model_len` | New KV tokens computed during prefill (excluding cached) |
| `vllm:iteration_tokens_total` | Histogram | `[1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]` | Total tokens per engine step (prompt + generation) |

> **Note:** The 1-2-5 bucket series is generated dynamically based on `max_model_len`. For example, with `max_model_len=4096`, buckets would be `[1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000]`.

## KV Cache Residency Metrics (Optional)

These metrics are only collected when `--kv-cache-metrics` is enabled in `ObservabilityConfig`. They use sampling (default 1% of blocks) to minimize overhead.

| Metric Name | Type | Description |
|-------------|------|-------------|
| `vllm:kv_block_lifetime_seconds` | Histogram | KV cache block lifetime from allocation to eviction |
| `vllm:kv_block_idle_before_evict_seconds` | Histogram | Idle time before KV cache block eviction |
| `vllm:kv_block_reuse_gap_seconds` | Histogram | Time gaps between consecutive KV cache block accesses |

All three use the same bucket set: `[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30, 60, 120, 300, 600, 1200, 1800]` seconds.

Enable with:
```bash
vllm serve <model> --kv-cache-metrics --kv-cache-metrics-sample 0.05
```

## Speculative Decoding Metrics

These metrics are only registered when speculative decoding is enabled.

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `vllm:spec_decode_num_drafts_total` | Counter | `model_name`, `engine` | Number of speculative decoding drafts |
| `vllm:spec_decode_num_draft_tokens_total` | Counter | `model_name`, `engine` | Number of draft tokens generated |
| `vllm:spec_decode_num_accepted_tokens_total` | Counter | `model_name`, `engine` | Number of accepted draft tokens |
| `vllm:spec_decode_num_accepted_tokens_per_pos_total` | Counter | `model_name`, `engine`, `position` | Accepted tokens per draft position |

### Speculative Decoding PromQL Queries

```promql
# Draft acceptance rate
rate(vllm:spec_decode_num_accepted_tokens_total[5m]) /
rate(vllm:spec_decode_num_draft_tokens_total[5m])

# Mean acceptance length (including bonus token)
1 + (
  rate(vllm:spec_decode_num_accepted_tokens_total[5m]) /
  rate(vllm:spec_decode_num_drafts_total[5m])
)

# Per-position acceptance rate vector
rate(vllm:spec_decode_num_accepted_tokens_per_pos_total[5m]) /
rate(vllm:spec_decode_num_drafts_total[5m])
```

## LoRA Metrics

When LoRA adapters are enabled, an additional gauge is registered:

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `vllm:lora_requests_info` | Gauge | `max_lora`, `waiting_lora_adapters`, `running_lora_adapters` | Running stats on LoRA requests |

## WebSocket Metrics (Realtime API)

When using the WebSocket-based Realtime API, these metrics are registered:

| Metric Name | Type | Description |
|-------------|------|-------------|
| `vllm:websocket_connections_active` | Gauge | Number of currently active WebSocket connections |
| `vllm:websocket_connections_total` | Counter | Total number of WebSocket connections |
| `vllm:websocket_connection_duration_seconds` | Histogram | Duration of WebSocket connections (buckets: `[0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600, 1800]` seconds) |

## Model FLOPs Utilization (MFU) Metrics

Enable with `--enable-mfu-metrics`. These counters support computing GPU utilization:

| Metric Name | Type | Description |
|-------------|------|-------------|
| `vllm:estimated_flops_per_gpu_total` | Counter | Estimated floating-point operations per GPU |
| `vllm:estimated_read_bytes_per_gpu_total` | Counter | Estimated bytes read from GPU memory |
| `vllm:estimated_write_bytes_per_gpu_total` | Counter | Estimated bytes written to GPU memory |

### MFU PromQL Queries

```promql
# Average TFLOPS per GPU
rate(vllm:estimated_flops_per_gpu_total[1m]) / 1e12

# Average memory bandwidth in GB/s
(
  rate(vllm:estimated_read_bytes_per_gpu_total[1m]) +
  rate(vllm:estimated_write_bytes_per_gpu_total[1m])
) / 1e9
```

## Configuration Info Metric

| Metric Name | Type | Description |
|-------------|------|-------------|
| `vllm:cache_config_info` | Gauge | Information about the LLMEngine CacheConfig (always set to 1, labels carry config values) |

## Multiprocess Prometheus Support

vLLM automatically configures Prometheus multiprocessing mode when running with multiple API server workers. The `PROMETHEUS_MULTIPROC_DIR` environment variable is set to a temporary directory that is cleaned up on exit.

```python
# vllm/v1/metrics/prometheus.py
def setup_multiprocess_prometheus():
    if "PROMETHEUS_MULTIPROC_DIR" not in os.environ:
        _prometheus_multiproc_dir = tempfile.TemporaryDirectory()
        os.environ["PROMETHEUS_MULTIPROC_DIR"] = _prometheus_multiproc_dir.name
```

> **Warning:** If you set `PROMETHEUS_MULTIPROC_DIR` manually, you must wipe the directory between vLLM runs to avoid stale metrics.

## Example Prometheus Queries

### Throughput

```promql
# Prompt token throughput (tokens/sec)
rate(vllm:prompt_tokens_total[1m])

# Generation token throughput (tokens/sec)
rate(vllm:generation_tokens_total[1m])

# Request rate (requests/sec)
rate(vllm:request_success_total[1m])
```

### Latency

```promql
# P50 time to first token
histogram_quantile(0.50, rate(vllm:time_to_first_token_seconds_bucket[5m]))

# P99 time to first token
histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m]))

# P95 end-to-end latency
histogram_quantile(0.95, rate(vllm:e2e_request_latency_seconds_bucket[5m]))

# P99 inter-token latency
histogram_quantile(0.99, rate(vllm:inter_token_latency_seconds_bucket[5m]))
```

### Cache Efficiency

```promql
# KV cache utilization percentage
vllm:kv_cache_usage_perc * 100

# Prefix cache hit rate
rate(vllm:prefix_cache_hits_total[5m]) /
rate(vllm:prefix_cache_queries_total[5m])

# External (KV connector) prefix cache hit rate
rate(vllm:external_prefix_cache_hits_total[5m]) /
rate(vllm:external_prefix_cache_queries_total[5m])
```

### Queue Health

```promql
# Current queue depth
vllm:num_requests_waiting

# Ratio of waiting to running requests
vllm:num_requests_waiting / vllm:num_requests_running
```

## Grafana Dashboard Hints

### Recommended Panels

1. **Throughput Row**
   - Prompt tokens/sec: `rate(vllm:prompt_tokens_total[1m])`
   - Generation tokens/sec: `rate(vllm:generation_tokens_total[1m])`
   - Request rate: `rate(vllm:request_success_total[1m])`

2. **Latency Row**
   - TTFT heatmap: `vllm:time_to_first_token_seconds_bucket`
   - E2E latency P50/P95/P99 using `histogram_quantile`
   - ITL P50/P99 using `histogram_quantile`

3. **Resource Utilization Row**
   - KV cache usage: `vllm:kv_cache_usage_perc * 100`
   - Running requests: `vllm:num_requests_running`
   - Waiting requests: `vllm:num_requests_waiting`

4. **Cache Efficiency Row**
   - Prefix cache hit rate
   - External cache hit rate (if KV connector enabled)
   - MM cache hit rate (if multimodal)

5. **Speculative Decoding Row** (if enabled)
   - Acceptance rate
   - Mean acceptance length
   - Draft vs accepted throughput

### Alert Rules

```yaml
# High queue depth alert
- alert: VllmHighQueueDepth
  expr: vllm:num_requests_waiting > 100
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "vLLM queue depth is high"

# High TTFT alert
- alert: VllmHighTTFT
  expr: |
    histogram_quantile(0.95,
      rate(vllm:time_to_first_token_seconds_bucket[5m])
    ) > 5.0
  for: 5m
  labels:
    severity: warning

# KV cache near full
- alert: VllmKVCacheNearFull
  expr: vllm:kv_cache_usage_perc > 0.9
  for: 1m
  labels:
    severity: critical
```

## Related Pages

- [FastAPI Instrumentator & `/metrics` Endpoint](instrumentator.md)
- [ObservabilityConfig Reference](observability-config.md)
- [OpenTelemetry Tracing](tracing.md)
