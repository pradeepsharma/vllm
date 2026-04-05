# Observability API Endpoints

vLLM exposes several HTTP endpoints for monitoring and introspection. This page documents the `/metrics` Prometheus scrape endpoint and the `/server_info` diagnostic endpoint.

## `/metrics` — Prometheus Scrape Endpoint

### Overview

| Property | Value |
|----------|-------|
| **Path** | `GET /metrics` |
| **Content-Type** | `text/plain; version=0.0.4; charset=utf-8` |
| **Authentication** | None |
| **Format** | Prometheus text exposition format |
| **Source** | `vllm/entrypoints/serve/instrumentator/metrics.py` |

The `/metrics` endpoint is the primary integration point for Prometheus. It returns all registered vLLM metrics in the standard Prometheus text format, which Prometheus can scrape and store in its time-series database.

### Example Request

```bash
curl http://localhost:8000/metrics
```

### Example Response

```
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 4.0

# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.0

# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 0.4219

# HELP vllm:prompt_tokens_total Number of prefill tokens processed.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine="0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 2097152.0
vllm:prompt_tokens_created{engine="0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 1743859381.0

# HELP vllm:generation_tokens_total Number of generation tokens processed.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="meta-llama/Llama-3.1-8B-Instruct"} 524288.0

# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.001",...} 0.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.005",...} 12.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.01",...} 89.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.1",...} 4521.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf",...} 5678.0
vllm:time_to_first_token_seconds_sum{...} 892.3
vllm:time_to_first_token_seconds_count{...} 5678.0

# HELP vllm:e2e_request_latency_seconds Histogram of e2e request latency in seconds.
# TYPE vllm:e2e_request_latency_seconds histogram
...

# HELP vllm:cache_config_info Information of the LLMEngine CacheConfig
# TYPE vllm:cache_config_info gauge
vllm:cache_config_info{block_size="16",cache_dtype="auto",...,engine="0"} 1.0
```

### Prometheus Scrape Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
    scrape_timeout: 10s
    # Optional: add labels to all metrics from this target
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
```

### Kubernetes ServiceMonitor (Prometheus Operator)

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vllm
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: vllm
  endpoints:
    - port: http
      path: /metrics
      interval: 15s
```

### Kubernetes Pod Annotations (Prometheus auto-discovery)

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8000"
    prometheus.io/path: "/metrics"
```

### Implementation Details

The `/metrics` endpoint is implemented using two mechanisms:

1. **`prometheus-fastapi-instrumentator`** — Instruments the FastAPI app and exposes HTTP request metrics (latency, status codes) for all non-excluded endpoints.

2. **`make_asgi_app(registry=registry)`** — A pure ASGI app mounted at `/metrics` that serves the Prometheus text format. This handles the actual vLLM metrics.

The `PrometheusResponse` class ensures the correct `Content-Type` header:

```python
class PrometheusResponse(Response):
    media_type = prometheus_client.CONTENT_TYPE_LATEST
    # Content-Type: text/plain; version=0.0.4; charset=utf-8
```

---

## `/server_info` — Server Diagnostic Endpoint

### Overview

| Property | Value |
|----------|-------|
| **Path** | `GET /server_info` |
| **Content-Type** | `application/json` |
| **Authentication** | None |
| **Source** | `vllm/entrypoints/serve/instrumentator/server_info.py` |

The `/server_info` endpoint returns comprehensive diagnostic information about the running vLLM server, including the full configuration, all VLLM environment variables, and system environment information.

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_format` | `"text"` or `"json"` | `"text"` | Format for the `vllm_config` field |

### Example Request

```bash
# Text format (default) — human-readable config string
curl http://localhost:8000/server_info

# JSON format — machine-readable config
curl "http://localhost:8000/server_info?config_format=json"
```

### Response Schema

```json
{
  "vllm_config": "<VllmConfig string or JSON object>",
  "vllm_env": {
    "VLLM_LOGGING_LEVEL": "INFO",
    "VLLM_LOG_STATS_INTERVAL": 10.0,
    "VLLM_WORKER_MULTIPROC_METHOD": "fork",
    "VLLM_USE_V1": true,
    "..."
  },
  "system_env": {
    "PyTorch Version": "2.5.1",
    "CUDA Version": "12.4",
    "GPU": "NVIDIA A100-SXM4-80GB",
    "Driver Version": "550.54.15",
    "..."
  }
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `vllm_config` | `string` or `object` | Full `VllmConfig` — as a string (text format) or JSON object (json format) |
| `vllm_env` | `object` | All `VLLM_*` environment variables and their current values |
| `system_env` | `object` | System information from `vllm.collect_env.get_env_info()` |

### Implementation

```python
# vllm/entrypoints/serve/instrumentator/server_info.py

@router.get("/server_info")
async def show_server_info(
    raw_request: Request,
    config_format: Annotated[Literal["text", "json"], Query()] = "text",
):
    vllm_config: VllmConfig = raw_request.app.state.vllm_config
    server_info = {
        "vllm_config": (
            str(vllm_config)
            if config_format == "text"
            else PydanticVllmConfig.dump_python(vllm_config, mode="json", fallback=str)
        ),
        "vllm_env": _get_vllm_env_vars(),
        "system_env": await asyncio.to_thread(_get_system_env_info_cached),
    }
    return JSONResponse(content=server_info)
```

Key implementation details:
- `vllm_config` is read from `app.state.vllm_config` (set during server startup)
- `vllm_env` collects all attributes from `vllm.envs` that start with `VLLM_` (excluding keys containing `KEY` for security)
- `system_env` is cached with `@functools.lru_cache(maxsize=1)` since system info doesn't change
- System info collection runs in a thread pool (`asyncio.to_thread`) to avoid blocking the event loop

### Use Cases

1. **Debugging configuration issues** — Verify that CLI flags were applied correctly
2. **Audit logging** — Record the exact configuration at deployment time
3. **Monitoring dashboards** — Display server metadata alongside metrics
4. **Support tickets** — Provide complete environment information

---

## `/load` — Server Load Metrics

| Property | Value |
|----------|-------|
| **Path** | `GET /load` |
| **Content-Type** | `application/json` |
| **Source** | `vllm/entrypoints/serve/instrumentator/basic.py` |

Returns the current server load metric, which tracks the number of active GPU-utilizing requests:

```bash
curl http://localhost:8000/load
```

```json
{"server_load": 4}
```

The `server_load` value counts active requests across these endpoints:
- `/v1/chat/completions`
- `/v1/completions`
- `/v1/embeddings`
- `/v1/responses`
- `/pooling`, `/classify`, `/score`, `/rerank`
- `/v1/audio/transcriptions`, `/v1/audio/translations`

---

## `/version` — Version Information

| Property | Value |
|----------|-------|
| **Path** | `GET /version` |
| **Content-Type** | `application/json` |

```bash
curl http://localhost:8000/version
```

```json
{"version": "0.9.0"}
```

---

## `/health` — Health Check

| Property | Value |
|----------|-------|
| **Path** | `GET /health` |
| **Content-Type** | `application/json` |

Returns HTTP 200 when the server is healthy and ready to serve requests. Used by Kubernetes liveness and readiness probes.

```bash
curl -f http://localhost:8000/health && echo "healthy"
```

---

## Endpoint Exclusions from Metrics

The following endpoints are excluded from HTTP request instrumentation (they do not appear in `http_requests_total` or `http_request_duration_seconds` metrics):

```python
excluded_handlers=[
    "/metrics",
    "/health",
    "/load",
    "/ping",
    "/version",
    "/server_info",
]
```

This prevents fast, non-inference endpoints from skewing latency histograms.

## Related Pages

- [Prometheus Metrics Reference](metrics.md)
- [FastAPI Instrumentator](instrumentator.md)
- [ObservabilityConfig Reference](observability-config.md)
