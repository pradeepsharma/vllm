---
description: >
  Step-by-step guide to setting up Prometheus and Grafana to monitor vLLM,
  including scrape configuration, key PromQL queries, alert rules, and
  OpenTelemetry tracing with Jaeger.
---

# Monitoring setup

This guide walks you through connecting vLLM's built-in observability to a
production-grade monitoring stack. It covers:

1. [Prometheus scraping](#prometheus-setup) — collecting metrics from vLLM
1. [Grafana dashboards](#grafana-dashboard) — visualising metrics over time
1. [Key PromQL queries](#key-promql-queries) — the most useful queries for
   LLM serving
1. [Alert rules](#alert-rules) — example Prometheus alerting rules
1. [OpenTelemetry tracing](#opentelemetry-tracing) — distributed request
   tracing with Jaeger

For a complete reference of every observability configuration option, see
[Observability configuration](../configuration/observability_config.md).

---

## Prerequisites

- A running vLLM server (see [Docker deployment](docker.md) or
  [Kubernetes deployment](k8s.md))
- [Docker](https://docs.docker.com/engine/install/) and
  [Docker Compose](https://docs.docker.com/compose/install/) for the
  quick-start examples
- Basic familiarity with [Prometheus](https://prometheus.io/) and
  [Grafana](https://grafana.com/)

---

## Prometheus setup

### Verifying the metrics endpoint

Prometheus metrics are enabled by default. No additional flags are required.
Start vLLM and verify the endpoint is reachable:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct

# In a second terminal
curl http://localhost:8000/metrics | head -40
```

You should see output similar to:

```text
# HELP vllm:num_requests_running Number of requests in model execution batches.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="meta-llama/Llama-3.1-8B-Instruct",engine="0"} 0.0

# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{model_name="meta-llama/Llama-3.1-8B-Instruct",engine="0"} 0.0

# HELP vllm:time_to_first_token_seconds Histogram of time to first token in seconds.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{le="0.001",...} 0.0
...
```

### Prometheus configuration

Create a `prometheus.yaml` scrape configuration that targets your vLLM
instance:

```yaml
# prometheus.yaml
global:
  scrape_interval: 5s       # How often Prometheus polls vLLM
  evaluation_interval: 30s  # How often alert rules are evaluated

scrape_configs:
  - job_name: vllm
    static_configs:
      - targets:
          - 'localhost:8000'   # Replace with your vLLM host:port
    # Optional: add labels to distinguish multiple instances
    # relabel_configs:
    #   - target_label: instance
    #     replacement: "prod-llama-3"
```

!!! tip
    A scrape interval of 5 seconds gives good resolution for latency histograms
    without adding significant overhead. For high-traffic deployments, 15–30
    seconds is also reasonable.

### Docker Compose quick start

The vLLM repository includes a ready-to-use Docker Compose configuration at
`examples/online_serving/prometheus_grafana/`. Use it to spin up Prometheus
and Grafana alongside a local vLLM server:

**1. Start vLLM:**

```bash
vllm serve mistralai/Mistral-7B-v0.1 --max-model-len 2048
```

**2. Start Prometheus and Grafana:**

```bash
cd examples/online_serving/prometheus_grafana
docker compose up
```

This starts:

- **Prometheus** at `http://localhost:9090` — scrapes `host.docker.internal:8000`
  every 5 seconds
- **Grafana** at `http://localhost:3000` — default credentials `admin` / `admin`

**3. Generate some traffic:**

```bash
vllm bench serve \
    --model mistralai/Mistral-7B-v0.1 \
    --endpoint /v1/completions \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --request-rate 3.0
```

### Scraping multiple vLLM instances

When running multiple vLLM instances (for example, with data parallelism or
multiple model replicas), add each instance as a separate target:

```yaml
scrape_configs:
  - job_name: vllm
    static_configs:
      - targets:
          - 'vllm-host-1:8000'
          - 'vllm-host-2:8000'
          - 'vllm-host-3:8000'
```

Every vLLM metric carries a `model_name` label, so you can aggregate or
filter by model in PromQL even when multiple models are served.

### Kubernetes ServiceMonitor

If you are using the Prometheus Operator on Kubernetes, create a
`ServiceMonitor` resource to scrape vLLM pods automatically:

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
      interval: 5s
  namespaceSelector:
    matchNames:
      - default
```

---

## Grafana dashboard

### Importing the reference dashboard

vLLM ships a reference Grafana dashboard at
`examples/online_serving/prometheus_grafana/grafana.json`.

**To import it:**

1. Open Grafana at `http://localhost:3000` and log in.
1. Navigate to **Connections → Data sources → Add data source** and select
   **Prometheus**.
1. Set the Prometheus server URL to `http://prometheus:9090` (when using
   Docker Compose) or your Prometheus host.
1. Click **Save & Test** — you should see a green success message.
1. Navigate to **Dashboards → Import**, upload `grafana.json`, and select
   the Prometheus data source you just added.

### Dashboard panels

The reference dashboard includes the following panels, which represent the
most operationally important metrics for LLM serving:

| Panel | Metric | Why it matters |
|---|---|---|
| End-to-end latency | `vllm:e2e_request_latency_seconds` | Primary user-facing SLO |
| Time to first token (TTFT) | `vllm:time_to_first_token_seconds` | Streaming responsiveness |
| Inter-token latency (TPOT) | `vllm:inter_token_latency_seconds` | Streaming smoothness |
| Requests running / waiting | `vllm:num_requests_running`, `vllm:num_requests_waiting` | Scheduler saturation |
| KV cache usage | `vllm:kv_cache_usage_perc` | Memory pressure |
| Prompt throughput | `vllm:prompt_tokens` | Prefill capacity |
| Generation throughput | `vllm:generation_tokens` | Decode capacity |
| Request prompt length | `vllm:request_prompt_tokens` | Input distribution |
| Request generation length | `vllm:request_generation_tokens` | Output distribution |
| Finished requests by reason | `vllm:request_success` | EOS vs length truncation |
| Queue time | `vllm:request_queue_time_seconds` | Scheduling backlog |
| Prefill time | `vllm:request_prefill_time_seconds` | Prefill performance |
| Decode time | `vllm:request_decode_time_seconds` | Decode performance |
| Max generation tokens | `vllm:request_max_num_generation_tokens` | Batch composition |

---

## Key PromQL queries

The following PromQL queries are useful starting points for building custom
dashboards and alert rules.

### Latency percentiles

```promql
# 50th, 95th, and 99th percentile TTFT over the past 5 minutes
histogram_quantile(0.50, rate(vllm:time_to_first_token_seconds_bucket[5m]))
histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket[5m]))
histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket[5m]))

# 95th percentile end-to-end latency
histogram_quantile(0.95, rate(vllm:e2e_request_latency_seconds_bucket[5m]))

# 95th percentile inter-token latency (TPOT)
histogram_quantile(0.95, rate(vllm:inter_token_latency_seconds_bucket[5m]))
```

### Throughput

```promql
# Prompt tokens per second (computed tokens only, excludes cache hits)
rate(vllm:prompt_tokens[1m])

# Generation tokens per second
rate(vllm:generation_tokens[1m])

# Total tokens per second (prompt + generation)
rate(vllm:prompt_tokens[1m]) + rate(vllm:generation_tokens[1m])
```

### Prefix cache hit rate

```promql
# Prefix cache hit rate over the past 5 minutes
rate(vllm:prefix_cache_hits[5m]) / rate(vllm:prefix_cache_queries[5m])

# External (KV connector) prefix cache hit rate
rate(vllm:external_prefix_cache_hits[5m]) / rate(vllm:external_prefix_cache_queries[5m])
```

### KV cache pressure

```promql
# Current KV cache utilisation (0–1)
vllm:kv_cache_usage_perc

# Alert threshold: cache above 90%
vllm:kv_cache_usage_perc > 0.9
```

### Request queue depth

```promql
# Number of requests waiting to be scheduled
vllm:num_requests_waiting

# Ratio of waiting to running requests (high ratio = saturation)
vllm:num_requests_waiting / (vllm:num_requests_running + 1)
```

### Request success and failure

```promql
# Requests finishing with EOS token per second
rate(vllm:request_success{finished_reason="stop"}[5m])

# Requests truncated at max length per second
rate(vllm:request_success{finished_reason="length"}[5m])

# Aborted requests per second
rate(vllm:request_success{finished_reason="abort"}[5m])
```

### HTTP-level metrics

vLLM instruments the FastAPI server with
[prometheus-fastapi-instrumentator](https://github.com/trallnag/prometheus-fastapi-instrumentator),
exposing HTTP-level metrics:

```promql
# Total HTTP requests per second by handler
rate(http_requests_total[1m])

# HTTP request duration 95th percentile
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# HTTP error rate (non-2xx responses)
rate(http_requests_total{status!~"2.."}[5m])
```

### Model FLOPs Utilization (MFU)

Requires `--enable-mfu-metrics`. Replace `<peak_flops>` with your GPU's
theoretical peak FLOPs per second (for example, `312e12` for an A100 80 GB
at BF16).

```promql
# MFU over the past 1 minute
rate(vllm:estimated_flops_per_gpu_total[1m]) / <peak_flops>

# Memory bandwidth utilisation (read)
rate(vllm:estimated_read_bytes_per_gpu_total[1m]) / <peak_memory_bandwidth>
```

---

## Alert rules

The following Prometheus alerting rules cover the most common production
failure modes. Add them to your Prometheus configuration under `rule_files`.

```yaml
# vllm-alerts.yaml
groups:
  - name: vllm
    rules:

      # KV cache nearly full — risk of increased preemptions
      - alert: VllmKVCacheHighUsage
        expr: vllm:kv_cache_usage_perc > 0.90
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "vLLM KV cache usage above 90%"
          description: >
            KV cache utilisation is {{ $value | humanizePercentage }} for
            model {{ $labels.model_name }}. Consider scaling out or reducing
            concurrent request load.

      # Large request backlog — scheduler is saturated
      - alert: VllmHighRequestQueueDepth
        expr: vllm:num_requests_waiting > 100
        for: 1m
        labels:
          severity: warning
        annotations:
          summary: "vLLM request queue depth above 100"
          description: >
            {{ $value }} requests are waiting to be scheduled for model
            {{ $labels.model_name }}. The engine may be saturated.

      # TTFT SLO breach — p95 above 5 seconds
      - alert: VllmHighTTFT
        expr: >
          histogram_quantile(0.95,
            rate(vllm:time_to_first_token_seconds_bucket[5m])
          ) > 5
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "vLLM p95 TTFT above 5 seconds"
          description: >
            The 95th percentile time-to-first-token for model
            {{ $labels.model_name }} is {{ $value | humanizeDuration }},
            which exceeds the 5-second SLO.

      # End-to-end latency SLO breach — p95 above 60 seconds
      - alert: VllmHighE2ELatency
        expr: >
          histogram_quantile(0.95,
            rate(vllm:e2e_request_latency_seconds_bucket[5m])
          ) > 60
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "vLLM p95 end-to-end latency above 60 seconds"
          description: >
            The 95th percentile end-to-end latency for model
            {{ $labels.model_name }} is {{ $value | humanizeDuration }}.

      # High abort rate — clients disconnecting or timeouts
      - alert: VllmHighAbortRate
        expr: >
          rate(vllm:request_success{finished_reason="abort"}[5m])
          / rate(vllm:request_success[5m]) > 0.05
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "vLLM abort rate above 5%"
          description: >
            More than 5% of requests for model {{ $labels.model_name }} are
            being aborted. Check for client timeouts or upstream load balancer
            issues.

      # Engine not responding — no metrics scraped
      - alert: VllmDown
        expr: up{job="vllm"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "vLLM metrics endpoint unreachable"
          description: >
            Prometheus cannot scrape the vLLM metrics endpoint at
            {{ $labels.instance }}. The server may be down.
```

Reference this file from your Prometheus configuration:

```yaml
# prometheus.yaml
rule_files:
  - "vllm-alerts.yaml"
```

---

## OpenTelemetry tracing

Distributed tracing captures the full lifecycle of individual requests,
complementing the aggregate view provided by Prometheus metrics. vLLM uses
[OpenTelemetry](https://opentelemetry.io/) and exports traces via the OTLP
protocol to any compatible backend (Jaeger, Tempo, Honeycomb, Datadog, etc.).

### Quick start with Jaeger

**1. Start Jaeger:**

```bash
docker run --rm --name jaeger \
    -p 16686:16686 \
    -p 4317:4317 \
    -p 4318:4318 \
    jaegertracing/all-in-one:1.57
```

**2. Export the Jaeger IP and start vLLM with tracing enabled:**

```bash
export JAEGER_IP=$(docker inspect \
    --format '{{ .NetworkSettings.IPAddress }}' jaeger)

export OTEL_SERVICE_NAME="vllm-server"
export OTEL_EXPORTER_OTLP_TRACES_INSECURE=true

vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --otlp-traces-endpoint="grpc://${JAEGER_IP}:4317"
```

**3. Send a request and view the trace:**

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": "Hello", "max_tokens": 50}'
```

Open the Jaeger UI at `http://localhost:16686`, select the `vllm-server`
service, and click **Find Traces**. Each request appears as a trace with
spans showing latency breakdowns.

### Enabling detailed traces

For deeper latency analysis, enable detailed traces for the model and worker
modules:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --otlp-traces-endpoint="grpc://${JAEGER_IP}:4317" \
    --collect-detailed-traces=model,worker
```

This adds `gen_ai.latency.time_in_model_forward` and
`gen_ai.latency.time_in_model_execute` attributes to each span, letting you
distinguish GPU compute time from synchronisation and sampling overhead.

!!! warning
    Detailed traces add blocking operations to the critical path. Enable only
    when actively diagnosing latency issues.

### Propagating trace context from clients

Pass W3C `traceparent` headers in your requests to link vLLM spans to your
application's trace:

```python
import requests

# traceparent format: 00-<trace-id>-<parent-id>-<flags>
headers = {
    "Content-Type": "application/json",
    "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
}

response = requests.post(
    "http://localhost:8000/v1/completions",
    headers=headers,
    json={
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "prompt": "Explain observability in one sentence.",
        "max_tokens": 100,
    },
)
```

### Using HTTP/Protobuf instead of gRPC

```bash
export OTEL_EXPORTER_OTLP_TRACES_PROTOCOL=http/protobuf

vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --otlp-traces-endpoint="http://${JAEGER_IP}:4318/v1/traces"
```

### FastAPI auto-instrumentation

You can instrument the FastAPI layer automatically using the
`opentelemetry-instrumentation-fastapi` package:

```bash
pip install opentelemetry-instrumentation-fastapi

opentelemetry-instrument vllm serve meta-llama/Llama-3.1-8B-Instruct \
    --otlp-traces-endpoint="grpc://${JAEGER_IP}:4317"
```

This adds HTTP-level spans (route, method, status code) to every trace.

---

## Logging-based metrics

In addition to Prometheus, vLLM logs a summary of key metrics to `INFO` every
5 seconds by default. This is useful for quick ad-hoc checks without a
Prometheus setup:

```text
Avg prompt throughput: 142.3 tokens/s, Avg generation throughput: 38.7 tokens/s,
Running: 8 reqs, Waiting: 2 reqs, GPU KV cache usage: 34.2%,
Prefix cache hit rate: 67.8%
```

To disable log-based metrics (for example, to reduce log volume in production
when Prometheus is the primary sink):

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --disable-log-stats
```

!!! note
    Disabling log stats also disables KV cache residency metrics
    (`--kv-cache-metrics`), which depend on the logging pipeline.

---

## Production checklist

Use this checklist before deploying vLLM to production:

- [ ] Prometheus is scraping `/metrics` at an appropriate interval (5–15 s).
- [ ] Grafana dashboard is imported and showing live data.
- [ ] Alert rules are configured for KV cache usage, queue depth, and TTFT.
- [ ] `--disable-log-stats` is **not** set if you rely on KV cache metrics.
- [ ] `PROMETHEUS_MULTIPROC_DIR` is **not** set manually (let vLLM manage it),
      or you have a process to wipe it between restarts.
- [ ] If using `--api-server-count > 1`, you have verified that process-level
      metrics (`python_info`, `process_*`) are not relied upon in dashboards.
- [ ] OTel tracing endpoint is configured if distributed tracing is required.
- [ ] `--collect-detailed-traces` is disabled in steady-state production.

---

## See also

- [Observability configuration reference](../configuration/observability_config.md)
  — all `ObservabilityConfig` options
- [Production metrics reference](../usage/metrics.md) — full list of all
  `vllm:` Prometheus metrics
- [Metrics design](../design/metrics.md) — architecture and design decisions
- [Prometheus + Grafana example](../../examples/online_serving/prometheus_grafana/README.md)
  — Docker Compose quick-start files
- [OpenTelemetry example](../../examples/online_serving/opentelemetry/README.md)
  — Jaeger tracing walkthrough
