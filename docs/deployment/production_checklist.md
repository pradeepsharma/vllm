---
description: >
  Production readiness checklist for vLLM deployments — security, reliability,
  observability, performance, and operational best practices.
---

# Production readiness checklist

Use this checklist before promoting a vLLM deployment to production. Each
section covers a specific concern area with actionable items and links to
detailed guidance.

---

## Security

### Authentication and authorisation

- [ ] **Enable API key authentication.** Pass `--api-key` (or multiple keys
  via repeated `--api-key` flags) to require a bearer token on every request.

    ```bash
    vllm serve mistralai/Mistral-7B-Instruct-v0.3 \
        --api-key sk-prod-key-1 \
        --api-key sk-prod-key-2
    ```

- [ ] **Rotate API keys regularly.** Store keys in a secrets manager (AWS
  Secrets Manager, HashiCorp Vault, Kubernetes Secrets) rather than in
  environment files or container images.

- [ ] **Restrict network access.** Bind vLLM to a private interface or use a
  firewall to prevent direct public access. Route external traffic through a
  load balancer or API gateway.

- [ ] **Disable the FastAPI docs UI in production.** The Swagger UI and ReDoc
  endpoints expose your API schema publicly.

    ```bash
    vllm serve ... --disable-fastapi-docs
    ```

### Transport security

- [ ] **Enable TLS.** Either terminate TLS at vLLM (`--ssl-keyfile`,
  `--ssl-certfile`) or at a reverse proxy / load balancer. Never serve plain
  HTTP on a public network.

- [ ] **Use certificates from a trusted CA.** Self-signed certificates are
  acceptable only for development.

- [ ] **Enable automatic certificate renewal.** Use Let's Encrypt + Certbot or
  cert-manager (Kubernetes) and enable `--enable-ssl-refresh` so vLLM reloads
  certificates without a restart.

- [ ] **Enforce TLS 1.2 or higher.** Disable SSLv3, TLS 1.0, and TLS 1.1 at
  the load balancer or Nginx configuration.

- [ ] **Use strong cipher suites.** Prefer ECDHE cipher suites that provide
  forward secrecy.

### Container security

- [ ] **Pin image tags.** Use a specific version tag (e.g.,
  `vllm/vllm-openai:v0.9.0`) instead of `latest`.

- [ ] **Scan images for vulnerabilities.** Integrate a container scanner
  (Trivy, Snyk, Grype) into your CI pipeline.

- [ ] **Run as a non-root user where possible.** The vLLM container runs as
  root by default. Evaluate whether your environment supports non-root
  execution.

- [ ] **Mount the filesystem read-only where possible.** Use
  `readOnlyRootFilesystem: true` in the Kubernetes security context and mount
  writable volumes only where needed (e.g., `/tmp`, model cache).

- [ ] **Limit container capabilities.** Drop all Linux capabilities and add
  back only those required.

### CORS configuration

- [ ] **Restrict CORS origins.** The default `--allowed-origins ["*"]` allows
  any origin. Restrict to your application's domain in production.

    ```bash
    vllm serve ... \
        --allowed-origins '["https://app.example.com"]' \
        --allowed-methods '["GET", "POST"]'
    ```

---

## Reliability

### Health probes

- [ ] **Configure liveness and readiness probes.** Use the `/health` endpoint.

    ```yaml
    # Kubernetes example
    readinessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 60
      periodSeconds: 10
      failureThreshold: 3

    livenessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 120
      periodSeconds: 15
      failureThreshold: 3
    ```

- [ ] **Set `initialDelaySeconds` proportional to model size.** A 70B model
  may take 5+ minutes to load. Set `initialDelaySeconds` to at least 300 for
  large models.

### Fault tolerance

- [ ] **Run at least two replicas.** A single replica is a single point of
  failure. Use `replicaCount: 2` or more in the Helm chart.

- [ ] **Configure a PodDisruptionBudget.** Prevent all replicas from being
  evicted simultaneously during cluster maintenance.

    ```yaml
    maxUnavailablePodDisruptionBudget: "1"
    ```

- [ ] **Use a rolling update strategy.** The default Helm chart strategy
  (`maxSurge: 100%, maxUnavailable: 0`) ensures zero downtime during updates.

- [ ] **Configure load balancer health checks.** Route traffic only to healthy
  instances. Use `proxy_next_upstream error timeout http_503` in Nginx to
  retry on failures.

### Resource limits

- [ ] **Set CPU and memory requests and limits.** Unset limits can cause a
  single pod to starve other workloads on the node.

    ```yaml
    resources:
      requests:
        cpu: "4"
        memory: "16Gi"
        nvidia.com/gpu: "1"
      limits:
        cpu: "8"
        memory: "32Gi"
        nvidia.com/gpu: "1"
    ```

- [ ] **Reserve shared memory for tensor parallelism.** Add a `shm` volume
  when using multi-GPU tensor parallelism:

    ```yaml
    volumes:
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: "10Gi"
    volumeMounts:
      - name: shm
        mountPath: /dev/shm
    ```

### Graceful shutdown

- [ ] **Configure a shutdown timeout.** vLLM handles `SIGTERM` gracefully and
  waits for in-flight requests to complete. Ensure your orchestrator's
  `terminationGracePeriodSeconds` is long enough.

    ```yaml
    # Kubernetes
    spec:
      terminationGracePeriodSeconds: 120
    ```

---

## Observability

### Logging

- [ ] **Configure structured logging.** vLLM uses Python's standard logging
  module. Redirect logs to your centralised log aggregator (Fluentd, Loki,
  CloudWatch).

- [ ] **Set the log level appropriately.** Use `--uvicorn-log-level warning`
  in production to reduce log volume. Use `info` or `debug` only when
  troubleshooting.

    ```bash
    vllm serve ... --uvicorn-log-level warning
    ```

- [ ] **Disable access logs if not needed.** High-traffic deployments generate
  large volumes of access log entries.

    ```bash
    vllm serve ... --disable-uvicorn-access-log
    ```

### Metrics

- [ ] **Enable Prometheus metrics.** vLLM exposes metrics at `/metrics`.
  Scrape this endpoint with Prometheus.

- [ ] **Monitor key metrics:**

    | Metric | Description |
    |---|---|
    | `vllm:num_requests_running` | Requests currently being processed |
    | `vllm:num_requests_waiting` | Requests in the queue |
    | `vllm:gpu_cache_usage_perc` | KV cache utilisation (%) |
    | `vllm:time_to_first_token_seconds` | Time to first token (TTFT) |
    | `vllm:time_per_output_token_seconds` | Inter-token latency |
    | `vllm:request_success_total` | Total successful requests |

- [ ] **Set up alerting.** Alert on high queue depth, high TTFT, and low KV
  cache availability.

### Tracing

- [ ] **Enable distributed tracing** if your organisation uses OpenTelemetry.
  vLLM supports OpenTelemetry trace export.

### Request IDs

- [ ] **Enable request ID headers** for end-to-end request tracing:

    ```bash
    vllm serve ... --enable-request-id-headers
    ```

    This adds an `X-Request-Id` header to every response.

---

## Performance

### Model loading

- [ ] **Pre-download model weights.** Use a Kubernetes Job or init container
  to download weights to a PVC before the server starts. Avoid downloading at
  startup time.

- [ ] **Enable Hugging Face transfer acceleration:**

    ```bash
    export HF_HUB_ENABLE_HF_TRANSFER=1
    ```

- [ ] **Use a local model cache.** Mount a PVC or host path at
  `/root/.cache/huggingface` to avoid re-downloading weights on pod restarts.

### GPU utilisation

- [ ] **Set `--gpu-memory-utilization` appropriately.** The default is `0.9`
  (90%). Reduce this if you observe OOM errors; increase it to maximise
  throughput.

    ```bash
    vllm serve ... --gpu-memory-utilization 0.85
    ```

- [ ] **Enable CUDA graphs** (enabled by default in V1 engine). Disable only
  if you encounter compatibility issues.

- [ ] **Choose the right parallelism strategy:**

    | Model size | Strategy |
    |---|---|
    | Fits on one GPU | No parallelism needed |
    | Fits on one node (multiple GPUs) | Tensor parallelism (`--tensor-parallel-size`) |
    | Requires multiple nodes | Pipeline + tensor parallelism |

### Quantisation

- [ ] **Consider quantisation for large models.** AWQ, GPTQ, or FP8
  quantisation can reduce memory usage by 2–4× with minimal accuracy loss.

    ```bash
    vllm serve ... --quantization awq
    ```

### KV cache

- [ ] **Enable prefix caching** for workloads with repeated prompt prefixes
  (e.g., system prompts):

    ```bash
    vllm serve ... --enable-prefix-caching
    ```

- [ ] **Tune `--max-model-len`** to match your workload's maximum sequence
  length. Larger values consume more KV cache memory.

### Concurrency

- [ ] **Set `--max-num-seqs`** to control the maximum number of sequences
  processed simultaneously. Tune based on your GPU memory and latency
  requirements.

- [ ] **Enable chunked prefill** for better GPU utilisation with mixed
  prefill/decode workloads:

    ```bash
    vllm serve ... --enable-chunked-prefill
    ```

---

## Operations

### Version management

- [ ] **Pin all dependency versions.** Use `docker/versions.json` as the
  source of truth for CUDA, Python, and library versions.

- [ ] **Test upgrades in a staging environment** before promoting to
  production.

- [ ] **Keep a rollback plan.** Maintain the previous image tag and Helm chart
  values so you can roll back quickly.

### Secrets management

- [ ] **Never store secrets in container images or ConfigMaps.** Use
  Kubernetes Secrets, AWS Secrets Manager, HashiCorp Vault, or a similar
  secrets manager.

- [ ] **Use workload identity** (AWS IRSA, GCP Workload Identity, Azure
  Managed Identity) instead of long-lived access keys where possible.

### Backup and recovery

- [ ] **Back up model weights.** If you store weights on a PVC, ensure the
  underlying storage is backed up or replicated.

- [ ] **Document the recovery procedure.** Know how to restore a failed
  deployment from scratch, including model download time.

### Capacity planning

- [ ] **Benchmark your workload.** Use `vllm bench` or the benchmarking
  scripts in `benchmarks/` to measure throughput and latency under realistic
  load.

- [ ] **Plan for peak load.** Size your cluster for peak traffic, not average
  traffic. Consider autoscaling (HPA) for variable workloads.

- [ ] **Monitor GPU memory headroom.** Keep KV cache utilisation below 85% to
  avoid OOM errors under burst traffic.

### Networking

- [ ] **Configure generous timeouts** at every layer (load balancer, Ingress,
  application). LLM requests can take minutes.

- [ ] **Disable response buffering** at the load balancer for streaming
  responses. Set `proxy_buffering off` in Nginx.

- [ ] **Use HTTP/1.1 keep-alive** between the load balancer and vLLM backends
  to reduce connection overhead.

---

## Pre-launch checklist

Run through this final checklist before going live:

- [ ] API key authentication is enabled
- [ ] TLS is configured and certificates are from a trusted CA
- [ ] CORS origins are restricted to your application's domain
- [ ] FastAPI docs UI is disabled (`--disable-fastapi-docs`)
- [ ] Health probes are configured with appropriate timeouts
- [ ] At least two replicas are running
- [ ] PodDisruptionBudget is configured
- [ ] Resource requests and limits are set
- [ ] Prometheus metrics are being scraped
- [ ] Alerting is configured for key metrics
- [ ] Model weights are pre-downloaded to a PVC
- [ ] Load balancer timeouts are set to at least 3600 seconds
- [ ] Response buffering is disabled at the load balancer
- [ ] A rollback procedure is documented and tested

---

## Next steps

- [Docker deployment](docker.md) — build and run vLLM Docker images
- [Kubernetes deployment](kubernetes.md) — Kubernetes + Helm deployment
- [SSL/TLS setup](ssl_tls.md) — enable HTTPS
- [Load balancing](load_balancing.md) — distribute traffic across instances
