---
description: >
  Overview of vLLM deployment options — Docker, Kubernetes, Helm, SSL/TLS,
  load balancing, and production hardening guides.
---

# Deployment

vLLM is designed to run in production at scale. This section covers everything
you need to deploy vLLM reliably — from a single Docker container to a
multi-node Kubernetes cluster with autoscaling, TLS, and load balancing.

---

## Deployment guides

<div class="grid cards" markdown>

-   :material-docker: **Docker**

    ---

    Run vLLM using the official pre-built images or build your own. Covers
    all image variants, build arguments, multi-platform builds with Bake, and
    the SageMaker image.

    [:octicons-arrow-right-24: Docker deployment](docker.md)

-   :material-kubernetes: **Kubernetes + Helm**

    ---

    Deploy vLLM on Kubernetes using the official Helm chart. Covers GPU
    scheduling, model download jobs, autoscaling, PodDisruptionBudgets, and
    secret management.

    [:octicons-arrow-right-24: Kubernetes deployment](kubernetes.md)

-   :material-lock: **SSL/TLS**

    ---

    Enable HTTPS for the vLLM API server. Covers certificate setup, mutual
    TLS, cipher suite configuration, and automatic certificate hot-reload.

    [:octicons-arrow-right-24: SSL/TLS setup](ssl_tls.md)

-   :material-scale-balance: **Load balancing**

    ---

    Distribute traffic across multiple vLLM instances. Covers Nginx
    round-robin and least-connections, elastic endpoint scaling, sleep mode,
    and Kubernetes-native approaches.

    [:octicons-arrow-right-24: Load balancing](load_balancing.md)

-   :material-shield-lock: **Security**

    ---

    Security best practices for production deployments — API key
    authentication, network isolation, CORS, endpoint hardening, container
    security, and vulnerability reporting.

    [:octicons-arrow-right-24: Security guide](security.md)

-   :material-clipboard-check: **Production checklist**

    ---

    A comprehensive checklist covering security, reliability, observability,
    performance, and operational best practices before going live.

    [:octicons-arrow-right-24: Production checklist](production_checklist.md)

-   :material-server-network: **Raw Kubernetes**

    ---

    Deploy vLLM on Kubernetes using native manifests (without Helm). Covers
    CPU and GPU deployments, PVCs, Secrets, and troubleshooting.

    [:octicons-arrow-right-24: Kubernetes guide](k8s.md)

</div>

---

## Choosing a deployment approach

| Scenario | Recommended approach |
|---|---|
| Local development or quick demo | [Docker quick start](docker.md#quick-start) |
| Single-server production | [Docker](docker.md) + [Nginx](nginx.md) + [SSL/TLS](ssl_tls.md) |
| Kubernetes (small team) | [Helm chart](kubernetes.md) |
| Kubernetes (enterprise) | [Helm chart](kubernetes.md) + [production checklist](production_checklist.md) |
| AWS SageMaker | [Docker SageMaker image](docker.md#aws-sagemaker-image) |
| Multi-instance with load balancing | [Load balancing guide](load_balancing.md) |
| Dynamic scaling | [Elastic endpoint scaling](load_balancing.md#elastic-endpoint-scaling) |

---

## Third-party integrations

vLLM integrates with many Kubernetes operators, cloud platforms, and serving
frameworks. See the integration-specific guides:

### Kubernetes operators and platforms

<div class="grid cards" markdown>

-   **NVIDIA Dynamo**

    [:octicons-arrow-right-24: Dynamo](integrations/dynamo.md)

-   **InftyAI / llmaz**

    [:octicons-arrow-right-24: llmaz](integrations/llmaz.md)

-   **llm-d**

    [:octicons-arrow-right-24: llm-d](integrations/llm-d.md)

-   **KAITO**

    [:octicons-arrow-right-24: KAITO](integrations/kaito.md)

-   **KServe**

    [:octicons-arrow-right-24: KServe](integrations/kserve.md)

-   **KubeRay**

    [:octicons-arrow-right-24: KubeRay](integrations/kuberay.md)

-   **KubeAI**

    [:octicons-arrow-right-24: KubeAI](integrations/kubeai.md)

-   **AIBrix**

    [:octicons-arrow-right-24: AIBrix](integrations/aibrix.md)

-   **Production Stack**

    [:octicons-arrow-right-24: Production Stack](integrations/production-stack.md)

-   **Kthena**

    [:octicons-arrow-right-24: Kthena](integrations/kthena.md)

-   **Llama Stack**

    [:octicons-arrow-right-24: Llama Stack](integrations/llamastack.md)

-   **LWS**

    [:octicons-arrow-right-24: LWS](frameworks/lws.md)

</div>

### Cloud and serving frameworks

<div class="grid cards" markdown>

-   **SkyPilot**

    [:octicons-arrow-right-24: SkyPilot](frameworks/skypilot.md)

-   **BentoML**

    [:octicons-arrow-right-24: BentoML](frameworks/bentoml.md)

-   **Modal**

    [:octicons-arrow-right-24: Modal](frameworks/modal.md)

-   **RunPod**

    [:octicons-arrow-right-24: RunPod](frameworks/runpod.md)

-   **Anyscale**

    [:octicons-arrow-right-24: Anyscale](frameworks/anyscale.md)

-   **Cerebrium**

    [:octicons-arrow-right-24: Cerebrium](frameworks/cerebrium.md)

-   **dstack**

    [:octicons-arrow-right-24: dstack](frameworks/dstack.md)

-   **Triton Inference Server**

    [:octicons-arrow-right-24: Triton](frameworks/triton.md)

-   **HF Inference Endpoints**

    [:octicons-arrow-right-24: HF Inference Endpoints](frameworks/hf_inference_endpoints.md)

-   **LiteLLM**

    [:octicons-arrow-right-24: LiteLLM](frameworks/litellm.md)

</div>

---

## Key concepts

### Image variants

vLLM ships three Docker image variants:

| Image | Purpose |
|---|---|
| `vllm/vllm-openai` | Production OpenAI-compatible API server |
| `vllm-sagemaker` | AWS SageMaker endpoint |
| `vllm:test` | CI / unit-test image |

### Health endpoint

All deployment configurations should use the `/health` endpoint for health
checks. It returns `200 OK` when the server is ready to accept requests.

```bash
curl http://localhost:8000/health
```

### Streaming support

vLLM supports Server-Sent Events (SSE) for streaming responses. Ensure your
load balancer and reverse proxy are configured to:

- Disable response buffering (`proxy_buffering off` in Nginx)
- Support chunked transfer encoding
- Use long read timeouts (at least 3600 seconds)

### GPU memory management

vLLM pre-allocates GPU memory for the KV cache at startup. The
`--gpu-memory-utilization` flag (default `0.9`) controls how much of the
available GPU memory is reserved. Monitor KV cache utilisation via the
`vllm:gpu_cache_usage_perc` Prometheus metric.

---

## Next steps

New to vLLM deployment? Start here:

1. **[Docker quick start](docker.md#quick-start)** — get a server running in
   minutes.
1. **[Production checklist](production_checklist.md)** — review before going
   live.
1. **[Kubernetes deployment](kubernetes.md)** — scale with Helm.
1. **[Load balancing](load_balancing.md)** — distribute traffic.
1. **[SSL/TLS setup](ssl_tls.md)** — secure your API.
1. **[Security guide](security.md)** — harden your deployment.
