---
description: >
  Deploy vLLM on Kubernetes using the official Helm chart — GPU scheduling,
  model download jobs, autoscaling, PodDisruptionBudgets, and more.
---

# Kubernetes deployment

vLLM provides a Helm chart (`examples/online_serving/chart-helm`) for
production-grade Kubernetes deployments. The chart handles GPU node affinity,
model weight downloads, health probes, autoscaling, and secret management.

For raw Kubernetes manifests without Helm, see the existing
[Kubernetes guide](k8s.md).

---

## Prerequisites

Before deploying, ensure you have:

- A running Kubernetes cluster (1.24+)
- [Helm 3](https://helm.sh/docs/intro/install/) installed
- The [NVIDIA GPU Operator](https://github.com/NVIDIA/gpu-operator) or
  [k8s-device-plugin](https://github.com/NVIDIA/k8s-device-plugin) installed
  and GPU nodes available
- (Optional) An S3-compatible bucket containing model weights, if using
  automatic model download

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm/examples/online_serving/chart-helm
```

### 2. Install the chart

```bash
helm upgrade --install --create-namespace \
    --namespace ns-vllm \
    test-vllm . \
    -f values.yaml \
    --set secrets.s3endpoint=$S3_ENDPOINT \
    --set secrets.s3bucketname=$S3_BUCKET \
    --set secrets.s3accesskeyid=$AWS_ACCESS_KEY_ID \
    --set secrets.s3accesskey=$AWS_SECRET_ACCESS_KEY
```

### 3. Verify the deployment

```bash
kubectl get pods -n ns-vllm
kubectl logs -n ns-vllm -l release=test -f
```

Wait for the pod to reach `Running` state and the log to show:

```text
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

### 4. Test the API

```bash
kubectl port-forward -n ns-vllm svc/test-vllm-service 8000:80

curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "opt-125m",
        "prompt": "San Francisco is a",
        "max_tokens": 16
    }'
```

---

## Chart structure

```
chart-helm/
├── Chart.yaml                  # Chart metadata (name, version)
├── values.yaml                 # Default values
├── values.schema.json          # JSON schema for values validation
├── lintconf.yaml               # YAML lint rules
├── ct.yaml                     # Chart testing config
└── templates/
    ├── _helpers.tpl            # Shared template helpers
    ├── deployment.yaml         # Main vLLM Deployment
    ├── service.yaml            # ClusterIP Service
    ├── configmap.yaml          # Optional ConfigMap for env vars
    ├── secrets.yaml            # Optional Secret (S3 credentials, etc.)
    ├── hpa.yaml                # HorizontalPodAutoscaler
    ├── pvc.yaml                # PersistentVolumeClaim for model storage
    ├── job.yaml                # Model download Job
    ├── poddisruptionbudget.yaml # PodDisruptionBudget
    └── custom-objects.yaml     # Arbitrary custom Kubernetes objects
```

---

## Values reference

### Image configuration

```yaml
image:
  repository: "vllm/vllm-openai"
  tag: "latest"
  command:
    - "vllm"
    - "serve"
    - "/data/"
    - "--served-model-name"
    - "opt-125m"
    - "--host"
    - "0.0.0.0"
    - "--port"
    - "8000"
```

!!! tip
    Pin `image.tag` to a specific release version in production. Using `latest`
    can cause unexpected behaviour when a new release changes defaults.

### Replicas and resources

```yaml
replicaCount: 1

resources:
  requests:
    cpu: 4
    memory: 16Gi
    nvidia.com/gpu: 1
  limits:
    cpu: 4
    memory: 16Gi
    nvidia.com/gpu: 1
```

### GPU node affinity

Set `gpuModels` to the GPU product label value on your nodes. The chart
automatically adds a `nodeAffinity` rule when GPU resources are requested:

```yaml
gpuModels:
  - "NVIDIA-A100-SXM4-80GB"
```

Find the label value on your nodes:

```bash
kubectl get nodes -o json | jq -r \
    '.items[].metadata.labels["nvidia.com/gpu.product"]' | sort -u
```

### Service configuration

```yaml
containerPort: 8000
servicePort: 80
serviceName: ""   # defaults to "<release>-service"
```

The chart creates a `ClusterIP` service. To expose the service externally,
use an Ingress or change the service type via a custom object.

---

## Model download

The chart includes a built-in model download workflow using an AWS CLI init
container and a Kubernetes Job.

### How it works

1. A Kubernetes **Job** (`job-download-model`) downloads model weights from S3
   to a `PersistentVolumeClaim`.
1. An **init container** (`wait-download-model`) in the main Deployment polls
   the PVC until the download is complete.
1. The main vLLM container starts and loads the model from `/data/`.

### Enable model download

```yaml
extraInit:
  modelDownload:
    enabled: true
    image:
      repository: "amazon/aws-cli"
      tag: "2.6.4"
      pullPolicy: "IfNotPresent"
  s3modelpath: "models/llama-3-8b"
  pvcStorage: "50Gi"
  awsEc2MetadataDisabled: true
```

### S3 credentials

Pass credentials as Helm `--set` arguments (they are stored in a Kubernetes
Secret):

```bash
helm upgrade --install test-vllm . \
    --set secrets.s3endpoint=https://s3.amazonaws.com \
    --set secrets.s3bucketname=my-model-bucket \
    --set secrets.s3accesskeyid=$AWS_ACCESS_KEY_ID \
    --set secrets.s3accesskey=$AWS_SECRET_ACCESS_KEY
```

### Custom download command

Override the default AWS CLI sync command for other storage backends:

```yaml
extraInit:
  modelDownload:
    enabled: true
    downloadJob:
      command: ["/bin/bash"]
      args:
        - "-eucx"
        - "gsutil -m cp -r gs://$GCS_BUCKET/$GCS_PATH /data"
      env:
        - name: GOOGLE_APPLICATION_CREDENTIALS
          value: "/var/secrets/google/key.json"
```

### Disable model download

If your model is already on a pre-populated PVC or you mount it via another
mechanism, disable the download job:

```yaml
extraInit:
  modelDownload:
    enabled: false
  pvcStorage: "50Gi"
```

---

## Health probes

The chart configures both readiness and liveness probes against the `/health`
endpoint:

```yaml
readinessProbe:
  initialDelaySeconds: 5
  periodSeconds: 5
  failureThreshold: 3
  httpGet:
    path: /health
    port: 8000

livenessProbe:
  initialDelaySeconds: 15
  periodSeconds: 10
  failureThreshold: 3
  httpGet:
    path: /health
    port: 8000
```

!!! warning
    Large models can take several minutes to load. If the readiness probe
    `failureThreshold` is too low, Kubernetes will restart the container before
    the model finishes loading. Increase `initialDelaySeconds` or
    `failureThreshold` proportionally to your model size.

    A 70B parameter model may need `initialDelaySeconds: 300` or more.

---

## Autoscaling

Enable the HorizontalPodAutoscaler to scale based on CPU utilisation:

```yaml
autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 4
  targetCPUUtilizationPercentage: 80
  # targetMemoryUtilizationPercentage: 80
```

!!! note
    CPU-based autoscaling is a coarse proxy for LLM serving load. For
    production, consider custom metrics (e.g., request queue depth or GPU
    utilisation) via the Kubernetes Custom Metrics API.

---

## PodDisruptionBudget

Protect against simultaneous pod evictions during cluster maintenance:

```yaml
maxUnavailablePodDisruptionBudget: "1"
```

This creates a `PodDisruptionBudget` that allows at most one pod to be
unavailable at a time.

---

## Deployment strategy

The default strategy is a rolling update with zero downtime:

```yaml
deploymentStrategy:
  rollingUpdate:
    maxSurge: 100%
    maxUnavailable: 0
```

Override for a recreate strategy (useful when GPU resources are scarce):

```yaml
deploymentStrategy:
  type: Recreate
```

---

## ConfigMaps and Secrets

### Injecting environment variables via ConfigMap

```yaml
configs:
  VLLM_WORKER_MULTIPROC_METHOD: "spawn"
  HF_HUB_ENABLE_HF_TRANSFER: "1"
```

### Injecting environment variables via Secret

```yaml
secrets:
  HF_TOKEN: "hf_your_token_here"
```

!!! warning
    Secret values in `values.yaml` are stored in plain text. Use
    `--set secrets.HF_TOKEN=$HF_TOKEN` on the command line, or use an external
    secrets manager (e.g., External Secrets Operator, Vault Agent) instead.

### Referencing external ConfigMaps and Secrets

```yaml
externalConfigs:
  - configMapRef:
      name: my-existing-configmap
  - secretRef:
      name: my-existing-secret
```

---

## Custom init containers

Add arbitrary init containers alongside or instead of the model download
container:

```yaml
extraInit:
  modelDownload:
    enabled: false
  initContainers:
    - name: model-cache-warmer
      image: busybox:latest
      command: ["/bin/sh", "-c", "echo 'Warming cache...'"]
      volumeMounts:
        - name: model-storage
          mountPath: /data
```

---

## Sidecar containers

Add sidecar containers to the main pod (e.g., a metrics exporter or proxy):

```yaml
extraContainers:
  - name: prometheus-exporter
    image: prom/node-exporter:latest
    ports:
      - containerPort: 9100
```

---

## Custom Kubernetes objects

Deploy arbitrary Kubernetes resources alongside the chart (e.g., an Ingress,
NetworkPolicy, or ServiceMonitor):

```yaml
customObjects:
  - apiVersion: networking.k8s.io/v1
    kind: Ingress
    metadata:
      name: vllm-ingress
      annotations:
        nginx.ingress.kubernetes.io/proxy-read-timeout: "3600"
    spec:
      rules:
        - host: vllm.example.com
          http:
            paths:
              - path: /
                pathType: Prefix
                backend:
                  service:
                    name: test-vllm-service
                    port:
                      number: 80
```

---

## Multi-GPU tensor parallelism

To run a large model across multiple GPUs on a single node, increase the GPU
count and add the tensor-parallel-size argument:

```yaml
resources:
  requests:
    nvidia.com/gpu: 4
  limits:
    nvidia.com/gpu: 4

image:
  command:
    - "vllm"
    - "serve"
    - "/data/"
    - "--served-model-name"
    - "llama-3-70b"
    - "--tensor-parallel-size"
    - "4"
    - "--host"
    - "0.0.0.0"
    - "--port"
    - "8000"
```

!!! note
    vLLM uses shared memory (`/dev/shm`) for inter-GPU communication. The chart
    does not mount a custom `emptyDir` for `/dev/shm` by default. If you
    encounter NCCL errors, add a shared memory volume via `extraContainers` or
    a custom pod spec patch.

---

## Uninstalling

```bash
helm uninstall test-vllm --namespace ns-vllm
```

!!! warning
    Uninstalling the chart deletes the PersistentVolumeClaim and all downloaded
    model weights. Back up or retain the PVC before uninstalling if you want to
    reuse the downloaded weights.

---

## Running chart tests

The chart includes unit tests using
[helm-unittest](https://github.com/helm-unittest/helm-unittest):

```bash
# Install the plugin (once)
helm plugin install https://github.com/helm-unittest/helm-unittest

# Run tests
cd examples/online_serving/chart-helm
helm unittest .
```

---

## Troubleshooting

### Pod stuck in `Pending`

```bash
kubectl describe pod -n ns-vllm <pod-name>
```

Common causes:

- **Insufficient GPU resources** — check `kubectl get nodes` for allocatable
  GPUs.
- **Node selector mismatch** — verify `gpuModels` matches the
  `nvidia.com/gpu.product` label on your nodes.
- **PVC not bound** — check `kubectl get pvc -n ns-vllm`.

### Model download Job failing

```bash
kubectl logs -n ns-vllm job/test-vllm-init-vllm
```

Verify that the S3 credentials and bucket path are correct.

### Container restarting due to probe failure

Increase the probe thresholds for large models:

```yaml
readinessProbe:
  initialDelaySeconds: 120
  failureThreshold: 10
  periodSeconds: 10

livenessProbe:
  initialDelaySeconds: 180
  failureThreshold: 5
  periodSeconds: 15
```

---

## Next steps

- [Docker deployment](docker.md) — build and run vLLM Docker images
- [SSL/TLS setup](ssl_tls.md) — enable HTTPS for the API server
- [Load balancing](load_balancing.md) — distribute traffic across multiple instances
- [Production checklist](production_checklist.md) — harden your deployment
- [Helm framework page](frameworks/helm.md) — full Helm values reference
