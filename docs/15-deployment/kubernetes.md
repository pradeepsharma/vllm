# Kubernetes Deployment

This page covers deploying vLLM on Kubernetes, including GPU node selectors, resource requests, health probes, and a complete example manifest.

## Prerequisites

- Kubernetes cluster with NVIDIA GPU nodes
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html) or `nvidia-device-plugin` installed
- `kubectl` configured to access the cluster
- A container registry accessible from the cluster (or use `vllm/vllm-openai` from Docker Hub)

## Complete Deployment Manifest

The following manifest deploys a single vLLM instance serving `meta-llama/Meta-Llama-3.1-8B-Instruct` on one A100 GPU:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
  labels:
    app: vllm-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
    spec:
      # Schedule on GPU nodes
      nodeSelector:
        accelerator: nvidia-a100

      # Tolerate GPU node taints
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule

      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          imagePullPolicy: IfNotPresent

          args:
            - --model
            - meta-llama/Meta-Llama-3.1-8B-Instruct
            - --tensor-parallel-size
            - "1"
            - --gpu-memory-utilization
            - "0.9"
            - --max-model-len
            - "8192"

          env:
            - name: HF_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: token
            - name: VLLM_LOGGING_LEVEL
              value: "INFO"

          ports:
            - name: http
              containerPort: 8000
              protocol: TCP

          resources:
            requests:
              cpu: "4"
              memory: "16Gi"
              nvidia.com/gpu: "1"
            limits:
              cpu: "8"
              memory: "32Gi"
              nvidia.com/gpu: "1"

          # Liveness probe — restarts the pod if the engine hangs
          livenessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 60
            periodSeconds: 30
            timeoutSeconds: 10
            failureThreshold: 3

          # Readiness probe — gates traffic until the model is loaded
          readinessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 5

          # Startup probe — allows extra time for large model downloads
          startupProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 30
            periodSeconds: 10
            timeoutSeconds: 5
            failureThreshold: 60   # 60 × 10s = 10 minutes max startup

          volumeMounts:
            - name: hf-cache
              mountPath: /root/.cache/huggingface
            - name: shm
              mountPath: /dev/shm

      volumes:
        - name: hf-cache
          persistentVolumeClaim:
            claimName: hf-model-cache-pvc
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: 16Gi

---
apiVersion: v1
kind: Service
metadata:
  name: vllm-server
spec:
  selector:
    app: vllm-server
  ports:
    - name: http
      port: 8000
      targetPort: http
  type: ClusterIP
```

## GPU Resource Requests

### Single GPU

```yaml
resources:
  requests:
    nvidia.com/gpu: "1"
  limits:
    nvidia.com/gpu: "1"
```

### Multi-GPU (Tensor Parallelism)

For tensor-parallel inference across multiple GPUs on the same node:

```yaml
resources:
  requests:
    nvidia.com/gpu: "4"
  limits:
    nvidia.com/gpu: "4"

args:
  - --tensor-parallel-size
  - "4"
```

> **Important:** The number of GPUs requested must equal `--tensor-parallel-size`. vLLM uses all visible GPUs.

### Shared Memory

PyTorch multiprocessing requires shared memory. Use either:

1. **`emptyDir` with `medium: Memory`** (recommended for Kubernetes):
   ```yaml
   volumes:
     - name: shm
       emptyDir:
         medium: Memory
         sizeLimit: 16Gi
   ```

2. **Host IPC** (not recommended in multi-tenant clusters):
   ```yaml
   spec:
     hostIPC: true
   ```

## Node Selectors and Tolerations

### Node Labels

Label GPU nodes with the GPU type for precise scheduling:

```bash
kubectl label node gpu-node-1 accelerator=nvidia-a100
kubectl label node gpu-node-2 accelerator=nvidia-h100
```

Then select in the deployment:

```yaml
nodeSelector:
  accelerator: nvidia-h100
```

### GPU Taints

GPU nodes are often tainted to prevent non-GPU workloads:

```bash
kubectl taint nodes gpu-node-1 nvidia.com/gpu=present:NoSchedule
```

Add the matching toleration:

```yaml
tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

### Node Affinity (Advanced)

For more complex scheduling rules:

```yaml
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
        - matchExpressions:
            - key: nvidia.com/gpu.product
              operator: In
              values:
                - NVIDIA-A100-SXM4-80GB
                - NVIDIA-H100-80GB-HBM3
```

## Health Probes

vLLM exposes a `/health` endpoint (defined in `vllm/entrypoints/serve/instrumentator/health.py`) that returns HTTP 200 when the engine is healthy.

```python
@router.get("/health", response_class=Response)
async def health(raw_request: Request) -> Response:
    await engine_client(raw_request).check_health()
    return Response(status_code=200)
```

### Probe Timing Guidelines

| Probe | `initialDelaySeconds` | `periodSeconds` | `failureThreshold` |
|-------|-----------------------|-----------------|-------------------|
| Startup | 30 | 10 | 60 (10 min) |
| Readiness | 30 | 10 | 5 |
| Liveness | 60 | 30 | 3 |

- **Startup probe** allows time for model download and loading. Set `failureThreshold × periodSeconds` to exceed your worst-case model load time.
- **Readiness probe** prevents traffic from reaching the pod until the model is ready.
- **Liveness probe** restarts the pod if the engine becomes unresponsive.

## Secrets Management

Store the HuggingFace token as a Kubernetes Secret:

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=token=hf_your_token_here
```

Reference in the deployment:

```yaml
env:
  - name: HF_TOKEN
    valueFrom:
      secretKeyRef:
        name: hf-token-secret
        key: token
```

## Persistent Volume for Model Cache

Create a PVC to cache downloaded models across pod restarts:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: hf-model-cache-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 200Gi
  storageClassName: fast-ssd
```

> **Note:** For multi-replica deployments, use `ReadWriteMany` with a shared filesystem (e.g., NFS, EFS) or pre-bake models into the container image.

## Horizontal Pod Autoscaler

Scale replicas based on CPU/custom metrics:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-server
  minReplicas: 1
  maxReplicas: 4
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

> **Note:** Each replica requires its own GPU(s). Ensure the cluster has sufficient GPU capacity before enabling HPA.

## Multi-Node Kubernetes Deployment

For models requiring multiple nodes (e.g., 405B parameter models), use a Ray cluster deployed via the [KubeRay operator](https://ray-project.github.io/kuberay/):

```yaml
apiVersion: ray.io/v1alpha1
kind: RayCluster
metadata:
  name: vllm-ray-cluster
spec:
  headGroupSpec:
    rayStartParams:
      num-gpus: "8"
    template:
      spec:
        containers:
          - name: ray-head
            image: vllm/vllm-openai:latest
            resources:
              limits:
                nvidia.com/gpu: "8"
  workerGroupSpecs:
    - replicas: 1
      rayStartParams:
        num-gpus: "8"
      template:
        spec:
          containers:
            - name: ray-worker
              image: vllm/vllm-openai:latest
              resources:
                limits:
                  nvidia.com/gpu: "8"
```

See [Multi-Node Deployment](multi-node.md) for the Ray cluster setup details.

## Environment Variables in Kubernetes

Pass vLLM environment variables via the `env` section:

```yaml
env:
  - name: VLLM_HOST_IP
    valueFrom:
      fieldRef:
        fieldPath: status.podIP
  - name: VLLM_LOGGING_LEVEL
    value: "INFO"
  - name: VLLM_WORKER_MULTIPROC_METHOD
    value: "spawn"
  - name: CUDA_VISIBLE_DEVICES
    value: "0,1,2,3"
```

> **Tip:** Set `VLLM_HOST_IP` to the pod's IP using the Downward API when running multi-process workers that need to communicate over the network.

## Related Pages

- [Docker Deployment](docker.md) — building and running the container image
- [Multi-Node Deployment](multi-node.md) — Ray cluster for multi-node inference
- [SSL/TLS Configuration](ssl-tls.md) — enabling HTTPS in Kubernetes with cert-manager
- [Distributed Inference](../07-distributed/README.md) — tensor and pipeline parallelism
