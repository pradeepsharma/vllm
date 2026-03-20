# Ray Cluster Configuration

This guide covers configuring a Ray cluster for vLLM distributed inference, including placement groups, runtime environments, worker configuration, and advanced tuning options.

---

## Why Ray?

Ray is vLLM's recommended distributed executor for:

- **Multi-node deployments** — Ray handles worker placement, health monitoring, and resource scheduling across nodes automatically.
- **TPU and XPU platforms** — Ray is the only supported executor backend for these accelerators.
- **Elastic expert parallelism** — dynamic scaling of expert parallel groups requires Ray's actor model.
- **Large-scale data parallelism** — Ray simplifies launching many DP replicas without per-node manual commands.

For single-node GPU deployments, the multiprocessing backend (`mp`) is typically preferred due to lower overhead.

---

## Installation

```bash
pip install "ray[default]>=2.43.0"

# For compiled DAG support (required for pipeline parallelism):
pip install "ray[cgraph]"

# cupy is required when using NCCL channel type with compiled DAGs:
pip install cupy-cuda12x  # adjust CUDA version as needed
```

Verify the installation:

```bash
python -c "import ray; print(ray.__version__)"
```

---

## Starting a Ray Cluster

### Manual Cluster Setup

**Head node:**

```bash
ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265
```

**Worker nodes:**

```bash
ray start \
    --address=<HEAD_NODE_IP>:6379 \
    --num-gpus=<NUM_GPUS_ON_THIS_NODE>
```

Verify the cluster:

```bash
ray status
# Expected output:
# Resources
# ---------------------------------------------------------------
# Usage:
#  0.0/32.0 CPU
#  0.0/16.0 GPU
#  ...
# Demands:
#  (no resource demands)
```

### Container-Based Setup

Use the provided helper script for Docker-based clusters:

```bash
# Head node
bash examples/online_serving/run_cluster.sh \
    vllm/vllm-openai \
    <HEAD_NODE_IP> \
    --head \
    /path/to/huggingface/cache \
    -e VLLM_HOST_IP=<HEAD_NODE_IP>

# Worker nodes (run on each worker)
bash examples/online_serving/run_cluster.sh \
    vllm/vllm-openai \
    <HEAD_NODE_IP> \
    --worker \
    /path/to/huggingface/cache \
    -e VLLM_HOST_IP=<WORKER_NODE_IP>
```

!!! important
    `VLLM_HOST_IP` must be set to the IP address of the **current node** on each node. This is the address that other nodes will use to reach this node's NCCL endpoints.

### KubeRay

For Kubernetes deployments, use [KubeRay](https://docs.ray.io/en/latest/cluster/kubernetes/index.html):

```yaml
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: vllm-cluster
spec:
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
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
      rayStartParams: {}
      template:
        spec:
          containers:
            - name: ray-worker
              image: vllm/vllm-openai:latest
              resources:
                limits:
                  nvidia.com/gpu: "8"
```

---

## Placement Groups

vLLM uses Ray placement groups to control where workers are scheduled. A placement group consists of one bundle per GPU worker.

### Automatic Placement Group

By default, vLLM creates a placement group automatically when `distributed_executor_backend="ray"`. The group uses `STRICT_PACK` strategy within each node and `PACK` across nodes.

### Custom Placement Group

You can provide a pre-created placement group via `ParallelConfig.placement_group`:

```python
import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

ray.init()

# Create a placement group with 8 GPU bundles
pg = placement_group(
    bundles=[{"GPU": 1, "CPU": 4} for _ in range(8)],
    strategy="STRICT_PACK",  # All on one node
)
ray.get(pg.ready())

from vllm import LLM
llm = LLM(
    model="meta-llama/Llama-3-70b",
    tensor_parallel_size=8,
    placement_group=pg,
)
```

### Bundle Index Control

To pin specific workers to specific bundles, set `VLLM_RAY_BUNDLE_INDICES`:

```bash
# Workers 0,1,2,3 → bundles 0,1,2,3
export VLLM_RAY_BUNDLE_INDICES="0,1,2,3"
```

This is useful when sharing a placement group with other Ray actors.

---

## Runtime Environment

Pass a Ray runtime environment to configure worker processes:

```python
from ray.runtime_env import RuntimeEnv
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3-70b",
    tensor_parallel_size=8,
    ray_runtime_env=RuntimeEnv(
        env_vars={
            "NCCL_IB_HCA": "mlx5_0",
            "NCCL_DEBUG": "WARN",
        },
        pip=["some-extra-package"],
    ),
)
```

Or via the CLI:

```bash
vllm serve meta-llama/Llama-3-70b \
    --tensor-parallel-size 8 \
    --ray-runtime-env '{"env_vars": {"NCCL_IB_HCA": "mlx5_0"}}'
```

---

## Worker Configuration

### GPU Resources per Worker

By default, each Ray worker requests 1 GPU. Override with:

```bash
export VLLM_RAY_PER_WORKER_GPUS=1.0  # fractional GPU allocation
```

### Worker-Specific Environment Variables

The following variables are **not** copied from the driver to workers (they are set per-worker by the executor):

- `VLLM_HOST_IP` / `VLLM_HOST_PORT`
- `LOCAL_RANK`
- `CUDA_VISIBLE_DEVICES`
- `HIP_VISIBLE_DEVICES` (AMD)
- `ROCR_VISIBLE_DEVICES` (AMD)

All other environment variables present on the driver are automatically propagated to workers.

### Nsight Profiling

Enable NVIDIA Nsight profiling for Ray workers:

```bash
vllm serve /path/to/model \
    --tensor-parallel-size 4 \
    --ray-workers-use-nsight
```

This configures each worker's Ray runtime environment with Nsight settings:

```python
{
    "nsight": {
        "t": "cuda,cudnn,cublas",
        "o": "'worker_process_%p'",
        "cuda-graph-trace": "node",
    }
}
```

---

## Compiled DAG (cgraph)

vLLM uses Ray's Compiled DAG for low-overhead inference dispatch. The DAG is built lazily on the first `execute_model` call.

### Channel Type

Control how tensors are transferred between pipeline stages:

```bash
# Options: "auto" (default), "nccl", "shm"
export VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=auto
```

| Value | Description | Use When |
|-------|-------------|----------|
| `auto` | Ray chooses automatically | Default |
| `nccl` | NCCL-based GPU-to-GPU transfer | Fast inter-node GPU links |
| `shm` | Shared memory (CPU) | TPU, XPU, or when NCCL unavailable |

### Communication Overlap

Enable overlapping GPU communication with computation:

```bash
export VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM=1
```

### PP Communication Backend

By default, vLLM uses Ray's built-in NCCL communicator for pipeline-parallel tensor transfers. To use vLLM's own PP `GroupCoordinator` instead:

```bash
export VLLM_USE_RAY_WRAPPED_PP_COMM=1
```

This wraps the vLLM `_PP GroupCoordinator` as a Ray accelerator context, which can improve performance in some configurations.

### cgraph Timeout

The default timeout for compiled DAG execution is 300 seconds. Adjust with:

```bash
export RAY_CGRAPH_get_timeout=600
```

---

## Data Parallel with Ray

Ray simplifies multi-node data parallel deployments — a single command on any node launches all DP ranks:

```bash
vllm serve $MODEL \
    --data-parallel-size 4 \
    --data-parallel-backend ray
```

### Spanning Multiple Nodes

When a single DP group requires multiple nodes (e.g., TP=8 per DP rank across 2 nodes):

```bash
export VLLM_RAY_DP_PACK_STRATEGY=span

vllm serve $MODEL \
    --data-parallel-size 2 \
    --tensor-parallel-size 8 \
    --data-parallel-backend ray
```

With `VLLM_RAY_DP_PACK_STRATEGY=span`, `--data-parallel-size-local` is ignored and determined automatically from available node resources.

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_HOST_IP` | auto-detected | IP address this node advertises to peers |
| `VLLM_HOST_PORT` | auto-assigned | Port for NCCL rendezvous |
| `VLLM_RAY_PER_WORKER_GPUS` | `1.0` | GPU fraction requested per Ray worker |
| `VLLM_RAY_BUNDLE_INDICES` | unset | Comma-separated bundle indices for worker placement |
| `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE` | `auto` | Compiled DAG channel type (`auto`/`nccl`/`shm`) |
| `VLLM_USE_RAY_COMPILED_DAG_OVERLAP_COMM` | `0` | Enable communication/computation overlap |
| `VLLM_USE_RAY_WRAPPED_PP_COMM` | `0` | Use vLLM PP communicator in Ray cgraph |
| `VLLM_RAY_DP_PACK_STRATEGY` | `pack` | DP placement strategy (`pack`/`span`) |
| `RAY_CGRAPH_get_timeout` | `300` | Compiled DAG execution timeout (seconds) |
| `RAY_USAGE_STATS_ENABLED` | `0` | Ray usage statistics (disabled by default) |

---

## Troubleshooting

### Ray Version Requirements

vLLM requires Ray ≥ 2.43.0. Check your version:

```bash
python -c "import ray; print(ray.__version__)"
```

### Workers Not Connecting

```bash
# Check Ray cluster status
ray status

# List all nodes
ray list nodes

# Check for resource availability
ray list resources
```

### NCCL Timeout in Compiled DAG

If the compiled DAG times out during execution:

```bash
export RAY_CGRAPH_get_timeout=600  # Increase to 10 minutes
```

### cupy Not Found

When using `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE=nccl`, cupy is required:

```bash
pip install cupy-cuda12x  # Match your CUDA version
```

### Placement Group Not Ready

```python
import ray
from ray.util.placement_group import placement_group

pg = placement_group([{"GPU": 1} for _ in range(8)])
# Wait with timeout
ready, _ = ray.wait([pg.ready()], timeout=60)
if not ready:
    raise RuntimeError("Placement group not ready after 60 seconds")
```

### Worker IP Address Conflicts

Each node must have a unique IP address. If `VLLM_HOST_IP` is set, ensure it is unique per node:

```bash
# On each node, set to that node's IP
export VLLM_HOST_IP=$(hostname -I | awk '{print $1}')
```

---

## See Also

- [Multi-Node Setup](multi_node.md) — general multi-node deployment guide
- [Distributed Inference Design](../design/distributed_inference.md) — internal architecture
- [Expert Parallel Deployment](../serving/expert_parallel_deployment.md) — MoE-specific setup
- [Ray Documentation](https://docs.ray.io/en/latest/) — official Ray docs
