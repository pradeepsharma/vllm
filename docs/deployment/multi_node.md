# Multi-Node Deployment

This guide covers deploying vLLM across multiple nodes using either **Ray** or native **multiprocessing** as the distributed executor backend.

!!! tip "When do you need multi-node?"
    Multi-node deployment is required when a model is too large to fit on the GPUs of a single node. The typical strategy is to set `tensor_parallel_size` to the number of GPUs per node and `pipeline_parallel_size` to the number of nodes.

---

## Prerequisites

Before deploying across multiple nodes, ensure:

1. **Identical environments** — every node must run the same vLLM version, Python version, and CUDA version. Container images are strongly recommended.
2. **Shared model storage** — all nodes must be able to read the model weights from the same path (NFS, object storage mount, or pre-downloaded to each node).
3. **Network connectivity** — all nodes must be able to reach each other by IP address on the ports used for NCCL and the distributed init method.
4. **Firewall rules** — open the required ports (see [Port Requirements](#port-requirements) below).

!!! warning "Security"
    NCCL and the distributed init method transmit data in an unencrypted format that can be exploited to execute arbitrary code. Restrict access to the inter-node network to trusted hosts only.

---

## Choosing a Backend

| Backend | Best For | Requirements |
|---------|----------|-------------|
| **Ray** (`ray`) | Multi-node, TPU/XPU, elastic scaling | Ray cluster running on all nodes |
| **Multiprocessing** (`mp`) | Multi-node without Ray overhead | Manual launch on each node |

The backend is selected automatically:

- If `world_size ≤ local_gpu_count` → `mp` (single node)
- If `world_size > local_gpu_count` → `ray` (multi-node)

Override with `--distributed-executor-backend ray` or `--distributed-executor-backend mp`.

---

## Multi-Node with Ray

Ray is the recommended backend for multi-node deployments. A single launch command on the head node is sufficient — Ray handles worker placement automatically.

### Step 1: Start the Ray Cluster

**Head node:**

```bash
ray start --head --port=6379
```

**Each worker node:**

```bash
ray start --address=<HEAD_NODE_IP>:6379
```

Verify the cluster is healthy:

```bash
ray status
```

You should see all nodes and their GPU resources listed.

### Step 2: Launch vLLM

From the head node (inside the Ray cluster), run a single `vllm serve` command:

```bash
# 2 nodes × 8 GPUs = 16 GPUs total
# TP=8 (within each node), PP=2 (across nodes)
vllm serve /path/to/model \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --distributed-executor-backend ray
```

Ray automatically places workers on the available nodes.

### Alternative: Pure Tensor Parallelism Across Nodes

If you prefer to use only tensor parallelism (no pipeline parallelism), set `tensor_parallel_size` to the total GPU count:

```bash
# 2 nodes × 8 GPUs = 16 GPUs total, all TP
vllm serve /path/to/model \
    --tensor-parallel-size 16 \
    --distributed-executor-backend ray
```

!!! note
    Pure tensor parallelism across nodes requires high-bandwidth inter-node networking (InfiniBand or NVLink). Pipeline parallelism is more tolerant of lower-bandwidth connections.

### Using Containers with Ray

The helper script `examples/online_serving/run_cluster.sh` starts Docker containers and initialises Ray automatically.

**Head node:**

```bash
bash examples/online_serving/run_cluster.sh \
    vllm/vllm-openai \
    <HEAD_NODE_IP> \
    --head \
    /path/to/huggingface/cache \
    -e VLLM_HOST_IP=<HEAD_NODE_IP>
```

**Each worker node:**

```bash
bash examples/online_serving/run_cluster.sh \
    vllm/vllm-openai \
    <HEAD_NODE_IP> \
    --worker \
    /path/to/huggingface/cache \
    -e VLLM_HOST_IP=<WORKER_NODE_IP>
```

!!! important
    `VLLM_HOST_IP` must be set to the IP address of each respective node. Keep the shell sessions open — closing them terminates the cluster.

---

## Multi-Node with Multiprocessing

The multiprocessing backend does not require Ray. You launch vLLM manually on each node, specifying the node's rank and the head node's address.

### Step 1: Head Node

```bash
vllm serve /path/to/model \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --nnodes 2 \
    --node-rank 0 \
    --master-addr <HEAD_NODE_IP> \
    --master-port 29501
```

### Step 2: Worker Nodes

On each additional node, run the same command with an incremented `--node-rank` and the `--headless` flag:

```bash
# Node 1
vllm serve /path/to/model \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --nnodes 2 \
    --node-rank 1 \
    --master-addr <HEAD_NODE_IP> \
    --master-port 29501 \
    --headless
```

The `--headless` flag tells the worker node not to start an HTTP server — only the head node serves API requests.

### Three-Node Example

```bash
# Node 0 (head, TP=8, PP=3)
vllm serve /path/to/model \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 3 \
    --nnodes 3 --node-rank 0 \
    --master-addr 10.0.0.1

# Node 1
vllm serve /path/to/model \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 3 \
    --nnodes 3 --node-rank 1 \
    --master-addr 10.0.0.1 --headless

# Node 2
vllm serve /path/to/model \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 3 \
    --nnodes 3 --node-rank 2 \
    --master-addr 10.0.0.1 --headless
```

### Distributed Timeout

For large models that take a long time to download, increase the distributed initialisation timeout:

```bash
--distributed-timeout-seconds 3600
```

---

## Data Parallel Multi-Node

Data parallelism can also span multiple nodes. Each node runs an independent model replica.

### Internal Load Balancing (single API endpoint)

```bash
# Node 0 (head, DP ranks 0 and 1)
vllm serve $MODEL \
    --data-parallel-size 4 \
    --data-parallel-size-local 2 \
    --data-parallel-address 10.0.0.1 \
    --data-parallel-rpc-port 13345

# Node 1 (DP ranks 2 and 3)
vllm serve $MODEL --headless \
    --data-parallel-size 4 \
    --data-parallel-size-local 2 \
    --data-parallel-start-rank 2 \
    --data-parallel-address 10.0.0.1 \
    --data-parallel-rpc-port 13345
```

### API Server on Dedicated Node

```bash
# Node 0 (API server only, no GPU workers)
vllm serve $MODEL \
    --data-parallel-size 4 \
    --data-parallel-size-local 0 \
    --data-parallel-address 10.0.0.1 \
    --data-parallel-rpc-port 13345

# Node 1 (all 4 GPU workers)
vllm serve $MODEL --headless \
    --data-parallel-size 4 \
    --data-parallel-size-local 4 \
    --data-parallel-address 10.0.0.1 \
    --data-parallel-rpc-port 13345
```

---

## Port Requirements

| Port | Protocol | Purpose |
|------|----------|---------|
| 6379 | TCP | Ray head node (default) |
| 29501 | TCP | `torch.distributed` init (multiprocessing, `--master-port`) |
| 29500 | TCP | Data parallel master port |
| 29550 | TCP | Data parallel RPC port |
| Dynamic | TCP | NCCL inter-node communication |

NCCL uses dynamically assigned ports for data transfer. To restrict the port range, set:

```bash
export NCCL_PORT_RANGE=<min>:<max>
```

---

## Network Optimisation

### InfiniBand

For tensor parallelism across nodes, InfiniBand dramatically reduces all-reduce latency. Enable it by passing the appropriate NCCL environment variables:

```bash
export NCCL_IB_HCA=mlx5_0   # Adjust to your IB HCA name
export NCCL_IB_GID_INDEX=3
```

When using Docker, add `--privileged` or `--cap-add=CAP_SYS_ADMIN` to enable IB performance counters.

### GPUDirect RDMA

GPUDirect RDMA allows network adapters to read/write GPU memory directly, bypassing the CPU. To enable it:

```bash
docker run --gpus all \
    --ipc=host \
    --shm-size=16G \
    -v /dev/shm:/dev/shm \
    vllm/vllm-openai
```

In Kubernetes, add to the pod spec:

```yaml
securityContext:
  capabilities:
    add: ["IPC_LOCK"]
volumes:
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: 16Gi
```

### Pipeline vs. Tensor Parallelism Trade-offs

| Aspect | Tensor Parallelism | Pipeline Parallelism |
|--------|-------------------|---------------------|
| Communication | All-reduce every layer | Send activations between stages |
| Bandwidth requirement | High (all-reduce is bandwidth-bound) | Lower (only activations transferred) |
| Latency | Lower (no pipeline bubble) | Higher (pipeline bubble) |
| Best network | NVLink / InfiniBand | Ethernet (100 GbE+) |

For nodes connected by Ethernet rather than InfiniBand, prefer pipeline parallelism over tensor parallelism.

---

## Troubleshooting

### Workers Hang at Initialisation

- Verify all nodes can reach each other: `ping <other_node_ip>`
- Check that `--master-addr` is reachable from all worker nodes
- Increase `--distributed-timeout-seconds`
- Ensure NCCL ports are not blocked by firewalls

### NCCL Errors

```bash
# Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### Ray Workers Not Found

```bash
# Check Ray cluster status
ray status
ray list nodes

# Verify GPU resources are visible
ray list resources
```

### Mismatched Environments

Ensure all nodes use identical:
- vLLM version (`pip show vllm`)
- CUDA version (`nvcc --version`)
- Python version (`python --version`)
- Model weights (same path, same files)

---

## See Also

- [Ray Cluster Configuration](ray_cluster.md) — detailed Ray setup
- [Parallelism and Scaling](../serving/parallelism_scaling.md) — choosing parallelism strategies
- [Distributed Inference Design](../design/distributed_inference.md) — internal architecture
- [Distributed Troubleshooting](../serving/distributed_troubleshooting.md) — common issues
