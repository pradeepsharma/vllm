# Parallel Configuration

`ParallelConfig` controls how vLLM distributes model computation across multiple GPUs and nodes. It covers tensor parallelism, pipeline parallelism, data parallelism, expert parallelism, and the distributed communication backend.

**Source:** `vllm/config/parallel.py`  
**CLI flags:** See [EngineArgs](engine_args.md) — Parallelism section.

---

## Parallelism Strategies Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Tensor Parallel (TP)  — splits each layer across N GPUs        │
│  Pipeline Parallel (PP) — splits layers across N GPU groups     │
│  Data Parallel (DP)    — runs N independent model replicas      │
│  Expert Parallel (EP)  — distributes MoE experts across GPUs    │
│  Context Parallel (CP) — splits long sequences across GPUs      │
└─────────────────────────────────────────────────────────────────┘
```

**World size** = `tensor_parallel_size × pipeline_parallel_size`

---

## Tensor Parallelism

### `tensor_parallel_size`

```
Type:    int
Default: 1
CLI:     --tensor-parallel-size
```

Number of GPUs to use for tensor parallelism. Each GPU holds a shard of every layer's weight matrix. Requires `world_size` GPUs to be available.

**When to use:** Large models that don't fit on a single GPU. Reduces per-GPU memory at the cost of inter-GPU communication overhead.

```bash
# 70B model on 4 GPUs
vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4
```

**Rules:**
- Must divide the model's `num_attention_heads` evenly
- Must divide the model's `num_key_value_heads` evenly (for GQA models)
- For MoE models, `tensor_parallel_size × data_parallel_size` must divide `num_experts`

---

## Pipeline Parallelism

### `pipeline_parallel_size`

```
Type:    int
Default: 1
CLI:     --pipeline-parallel-size
```

Number of pipeline stages. Layers are split across `pipeline_parallel_size` groups of `tensor_parallel_size` GPUs each.

**When to use:** Very large models (405B+) that require more GPUs than TP alone can provide, or multi-node setups.

```bash
# 405B model: 4 TP × 2 PP = 8 GPUs total
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2
```

**Note:** Pipeline parallelism introduces pipeline bubbles (idle time between micro-batches). For most workloads, maximizing TP before using PP gives better throughput.

---

## Data Parallelism

### `data_parallel_size`

```
Type:    int
Default: 1
CLI:     --data-parallel-size
```

Number of data parallel replicas. Each replica is a full model instance (with its own TP/PP group). Requests are load-balanced across replicas.

**When to use:** High-throughput serving where you want multiple model replicas to handle concurrent requests.

```bash
# 2 replicas of an 8B model, each using 2 GPUs (4 GPUs total)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 2 \
  --data-parallel-size 2
```

### `data_parallel_backend`

```
Type:    Literal["ray", "mp"]
Default: "mp"
CLI:     --data-parallel-backend
```

Backend for data parallel coordination:
- `mp` — Multiprocessing (single-node)
- `ray` — Ray (multi-node capable)

### `data_parallel_external_lb`

```
Type:    bool
Default: False
CLI:     --data-parallel-external-lb
```

Use external load balancing (e.g., Kubernetes service). Useful for "one-pod-per-rank" wide-EP setups. Set implicitly when `--data-parallel-rank` is provided explicitly.

### `data_parallel_hybrid_lb`

```
Type:    bool
Default: False
CLI:     --data-parallel-hybrid-lb
```

Hybrid load balancing: vLLM balances between local DP ranks, while an external LB balances between vLLM nodes/replicas.

---

## Expert Parallelism (MoE)

Expert parallelism distributes Mixture-of-Experts (MoE) layers across GPUs instead of using tensor parallelism for those layers.

### `enable_expert_parallel`

```
Type:    bool
Default: False
CLI:     --enable-expert-parallel
```

Enable expert parallelism for MoE layers. When enabled, each GPU holds a subset of experts rather than a shard of each expert.

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel
```

### `all2all_backend`

```
Type:    Literal["naive", "allgather_reducescatter", "deepep_high_throughput", "deepep_low_latency", "mori", "flashinfer_all2allv"]
Default: "allgather_reducescatter"
CLI:     --all2all-backend
```

All-to-All communication backend for MoE expert routing:

| Backend | Description |
|---|---|
| `naive` | Naive broadcast-based all2all |
| `allgather_reducescatter` | Default — allgather + reducescatter |
| `deepep_high_throughput` | DeepEP high-throughput kernels (large batches) |
| `deepep_low_latency` | DeepEP low-latency kernels (small batches) |
| `mori` | Mori kernels |
| `flashinfer_all2allv` | FlashInfer alltoallv for MNNVL |

### `expert_placement_strategy`

```
Type:    Literal["linear", "round_robin"]
Default: "linear"
CLI:     --expert-placement-strategy
```

How experts are distributed across ranks:

- **`linear`**: Contiguous placement. With 4 experts and 2 ranks: rank 0 → experts [0,1], rank 1 → experts [2,3].
- **`round_robin`**: Interleaved placement. With 4 experts and 2 ranks: rank 0 → experts [0,2], rank 1 → experts [1,3]. Better load balancing for grouped expert models.

### `enable_eplb`

```
Type:    bool
Default: False
CLI:     --enable-eplb
```

Enable Expert Parallel Load Balancing (EPLB). Dynamically rebalances expert assignments based on observed load. Requires `enable_expert_parallel=True` and CUDA/ROCm.

### `eplb_config`

```
Type:    EPLBConfig
Default: EPLBConfig()
CLI:     --eplb-config (JSON)
```

EPLB configuration:

| Field | Default | Description |
|---|---|---|
| `window_size` | `1000` | Steps of expert load history to track |
| `step_interval` | `3000` | Steps between expert rebalancing |
| `num_redundant_experts` | `0` | Extra redundant expert copies |
| `log_balancedness` | `False` | Log load balancedness each step |
| `log_balancedness_interval` | `1` | Interval for balancedness logging |
| `use_async` | `False` | Non-blocking EPLB updates |
| `policy` | `"default"` | EPLB policy type |

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-config '{"window_size": 2000, "num_redundant_experts": 2}'
```

### `enable_elastic_ep`

```
Type:    bool
Default: False
CLI:     --enable-elastic-ep
```

Enable elastic expert parallelism with stateless NCCL groups for DP/EP. Allows dynamic expert group reconfiguration.

---

## Context Parallelism

### `prefill_context_parallel_size`

```
Type:    int
Default: 1
CLI:     --prefill-context-parallel-size
```

Number of context parallel groups for prefill. Splits long sequences across GPUs during the prefill phase.

### `decode_context_parallel_size`

```
Type:    int
Default: 1
CLI:     --decode-context-parallel-size
```

Number of context parallel groups for decode. Reuses TP GPUs (does not increase total GPU count). `tensor_parallel_size` must be divisible by `decode_context_parallel_size`.

### `dcp_comm_backend`

```
Type:    Literal["ag_rs", "a2a"]
Default: "ag_rs"
CLI:     --dcp-comm-backend
```

Communication backend for Decode Context Parallel (DCP):
- `ag_rs` — AllGather + ReduceScatter (default)
- `a2a` — All-to-All exchange (reduces NCCL calls from 3 to 2 per layer for MLA models)

### `cp_kv_cache_interleave_size`

```
Type:    int
Default: 1
CLI:     --cp-kv-cache-interleave-size
```

KV cache interleave size for context parallelism. Controls how tokens are distributed across CP ranks:
- `1` — Token-level alignment (token `i` on rank `i % cp_world_size`)
- `block_size` — Block-level alignment

---

## Distributed Backend

### `distributed_executor_backend`

```
Type:    str | None
Default: None (auto-select)
CLI:     --distributed-executor-backend
```

Backend for distributed model workers:

| Value | Description |
|---|---|
| `mp` | Multiprocessing — single-node, no Ray required |
| `ray` | Ray — multi-node capable, requires Ray cluster |
| `uni` | Uniprocess — single process (no parallelism) |
| `external_launcher` | External process launcher |

Auto-selection: `mp` if all GPUs are on one host, otherwise requires explicit `ray`.

### `master_addr`

```
Type:    str
Default: "127.0.0.1"
CLI:     --master-addr
```

Master node IP address for multi-node distributed inference (MP backend).

### `master_port`

```
Type:    int
Default: 29501
CLI:     --master-port
```

Master node port for multi-node distributed inference (MP backend).

### `nnodes`

```
Type:    int
Default: 1
CLI:     --nnodes
```

Total number of nodes for multi-node inference.

### `node_rank`

```
Type:    int
Default: 0
CLI:     --node-rank
```

Rank of this node (0-indexed) in multi-node inference.

### `distributed_timeout_seconds`

```
Type:    int | None
Default: None (PyTorch default: 600s for NCCL)
CLI:     --distributed-timeout-seconds
```

Timeout for distributed operations like `init_process_group`. Increase for multi-node setups where model downloads may be slow.

---

## Communication Optimizations

### `disable_custom_all_reduce`

```
Type:    bool
Default: False
CLI:     --disable-custom-all-reduce
```

Disable vLLM's custom all-reduce kernel and fall back to NCCL. The custom kernel is faster for small tensors on NVLink-connected GPUs.

### `disable_nccl_for_dp_synchronization`

```
Type:    bool | None
Default: None (True when async scheduling is enabled)
CLI:     --disable-nccl-for-dp-synchronization
```

Force DP synchronization to use Gloo instead of NCCL. Defaults to `True` when async scheduling is enabled.

### `enable_dbo`

```
Type:    bool
Default: False
CLI:     --enable-dbo
```

Enable Dual Batch Overlap (DBO) for the model executor. Overlaps computation and communication using micro-batching.

### `dbo_decode_token_threshold`

```
Type:    int
Default: 32
CLI:     --dbo-decode-token-threshold
```

Token count threshold for DBO in decode-only batches. Batches above this threshold use micro-batching.

### `dbo_prefill_token_threshold`

```
Type:    int
Default: 512
CLI:     --dbo-prefill-token-threshold
```

Token count threshold for DBO in batches containing prefills.

---

## Loading

### `max_parallel_loading_workers`

```
Type:    int | None
Default: None
CLI:     --max-parallel-loading-workers
```

Maximum number of parallel workers for loading model weights. Useful for large models with tensor parallelism to avoid CPU RAM OOM during loading.

---

## Worker Configuration

### `worker_cls`

```
Type:    str
Default: "auto"
CLI:     --worker-cls
```

Fully-qualified class name of the worker to use. `"auto"` selects based on the platform.

### `worker_extension_cls`

```
Type:    str
Default: ""
CLI:     --worker-extension-cls
```

Fully-qualified class name of a worker extension. The extension is dynamically inherited by the worker class, allowing injection of new attributes and methods for `collective_rpc` calls.

### `ray_workers_use_nsight`

```
Type:    bool
Default: False
CLI:     --ray-workers-use-nsight
```

Profile Ray workers with Nsight Systems. See [Ray profiling docs](https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html).

---

## Multi-Node Setup Examples

### Two-node tensor parallel (MP backend)

```bash
# Node 0 (master)
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr 10.0.0.1 \
  --master-port 29501

# Node 1
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr 10.0.0.1 \
  --master-port 29501
```

### Ray-based multi-node

```bash
# Start Ray cluster first, then:
vllm serve meta-llama/Llama-3.1-405B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend ray
```

### MoE model with expert parallelism

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --all2all-backend deepep_high_throughput \
  --enable-eplb \
  --eplb-config '{"num_redundant_experts": 4}'
```

### Data parallel with multiple replicas

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --tensor-parallel-size 2 \
  --data-parallel-size 4 \
  --data-parallel-backend ray
```
