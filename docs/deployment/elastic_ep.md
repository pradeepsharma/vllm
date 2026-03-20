# Elastic Expert Parallelism

Elastic Expert Parallelism (Elastic EP) allows vLLM to dynamically scale the number of expert-parallel ranks at runtime — adding or removing GPU workers without restarting the serving process. This is particularly valuable for MoE models like DeepSeek in production environments where demand fluctuates.

!!! warning "Experimental Feature"
    Elastic EP is an experimental feature. APIs and behaviour may change in future releases.

---

## Overview

Standard expert parallelism (EP) assigns a fixed number of GPUs to serve a model. Elastic EP extends this with:

- **Scale-up**: Add new GPU workers to an existing EP group, redistributing expert weights.
- **Scale-down**: Remove GPU workers from an EP group, consolidating expert weights onto remaining GPUs.
- **Expert Load Balancing (EPLB)**: Continuously monitor expert utilisation and rearrange expert weights across GPUs to minimise load imbalance.

These two features are complementary but independent:

| Feature | What it does | When to use |
|---------|-------------|-------------|
| **EPLB** | Rebalances experts across a *fixed* set of GPUs | Always-on for MoE models with uneven expert load |
| **Elastic EP** | Changes the *number* of GPUs in the EP group | Scaling in/out based on traffic demand |

---

## Expert Parallelism Load Balancing (EPLB)

### Motivation

In MoE models, different experts receive different numbers of tokens depending on the input distribution. Without load balancing, some GPUs become bottlenecks while others are underutilised. EPLB addresses this by:

1. **Monitoring** expert utilisation over a sliding window of forward passes.
2. **Computing** an optimal expert placement using a balanced packing algorithm.
3. **Rearranging** expert weights across GPUs via peer-to-peer NCCL transfers.

### Key Concepts

**Logical Expert**: An expert defined in the model architecture (e.g., DeepSeek-R1 has 256 logical experts per MoE layer).

**Physical Expert**: An instantiation of a logical expert on a specific GPU. With redundant experts, popular logical experts can have multiple physical copies on different GPUs.

**Redundant Experts**: Additional copies of high-load experts. For example, with 256 logical experts and 32 redundant experts, there are 288 physical experts total. On 32 EP ranks, each GPU holds 9 physical experts.

### Enabling EPLB

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel \
    --enable-eplb \
    --num-redundant-experts 32
```

Or via Python:

```python
from vllm import LLM

llm = LLM(
    model="deepseek-ai/DeepSeek-V3",
    tensor_parallel_size=1,
    data_parallel_size=8,
    enable_expert_parallel=True,
    enable_eplb=True,
    num_redundant_experts=32,
)
```

### EPLB Configuration

All EPLB settings are grouped under `EPLBConfig`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 1000 | Number of forward passes in the load-tracking sliding window |
| `step_interval` | 3000 | Forward passes between rebalancing operations |
| `num_redundant_experts` | 0 | Extra physical expert slots for popular experts |
| `log_balancedness` | False | Log load imbalance metrics (adds communication overhead) |
| `log_balancedness_interval` | 1 | Steps between balancedness log entries |
| `use_async` | False | Run rebalancing in a background thread |
| `policy` | `"default"` | Rebalancing algorithm (`"default"` only currently) |

Configure via CLI:

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
    --enable-expert-parallel \
    --enable-eplb \
    --eplb-window-size 500 \
    --eplb-step-interval 1000 \
    --num-redundant-experts 16
```

### Rebalancing Algorithm

The default EPLB policy (`DefaultEplbPolicy`) uses a two-phase algorithm adapted from [DeepSeek EPLB](https://github.com/deepseek-ai/eplb):

**Phase 1 — Expert Replication** (`replicate_experts`):
Given the load of each logical expert, determine how many physical copies each expert should have. Popular experts receive more copies. The algorithm minimises the maximum load across all physical experts.

**Phase 2 — Balanced Packing** (`balanced_packing`):
Assign physical experts to GPU packs such that each GPU holds exactly `num_physical_experts / num_gpus` experts and the total load per GPU is as balanced as possible. Uses a greedy bin-packing approach.

### Expert Weight Transfer

When a rebalancing decision is made, expert weights are exchanged between GPUs using batched P2P NCCL operations (`batch_isend_irecv`). The transfer is coordinated by `rearrange_expert_weights_inplace` in `vllm/distributed/eplb/rebalance_execute.py`.

**Synchronous mode** (default): The main inference thread pauses while weights are transferred.

**Asynchronous mode** (`use_async=True`): A background thread handles weight transfers while inference continues. The main thread waits only when it needs to consume the updated weights.

```bash
# Enable async EPLB
vllm serve deepseek-ai/DeepSeek-V3 \
    --enable-expert-parallel \
    --enable-eplb \
    --eplb-use-async
```

### Expert Placement Strategies

Control the initial assignment of experts to GPUs:

```bash
# Linear (default): GPU 0 gets experts [0..k-1], GPU 1 gets [k..2k-1], etc.
--expert-placement-strategy linear

# Round-robin: GPU 0 gets experts [0, ep_size, 2*ep_size, ...], etc.
--expert-placement-strategy round_robin
```

Round-robin can improve initial load balance for grouped expert models without redundant experts.

### Monitoring Load Balancedness

Enable balancedness logging to observe how well experts are balanced:

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
    --enable-expert-parallel \
    --enable-eplb \
    --eplb-log-balancedness \
    --eplb-log-balancedness-interval 100
```

!!! note
    Balancedness logging requires an all-reduce operation and adds communication overhead. Use only for diagnostics.

---

## Elastic EP (Dynamic Scaling)

Elastic EP enables changing the number of EP ranks at runtime. It uses **stateless NCCL groups** — process groups that can be created and destroyed independently of the main `torch.distributed` world group.

### Prerequisites

- CUDA-capable GPUs only (no CPU or XPU support)
- Ray executor backend (`--distributed-executor-backend ray`)
- Expert parallelism enabled (`--enable-expert-parallel`)
- Single-node TP/PP (multi-node TP/PP is not yet supported with Elastic EP)

### Enabling Elastic EP

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
    --enable-expert-parallel \
    --enable-elastic-ep \
    --data-parallel-size 8 \
    --distributed-executor-backend ray
```

### Stateless Process Groups

Standard `torch.distributed` process groups are tied to a fixed world size established at startup. Elastic EP uses `StatelessGroupCoordinator` — groups bootstrapped via dedicated TCP stores — for the DP, EP, and EPLB communicators. This allows these groups to be torn down and recreated with a different membership without affecting TP and PP groups.

Port allocation for stateless groups is handled automatically by `ParallelConfig.allocate_elastic_ep_ports()`, which pre-allocates all required ports after `ray.init()` to avoid conflicts.

### Scale-Up Sequence

When new GPU workers are added:

```
Existing engines                    New engines
─────────────────────────────────────────────────────
WAIT_NEW_CORE_ENGINES_INIT
                                    PREPARE
CREATE_STANDBY_GROUPS  ←──────────→ (standby groups created)
TRANSFER_EXPERT_MAPPING ──────────→ (expert mapping sent)
WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT
                                    (weights initialised)
TRANSFER_WEIGHTS ─────────────────→ (weights transferred)
SYNC_KV_CACHE_MEMORY_SIZE ←───────→
SWITCH_AND_PREPARE
EPLB_RESHUFFLE ←──────────────────→ EPLB_RESHUFFLE
COMPLETE                            COMPLETE
```

### Scale-Down Sequence

When GPU workers are removed:

```
Remaining engines                   Removing engines
─────────────────────────────────────────────────────
PREPARE                             PREPARE
EPLB_RESHUFFLE ←──────────────────→ EPLB_RESHUFFLE
SWITCH_AND_PREPARE
COMPLETE                            COMPLETE
```

### Launching New Workers

New workers are launched with the `VLLM_ELASTIC_EP_SCALE_UP_LAUNCH` environment variable set:

```bash
export VLLM_ELASTIC_EP_SCALE_UP_LAUNCH=1
vllm serve deepseek-ai/DeepSeek-V3 \
    --enable-expert-parallel \
    --enable-elastic-ep \
    --data-parallel-size 16  # increased from 8
```

### Standby Groups

During scale-up, "standby" process groups are created that include both existing and new ranks. These groups are used for weight transfer and expert mapping synchronisation before the new ranks become active. After the transition completes, standby groups are destroyed and the active groups are replaced atomically via `_replace_active_groups()`.

---

## Requirements and Constraints

| Constraint | Details |
|-----------|---------|
| Platform | CUDA only (EPLB); CUDA + Ray (Elastic EP) |
| TP/PP with Elastic EP | Single-node only (multi-node TP/PP not supported) |
| `enable_expert_parallel` | Must be True for both EPLB and Elastic EP |
| `tensor_parallel_size × data_parallel_size` | Must be > 1 for EPLB |
| `num_redundant_experts` | Must be 0 if EPLB is disabled |
| Max redundant experts | ≤ 1023 per GPU |

---

## Example: DeepSeek-V3 with EPLB

A complete example for serving DeepSeek-V3 on 8 H100 GPUs with EPLB:

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel \
    --all2all-backend deepep_high_throughput \
    --enable-eplb \
    --num-redundant-experts 32 \
    --eplb-window-size 1000 \
    --eplb-step-interval 3000 \
    --max-model-len 8192
```

For decode-optimised serving:

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --enable-expert-parallel \
    --all2all-backend deepep_low_latency \
    --enable-eplb \
    --num-redundant-experts 16 \
    --eplb-use-async
```

---

## Troubleshooting

### EPLB Hangs During Rebalancing

All EP ranks must call the rebalancing collective simultaneously. If ranks are out of sync, the operation will hang. Ensure:
- All DP ranks are processing requests (or dummy batches) at the same step.
- `step_interval` is consistent across all ranks.

### "EPLB requires tensor_parallel_size or data_parallel_size > 1"

EPLB needs at least 2 GPUs in the EP group:

```bash
# Minimum: TP=2 or DP=2
--tensor-parallel-size 2 --enable-expert-parallel --enable-eplb
```

### Elastic EP: "Multi-node TP/PP not supported"

Elastic EP currently requires all TP and PP ranks to be on the same node. Use single-node TP with multi-node DP instead:

```bash
# Correct: TP=8 on one node, DP spans nodes
--tensor-parallel-size 8 --data-parallel-size 4 --enable-elastic-ep
```

### Port Conflicts During Elastic EP Initialisation

Elastic EP pre-allocates ports after `ray.init()`. If you see `EADDRINUSE` errors, ensure no other processes are using the port range. The system retries up to 5 times with fresh ports automatically.

---

## See Also

- [Expert Parallel Deployment](../serving/expert_parallel_deployment.md) — EP setup guide
- [Data Parallel Deployment](../serving/data_parallel_deployment.md) — DP configuration
- [Distributed Inference Design](../design/distributed_inference.md) — internal architecture
- [DeepSeek EPLB](https://github.com/deepseek-ai/eplb) — upstream algorithm reference
