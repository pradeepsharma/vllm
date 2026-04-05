# Elastic Expert Parallelism (EPLB)

Elastic Expert Parallelism Load Balancing (EPLB) dynamically rebalances MoE experts across GPUs based on observed token routing statistics. It addresses the load imbalance problem where some experts receive far more tokens than others, causing GPU utilization to be bottlenecked by the busiest expert.

## Motivation

In MoE models like DeepSeek-R1/V3, the router sends tokens to experts non-uniformly. Some experts may receive 10× more tokens than others in a given batch. With static expert placement, the GPU holding the hot expert becomes a bottleneck while other GPUs sit idle.

EPLB solves this by:
1. **Monitoring** expert load over a sliding window of forward passes
2. **Replicating** hot experts onto multiple GPUs (redundant experts)
3. **Rebalancing** the physical-to-logical expert mapping periodically

## Key Concepts

### Logical vs Physical Experts

From `vllm/distributed/eplb/eplb_state.py`:

- **Logical Expert**: An expert defined by the model architecture (e.g., DeepSeek-R1 has 256 logical experts per MoE layer)
- **Physical Expert**: An instantiated copy of a logical expert on a specific GPU
- **Redundant Expert**: An additional copy of a popular logical expert for load balancing

For example, with 256 logical experts, 32 redundant experts, and 32 EP ranks:
- Total physical experts = 256 + 32 = 288
- Physical experts per GPU = 288 / 32 = 9

### Expert Mapping

EPLB maintains two mapping tensors per MoE layer:

```python
# Shape: (num_moe_layers, num_physical_experts)
physical_to_logical_map: torch.Tensor
# Maps each physical expert slot to its logical expert index

# Shape: (num_moe_layers, num_logical_experts, max_replicas + 1)
logical_to_physical_map: torch.Tensor
# Maps each logical expert to its physical expert slots (-1 = unused)
```

Example for a 2-layer model with 6 physical experts and 4 logical experts:

```
physical_to_logical_map:
[[0, 1, 2, 3, 0, 1],   # Layer 0: experts 0,1 are replicated
 [0, 2, 0, 1, 0, 3]]   # Layer 1: expert 0 has 3 replicas

logical_to_physical_map:
[[[0, 4, -1],   # Layer 0, logical 0 → physical slots 0, 4
  [1, 5, -1],   # Layer 0, logical 1 → physical slots 1, 5
  [2, -1, -1],  # Layer 0, logical 2 → physical slot 2
  [3, -1, -1]], # Layer 0, logical 3 → physical slot 3
 ...]
```

## EPLB Algorithm

The default policy (`DefaultEplbPolicy` in `vllm/distributed/eplb/policy/default.py`) is adapted from [DeepSeek EPLB](https://github.com/deepseek-ai/eplb):

### Step 1: Collect Load Statistics

Each forward pass, the number of tokens routed to each expert is recorded:

```python
# Shape: (window_size, num_moe_layers, num_physical_experts)
expert_load_window: torch.Tensor
```

An all-reduce aggregates load statistics across all EP ranks.

### Step 2: Replicate Hot Experts

The `replicate_experts()` method determines how many replicas each logical expert should have, proportional to its load:

```python
@classmethod
def replicate_experts(cls, weight: np.ndarray, num_phy: int):
    """Replicate num_log experts to num_phy replicas,
    such that the maximum load across all replicas is minimized."""
```

### Step 3: Pack Experts onto GPUs

The `balanced_packing()` method assigns physical experts to GPUs to minimize the maximum load per GPU:

```python
@classmethod
def balanced_packing(cls, weight: np.ndarray, num_packs: int):
    """Pack n weighted objects to m packs, such that each bin
    contains exactly n/m objects and weights are as balanced as possible."""
```

### Step 4: Transfer Expert Weights

The `rearrange_expert_weights_inplace()` function in `vllm/distributed/eplb/rebalance_execute.py` performs the actual weight transfer using P2P NCCL operations:

```python
# Uses batch_isend_irecv for efficient parallel weight exchange
p2p_ops = [P2POp(isend, weight, dst_rank), P2POp(irecv, buffer, src_rank), ...]
batch_isend_irecv(p2p_ops)
```

## Configuration

```python
from vllm import LLM

llm = LLM(
    model="deepseek-ai/DeepSeek-R1",
    tensor_parallel_size=8,
    enable_expert_parallel=True,
    enable_eplb=True,
    eplb_config={
        "window_size": 1000,        # Steps to average load over
        "step_interval": 3000,      # Steps between rebalancing
        "num_redundant_experts": 32, # Extra expert slots for replication
        "use_async": False,          # Async background transfer
        "policy": "default",
    },
)
```

CLI:

```bash
vllm serve deepseek-ai/DeepSeek-R1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \
  --enable-eplb \
  --eplb-window-size 1000 \
  --eplb-step-interval 3000 \
  --num-redundant-experts 32
```

### EPLBConfig Parameters

From `vllm/config/parallel.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `window_size` | 1000 | Steps to average expert load over |
| `step_interval` | 3000 | Steps between expert rebalancing |
| `num_redundant_experts` | 0 | Extra physical expert slots for replication |
| `log_balancedness` | False | Log load balance metrics (adds communication overhead) |
| `log_balancedness_interval` | 1 | Interval for balancedness logging |
| `use_async` | False | Use background thread for weight transfer |
| `policy` | `"default"` | Rebalancing algorithm |

## Async EPLB

When `use_async=True`, expert weight transfer happens in a background thread, overlapping with the main inference forward pass:

```python
# From vllm/distributed/eplb/async_worker.py
def start_async_worker(state: "EplbState") -> threading.Thread:
    """Start background thread for async expert weight transfer."""
    def thread_target():
        torch.cuda.set_device(device_index)
        cuda_stream = torch.cuda.Stream(device=device_index)
        loop = asyncio.new_event_loop()
        loop.run_until_complete(transfer_run_periodically(...))
    thread = threading.Thread(target=thread_target, daemon=True)
    thread.start()
    return thread
```

### CUDA Event Synchronization

The async worker uses CUDA events to coordinate with the main thread:

```python
buffer_ready_event: torch.cuda.Event   # Async worker signals buffer is filled
buffer_consumed_event: torch.cuda.Event # Main thread signals buffer is consumed
window_ready_event: torch.cuda.Event   # Main thread signals load window is ready
```

> **Warning:** When using async EPLB with `deepep_low_latency` backend, `NCCL_MAX_CTAS=8` is automatically set to prevent deadlocks between NCCL collectives and DeepEP cooperative kernels.

## EPLB Process Group

EPLB uses a **separate** process group (`_EPLB`) from the EP group to avoid deadlocks:

```python
# From vllm/distributed/parallel_state.py
# EPLB group has same ranks as EP but separate NCCL communicator
if config.parallel_config.enable_eplb:
    _EPLB = init_model_parallel_group(group_ranks, ..., group_name="eplb")
```

This isolation prevents EPLB's weight transfer collectives from interfering with MoE forward pass collectives.

## Elastic EP (Dynamic Scaling)

The `elastic_ep/` module extends EPLB with the ability to dynamically add or remove EP ranks at runtime. This enables:

- **Scale-up**: Add new GPU nodes to increase EP size
- **Scale-down**: Remove GPU nodes to reduce EP size

### Scale-Up State Machine

From `vllm/distributed/elastic_ep/elastic_state.py`:

```python
class ScaleUpExistingEngineState(enum.IntEnum):
    WAIT_NEW_CORE_ENGINES_INIT = 0
    CREATE_STANDBY_GROUPS = 1
    TRANSFER_EXPERT_MAPPING = 2
    WAIT_NEW_CORE_ENGINES_WEIGHTS_INIT = 3
    TRANSFER_WEIGHTS = 4
    SYNC_KV_CACHE_MEMORY_SIZE = 5
    SWITCH_AND_PREPARE = 6
    EPLB_RESHUFFLE = 7
    COMPLETE = 8
```

### Stateless Groups

Elastic EP uses stateless NCCL groups (no persistent process group state) to allow dynamic reconfiguration:

```python
# From vllm/config/parallel.py
enable_elastic_ep: bool = False
"""Enable elastic expert parallelism with stateless NCCL groups for DP/EP."""
```

When `enable_elastic_ep=True`, the DP and EP groups are created as `StatelessGroupCoordinator` instances that can be torn down and recreated without restarting the process.

## Load Balancedness Logging

Enable load balance monitoring:

```bash
vllm serve ... --eplb-log-balancedness --eplb-log-balancedness-interval 100
```

This logs the coefficient of variation of expert loads across the EP group, helping you tune `window_size` and `step_interval`.

## Related Pages

- [Expert Parallelism](expert-parallelism.md) — Static expert sharding
- [Data Parallelism](data-parallelism.md) — Wide EP with DP
- [Ray Executor](ray-executor.md) — Multi-node execution
