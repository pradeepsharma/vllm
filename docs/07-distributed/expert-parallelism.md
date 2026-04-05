# Expert Parallelism

Expert parallelism (EP) is a specialized parallelism strategy for Mixture-of-Experts (MoE) models. Instead of replicating all expert weights across GPUs (as tensor parallelism would), EP shards the experts themselves — each GPU holds a distinct subset of experts.

## MoE Expert Sharding

In a MoE model, each layer contains `N` expert networks. With EP, these experts are distributed across the EP group:

```
EP rank 0: experts [0, 1, ..., N/ep_size - 1]
EP rank 1: experts [N/ep_size, ..., 2N/ep_size - 1]
...
EP rank k: experts [k*N/ep_size, ..., (k+1)*N/ep_size - 1]
```

For example, DeepSeek-V3 has 256 routed experts. With `ep_size=32`, each GPU holds 8 experts.

## EP vs TP for MoE Layers

By default, vLLM uses tensor parallelism for MoE layers (replicating all experts and sharding their weight matrices). Expert parallelism is an alternative that avoids weight replication:

| Strategy | Memory per GPU | Communication |
|----------|---------------|---------------|
| TP (default) | `expert_weights / tp_size` | All-reduce after each expert |
| EP | `expert_weights / ep_size` | All-to-all for token routing |

Enable EP with:

```bash
vllm serve deepseek-ai/DeepSeek-V3 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel
```

## EP Group Size

The EP group size is derived from `data_parallel_size × tensor_parallel_size`:

```python
# From vllm/config/parallel.py
@property
def expert_parallel_size(self) -> int:
    dp_size = self.data_parallel_size
    ep_size = self.data_parallel_size * self.world_size_across_dp
    return ep_size
```

The EP group spans all DP ranks and TP ranks, so every GPU in the deployment participates in expert routing.

## Process Group Initialization

The EP group is created in `initialize_model_parallel()` (see `vllm/distributed/parallel_state.py`):

```python
# EP group spans DP × PCP × TP dimensions
group_ranks = (
    all_ranks.transpose(1, 2)  # move DP dim
    .reshape(-1, data_parallel_size * prefill_context_model_parallel_size * tensor_model_parallel_size)
    .unbind(0)
)
_EP = init_model_parallel_group(group_ranks, ..., group_name="ep")
```

> **Note:** The EP group is only created for MoE models (`config.model_config.is_moe == True`). For dense models, `get_ep_group()` will raise an assertion error.

## All-to-All Communication

When a token is routed to an expert on a different GPU, it must be transferred via an all-to-all collective. vLLM supports multiple all-to-all backends:

| Backend | Description |
|---------|-------------|
| `allgather_reducescatter` | Default; all-gather + reduce-scatter |
| `naive` | Simple broadcast-based implementation |
| `deepep_high_throughput` | DeepEP high-throughput kernels |
| `deepep_low_latency` | DeepEP low-latency kernels |
| `mori` | Mori kernels |
| `flashinfer_all2allv` | FlashInfer alltoallv for MNNVL |

Configure via:

```bash
vllm serve ... --all2all-backend deepep_high_throughput
```

## Expert Placement Strategies

Two strategies control how experts are initially assigned to GPUs:

### Linear (default)

Experts are assigned contiguously. With 4 experts and 2 ranks:
- Rank 0: experts [0, 1]
- Rank 1: experts [2, 3]

### Round-Robin

Experts are interleaved. With 4 experts and 2 ranks:
- Rank 0: experts [0, 2]
- Rank 1: experts [1, 3]

Round-robin can improve load balance for grouped expert models without redundant experts.

```bash
vllm serve ... --expert-placement-strategy round_robin
```

## Configuration Reference

From `vllm/config/parallel.py`:

```python
@config
class ParallelConfig:
    enable_expert_parallel: bool = False
    """Use expert parallelism instead of tensor parallelism for MoE layers."""

    expert_placement_strategy: ExpertPlacementStrategy = "linear"
    """'linear' or 'round_robin'"""

    all2all_backend: All2AllBackend = "allgather_reducescatter"
    """All2All backend for MoE expert parallel communication."""
```

## Interaction with EPLB

Expert parallelism is a prerequisite for the Expert Parallelism Load Balancer (EPLB). EPLB dynamically rebalances experts across GPUs based on observed token routing statistics. See [Elastic Expert Parallelism](elastic-ep.md) for details.

## Interaction with Data Parallelism

When both EP and DP are enabled, the EP group spans all DP ranks. This means all DP replicas share the same expert assignment — a single all-to-all collective routes tokens from all DP ranks to the appropriate expert GPUs.

This "wide EP" configuration is particularly effective for large MoE models like DeepSeek-R1 where the number of experts (256) is much larger than the number of GPUs per DP replica.

## Related Pages

- [Elastic Expert Parallelism](elastic-ep.md) — Dynamic load balancing
- [Tensor Parallelism](tensor-parallelism.md) — Alternative for MoE layers
- [Data Parallelism](data-parallelism.md) — Interaction with DP
