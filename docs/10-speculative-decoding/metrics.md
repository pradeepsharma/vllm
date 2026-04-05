# Speculative Decoding Metrics

vLLM provides comprehensive metrics for monitoring speculative decoding performance. These metrics help you understand the effectiveness of your speculative decoding configuration and tune it for optimal throughput.

## Key Metrics

### Acceptance Rate

The **acceptance rate** (α) is the probability that a single draft token is accepted by the target model's distribution. It is the most important metric for evaluating speculative decoding effectiveness.

```
acceptance_rate = num_accepted_tokens / num_draft_tokens
```

A higher acceptance rate means more draft tokens are accepted per target model forward pass, leading to greater speedup.

### Mean Acceptance Length (MAL)

The **mean acceptance length** is the average number of tokens generated per target model forward pass, including the mandatory bonus token:

```
MAL = 1 + (num_accepted_tokens / num_drafts)
```

For example, with `num_speculative_tokens=5` and an acceptance rate of 0.8:
- Expected accepted tokens per draft: 0.8 + 0.64 + 0.512 + 0.41 + 0.33 ≈ 2.69
- MAL ≈ 1 + 2.69 = 3.69

This means each target model forward pass produces ~3.69 tokens on average, compared to 1.0 without speculative decoding.

### Per-Position Acceptance Rate

The per-position acceptance rate shows how acceptance probability degrades across draft positions:

```
position_0_rate = accepted_at_pos_0 / num_drafts
position_1_rate = accepted_at_pos_1 / num_drafts
...
```

Typically, acceptance rates decrease monotonically with position. A steep drop-off suggests reducing `num_speculative_tokens`.

### Draft and Accepted Throughput

- **Draft throughput**: Number of draft tokens generated per second
- **Accepted throughput**: Number of accepted tokens per second (= effective token generation rate)

## Implementation

### SpecDecodingStats

Per-step statistics are collected in `SpecDecodingStats` (`vllm/v1/spec_decode/metrics.py`):

```python
@dataclass
class SpecDecodingStats:
    """Per-step iteration decoding stats from scheduler."""

    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int] = field(default_factory=list)

    def observe_draft(self, num_draft_tokens: int, num_accepted_tokens: int):
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        # Track per-position acceptance
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1
```

These stats are aggregated by the scheduler each step and returned to the frontend via `EngineCoreOutputs → SchedulerStats`.

### SpecDecodingLogging

The `SpecDecodingLogging` class aggregates metrics over a time interval and logs them:

```python
class SpecDecodingLogging:
    def log(self, log_fn=logger.info):
        # Compute aggregated metrics
        draft_acceptance_rate = num_accepted_tokens / num_draft_tokens * 100
        mean_acceptance_length = 1 + (num_accepted_tokens / num_drafts)

        # Per-position acceptance rates
        pos_matrix = np.array(self.accepted_tokens_per_pos_lists)
        acceptance_rates = np.sum(pos_matrix, axis=0) / num_drafts
        rates_str = ", ".join(f"{p:.3f}" for p in acceptance_rates)

        log_fn(
            "SpecDecoding metrics: "
            "Mean acceptance length: %.2f, "
            "Accepted throughput: %.2f tokens/s, "
            "Drafted throughput: %.2f tokens/s, "
            "Accepted: %d tokens, "
            "Drafted: %d tokens, "
            "Per-position acceptance rate: %s, "
            "Avg Draft acceptance rate: %.1f%%",
            mean_acceptance_length,
            accepted_throughput,
            draft_throughput,
            num_accepted_tokens,
            num_draft_tokens,
            rates_str,
            draft_acceptance_rate,
        )
```

### Sample Log Output

```
SpecDecoding metrics: Mean acceptance length: 3.72, Accepted throughput: 1847.3 tokens/s,
Drafted throughput: 2309.1 tokens/s, Accepted: 18473 tokens, Drafted: 23091 tokens,
Per-position acceptance rate: 0.891, 0.794, 0.708, 0.631, 0.563,
Avg Draft acceptance rate: 80.0%
```

## Prometheus Metrics

vLLM exposes speculative decoding metrics via Prometheus for production monitoring.

### Available Counters

| Metric Name | Type | Description |
|-------------|------|-------------|
| `vllm:spec_decode_num_drafts` | Counter | Total number of speculative decoding drafts |
| `vllm:spec_decode_num_draft_tokens` | Counter | Total number of draft tokens generated |
| `vllm:spec_decode_num_accepted_tokens` | Counter | Total number of accepted tokens |
| `vllm:spec_decode_num_accepted_tokens_per_pos` | Counter | Accepted tokens per draft position (labeled by `position`) |

### PromQL Queries

**Acceptance rate** (rolling window):
```promql
rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
rate(vllm:spec_decode_num_draft_tokens_total[$interval])
```

**Mean acceptance length** (including bonus token):
```promql
1 + (
  rate(vllm:spec_decode_num_accepted_tokens_total[$interval]) /
  rate(vllm:spec_decode_num_drafts[$interval])
)
```

**Per-position acceptance rate vector**:
```promql
vllm:spec_decode_num_accepted_tokens_per_pos[$interval] /
vllm:spec_decode_num_drafts[$interval]
```

**Effective speedup** (ratio of accepted to drafted throughput):
```promql
rate(vllm:spec_decode_num_accepted_tokens_total[5m]) /
rate(vllm:spec_decode_num_draft_tokens_total[5m])
```

### Implementation

The `SpecDecodingProm` class (`vllm/v1/spec_decode/metrics.py`) manages Prometheus counter registration and updates:

```python
class SpecDecodingProm:
    def __init__(
        self,
        speculative_config: SpeculativeConfig | None,
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ):
        # Register counters for each metric
        counter_drafts = prometheus_client.Counter(
            name="vllm:spec_decode_num_drafts",
            documentation="Number of spec decoding drafts.",
            labelnames=labelnames,
        )
        # ... similar for other counters

        # Per-position counters (one per speculative token position)
        num_spec_tokens = speculative_config.num_speculative_tokens
        base_counter = prometheus_client.Counter(
            name="vllm:spec_decode_num_accepted_tokens_per_pos",
            documentation="Accepted tokens per draft position.",
            labelnames=labelnames + ["position"],
        )
```

## SpecDecodeMetadata

The `SpecDecodeMetadata` dataclass (`vllm/v1/spec_decode/metadata.py`) carries per-batch information needed for the verification step:

```python
@dataclass
class SpecDecodeMetadata:
    draft_token_ids: torch.Tensor        # [num_tokens] - all draft token IDs
    num_draft_tokens: list[int]          # [batch_size] - per-request draft counts
    cu_num_draft_tokens: torch.Tensor    # [batch_size] - cumulative draft counts
    cu_num_sampled_tokens: torch.Tensor  # [batch_size] - cumulative sampled counts
    target_logits_indices: torch.Tensor  # [num_tokens] - indices into target logits
    bonus_logits_indices: torch.Tensor   # [batch_size] - bonus token logit indices
    logits_indices: torch.Tensor         # [num_tokens + batch_size] - combined indices
```

This metadata is used by the rejection sampler to efficiently extract the relevant logits from the target model's output and perform token acceptance/rejection.

## Tuning for Performance

### Choosing `num_speculative_tokens`

The optimal number of speculative tokens depends on your acceptance rate:

| Acceptance Rate | Recommended `num_speculative_tokens` |
|----------------|--------------------------------------|
| > 0.85 | 5–10 |
| 0.70–0.85 | 3–5 |
| 0.50–0.70 | 2–3 |
| < 0.50 | 1–2 (or disable) |

Monitor the per-position acceptance rates. If the rate at position `k` drops below ~0.3, reducing `num_speculative_tokens` to `k` will improve efficiency.

### Token Budget

The effective token budget per step is:
```
tokens_per_step = 1 (base) + num_speculative_tokens (draft) + 1 (bonus)
```

For a batch of `B` requests with `k` speculative tokens, the target model processes `B × (k + 1)` tokens per forward pass. Ensure your GPU memory and compute budget can handle this.

### Monitoring Acceptance Rate Over Time

Acceptance rates can vary with:
- **Input distribution**: Different prompt types have different acceptance rates
- **Temperature**: Higher temperature → lower acceptance rate
- **Context length**: Very long contexts may reduce acceptance rates for some methods

Use the Prometheus metrics to track acceptance rates in production and adjust `num_speculative_tokens` accordingly.

## Related Pages

- [Overview](README.md) — Speculative decoding overview
- [Configuration](configuration.md) — Full `SpeculativeConfig` reference
- [EAGLE](eagle.md) — EAGLE speculative decoding
- [N-gram Proposer](ngram.md) — N-gram prompt lookup
