# Benchmark Result Interpretation Guide

This guide explains how to read and interpret the metrics produced by vLLM's benchmark tools, and how to use them to make informed decisions about performance optimization.

## Core Metrics

### Time to First Token (TTFT)

TTFT measures the time from when a request is sent to when the first output token is received. It is dominated by the **prefill** phase.

```
TTFT = network_latency + queue_wait + prefill_time
```

**What it tells you:**
- High TTFT → prefill is the bottleneck (long prompts, insufficient batched tokens)
- Low TTFT → good responsiveness for interactive applications

**Typical targets:**
| Application | Target TTFT |
|-------------|-------------|
| Interactive chat | < 200ms |
| Code completion | < 500ms |
| Document QA | < 2000ms |
| Batch processing | Not critical |

### Time per Output Token (TPOT)

TPOT measures the average time between consecutive output tokens (excluding the first token). It reflects **decode** throughput.

```
TPOT = total_decode_time / (num_output_tokens - 1)
```

**What it tells you:**
- High TPOT → decode is the bottleneck (too many concurrent sequences, memory bandwidth limited)
- Low TPOT → fast token generation, good for streaming applications

**Typical targets:**
| Application | Target TPOT |
|-------------|-------------|
| Real-time streaming | < 50ms (> 20 tok/s) |
| Background generation | < 200ms |
| Batch processing | Not critical |

### Inter-Token Latency (ITL)

ITL is similar to TPOT but measured from the client side as the time between receiving consecutive tokens. In practice, ITL ≈ TPOT for well-behaved servers.

```
ITL = time_between_received_tokens
```

### End-to-End Latency (E2EL)

E2EL is the total time from sending a request to receiving the complete response.

```
E2EL = TTFT + (num_output_tokens - 1) × TPOT
```

**What it tells you:**
- The overall user experience quality
- Used for SLO (Service Level Objective) compliance

### Throughput Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| `request_throughput` | `completed / duration` | Requests per second |
| `output_throughput` | `total_output_tokens / duration` | Output tokens per second |
| `total_token_throughput` | `(input + output) / duration` | Total tokens per second |
| `request_goodput` | `slo_compliant / duration` | Requests/s meeting SLOs |

## Reading `vllm bench serve` Output

```
============================================================
Traffic request rate: 10.0
Burstiness factor: 1.0 (Poisson process)
--------------------Time to First Token--------------------
Mean TTFT (ms):                          45.23    ← Average TTFT
Median TTFT (ms):                        42.10    ← 50th percentile
P99 TTFT (ms):                           98.45    ← 99th percentile (tail latency)
-----------Time per Output Token (excl. 1st token)---------
Mean TPOT (ms):                          12.34
Median TPOT (ms):                        11.89
P99 TPOT (ms):                           18.23
--------------------Inter-token Latency--------------------
Mean ITL (ms):                           12.34
--------------------End-to-end Latency---------------------
Mean E2EL (ms):                         234.56
Median E2EL (ms):                       221.34
P99 E2EL (ms):                          456.78
==================================================
Successful requests:                     1000
Failed requests:                         0
Request throughput (req/s):              9.87     ← Actual vs target 10.0
Request goodput (req/s):                 9.45     ← Requests meeting SLOs
Output token throughput (tok/s):         1264.32
Total token throughput (tok/s):          6400.00
```

### Key Observations

1. **Request throughput < request rate** — The server cannot keep up; reduce request rate or optimize server
2. **High P99 vs median** — High tail latency; consider chunked prefill or reducing `max-num-seqs`
3. **Goodput < throughput** — Some requests miss SLOs; tighten server parameters or relax SLOs
4. **Failed requests > 0** — Server errors; check logs for OOM or timeout issues

## Percentile Analysis

Percentiles reveal the distribution of latency:

| Percentile | Meaning |
|------------|---------|
| P50 (median) | Half of requests are faster than this |
| P90 | 90% of requests are faster than this |
| P99 | 99% of requests are faster than this (tail latency) |
| P99.9 | 99.9% of requests are faster than this |

**Healthy distribution:** P99 / P50 < 3× (low tail latency)

**Problematic distribution:** P99 / P50 > 10× (high tail latency, likely caused by head-of-line blocking from long prefills)

```bash
# Report more percentiles
vllm bench serve \
    --percentile-metrics ttft,tpot,e2el \
    --metric-percentiles 50,90,95,99,99.9
```

## Throughput vs Latency Trade-off

As request rate increases, latency increases due to queuing:

```
Low QPS:   Low latency, low throughput (underutilized)
Medium QPS: Balanced latency and throughput
High QPS:  High latency, high throughput (saturated)
```

To find the optimal operating point:

```bash
# Sweep request rates
for qps in 1 2 4 8 16 32 64; do
    vllm bench serve \
        --request-rate $qps \
        --num-prompts 500 \
        --save-result \
        --result-filename "qps_${qps}.json"
done
```

Plot TTFT and throughput vs QPS to find the "knee" of the curve — the point where latency starts increasing rapidly.

## Goodput and SLOs

Goodput measures the fraction of requests that meet all SLO constraints:

```bash
# Define SLOs: TTFT < 200ms, E2EL < 1000ms
vllm bench serve \
    --request-rate 10 \
    --goodput ttft:200 e2el:1000
```

**Interpreting goodput:**
- `goodput ≈ throughput` → All requests meet SLOs; can increase load
- `goodput << throughput` → Many requests miss SLOs; reduce load or optimize

## Reading Attention Benchmark Results

The attention benchmark outputs a Rich table:

```
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│ Batch Spec      │ FLASH_ATTN   │ TRITON_ATTN  │ FLASHINFER   │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ q2k             │ 1.234ms ✓    │ 1.456ms      │ 1.389ms      │
│ 8q1s1k          │ 0.234ms      │ 0.267ms      │ 0.198ms ✓    │
│ 2q2k_32q1s1k    │ 0.567ms      │ 0.612ms      │ 0.523ms ✓    │
└─────────────────┴──────────────┴──────────────┴──────────────┘
✓ = fastest for this batch spec
```

**Key insights:**
- Different backends win for different workloads
- Prefill-heavy → Flash Attention often wins
- Decode-heavy → FlashInfer often wins
- Mixed batches → depends on ratio

## Reading CUTLASS Benchmark Results

```
[------------- scaled-torch.float8_e4m3fn-gemm -------------]
                                  |  pytorch_bf16  |  cutlass_fp8_bf16  |  triton_fp8_blockwise
1 threads: -----------------------------------------------
      MKN=(1x4096x14336)         |    1234.5 us   |      456.7 us      |       512.3 us
      MKN=(16x4096x14336)        |    2345.6 us   |      789.0 us      |       834.5 us
      MKN=(128x4096x14336)       |    8901.2 us   |     2345.6 us      |      2456.7 us
```

**Speedup calculation:**
```
Speedup = pytorch_bf16_time / cutlass_fp8_time
= 1234.5 / 456.7 = 2.7× speedup for M=1
```

**When to use each kernel:**
- Small M (decode): CUTLASS FP8 often wins
- Large M (prefill): Triton block-wise may win for block-quantized models

## Reading `auto_tune.sh` Results

```
hash:a1b2c3d4...
max_num_seqs: 128, max_num_batched_tokens: 2048, request_rate: 10.0, e2el: 450.5, throughput: 9.8, goodput: 9.8
max_num_seqs: 128, max_num_batched_tokens: 4096, request_rate: 10.0, e2el: 380.2, throughput: 9.9, goodput: 9.9
max_num_seqs: 256, max_num_batched_tokens: 2048, request_rate: 10.0, e2el: 520.1, throughput: 9.7, goodput: 8.2
max_num_seqs: 256, max_num_batched_tokens: 4096 does not meet latency requirement 500
best_max_num_seqs: 128, best_num_batched_tokens: 4096, best_throughput: 9.9
```

**Interpretation:**
- `max_num_seqs=256, max_num_batched_tokens=2048` → goodput drops (some requests miss SLO)
- `max_num_seqs=256, max_num_batched_tokens=4096` → exceeds latency limit entirely
- **Best:** `max_num_seqs=128, max_num_batched_tokens=4096` → highest throughput within SLO

## LM Eval Accuracy Results

```python
results = lm_eval.simple_evaluate(model="vllm", tasks="gsm8k", ...)
score = results["results"]["gsm8k"]["exact_match,strict-match"]
# e.g., 0.68 = 68% accuracy on GSM8K
```

**Interpreting accuracy:**
- Compare against published baselines for the model
- ±3% tolerance is typical for non-deterministic sampling
- Significant drops (> 5%) indicate a regression

## Common Performance Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| High TTFT, low TPOT | Prefill bottleneck | Increase `max-num-batched-tokens`, enable chunked prefill |
| Low TTFT, high TPOT | Decode bottleneck | Reduce `max-num-seqs`, use FP8 KV cache |
| High P99/P50 ratio | Head-of-line blocking | Enable chunked prefill |
| Throughput < request rate | Server saturated | Reduce QPS or scale horizontally |
| High failed requests | OOM or timeout | Reduce `gpu-memory-utilization` or `max-num-seqs` |
| Goodput << throughput | SLO violations | Relax SLOs or reduce load |

## Related Pages

- [Serving Benchmarks](serving-benchmarks.md) — Running benchmarks
- [Performance Tuning Guide](performance-tuning.md) — Optimization strategies
- [Auto-Tune Scripts](auto-tune.md) — Automated parameter search
- [Observability Metrics](../11-observability/metrics.md) — Production monitoring
