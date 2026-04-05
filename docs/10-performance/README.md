# Performance & Benchmarking

This section provides an overview of vLLM's performance characteristics and links to the comprehensive benchmarking documentation.

> For the full benchmarking suite documentation, see [Benchmarking & Performance](../15-benchmarking/README.md).

## Performance Overview

vLLM achieves high throughput and low latency through several complementary mechanisms:

| Mechanism | Benefit |
|-----------|---------|
| PagedAttention | Eliminates KV cache fragmentation; enables prefix sharing |
| Continuous batching | Maximizes GPU utilization across concurrent requests |
| Chunked prefill | Reduces TTFT for concurrent requests during long prefills |
| `torch.compile` (mode 3) | Eliminates kernel launch overhead via CUDA graph capture |
| FlashAttention / FlashInfer | Hardware-optimized attention kernels |
| Speculative decoding | Up to 3× speedup for low-entropy generation |
| FP8 quantization | ~2× throughput on H100/H200 with near-zero accuracy loss |
| Prefix caching | Avoids recomputing KV for repeated prompt prefixes |

## Quick Benchmarks

```bash
# Measure offline throughput
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --input-len 512 \
  --output-len 128

# Measure online serving latency
vllm serve meta-llama/Llama-3.1-8B-Instruct &
vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --request-rate 10 \
  --num-prompts 200
```

## Key Performance Metrics

| Metric | Description |
|--------|-------------|
| **TTFT** | Time to first token — latency from request arrival to first output token |
| **ITL** | Inter-token latency — time between consecutive output tokens |
| **E2E latency** | Total request latency from submission to completion |
| **Throughput** | Output tokens per second across all concurrent requests |
| **Goodput** | Throughput for requests meeting SLA latency targets |

## See Also

- [Benchmarking & Performance](../15-benchmarking/README.md) — full benchmark suite
- [Performance Tuning Guide](../15-benchmarking/performance-tuning.md) — tuning recommendations
- [Serving Benchmarks](../15-benchmarking/serving-benchmarks.md) — latency/throughput CLI
- [Compilation & Optimization](../10-compilation/README.md) — `torch.compile` details
- [Speculative Decoding](../10-speculative-decoding/README.md) — latency acceleration
- [Quantization](../14-quantization/README.md) — memory and throughput optimization
