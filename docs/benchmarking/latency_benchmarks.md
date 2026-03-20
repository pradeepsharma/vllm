# Latency Benchmarks

The `vllm bench latency` command measures the end-to-end latency of processing a **single batch** of requests through the vLLM engine. Unlike the online serving benchmark (`vllm bench serve`), latency benchmarks bypass the HTTP server and call the LLM engine directly, giving a clean measurement of pure inference time.

---

## When to Use Latency Benchmarks

Use `vllm bench latency` when you want to:

- Measure the raw inference latency of a specific batch size and sequence length
- Evaluate the impact of engine configuration changes (quantization, tensor parallelism, etc.)
- Profile a single batch with the PyTorch or CUDA profiler
- Establish a baseline before running online serving benchmarks
- Detect latency regressions in CI

For measuring latency under realistic concurrent load, use [`vllm bench serve`](performance_benchmarks.md) instead.

---

## Quick Start

```bash
vllm bench latency \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 32 \
  --output-len 128 \
  --batch-size 8
```

### Example Output

```text
Warming up...
Warmup iterations: 100%|████████████████| 10/10 [00:12<00:00,  1.24s/it]
Bench iterations: 100%|████████████████| 30/30 [00:37<00:00,  1.24s/it]
Avg latency: 1.2341 seconds
10% percentile latency: 1.2198 seconds
25% percentile latency: 1.2267 seconds
50% percentile latency: 1.2334 seconds
75% percentile latency: 1.2401 seconds
90% percentile latency: 1.2478 seconds
99% percentile latency: 1.2591 seconds
```

---

## CLI Reference

```bash
vllm bench latency --help
```

### Core Options

| Option | Default | Description |
|---|---|---|
| `--model` | *(required)* | HuggingFace model ID or local path |
| `--input-len` | `32` | Number of input tokens per request |
| `--output-len` | `128` | Number of output tokens to generate |
| `--batch-size` | `8` | Number of requests in the batch |
| `--n` | `1` | Number of output sequences per prompt |
| `--use-beam-search` | `false` | Use beam search instead of sampling |

### Warmup and Iteration Control

| Option | Default | Description |
|---|---|---|
| `--num-iters-warmup` | `10` | Warmup iterations (discarded from measurement) |
| `--num-iters` | `30` | Measurement iterations |

Warmup is critical for accurate latency measurement. It allows:

- CUDA kernels to be compiled and cached (torch.compile)
- GPU caches to reach steady state
- JIT compilation overhead to be excluded from results

### Output Options

| Option | Default | Description |
|---|---|---|
| `--output-json` | *(none)* | Path to save results as JSON |
| `--disable-detokenize` | `false` | Exclude detokenization time from measurement |

### Profiling Options

| Option | Default | Description |
|---|---|---|
| `--profile` | `false` | Enable profiling for a single batch after warmup |

When `--profile` is set, the benchmark runs warmup iterations normally, then profiles exactly one batch. The profiler type and output directory are controlled by `--profiler-config` (see [Profiling](profiling.md)).

---

## Common Benchmark Scenarios

### Decode-Heavy Workload (Small Input, Large Output)

```bash
vllm bench latency \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 32 \
  --output-len 512 \
  --batch-size 16 \
  --num-iters 50
```

### Prefill-Heavy Workload (Large Input, Small Output)

```bash
vllm bench latency \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 2048 \
  --output-len 16 \
  --batch-size 4 \
  --num-iters 30
```

### Long-Context Workload

```bash
vllm bench latency \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 8192 \
  --output-len 256 \
  --batch-size 1 \
  --max-model-len 16384 \
  --num-iters 20
```

### Beam Search

```bash
vllm bench latency \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 64 \
  --output-len 128 \
  --batch-size 4 \
  --n 4 \
  --use-beam-search
```

### Quantized Model

```bash
vllm bench latency \
  --model neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
  --input-len 512 \
  --output-len 128 \
  --batch-size 8 \
  --dtype float16
```

### Multi-GPU (Tensor Parallelism)

```bash
vllm bench latency \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --input-len 512 \
  --output-len 128 \
  --batch-size 4 \
  --tensor-parallel-size 4
```

---

## Saving Results to JSON

```bash
vllm bench latency \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 512 \
  --output-len 128 \
  --batch-size 8 \
  --output-json latency_results.json
```

The JSON output contains:

```json
{
    "avg_latency": 1.2341,
    "latencies": [1.2198, 1.2267, 1.2334, ...],
    "percentiles": {
        "10": 1.2198,
        "25": 1.2267,
        "50": 1.2334,
        "75": 1.2401,
        "90": 1.2478,
        "99": 1.2591
    }
}
```

A companion `latency_results.pytorch.json` file is also generated in PyTorch benchmark format for integration with the PyTorch benchmark infrastructure.

---

## Profiling a Single Batch

To profile the latency benchmark with the PyTorch profiler:

```bash
vllm bench latency \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 512 \
  --output-len 128 \
  --batch-size 8 \
  --profile \
  --profiler-config.profiler torch \
  --profiler-config.torch-profiler-dir /tmp/latency_profile
```

After the benchmark completes, open the trace in TensorBoard:

```bash
tensorboard --logdir /tmp/latency_profile
```

For CUDA profiler integration:

```bash
vllm bench latency \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 512 \
  --output-len 128 \
  --batch-size 8 \
  --profile \
  --profiler-config.profiler cuda
```

See [Profiling](profiling.md) for a complete guide to interpreting profiler output.

---

## Disabling Prefix Caching

The vLLM V1 engine enables prefix caching by default, which can skew latency measurements by reusing KV cache across iterations. The latency benchmark automatically disables prefix caching to ensure clean measurements:

```python
# This is set automatically by vllm bench latency
parser.set_defaults(enable_prefix_caching=False)
```

If you want to measure latency **with** prefix caching enabled (e.g., to evaluate cache hit performance), pass it explicitly:

```bash
vllm bench latency \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 512 \
  --output-len 128 \
  --batch-size 8 \
  --enable-prefix-caching
```

---

## Understanding Latency Components

A single batch latency measurement includes:

```
Total Batch Latency
├── Prefill phase
│   ├── Tokenization (if not pre-tokenized)
│   ├── Attention computation (scales with input_len²)
│   └── KV cache population
└── Decode phase (repeated output_len times)
    ├── Attention computation (scales with sequence_len)
    ├── Sampling
    └── Detokenization (unless --disable-detokenize)
```

**Prefill time** scales roughly as O(input_len²) for standard attention and O(input_len) for linear attention variants.

**Decode time** scales roughly as O(sequence_len) per token due to KV cache reads.

For long-output workloads, decode time dominates. For long-input workloads, prefill time dominates.

---

## Latency vs. Throughput Trade-off

Latency and throughput are inversely related:

- **Small batch sizes** → lower latency, lower throughput
- **Large batch sizes** → higher latency, higher throughput

Use `vllm bench latency` to find the minimum achievable latency (batch size = 1), then use `vllm bench serve` or `vllm bench sweep serve_workload` to find the throughput-latency Pareto frontier.

---

## Attention Kernel Latency

For fine-grained attention kernel benchmarks (not full model), use the attention benchmark suite:

```bash
cd benchmarks/attention_benchmarks

# Decode latency for different batch sizes
python benchmark.py \
  --backends flash flashinfer \
  --batch-specs "1q1s1k" "8q1s1k" "32q1s1k" "64q1s1k" "128q1s1k" \
  --output-csv decode_latency.csv

# Prefill latency for different sequence lengths
python benchmark.py \
  --backends flash triton \
  --batch-specs "q512" "q1k" "q2k" "q4k" "q8k" \
  --output-csv prefill_latency.csv
```

---

## Related Pages

- [Throughput Benchmarks](throughput_benchmarks.md) — Offline batch throughput
- [Performance Benchmarks](performance_benchmarks.md) — Online serving benchmarks
- [Profiling](profiling.md) — Deep-dive profiler integration
- [Benchmark CLI Reference](cli.md) — Full CLI reference
