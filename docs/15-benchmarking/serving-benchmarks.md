# Serving Benchmarks

vLLM provides a unified `vllm bench` CLI for measuring latency, throughput, and online serving performance. The benchmark implementations live in `vllm/benchmarks/` and are invoked via `vllm bench latency`, `vllm bench throughput`, and `vllm bench serve`.

> **Migration note:** The legacy `benchmarks/benchmark_latency.py`, `benchmark_throughput.py`, and `benchmark_serving.py` scripts are deprecated. Use `vllm bench latency`, `vllm bench throughput`, and `vllm bench serve` instead.

## CLI Overview

```
vllm bench <subcommand> [options]

Subcommands:
  latency      Benchmark offline batch latency
  throughput   Benchmark offline inference throughput
  serve        Benchmark online serving throughput
  sweep        Sweep over parameter combinations
  startup      Benchmark server startup time
  mm-processor Benchmark multimodal processor
```

## `vllm bench latency`

Measures the time to process a single batch of requests offline (no server required). Useful for measuring raw model throughput without network overhead.

**Implementation:** `vllm/benchmarks/latency.py`

### Usage

```bash
vllm bench latency \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --input-len 512 \
    --output-len 128 \
    --batch-size 8 \
    --num-iters 30 \
    --num-iters-warmup 10 \
    --output-json latency_results.json
```

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `--input-len` | 32 | Number of input tokens per request |
| `--output-len` | 128 | Number of output tokens to generate |
| `--batch-size` | 8 | Number of requests in the batch |
| `--n` | 1 | Number of generated sequences per prompt |
| `--num-iters` | 30 | Number of timed iterations |
| `--num-iters-warmup` | 10 | Warmup iterations (not timed) |
| `--use-beam-search` | False | Use beam search instead of sampling |
| `--profile` | False | Enable profiler during one iteration |
| `--output-json` | None | Save results to JSON file |
| `--disable-detokenize` | False | Exclude detokenization from timing |

Plus all `EngineArgs` (model, dtype, tensor-parallel-size, etc.).

> **Note:** Prefix caching is disabled by default in latency benchmarks to avoid cache hits skewing results.

### Output

```
Warming up...
Bench iterations: 100%|████████| 30/30
Avg latency: 2.345 seconds
10% percentile latency: 2.301 seconds
25% percentile latency: 2.318 seconds
50% percentile latency: 2.340 seconds
75% percentile latency: 2.367 seconds
90% percentile latency: 2.389 seconds
99% percentile latency: 2.412 seconds
```

## `vllm bench throughput`

Measures offline inference throughput by processing a dataset of prompts as fast as possible.

**Implementation:** `vllm/benchmarks/throughput.py`

### Usage

```bash
vllm bench throughput \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 \
    --output-json throughput_results.json
```

### Supported Datasets

| Dataset | Description |
|---------|-------------|
| `random` | Synthetic random token sequences |
| `sharegpt` | ShareGPT conversation dataset |
| `sonnet` | Shakespeare sonnets (for prefix testing) |
| `burstgpt` | BurstGPT traffic patterns |
| `aimo` | AIMO math competition dataset |
| `instruct-coder` | Code instruction dataset |
| `conversation` | Multi-turn conversation dataset |
| `vision-arena` | Vision Arena multimodal dataset |

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `--dataset-name` | `random` | Dataset to use |
| `--num-prompts` | 1000 | Number of prompts to process |
| `--n` | 1 | Sequences per prompt |
| `--use-beam-search` | False | Use beam search |
| `--enable-lora` | False | Enable LoRA adapter loading |
| `--output-json` | None | Save results to JSON |

### Output

```
Throughput: 1234.5 requests/s, 98765.4 tokens/s
```

## `vllm bench serve`

Benchmarks online serving by sending requests to a running vLLM server at a specified rate and measuring latency metrics.

**Implementation:** `vllm/benchmarks/serve.py`

### Setup

```bash
# Start the server
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# Run the benchmark
vllm bench serve \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --backend openai \
    --dataset-name random \
    --request-rate 10 \
    --num-prompts 1000 \
    --save-result \
    --result-dir ./results
```

### Key Options

| Option | Default | Description |
|--------|---------|-------------|
| `--backend` | `openai` | Backend type (see table below) |
| `--host` | `127.0.0.1` | Server host |
| `--port` | `8000` | Server port |
| `--model` | (from server) | Model name |
| `--request-rate` | `inf` | Requests per second (Poisson process) |
| `--burstiness` | `1.0` | Burstiness factor (1.0 = Poisson, >1 = bursty) |
| `--num-prompts` | `1000` | Total number of requests |
| `--max-concurrency` | None | Maximum concurrent requests |
| `--input-len` | None | General input length |
| `--output-len` | None | General output length |
| `--goodput` | None | SLO constraints (e.g., `ttft:200 e2el:1000`) |
| `--save-result` | False | Save results to JSON |
| `--result-dir` | None | Directory for result files |
| `--result-filename` | (auto) | Custom result filename |
| `--profile` | False | Enable server-side profiling |
| `--num-warmups` | 0 | Warmup requests before timing |
| `--percentile-metrics` | `ttft,itl,e2el` | Metrics to compute percentiles for |
| `--metric-percentiles` | `99` | Percentile values to report |

### Supported Backends

| Backend | Endpoint | Description |
|---------|----------|-------------|
| `openai` | `/v1/completions` | OpenAI-compatible completions |
| `openai-chat` | `/v1/chat/completions` | OpenAI-compatible chat |
| `vllm` | `/v1/completions` | vLLM-specific (with extra fields) |
| `tgi` | `/generate` | Text Generation Inference |
| `lmdeploy` | `/v1/completions` | LMDeploy |
| `deepspeed-mii` | `/mii/completions` | DeepSpeed-MII |
| `openai-embeddings` | `/v1/embeddings` | Embedding endpoint |

### Traffic Patterns

**Constant rate (Poisson process):**
```bash
vllm bench serve --request-rate 10 --burstiness 1.0
```

**Bursty traffic (Gamma distribution):**
```bash
vllm bench serve --request-rate 10 --burstiness 3.0
```

**Maximum throughput (all requests at once):**
```bash
vllm bench serve --request-rate inf
```

**Ramp-up (linear):**
```bash
vllm bench serve \
    --ramp-up-strategy linear \
    --ramp-up-start-rps 1 \
    --ramp-up-end-rps 20
```

**Concurrency-limited:**
```bash
vllm bench serve --request-rate 100 --max-concurrency 50
```

### Goodput Measurement

Goodput measures the fraction of requests that meet SLO constraints:

```bash
vllm bench serve \
    --request-rate 10 \
    --goodput ttft:200 tpot:50 e2el:1000
```

This reports both raw throughput and goodput (requests/s that met all SLOs).

### Output Metrics

```
============================================================
Traffic request rate: 10.0
Burstiness factor: 1.0 (Poisson process)
Maximum request concurrency: None
--------------------Time to First Token--------------------
Mean TTFT (ms):                          45.23
Median TTFT (ms):                        42.10
P99 TTFT (ms):                           98.45
-----------Time per Output Token (excl. 1st token)---------
Mean TPOT (ms):                          12.34
Median TPOT (ms):                        11.89
P99 TPOT (ms):                           18.23
--------------------Inter-token Latency--------------------
Mean ITL (ms):                           12.34
Median ITL (ms):                         11.89
P99 ITL (ms):                            18.23
--------------------End-to-end Latency---------------------
Mean E2EL (ms):                         234.56
Median E2EL (ms):                       221.34
P99 E2EL (ms):                          456.78
==================================================
Successful requests:                     1000
Failed requests:                         0
Total input tokens:                      512000
Total generated tokens:                  128000
Request throughput (req/s):              9.87
Request goodput (req/s):                 9.45
Output token throughput (tok/s):         1264.32
Total token throughput (tok/s):          6400.00
```

## `BenchmarkMetrics` Data Class

The `BenchmarkMetrics` dataclass (in `vllm/benchmarks/serve.py`) captures all metrics:

```python
@dataclass
class BenchmarkMetrics:
    completed: int              # Successful requests
    failed: int                 # Failed requests
    total_input: int            # Total input tokens
    total_output: int           # Total output tokens
    request_throughput: float   # req/s
    request_goodput: float      # req/s meeting SLOs
    output_throughput: float    # output tok/s
    total_token_throughput: float  # (input + output) tok/s
    mean_ttft_ms: float         # Mean Time to First Token
    median_ttft_ms: float
    std_ttft_ms: float
    percentiles_ttft_ms: list[tuple[float, float]]
    mean_tpot_ms: float         # Mean Time per Output Token
    median_tpot_ms: float
    std_tpot_ms: float
    percentiles_tpot_ms: list[tuple[float, float]]
    mean_itl_ms: float          # Mean Inter-token Latency
    median_itl_ms: float
    std_itl_ms: float
    percentiles_itl_ms: list[tuple[float, float]]
    mean_e2el_ms: float         # Mean End-to-end Latency
    median_e2el_ms: float
    std_e2el_ms: float
    percentiles_e2el_ms: list[tuple[float, float]]
    max_output_tokens_per_s: float  # Peak output tokens/s
    max_concurrent_requests: int    # Concurrent requests at peak
    rtfx: float                     # Real-time factor (ASR)
```

## Result File Format

When `--save-result` is used, results are saved as JSON:

```json
{
  "completed": 1000,
  "total_input_tokens": 512000,
  "total_output_tokens": 128000,
  "request_throughput": 9.87,
  "output_throughput": 1264.32,
  "mean_ttft_ms": 45.23,
  "median_ttft_ms": 42.10,
  "p99_ttft_ms": 98.45,
  "mean_tpot_ms": 12.34,
  "p99_tpot_ms": 18.23,
  "mean_e2el_ms": 234.56,
  "p99_e2el_ms": 456.78
}
```

A PyTorch benchmark format file (`.pytorch.json`) is also saved for integration with benchmark tracking systems.

## Related Pages

- [Result Interpretation](result-interpretation.md) — Understanding metrics
- [Auto-Tune Scripts](auto-tune.md) — Automated parameter search
- [Performance Tuning Guide](performance-tuning.md) — Optimization strategies
- [Disaggregated Benchmarks](disagg-benchmarks.md) — Disaggregated prefill benchmarks
