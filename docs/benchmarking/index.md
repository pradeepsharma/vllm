# Benchmarking Overview

vLLM ships a comprehensive benchmarking suite that covers every dimension of LLM serving performance — from raw kernel throughput to end-to-end online serving latency. This section explains what tools are available, when to use each one, and how they fit together.

---

## Why Benchmark?

Benchmarking answers three distinct questions:

| Question | Tool |
|---|---|
| **How fast is my server under realistic load?** | `vllm bench serve` (online serving) |
| **How many tokens can I generate per second in batch mode?** | `vllm bench throughput` (offline throughput) |
| **How long does a single batch take end-to-end?** | `vllm bench latency` (latency) |
| **How do different configurations compare?** | `vllm bench sweep` (parameter sweeps) |
| **Where is time being spent inside the model?** | vLLM Profiler (PyTorch / CUDA) |
| **How does my model score on standard NLP tasks?** | LM Evaluation Harness |

---

## Benchmark Tools at a Glance

### `vllm bench` CLI

The primary entry point for all performance benchmarks. All legacy `benchmark_*.py` scripts have been migrated into the unified `vllm bench` CLI.

```
vllm bench <subcommand> [options]
```

| Subcommand | Purpose |
|---|---|
| `serve` | Online serving benchmark — measures TTFT, TPOT, ITL, E2EL, and throughput against a live server |
| `throughput` | Offline batch throughput — measures tokens/s for a fixed request set |
| `latency` | Single-batch latency — measures per-batch latency with warmup and percentile reporting |
| `startup` | Model startup time — measures cold and warm startup latency |
| `sweep serve` | Parameter sweep over server and benchmark configurations |
| `sweep serve_workload` | Workload explorer — finds the latency/throughput Pareto frontier |
| `sweep startup` | Startup time sweep across engine configurations |
| `sweep plot` | Visualize sweep results as performance curves |
| `sweep plot_pareto` | Visualize per-user vs per-GPU throughput Pareto chart |

### Specialized Benchmark Scripts

Located in `benchmarks/`, these scripts target specific features:

| Script / Directory | What it measures |
|---|---|
| `benchmarks/attention_benchmarks/` | Attention kernel performance (Flash, Triton, FlashInfer, MLA backends) |
| `benchmarks/kernels/` | Individual CUDA kernel benchmarks (GEMM, quantization, MoE, RoPE, etc.) |
| `benchmarks/disagg_benchmarks/` | Disaggregated prefill vs. chunked prefill comparison |
| `benchmarks/multi_turn/` | Multi-turn conversation serving with KV cache offloading |
| `benchmarks/auto_tune/` | Automated server parameter tuning (`max-num-seqs`, `max-num-batched-tokens`) |
| `benchmarks/cutlass_benchmarks/` | CUTLASS W8A8 and sparse kernel benchmarks |
| `benchmarks/fused_kernels/` | Fused LayerNorm / RMSNorm benchmarks |

---

## Key Metrics Explained

### Online Serving Metrics (`vllm bench serve`)

| Metric | Abbreviation | Definition |
|---|---|---|
| Time to First Token | **TTFT** | Wall-clock time from sending a request to receiving the first output token |
| Time per Output Token | **TPOT** | Average time between consecutive output tokens (excludes first token) |
| Inter-Token Latency | **ITL** | Time between any two consecutive tokens (includes first token) |
| End-to-End Latency | **E2EL** | Total time from request send to final token received |
| Request Throughput | **req/s** | Completed requests per second |
| Output Token Throughput | **tok/s** | Output tokens generated per second |
| Total Token Throughput | **total tok/s** | Input + output tokens processed per second |
| Request Goodput | **goodput** | Throughput counting only requests that meet SLA constraints |

### Offline / Latency Metrics

| Metric | Definition |
|---|---|
| Batch latency | Wall-clock time to process one complete batch |
| Tokens/s | Total tokens (prompt + output) processed per second |
| Output tokens/s | Output tokens generated per second |
| Percentile latencies | P10, P25, P50, P75, P90, P99 batch latency |

---

## Choosing the Right Benchmark

```
Are you measuring a live server?
├── YES → vllm bench serve
│         ├── Single config → vllm bench serve
│         └── Multiple configs → vllm bench sweep serve
│
└── NO (offline / batch)
    ├── Throughput (tokens/s) → vllm bench throughput
    ├── Per-batch latency → vllm bench latency
    ├── Startup time → vllm bench startup
    └── Kernel-level → benchmarks/kernels/ or benchmarks/attention_benchmarks/
```

---

## Datasets

The benchmark suite supports a wide range of datasets for realistic workload simulation:

| Dataset | Type | Notes |
|---|---|---|
| **ShareGPT** | Real conversations | Most commonly used; covers diverse prompt lengths |
| **BurstGPT** | Real traffic traces | Captures bursty arrival patterns |
| **Random** | Synthetic | Fully configurable input/output lengths |
| **Sonnet** | Synthetic (deprecated) | Fixed-length prompts from Shakespeare sonnets |
| **HuggingFace** | Various | VisionArena, MMVU, InstructCoder, AIMO, MT-Bench, etc. |
| **Custom** | JSONL | Bring your own prompts |
| **Custom MM** | JSONL | Bring your own multimodal prompts |
| **Spec Bench** | Speculative decoding | Standard speculative decoding evaluation |

See [Performance Benchmarks](performance_benchmarks.md) for dataset download instructions.

---

## Continuous Performance Dashboard

vLLM's CI automatically runs benchmarks on every commit and publishes results to the [vLLM Performance Dashboard](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm). The dashboard tracks:

- Serving throughput and latency across models
- Regression detection between commits
- Multi-GPU and multi-node configurations

See [Performance Dashboard](dashboard.md) for details on triggering manual runs and interpreting results.

---

## In This Section

<div class="grid cards" markdown>

-   :material-speedometer: **Performance Benchmarks**

    ---

    Running the full benchmark suite, dataset setup, and interpreting results.

    [:octicons-arrow-right-24: Performance Benchmarks](performance_benchmarks.md)

-   :material-timer-outline: **Latency Benchmarks**

    ---

    Measuring single-batch latency with warmup, percentile reporting, and profiling integration.

    [:octicons-arrow-right-24: Latency Benchmarks](latency_benchmarks.md)

-   :material-chart-line: **Throughput Benchmarks**

    ---

    Offline batch throughput measurement across backends, datasets, and configurations.

    [:octicons-arrow-right-24: Throughput Benchmarks](throughput_benchmarks.md)

-   :material-tune: **Parameter Sweeps**

    ---

    Automated multi-configuration sweeps with visualization and Pareto analysis.

    [:octicons-arrow-right-24: Parameter Sweeps](sweeps.md)

-   :material-flask: **LM Evaluation Harness**

    ---

    Accuracy evaluation on standard NLP benchmarks using the lm-evaluation-harness framework.

    [:octicons-arrow-right-24: LM Eval Harness](lm_eval_harness.md)

-   :material-magnify: **Profiling**

    ---

    Deep-dive performance analysis with PyTorch and CUDA profilers.

    [:octicons-arrow-right-24: Profiling](profiling.md)

-   :material-chart-bar: **Benchmark CLI Reference**

    ---

    Full CLI reference for all `vllm bench` subcommands and dataset options.

    [:octicons-arrow-right-24: Benchmark CLI](cli.md)

-   :material-monitor-dashboard: **Performance Dashboard**

    ---

    Continuous benchmarking CI and the public performance dashboard.

    [:octicons-arrow-right-24: Performance Dashboard](dashboard.md)

</div>
