# Performance Benchmarks

This guide covers the full vLLM benchmark suite — how to set up datasets, run online and offline benchmarks, interpret results, and compare configurations.

!!! tip "Production benchmarking"
    For production vLLM servers, consider [GuideLLM](https://github.com/vllm-project/guidellm), an established performance benchmarking framework with live progress updates and automatic report generation. It is more flexible than `vllm bench serve` in terms of dataset loading, request formatting, and workload patterns.

---

## Prerequisites

Install vLLM and ensure the `vllm` CLI is available:

```bash
pip install vllm
vllm bench --help
```

---

## Dataset Setup

### ShareGPT (Most Common)

The ShareGPT dataset is the standard benchmark dataset for conversational workloads:

```bash
wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

### BurstGPT (Bursty Traffic)

Captures real-world bursty request arrival patterns:

```bash
wget https://github.com/HPMLL/BurstGPT/releases/download/v1.1/BurstGPT_without_fails_2.csv
```

### Sonnet (Deprecated)

A local file included in the repository at `benchmarks/sonnet.txt`. Use `--dataset-name sonnet` with `--dataset-path benchmarks/sonnet.txt`.

### Random (Synthetic)

No download required. Use `--dataset-name random` with `--random-input-len` and `--random-output-len` to control prompt and output lengths.

### HuggingFace Datasets

Many datasets are loaded directly from HuggingFace Hub. Use `--dataset-name hf` with `--dataset-path <hf-dataset-id>`:

```bash
# VisionArena (multimodal)
--dataset-name hf --dataset-path lmarena-ai/VisionArena-Chat

# InstructCoder (code generation)
--dataset-name hf --dataset-path likaixin/InstructCoder

# MT-Bench (multi-turn)
--dataset-name hf --dataset-path philschmid/mt-bench

# AIMO (math reasoning)
--dataset-name hf --dataset-path AI-MO/aimo-validation-aime
```

### Custom Dataset (JSONL)

Bring your own prompts in JSONL format with a `"prompt"` field:

```json
{"prompt": "Explain the theory of relativity in simple terms."}
{"prompt": "Write a Python function to compute Fibonacci numbers."}
{"prompt": "What are the main causes of climate change?"}
```

Use `--dataset-name custom --dataset-path /path/to/data.jsonl`.

### Custom Multimodal Dataset (JSONL)

For multimodal benchmarks, include `"image_files"` alongside `"prompt"`:

```json
{"prompt": "Describe this image.", "image_files": ["/path/to/image.jpg"]}
```

Use `--dataset-name custom_mm --dataset-path /path/to/mm_data.jsonl`.

---

## Online Serving Benchmark (`vllm bench serve`)

The online serving benchmark measures how a live vLLM server handles concurrent requests. It reports TTFT, TPOT, ITL, E2EL, and throughput.

### Step 1: Start the Server

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

### Step 2: Run the Benchmark

```bash
vllm bench serve \
  --backend vllm \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --endpoint /v1/completions \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000
```

### Example Output

```text
============ Serving Benchmark Result ============
Successful requests:                     1000
Benchmark duration (s):                  47.23
Total input tokens:                      218432
Total generated tokens:                  312891
Request throughput (req/s):              21.17
Output token throughput (tok/s):         6625.4
Total token throughput (tok/s):          11248.7
---------------Time to First Token----------------
Mean TTFT (ms):                          71.54
Median TTFT (ms):                        73.88
P99 TTFT (ms):                           79.49
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.91
Median TPOT (ms):                        7.96
P99 TPOT (ms):                           8.03
---------------Inter-token Latency----------------
Mean ITL (ms):                           7.74
Median ITL (ms):                         7.70
P99 ITL (ms):                            8.39
-----------End-to-End Latency (E2EL)-------------
Mean E2EL (ms):                          2481.3
Median E2EL (ms):                        2390.1
P99 E2EL (ms):                           4821.7
==================================================
```

### Key Options

| Option | Default | Description |
|---|---|---|
| `--backend` | `openai` | Backend type: `openai`, `vllm`, `openai-chat`, `tgi`, etc. |
| `--model` | *(from server)* | Model name (auto-detected from `/v1/models` if omitted) |
| `--num-prompts` | `1000` | Number of requests to send |
| `--request-rate` | `inf` | Requests per second (Poisson process); `inf` sends all at once |
| `--burstiness` | `1.0` | Gamma distribution shape for inter-arrival times (1.0 = Poisson) |
| `--max-concurrency` | *(none)* | Cap on concurrent in-flight requests |
| `--input-len` | *(dataset)* | Override input length for synthetic datasets |
| `--output-len` | *(dataset)* | Override output length |
| `--percentile-metrics` | `ttft,tpot,itl` | Metrics to report percentiles for |
| `--save-result` | `false` | Save results to JSON |
| `--result-dir` | `.` | Directory for result JSON files |
| `--num-warmups` | `0` | Number of warmup requests before measurement |
| `--profile` | `false` | Trigger vLLM profiler during benchmark |

### Controlled Request Rate

To simulate a specific arrival rate (e.g., 10 requests/second):

```bash
vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --request-rate 10 \
  --num-prompts 500
```

### Saving Results

```bash
vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --save-result \
  --result-dir ./results/ \
  --metadata version=0.9.0 tp=1
```

Results are saved as `{label}-{request_rate}qps-{model}-{datetime}.json`.

---

## Offline Throughput Benchmark (`vllm bench throughput`)

The offline throughput benchmark measures how many tokens vLLM can process per second when all requests are submitted at once (no rate limiting).

```bash
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000
```

### Example Output

```text
Throughput: 21.34 requests/s, 8421.7 total tokens/s, 5832.1 output tokens/s
Total num prompt tokens:  218432
Total num output tokens:  273891
```

### Comparing Backends

```bash
# vLLM backend (default)
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend vllm \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 1000

# HuggingFace Transformers backend (baseline comparison)
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend hf \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 200 \
  --hf-max-batch-size 8
```

---

## Specialized Benchmark Scripts

### Attention Kernel Benchmarks

The `benchmarks/attention_benchmarks/` suite benchmarks attention backends (Flash, Triton, FlashInfer, MLA) with a concise batch specification grammar:

```bash
cd benchmarks/attention_benchmarks

# Standard attention backends
python benchmark.py \
  --backends flash triton flashinfer \
  --batch-specs "q2k" "8q1s1k" "2q2k_32q1s1k" \
  --output-csv results.csv

# MLA backends (DeepSeek-style)
python benchmark.py \
  --backends cutlass_mla flashinfer_mla flashattn_mla flashmla \
  --batch-specs "64q1s1k" "64q1s4k" \
  --output-csv mla_results.csv
```

**Batch specification grammar:**

| Spec | Meaning |
|---|---|
| `q2k` | 2048-token prefill (q_len=2048, seq_len=2048) |
| `q1s1k` | Decode: 1 new token, 1K sequence length |
| `8q1s1k` | 8 decode requests |
| `q4s1k` | 4-token extend (speculative decode) |
| `2q2k_32q1s1k` | Mixed: 2 prefills + 32 decodes |

Pre-configured YAML benchmarks:

```bash
python benchmark.py --config configs/mla_decode.yaml
python benchmark.py --config configs/mla_mixed_batch.yaml
python benchmark.py --config configs/speculative_decode.yaml
python benchmark.py --config configs/standard_attention.yaml
```

### Disaggregated Prefill Benchmarks

Compare disaggregated prefill against chunked prefill (requires 2 GPUs):

```bash
cd benchmarks/disagg_benchmarks
bash disagg_performance_benchmark.sh
```

### Multi-Turn Conversation Benchmarks

Benchmark KV cache offloading with multi-turn conversations:

```bash
# Start the server
vllm serve meta-llama/Meta-Llama-3.1-8B-Instruct --served-model-name Llama

# Run the multi-turn benchmark
cd benchmarks/multi_turn
python benchmark_serving_multi_turn.py \
  --model /models/meta-llama/Meta-Llama-3.1-8B-Instruct \
  --served-model-name Llama \
  --input-file generate_multi_turn.json \
  --num-clients 2 \
  --max-active-conversations 6
```

### Automated Parameter Tuning

Find optimal `max-num-seqs` and `max-num-batched-tokens` automatically:

```bash
cd benchmarks/auto_tune
MODEL=meta-llama/Llama-3.1-8B-Instruct \
SYSTEM=GPU \
TP=1 \
INPUT_LEN=512 \
OUTPUT_LEN=128 \
MAX_MODEL_LEN=1024 \
NUM_SEQS_LIST="64 128 256" \
NUM_BATCHED_TOKENS_LIST="1024 2048 4096" \
bash auto_tune.sh
```

Results are saved to `$BASE/auto-benchmark/YYYY_MM_DD_HH_MM/result.txt`.

---

## Kernel-Level Benchmarks

Individual CUDA kernel benchmarks are in `benchmarks/kernels/`. These are useful for evaluating specific operations in isolation:

```bash
# Paged attention kernel
python benchmarks/kernels/benchmark_paged_attention.py

# MoE (Mixture of Experts) kernel
python benchmarks/kernels/benchmark_moe.py

# FP8 GEMM
python benchmarks/kernels/benchmark_fp8_gemm.py

# RoPE (Rotary Position Embedding)
python benchmarks/kernels/benchmark_rope.py

# LayerNorm / RMSNorm
python benchmarks/kernels/benchmark_rmsnorm.py

# LoRA kernel
python benchmarks/kernels/benchmark_lora.py
```

### DeepGEMM Benchmark

Compare DeepSeek's DeepGEMM against vLLM's Triton and CUTLASS implementations:

```bash
# Install DeepGEMM first
git clone --recursive https://github.com/deepseek-ai/DeepGEMM
cd DeepGEMM && python setup.py install

# Run the benchmark
python benchmarks/kernels/deepgemm/benchmark_fp8_block_dense_gemm.py
```

---

## Interpreting Results

### What to Look For

**TTFT (Time to First Token)** — Dominated by prefill compute. High TTFT indicates:
- Long input prompts
- Insufficient GPU memory bandwidth
- Chunked prefill not enabled

**TPOT (Time per Output Token)** — Dominated by decode compute. High TPOT indicates:
- Large batch sizes causing memory bandwidth saturation
- Insufficient KV cache memory

**ITL (Inter-Token Latency)** — Similar to TPOT but includes the first token. Useful for streaming applications.

**E2EL (End-to-End Latency)** — Total request latency from client perspective. Includes network overhead.

**Throughput** — Higher is better. Throughput and latency trade off against each other; use `vllm bench sweep serve_workload` to find the Pareto frontier.

### Goodput vs. Throughput

Goodput counts only requests that meet SLA constraints (e.g., TTFT < 500ms). Use `--goodput` to specify SLA thresholds:

```bash
vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --goodput ttft:500 tpot:50
```

---

## Comparing Results

Use the `compare-json-results.py` script from the performance dashboard to compare two benchmark runs:

```bash
python .buildkite/performance-benchmarks/scripts/compare-json-results.py \
  -f results_baseline/benchmark_results.json \
  -f results_new/benchmark_results.json
```

This produces a table showing throughput ratios and latency comparisons across all tested configurations.

---

## Related Pages

- [Latency Benchmarks](latency_benchmarks.md) — Single-batch latency measurement
- [Throughput Benchmarks](throughput_benchmarks.md) — Detailed offline throughput guide
- [Parameter Sweeps](sweeps.md) — Automated multi-configuration sweeps
- [Profiling](profiling.md) — Deep-dive performance analysis
- [Benchmark CLI Reference](cli.md) — Full CLI reference
