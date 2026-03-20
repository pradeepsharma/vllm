---
description: >
  Complete reference for `vllm bench` — benchmarking latency, throughput,
  online serving, startup time, and multimodal processor performance.
---

# `vllm bench` — Benchmarking Reference

`vllm bench` is a suite of benchmarking subcommands for measuring vLLM performance across different dimensions: single-batch latency, offline throughput, online serving throughput, model startup time, multimodal processor latency, and parameter sweeps.

```
vllm bench <bench_type> [options]
```

!!! note "Installation"
    Benchmarking tools require extra dependencies. Install them with:
    ```bash
    pip install vllm[bench]
    ```

!!! tip "Platform detection"
    When running `vllm bench` on a machine without a GPU, vLLM automatically switches to the CPU platform to avoid device-type inference errors.

---

## :compass: Available Subcommands

| Subcommand | Description |
|---|---|
| [`vllm bench latency`](#vllm-bench-latency) | Measure single-batch generation latency |
| [`vllm bench throughput`](#vllm-bench-throughput) | Measure offline inference throughput |
| [`vllm bench serve`](#vllm-bench-serve) | Measure online serving throughput |
| [`vllm bench startup`](#vllm-bench-startup) | Measure model startup time |
| [`vllm bench mm-processor`](#vllm-bench-mm-processor) | Benchmark multimodal processor latency |
| [`vllm bench sweep`](#vllm-bench-sweep) | Parameter sweep across configurations |

---

## `vllm bench latency`

Measures the end-to-end latency of processing a single batch of requests through the vLLM engine. Runs multiple warmup iterations followed by timed benchmark iterations and reports average and percentile latencies.

```
vllm bench latency [options]
```

### Quick Examples

```bash
# Basic latency benchmark
vllm bench latency \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 32 \
    --output-len 128

# Fast test with dummy weights (no download required)
vllm bench latency \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy

# Beam search latency
vllm bench latency \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --input-len 64 \
    --output-len 128 \
    --n 4 \
    --use-beam-search

# Save results to JSON
vllm bench latency \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --output-json latency_results.json
```

### Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--input-len` | `int` | `32` | Number of input (prompt) tokens per request. |
| `--output-len` | `int` | `128` | Number of output tokens to generate per request. |
| `--batch-size` | `int` | `8` | Number of requests in each batch. |
| `--n` | `int` | `1` | Number of generated sequences per prompt. |
| `--use-beam-search` | flag | `false` | Use beam search instead of sampling. |
| `--num-iters-warmup` | `int` | `10` | Number of warmup iterations (not included in measurements). |
| `--num-iters` | `int` | `30` | Number of benchmark iterations to measure. |
| `--profile` | flag | `false` | Profile a single batch using the configured profiler (torch or CUDA). |
| `--output-json` | `str` | — | Path to save results in JSON format. Also generates a PyTorch benchmark format file (`*.pytorch.json`). |
| `--disable-detokenize` | flag | `false` | Skip detokenization (excludes detokenization time from measurements). |

Plus all [engine arguments](#engine-arguments-common) (model, dtype, quantization, parallelism, etc.).

!!! note "Prefix caching"
    Prefix caching is disabled by default for latency benchmarks to avoid skewing results. Override with `--enable-prefix-caching`.

### Output

```
Warming up...
Warmup iterations: 100%|████████████████| 10/10
Bench iterations: 100%|█████████████████| 30/30
Avg latency: 0.4231 seconds
10% percentile latency: 0.4102 seconds
25% percentile latency: 0.4178 seconds
50% percentile latency: 0.4219 seconds
75% percentile latency: 0.4267 seconds
90% percentile latency: 0.4341 seconds
99% percentile latency: 0.4489 seconds
```

---

## `vllm bench throughput`

Measures offline inference throughput by processing a dataset of prompts and computing tokens per second. Supports multiple backends and dataset types.

```
vllm bench throughput [options]
```

### Quick Examples

```bash
# Throughput with ShareGPT dataset
vllm bench throughput \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset-name sharegpt \
    --dataset-path /path/to/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000

# Random dataset (no download required)
vllm bench throughput \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset-name random \
    --input-len 512 \
    --output-len 128 \
    --num-prompts 500

# With dummy weights for quick testing
vllm bench throughput \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset-name random \
    --input-len 32 \
    --output-len 1 \
    --enforce-eager \
    --load-format dummy

# Using async engine
vllm bench throughput \
    --model meta-llama/Llama-3.2-8B-Instruct \
    --dataset-name sharegpt \
    --dataset-path /path/to/ShareGPT.json \
    --async-engine

# Save results to JSON
vllm bench throughput \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --dataset-name random \
    --output-json throughput_results.json
```

### Arguments

#### Backend

| Flag | Type | Default | Description |
|---|---|---|---|
| `--backend` | `str` | `vllm` | Inference backend. Options: `vllm`, `hf` (HuggingFace Transformers), `mii` (DeepSpeed-MII), `vllm-chat`. |
| `--async-engine` | flag | `false` | Use the vLLM async engine instead of the synchronous `LLM` class. |
| `--disable-frontend-multiprocessing` | flag | `false` | Disable the decoupled async engine frontend. |

#### Dataset

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dataset-name` | `str` | `sharegpt` | Dataset to benchmark on. Options: `sharegpt`, `random`, `sonnet`, `burstgpt`, `hf`, `prefix_repetition`, `random-mm`, `random-rerank`. |
| `--dataset` | `str` | — | *(Deprecated)* Path to a ShareGPT JSON file. Use `--dataset-path` instead. |
| `--dataset-path` | `str` | — | Path to the dataset file. |
| `--input-len` | `int` | — | Input prompt length (for `random` dataset). |
| `--output-len` | `int` | — | Output length per request. Overrides dataset output lengths. |
| `--num-prompts` | `int` | `1000` | Number of prompts to process. |
| `--n` | `int` | `1` | Number of generated sequences per prompt. |
| `--hf-subset` | `str` | — | Subset of the HuggingFace dataset. |
| `--hf-split` | `str` | — | Split of the HuggingFace dataset (e.g., `train`, `test`). |
| `--prefix-len` | `int` | `0` | Number of fixed prefix tokens before random context. |

#### HuggingFace Backend

| Flag | Type | Default | Description |
|---|---|---|---|
| `--hf-max-batch-size` | `int` | — | Maximum batch size for the HF backend. |
| `--hf-enable-torch-compile` | flag | `false` | Enable `torch.compile` for the HF backend. |

#### LoRA

| Flag | Type | Default | Description |
|---|---|---|---|
| `--lora-path` | `str` | — | Path to a LoRA adapter (absolute, relative, or HuggingFace model ID). |

#### Output

| Flag | Type | Default | Description |
|---|---|---|---|
| `--output-json` | `str` | — | Path to save throughput results in JSON format. |
| `--disable-detokenize` | flag | `false` | Skip detokenization (excludes it from throughput measurement). |
| `--profile` | flag | `false` | Profile using the configured profiler. |

Plus all [engine arguments](#engine-arguments-common).

### Output

```
Throughput: 1234.56 requests/s, 98765.43 tokens/s
```

---

## `vllm bench serve`

Measures online serving throughput by sending requests to a running `vllm serve` instance and measuring latency metrics including Time to First Token (TTFT), Time Per Output Token (TPOT), Inter-Token Latency (ITL), and end-to-end latency (E2EL).

```
vllm bench serve [options]
```

!!! warning "Requires a running server"
    `vllm bench serve` sends requests to an existing API server. Start one first with `vllm serve`, then run the benchmark in a separate terminal.

### Quick Examples

```bash
# Basic online serving benchmark
vllm bench serve \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --host 127.0.0.1 \
    --port 8000

# With specific request rate and dataset
vllm bench serve \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --host 127.0.0.1 \
    --port 8000 \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 128 \
    --num-prompts 500 \
    --request-rate 10

# Save results to JSON
vllm bench serve \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --save-result \
    --result-dir ./bench_results

# With goodput SLOs
vllm bench serve \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --goodput ttft:200 tpot:50 e2el:5000
```

### Arguments

#### Server Connection

| Flag | Type | Default | Description |
|---|---|---|---|
| `--host` | `str` | `127.0.0.1` | API server hostname. |
| `--port` | `int` | `8000` | API server port. |
| `--base-url` | `str` | — | Full base URL (overrides `--host` and `--port`). |
| `--endpoint` | `str` | `/v1/completions` | API endpoint path. |
| `--backend` | `str` | `openai` | Request backend type. Options: `openai`, `openai-chat`, `tgi`, `vllm`, `lmdeploy`, `deepspeed-mii`, `ray-serve`, `sglang`, `sglang-chat`, `openai-embedding`, `azure-openai`, `vertex-ai`, `sagemaker`, `torchserve-llm`, `friendliai`, `triton`, `openai-audio`. |
| `--header` | `KEY=VALUE` (repeatable) | — | Additional HTTP headers to include with each request. |
| `--model` | `str` | — | Model name. If not specified, fetches the first model from `/v1/models`. |

#### Dataset

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dataset-name` | `str` | — | Dataset to use. Options: `sharegpt`, `random`, `sonnet`, `burstgpt`, `hf`, `prefix_repetition`, `random-mm`, `random-rerank`, `conversation`, `instruct-coder`, `aimo`, `vision-arena`. |
| `--dataset-path` | `str` | — | Path to the dataset file. |
| `--input-len` | `int` | — | General input length (maps to dataset-specific input length args). |
| `--output-len` | `int` | — | General output length (maps to dataset-specific output length args). |
| `--num-prompts` | `int` | — | Number of prompts to send. |
| `--tokenizer` | `str` | — | Tokenizer name or path. |
| `--tokenizer-mode` | `str` | `auto` | Tokenizer mode: `auto`, `hf`, `slow`, `mistral`, `deepseek_v32`, `qwen_vl`. |

#### Request Rate

| Flag | Type | Default | Description |
|---|---|---|---|
| `--request-rate` | `float` | `inf` | Requests per second. `inf` sends all requests at time 0. Otherwise uses Poisson process or gamma distribution. |
| `--burstiness` | `float` | `1.0` | Burstiness factor for request arrival. `1.0` = Poisson. Values < 1 = more bursty; values > 1 = more uniform. |
| `--max-concurrency` | `int` | — | Maximum concurrent requests. Limits actual concurrency even if `--request-rate` is higher. |

#### Sampling Parameters

| Flag | Type | Default | Description |
|---|---|---|---|
| `--top-p` | `float` | — | Top-p sampling parameter. |
| `--top-k` | `int` | — | Top-k sampling parameter. |
| `--min-p` | `float` | — | Min-p sampling parameter. |
| `--temperature` | `float` | — | Temperature for sampling. |
| `--use-beam-search` | flag | `false` | Use beam search. |
| `--logprobs` | `int` | — | Number of logprobs per token to compute and return. |
| `--ignore-eos` | flag | `false` | Set `ignore_eos` flag on requests. |

#### Results & Output

| Flag | Type | Default | Description |
|---|---|---|---|
| `--save-result` | flag | `false` | Save benchmark results to a JSON file. |
| `--save-detailed` | flag | `false` | Include per-request details (response, errors, TTFTs, TPOTs) in saved results. |
| `--append-result` | flag | `false` | Append results to an existing JSON file instead of overwriting. |
| `--result-dir` | `str` | `.` | Directory for saving result JSON files. |
| `--result-filename` | `str` | — | Filename for results. Defaults to `{label}-{request_rate}qps-{model}-{datetime}.json`. |
| `--label` | `str` | `--backend` value | Label prefix for result files. |
| `--metadata` | `KEY=VALUE` (repeatable) | — | Key-value metadata to embed in result JSON for record-keeping. |
| `--disable-tqdm` | flag | `false` | Disable the tqdm progress bar. |
| `--num-warmups` | `int` | `0` | Number of warmup requests (not included in measurements). |
| `--profile` | flag | `false` | Use vLLM profiling (requires `--profiler-config` on the server). |

#### Metrics

| Flag | Type | Default | Description |
|---|---|---|---|
| `--percentile-metrics` | `str` | `ttft,tpot,itl` | Comma-separated metrics for percentile reporting. Options: `ttft`, `tpot`, `itl`, `e2el`. |
| `--metric-percentiles` | `str` | `99` | Comma-separated percentile values to report (e.g., `25,50,75,99`). |
| `--goodput` | `KEY:VALUE` (repeatable) | — | Service level objectives for goodput. Keys: `ttft`, `tpot`, `e2el`. Values in milliseconds. Example: `ttft:200 tpot:50`. |
| `--request-id-prefix` | `str` | `bench-{uuid}-` | Prefix for request IDs. |

### Key Metrics

| Metric | Description |
|---|---|
| **TTFT** | Time to First Token — latency from request submission to first token received |
| **TPOT** | Time Per Output Token — average time between consecutive output tokens |
| **ITL** | Inter-Token Latency — latency between consecutive tokens (similar to TPOT) |
| **E2EL** | End-to-End Latency — total time from request submission to last token |
| **Goodput** | Fraction of requests meeting specified SLO thresholds |

---

## `vllm bench startup`

Measures model startup time — the time from process start to the model being ready to serve requests. Runs both cold (fresh process) and warm (cached weights) startup iterations.

```
vllm bench startup [options]
```

### Quick Examples

```bash
# Basic startup benchmark
vllm bench startup \
    --model meta-llama/Llama-3.2-1B-Instruct

# More iterations for statistical stability
vllm bench startup \
    --model meta-llama/Llama-3.2-8B-Instruct \
    --num-iters-cold 5 \
    --num-iters-warm 5 \
    --output-json startup_results.json
```

### Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--num-iters-cold` | `int` | `3` | Number of cold startup iterations (fresh process, no cached weights). |
| `--num-iters-warmup` | `int` | `1` | Number of warmup iterations before measuring warm startups. |
| `--num-iters-warm` | `int` | `3` | Number of warm startup iterations (weights already cached). |
| `--output-json` | `str` | — | Path to save startup time results in JSON format. |

Plus all [engine arguments](#engine-arguments-common).

!!! note "Process isolation"
    Each startup iteration runs in a separate subprocess with `multiprocessing.set_start_method('spawn')` to ensure complete isolation and accurate cold-start measurements.

---

## `vllm bench mm-processor`

Benchmarks the multimodal processor — the component that converts images, videos, and audio into token embeddings before they reach the language model. Useful for identifying preprocessing bottlenecks in multimodal workloads.

```
vllm bench mm-processor [options]
```

### Quick Examples

```bash
# Benchmark with random multimodal data
vllm bench mm-processor \
    --model llava-hf/llava-1.5-7b-hf \
    --dataset-name random-mm \
    --num-prompts 50

# Benchmark with a HuggingFace dataset
vllm bench mm-processor \
    --model llava-hf/llava-1.5-7b-hf \
    --dataset-name hf \
    --dataset-path yale-nlp/MMVU \
    --hf-subset Science \
    --hf-split test
```

### Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--dataset-name` | `str` | `random-mm` | Dataset to benchmark on. Options: `random-mm`, `hf`. |
| `--num-prompts` | `int` | `10` | Number of prompts to process. |
| `--num-warmups` | `int` | `1` | Number of warmup prompts. |
| `--dataset-path` | `str` | — | Path or HuggingFace dataset name (e.g., `yale-nlp/MMVU`). |
| `--hf-subset` | `str` | — | Subset of the HuggingFace dataset. |
| `--hf-split` | `str` | — | Split of the HuggingFace dataset (e.g., `train`, `test`). |
| `--output-len` | `int` | — | Number of output tokens per request. |

Plus random multimodal dataset arguments and all [engine arguments](#engine-arguments-common).

!!! note "MM processor stats"
    `--enable-mm-processor-stats` is set to `true` by default for this benchmark to collect detailed processor timing information.

---

## `vllm bench sweep`

Runs parameter sweeps across multiple configurations, automatically varying parameters like request rate, concurrency, or model settings and collecting results for each combination. Useful for finding optimal operating points.

```
vllm bench sweep <sweep_type> [options]
```

### Sweep Subcommands

| Subcommand | Description |
|---|---|
| `vllm bench sweep serve` | Sweep online serving parameters |
| `vllm bench sweep serve-workload` | Sweep workload parameters for serving |
| `vllm bench sweep startup` | Sweep startup configurations |
| `vllm bench sweep plot` | Plot sweep results |
| `vllm bench sweep plot-pareto` | Plot Pareto frontier of sweep results |

### Quick Example

```bash
# Sweep request rates for a serving benchmark
vllm bench sweep serve \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --host 127.0.0.1 \
    --port 8000

# Plot the results
vllm bench sweep plot --result-dir ./sweep_results
```

For detailed sweep arguments, run:

```bash
vllm bench sweep serve --help
vllm bench sweep plot --help
```

---

## :gear: Engine Arguments (Common)

All `vllm bench` subcommands that run local inference (i.e., all except `vllm bench serve`) accept the standard vLLM engine arguments. The most commonly used ones are:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | `str` | — | HuggingFace model ID or local path. **Required.** |
| `--tokenizer` | `str` | — | Tokenizer ID or path. |
| `--dtype` | `str` | `auto` | Model weight dtype: `auto`, `half`, `float16`, `bfloat16`, `float32`. |
| `--max-model-len` | `int` | — | Maximum sequence length. |
| `--tensor-parallel-size` | `int` | `1` | Number of GPUs for tensor parallelism. |
| `--pipeline-parallel-size` | `int` | `1` | Number of pipeline stages. |
| `--gpu-memory-utilization` | `float` | `0.90` | Fraction of GPU memory for the KV cache. |
| `--quantization` | `str` | — | Quantization method (e.g., `awq`, `gptq`, `fp8`). |
| `--load-format` | `str` | `auto` | Weight loading format. Use `dummy` for quick tests without downloading weights. |
| `--enforce-eager` | flag | `false` | Disable CUDA graph capture (useful for quick tests). |
| `--enable-prefix-caching` | flag | — | Enable automatic prefix caching. |
| `--trust-remote-code` | flag | `false` | Allow remote code execution from HuggingFace Hub. |
| `--seed` | `int` | `0` | Random seed. |

For the full list of engine arguments, see the [`vllm serve` reference](vllm_serve.md).

---

## :bulb: Tips

### Quick testing without downloading weights

Use `--load-format dummy` to skip weight downloading and test the benchmark pipeline:

```bash
vllm bench latency \
    --model meta-llama/Llama-3.2-70B-Instruct \
    --load-format dummy \
    --enforce-eager \
    --input-len 32 \
    --output-len 1
```

### Comparing configurations

Use `--output-json` to save results from multiple runs, then compare:

```bash
# Run 1: baseline
vllm bench latency --model my-model --output-json baseline.json

# Run 2: with quantization
vllm bench latency --model my-model --quantization awq --output-json quantized.json
```

### Profiling

Enable profiling to capture detailed traces:

```bash
vllm bench latency \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --profile \
    --compilation-config.profiler torch \
    --compilation-config.torch_profiler_dir ./traces
```

---

## :link: Related

- [CLI Overview](index.md) — All vLLM CLI commands
- [Benchmarking guide](../benchmarking/) — Interpreting benchmark results
- [`vllm serve` reference](vllm_serve.md) — Server arguments for `vllm bench serve`
