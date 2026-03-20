# Throughput Benchmarks

The `vllm bench throughput` command measures **offline batch inference throughput** — how many tokens vLLM can process per second when all requests are submitted simultaneously, without any rate limiting or concurrency constraints.

This is the standard benchmark for comparing vLLM against other inference frameworks and for evaluating the impact of hardware, quantization, and engine configuration on raw throughput.

---

## When to Use Throughput Benchmarks

Use `vllm bench throughput` when you want to:

- Measure peak tokens/second for a given model and hardware configuration
- Compare vLLM against HuggingFace Transformers or other backends
- Evaluate the throughput impact of quantization, LoRA, or tensor parallelism
- Establish a throughput baseline before tuning serving parameters
- Run CI regression tests for throughput

For measuring throughput under realistic concurrent load with rate-limited requests, use [`vllm bench serve`](performance_benchmarks.md) instead.

---

## Quick Start

```bash
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 1000
```

### Example Output

```text
Throughput: 21.34 requests/s, 8421.7 total tokens/s, 5832.1 output tokens/s
Total num prompt tokens:  512000
Total num output tokens:  128000
```

---

## CLI Reference

```bash
vllm bench throughput --help
```

### Core Options

| Option | Default | Description |
|---|---|---|
| `--model` | *(required)* | HuggingFace model ID or local path |
| `--backend` | `vllm` | Inference backend: `vllm`, `hf`, `mii` |
| `--dataset-name` | `sharegpt` | Dataset to use for sampling requests |
| `--num-prompts` | `1000` | Number of prompts to process |
| `--input-len` | *(dataset)* | Override input length (maps to dataset-specific args) |
| `--output-len` | *(dataset)* | Override output length |
| `--n` | `1` | Number of output sequences per prompt |

### Dataset Options

| Option | Description |
|---|---|
| `--dataset-path` | Path to dataset file (ShareGPT JSON, CSV, etc.) |
| `--random-input-len` | Input length for random dataset |
| `--random-output-len` | Output length for random dataset |
| `--hf-subset` | HuggingFace dataset subset |
| `--hf-split` | HuggingFace dataset split (e.g., `train`, `test`) |

### Engine Options

| Option | Default | Description |
|---|---|---|
| `--tensor-parallel-size` | `1` | Number of GPUs for tensor parallelism |
| `--dtype` | `auto` | Model dtype: `auto`, `float16`, `bfloat16`, `float32` |
| `--quantization` | *(none)* | Quantization method: `fp8`, `awq`, `gptq`, etc. |
| `--max-model-len` | *(model config)* | Maximum sequence length |
| `--gpu-memory-utilization` | `0.9` | Fraction of GPU memory to use for KV cache |
| `--enable-prefix-caching` | `false` | Enable automatic prefix caching |

### Output Options

| Option | Default | Description |
|---|---|---|
| `--output-json` | *(none)* | Path to save results as JSON |
| `--disable-detokenize` | `false` | Exclude detokenization time from measurement |
| `--profile` | `false` | Enable vLLM profiler during benchmark |

### Advanced Options

| Option | Default | Description |
|---|---|---|
| `--async-engine` | `false` | Use the async engine instead of the synchronous LLM class |
| `--disable-frontend-multiprocessing` | `false` | Disable decoupled async engine frontend |
| `--hf-max-batch-size` | *(none)* | Maximum batch size for HF backend |
| `--hf-enable-torch-compile` | `false` | Enable `torch.compile` for HF backend |
| `--lora-path` | *(none)* | Path to LoRA adapter |
| `--prefix-len` | `0` | Fixed prefix tokens before random context |

---

## Common Benchmark Scenarios

### Standard Throughput (ShareGPT)

```bash
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000
```

### Synthetic Random Workload

```bash
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 512 \
  --num-prompts 1000
```

### Long-Context Throughput

```bash
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 8192 \
  --random-output-len 256 \
  --num-prompts 200 \
  --max-model-len 16384
```

### Multi-GPU Throughput

```bash
vllm bench throughput \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 1000 \
  --tensor-parallel-size 4
```

### FP8 Quantized Model

```bash
vllm bench throughput \
  --model neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8 \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 1000 \
  --dtype float16
```

### LoRA Throughput

```bash
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 1000 \
  --enable-lora \
  --lora-path /path/to/lora/adapter
```

### Prefix Caching Throughput

Measure throughput improvement from prefix caching with repeated prefixes:

```bash
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 1000 \
  --prefix-len 256 \
  --enable-prefix-caching
```

### Async Engine Throughput

Use the async engine for higher concurrency:

```bash
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 1000 \
  --async-engine
```

---

## Comparing Against HuggingFace Transformers

The `hf` backend runs the same requests through HuggingFace Transformers for a direct comparison:

```bash
# vLLM
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend vllm \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 500 \
  --output-json vllm_results.json

# HuggingFace Transformers
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --backend hf \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 200 \
  --hf-max-batch-size 8 \
  --output-json hf_results.json
```

!!! note
    The HF backend processes requests in mini-batches (`--hf-max-batch-size`) and does not support tensor parallelism. Use a smaller `--num-prompts` to keep runtime reasonable.

---

## Saving Results to JSON

```bash
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 1000 \
  --output-json throughput_results.json
```

The JSON output contains:

```json
{
    "elapsed_time": 46.87,
    "num_requests": 1000,
    "total_num_tokens": 640000,
    "requests_per_second": 21.34,
    "tokens_per_second": 13653.2
}
```

A companion `throughput_results.pytorch.json` is also generated in PyTorch benchmark format.

---

## Profiling During Throughput Benchmark

Enable the vLLM profiler to capture a trace during the throughput run:

```bash
# Start the server with profiler config
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --profiler-config.profiler torch \
  --profiler-config.torch-profiler-dir /tmp/throughput_profile

# Run throughput benchmark with profiling
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 100 \
  --profile
```

See [Profiling](profiling.md) for details on interpreting profiler output.

---

## Multimodal Throughput

### Image + Text (VLM)

```bash
vllm bench throughput \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset-name hf \
  --dataset-path lmarena-ai/VisionArena-Chat \
  --hf-split train \
  --num-prompts 200
```

### Random Multimodal (Synthetic)

```bash
vllm bench throughput \
  --model Qwen/Qwen2-VL-7B-Instruct \
  --dataset-name random-mm \
  --random-input-len 256 \
  --random-output-len 128 \
  --num-prompts 200
```

!!! warning
    For multimodal models, use the `vllm-chat` backend with `vllm bench serve` for accurate token counting. The throughput benchmark may undercount image tokens.

---

## Throughput Metrics Explained

| Metric | Formula | Interpretation |
|---|---|---|
| **Requests/s** | `num_requests / elapsed_time` | How many complete requests are processed per second |
| **Total tokens/s** | `(prompt_tokens + output_tokens) / elapsed_time` | Total token processing rate (input + output) |
| **Output tokens/s** | `output_tokens / elapsed_time` | Generation rate (most relevant for decode-heavy workloads) |

**Which metric to use?**

- **Output tokens/s** is the most commonly cited metric for generation throughput
- **Total tokens/s** is useful when comparing prefill-heavy vs. decode-heavy workloads
- **Requests/s** is useful for fixed-length workloads

---

## Throughput Optimization Tips

### 1. Increase Batch Size

vLLM uses continuous batching, so throughput scales with the number of concurrent requests. Increase `--max-num-seqs` on the server:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --max-num-seqs 256 \
  --max-num-batched-tokens 8192
```

### 2. Enable Prefix Caching

For workloads with repeated system prompts or shared prefixes:

```bash
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching \
  --prefix-len 512 \
  --dataset-name random \
  --random-input-len 1024 \
  --random-output-len 128 \
  --num-prompts 1000
```

### 3. Use Chunked Prefill

Chunked prefill improves throughput for mixed prefill/decode batches:

```bash
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --enable-chunked-prefill \
  --max-num-batched-tokens 2048 \
  --dataset-name sharegpt \
  --dataset-path ShareGPT_V3_unfiltered_cleaned_split.json \
  --num-prompts 1000
```

### 4. Tune GPU Memory Utilization

Higher memory utilization allows more KV cache, enabling larger batches:

```bash
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --gpu-memory-utilization 0.95 \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 1000
```

### 5. Automated Parameter Tuning

Use the auto-tune script to find optimal `max-num-seqs` and `max-num-batched-tokens`:

```bash
cd benchmarks/auto_tune
MODEL=meta-llama/Llama-3.1-8B-Instruct \
SYSTEM=GPU TP=1 \
INPUT_LEN=512 OUTPUT_LEN=128 MAX_MODEL_LEN=1024 \
NUM_SEQS_LIST="64 128 256" \
NUM_BATCHED_TOKENS_LIST="1024 2048 4096" \
bash auto_tune.sh
```

---

## Disaggregated Prefill Throughput

Compare disaggregated prefill against chunked prefill for long-context workloads:

```bash
cd benchmarks/disagg_benchmarks
bash disagg_performance_benchmark.sh
```

This script:
1. Launches two vLLM instances with chunked prefill (baseline)
2. Launches disaggregated prefill (1 prefill instance + 1 decode instance)
3. Runs `vllm bench serve` against both configurations
4. Reports throughput comparison

---

## Related Pages

- [Latency Benchmarks](latency_benchmarks.md) — Single-batch latency measurement
- [Performance Benchmarks](performance_benchmarks.md) — Online serving benchmarks
- [Parameter Sweeps](sweeps.md) — Automated multi-configuration sweeps
- [Profiling](profiling.md) — Deep-dive profiler integration
- [Benchmark CLI Reference](cli.md) — Full CLI reference
