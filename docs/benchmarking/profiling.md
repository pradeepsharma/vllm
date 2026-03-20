# Profiling with vLLM

vLLM provides built-in profiling support through the PyTorch profiler and CUDA profiler. Profiling helps you understand where time is being spent during inference — whether in attention kernels, GEMM operations, memory transfers, or Python overhead.

---

## Profiler Overview

vLLM supports two profiler backends:

| Profiler | Flag | Output | Best For |
|---|---|---|---|
| **PyTorch** | `--profiler-config.profiler torch` | Chrome trace (`.json.gz`) | Detailed CPU + GPU timeline, kernel-level analysis |
| **CUDA** | `--profiler-config.profiler cuda` | CUDA profiler output | Low-overhead GPU-only profiling |

Both profilers are controlled through the `ProfilerConfig` and can be triggered:
- **During serving** — via HTTP API endpoints (`/start_profile`, `/stop_profile`)
- **During benchmarks** — via `--profile` flag on `vllm bench latency` and `vllm bench throughput`

---

## Configuring the Profiler

### Server-Side Configuration

Pass profiler configuration when starting the vLLM server:

```bash
# PyTorch profiler
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --profiler-config.profiler torch \
  --profiler-config.torch-profiler-dir /tmp/vllm_profiles

# CUDA profiler
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --profiler-config.profiler cuda
```

### ProfilerConfig Options

| Option | Default | Description |
|---|---|---|
| `profiler` | `null` | Profiler backend: `torch` or `cuda` |
| `torch_profiler_dir` | *(required for torch)* | Directory to save trace files |
| `torch_profiler_with_stack` | `false` | Enable Python stack tracing (higher overhead) |
| `torch_profiler_with_flops` | `false` | Enable FLOPS counting |
| `torch_profiler_use_gzip` | `true` | Save traces in gzip format |
| `torch_profiler_dump_cuda_time_total` | `true` | Include total CUDA time in traces |
| `torch_profiler_record_shapes` | `false` | Record tensor shapes |
| `torch_profiler_with_memory` | `false` | Enable memory profiling |
| `delay_iterations` | `0` | Skip N engine iterations before starting profiling |
| `max_iterations` | `0` | Stop after N iterations (0 = no limit) |
| `warmup_iterations` | `0` | PyTorch profiler schedule warmup iterations |
| `active_iterations` | `5` | PyTorch profiler schedule active iterations |
| `wait_iterations` | `0` | PyTorch profiler schedule wait iterations |
| `ignore_frontend` | `false` | Disable frontend (AsyncLLM) profiling |

### Environment Variable

Enable stack tracing via environment variable:

```bash
VLLM_TORCH_PROFILER_WITH_STACK=1 vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --profiler-config.profiler torch \
  --profiler-config.torch-profiler-dir /tmp/profiles
```

---

## Profiling During Serving

### Triggering via HTTP API

Once the server is running with a profiler configured, use the `/start_profile` and `/stop_profile` endpoints:

```bash
# Start profiling
curl -X POST http://localhost:8000/start_profile

# Send some requests (the profiler captures them)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.1-8B-Instruct", "prompt": "Hello", "max_tokens": 100}'

# Stop profiling
curl -X POST http://localhost:8000/stop_profile
```

Trace files are saved to the `torch_profiler_dir` directory.

### Triggering via Python API

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    profiler_config={
        "profiler": "torch",
        "torch_profiler_dir": "/tmp/vllm_profiles",
    }
)

# Start profiling
llm.start_profile()

# Run inference
outputs = llm.generate(
    ["Explain quantum computing in simple terms."],
    SamplingParams(max_tokens=200)
)

# Stop profiling
llm.stop_profile()
```

With a custom trace prefix:

```python
llm.start_profile(profile_prefix="my_experiment")
```

This names trace files as `my_experiment_dp0_pp0_tp0.json.gz`.

---

## Profiling During Benchmarks

### Latency Benchmark with Profiling

```bash
vllm bench latency \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 512 \
  --output-len 128 \
  --batch-size 8 \
  --num-iters-warmup 10 \
  --profile \
  --profiler-config.profiler torch \
  --profiler-config.torch-profiler-dir /tmp/latency_profile
```

When `--profile` is set, the benchmark:
1. Runs `--num-iters-warmup` warmup iterations normally
2. Profiles exactly **one** batch
3. Saves the trace to `torch_profiler_dir`

### Throughput Benchmark with Profiling

```bash
vllm bench throughput \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 100 \
  --profile \
  --profiler-config.profiler torch \
  --profiler-config.torch-profiler-dir /tmp/throughput_profile
```

### Online Serving Benchmark with Profiling

```bash
# Start server with profiler
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --profiler-config.profiler torch \
  --profiler-config.torch-profiler-dir /tmp/serve_profile

# Run benchmark with profiling
vllm bench serve \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 128 \
  --num-prompts 100 \
  --profile
```

---

## Viewing PyTorch Profiler Traces

### TensorBoard

```bash
pip install tensorboard torch-tb-profiler

tensorboard --logdir /tmp/vllm_profiles
```

Open `http://localhost:6006` in your browser and navigate to the **PyTorch Profiler** tab.

### Chrome Trace Viewer

1. Open Chrome and navigate to `chrome://tracing`
2. Click **Load** and select the `.json.gz` trace file
3. Use the timeline view to inspect CPU and GPU activity

### Perfetto

For large traces, [Perfetto](https://ui.perfetto.dev/) provides better performance than Chrome's built-in viewer:

1. Navigate to [ui.perfetto.dev](https://ui.perfetto.dev/)
2. Click **Open trace file** and select the `.json.gz` file

---

## Profiler Schedule

For fine-grained control over which iterations are profiled, use the schedule options:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --profiler-config.profiler torch \
  --profiler-config.torch-profiler-dir /tmp/profiles \
  --profiler-config.delay-iterations 5 \
  --profiler-config.max-iterations 10 \
  --profiler-config.warmup-iterations 2 \
  --profiler-config.active-iterations 5 \
  --profiler-config.wait-iterations 1
```

The schedule works as follows after `/start_profile` is received:

```
Iteration:  1  2  3  4  5  6  7  8  9  10  11  12  13
            |--delay (5)--|  |wait|  |warmup|  |active (5)|
                              off    discard    recording
```

- **delay_iterations**: Skip N iterations before the profiler activates
- **wait_iterations**: Profiler is completely off (zero overhead)
- **warmup_iterations**: Profiler runs but data is discarded (JIT warmup)
- **active_iterations**: Profiler records data

!!! warning
    Using `delay_iterations` or `max_iterations` with `ignore_frontend=False` (the default) may cause high overhead because the frontend profiling does not track iterations. Set `--profiler-config.ignore-frontend true` when using these options.

---

## Layerwise Profiling

vLLM includes a layerwise profiler (`vllm.profiler.layerwise_profile`) that provides a structured breakdown of time spent in each model layer:

```python
from vllm.profiler.layerwise_profile import LayerwiseProfileResults
from torch.profiler import ProfilerActivity, profile

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=True,
) as prof:
    # Run inference
    llm.generate(prompts, sampling_params)

results = LayerwiseProfileResults(prof._kineto_results)

# Print per-layer breakdown
results.print_model_table()

# Print summary by operation type
results.print_summary_table()

# Export to CSV
results.export_model_stats_table_csv("model_stats.csv")
results.export_summary_stats_table_csv("summary_stats.csv")
```

### Model Table Output

```text
name                                                         cpu_time_us  cuda_time_us  pct_cuda_time  trace
|-- LlamaForCausalLM                                         12453.2      98234.1       100.0%         ...
|---- model                                                  11234.5      97123.4       98.9%          ...
|------ embed_tokens                                         234.1        1234.5        1.3%           ...
|------ layers.0                                             1023.4       8234.1        8.4%           ...
|-------- self_attn                                          512.3        4123.4        4.2%           ...
|---------- q_proj                                           123.4        1234.5        1.3%           ...
...
```

### Summary Table Output

```text
name                                                         cuda_time_us  pct_cuda_time  invocations
flash_attn_varlen_func                                       45234.1       46.1%          32
cutlass_gemm_with_blkscaled_epilogue                         23123.4       23.5%          96
rms_norm_kernel                                              8234.1        8.4%           64
...
```

---

## CUDA Profiler

The CUDA profiler (`--profiler-config.profiler cuda`) uses NVIDIA's CUDA profiling API for low-overhead GPU profiling:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --profiler-config.profiler cuda
```

Trigger profiling:

```bash
curl -X POST http://localhost:8000/start_profile
# ... send requests ...
curl -X POST http://localhost:8000/stop_profile
```

Use Nsight Systems to view CUDA profiler output:

```bash
nsys profile --trace=cuda,nvtx python -c "
from vllm import LLM, SamplingParams
llm = LLM('meta-llama/Llama-3.1-8B-Instruct')
llm.generate(['Hello world'], SamplingParams(max_tokens=50))
"
```

---

## Profiling Specific Scenarios

### Profiling Prefill vs. Decode

To profile prefill and decode separately, use `delay_iterations` and `max_iterations`:

```bash
# Profile only the first iteration (prefill-heavy)
vllm bench latency \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 2048 \
  --output-len 1 \
  --batch-size 8 \
  --profile \
  --profiler-config.profiler torch \
  --profiler-config.torch-profiler-dir /tmp/prefill_profile

# Profile decode (long output, short input)
vllm bench latency \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-len 32 \
  --output-len 512 \
  --batch-size 8 \
  --profile \
  --profiler-config.profiler torch \
  --profiler-config.torch-profiler-dir /tmp/decode_profile
```

### Profiling with Memory Tracking

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --profiler-config.profiler torch \
  --profiler-config.torch-profiler-dir /tmp/memory_profile \
  --profiler-config.torch-profiler-with-memory true
```

Memory profiling adds memory allocation/deallocation events to the trace, useful for diagnosing memory fragmentation or unexpected allocations.

### Profiling with Shape Recording

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --profiler-config.profiler torch \
  --profiler-config.torch-profiler-dir /tmp/shapes_profile \
  --profiler-config.torch-profiler-record-shapes true
```

Shape recording adds tensor shapes to each operation, useful for understanding which kernel configurations are being used.

### Profiling Multi-GPU Inference

For tensor-parallel inference, each GPU worker saves its own trace file:

```bash
vllm serve meta-llama/Llama-3.1-70B-Instruct \
  --tensor-parallel-size 4 \
  --profiler-config.profiler torch \
  --profiler-config.torch-profiler-dir /tmp/tp4_profile
```

Trace files are named `{prefix}_dp0_pp0_tp{rank}.json.gz` for each tensor-parallel rank.

---

## Auto-Tune with Profiling

The auto-tune script automatically saves profiler traces for the best-performing configuration:

```bash
cd benchmarks/auto_tune
MODEL=meta-llama/Llama-3.1-8B-Instruct \
SYSTEM=GPU TP=1 \
INPUT_LEN=512 OUTPUT_LEN=128 MAX_MODEL_LEN=1024 \
NUM_SEQS_LIST="64 128 256" \
NUM_BATCHED_TOKENS_LIST="1024 2048 4096" \
bash auto_tune.sh
```

The profiler trace for the best run is saved to `$BASE/auto-benchmark/YYYY_MM_DD_HH_MM/profile/`.

---

## Python-Level Profiling (cProfile)

For profiling Python overhead (scheduling, tokenization, etc.), use the built-in `cprofile` utilities:

```python
from vllm.utils.profiling import cprofile_context

with cprofile_context(save_file="profile.stats"):
    # Run inference
    outputs = llm.generate(prompts, sampling_params)
```

Or as a decorator:

```python
from vllm.utils.profiling import cprofile

@cprofile(save_file="my_function.stats")
def my_function():
    # ...
    pass
```

Analyze the output:

```python
import pstats

stats = pstats.Stats("profile.stats")
stats.sort_stats("cumulative")
stats.print_stats(20)  # Top 20 functions by cumulative time
```

---

## Interpreting Profiler Output

### Key Things to Look For

**GPU utilization** — Is the GPU busy most of the time? Low GPU utilization suggests CPU bottlenecks (scheduling, tokenization, Python overhead).

**Memory bandwidth** — For decode-heavy workloads, memory bandwidth is often the bottleneck. Look for high `GB/s` in attention kernels.

**Kernel efficiency** — Compare actual TFLOPS against theoretical peak. Low efficiency may indicate suboptimal kernel configurations.

**Synchronization points** — Excessive CPU-GPU synchronization can cause bubbles in the GPU timeline.

**Attention vs. GEMM ratio** — For long sequences, attention dominates. For large batch sizes, GEMM dominates.

### Common Bottlenecks

| Symptom | Likely Cause | Solution |
|---|---|---|
| Low GPU utilization | CPU scheduling overhead | Increase batch size, use async engine |
| High TTFT | Slow prefill | Enable chunked prefill, use FlashAttention |
| High TPOT | Memory bandwidth saturation | Reduce batch size, use quantization |
| Memory OOM | KV cache too large | Reduce `max_model_len`, use FP8 KV cache |
| Slow startup | Model loading / compilation | Enable `--load-format safetensors`, warm cache |

---

## Related Pages

- [Latency Benchmarks](latency_benchmarks.md) — Single-batch latency measurement
- [Throughput Benchmarks](throughput_benchmarks.md) — Offline batch throughput
- [Performance Benchmarks](performance_benchmarks.md) — Online serving benchmarks
- [Performance Dashboard](dashboard.md) — Continuous benchmarking CI
