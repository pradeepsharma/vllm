# LM Evaluation Harness Integration

The [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) (`lm_eval`) is the standard framework for evaluating language model accuracy on a wide range of NLP benchmarks. vLLM integrates with `lm_eval` through two backends:

- **`vllm`** — Runs evaluation directly through the vLLM Python API (offline, no server required)
- **`local-completions`** — Runs evaluation against a live vLLM OpenAI-compatible server

This page explains how to use both backends, how vLLM's CI uses `lm_eval` for accuracy regression testing, and how to write your own evaluation configs.

---

## Installation

```bash
pip install lm-eval
```

For additional task dependencies (e.g., math benchmarks):

```bash
pip install lm-eval[math]
```

---

## Quick Start

### Offline Evaluation (vllm backend)

The `vllm` backend loads the model directly using the vLLM Python API:

```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="vllm",
    model_args="pretrained=meta-llama/Llama-3.1-8B-Instruct,max_model_len=4096",
    tasks="gsm8k",
    batch_size="auto",
)

print(results["results"])
```

### Online Evaluation (local-completions backend)

The `local-completions` backend sends requests to a running vLLM server:

```bash
# Step 1: Start the vLLM server
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 4096

# Step 2: Run evaluation
python -c "
import lm_eval

results = lm_eval.simple_evaluate(
    model='local-completions',
    model_args='model=meta-llama/Llama-3.1-8B-Instruct,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=500,tokenized_requests=False',
    tasks='gsm8k',
)
print(results['results'])
"
```

---

## Command-Line Usage

The `lm_eval` CLI provides a convenient interface for running evaluations:

### Offline (vllm backend)

```bash
lm_eval \
  --model vllm \
  --model_args "pretrained=meta-llama/Llama-3.1-8B-Instruct,max_model_len=4096" \
  --tasks gsm8k \
  --batch_size auto \
  --output_path results/llama-3.1-8b/
```

### Online (local-completions backend)

```bash
# Start server first
vllm serve meta-llama/Llama-3.1-8B-Instruct --max-model-len 4096

# Run evaluation
lm_eval \
  --model local-completions \
  --model_args "model=meta-llama/Llama-3.1-8B-Instruct,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=500,tokenized_requests=False" \
  --tasks gsm8k \
  --output_path results/llama-3.1-8b-server/
```

---

## Supported Tasks

`lm_eval` supports hundreds of tasks. Common ones used with vLLM:

| Task | Description | Metric |
|---|---|---|
| `gsm8k` | Grade school math word problems | `exact_match,strict-match` |
| `mmlu` | Massive Multitask Language Understanding | `acc` |
| `hellaswag` | Commonsense NLI | `acc_norm` |
| `arc_easy` / `arc_challenge` | AI2 Reasoning Challenge | `acc_norm` |
| `winogrande` | Winograd schema challenge | `acc` |
| `truthfulqa_mc1` | TruthfulQA multiple choice | `acc` |
| `humaneval` | Code generation | `pass@1` |
| `math` | MATH benchmark | `exact_match` |

List all available tasks:

```bash
lm_eval --tasks list
```

---

## Model Arguments

### vllm Backend Arguments

Pass engine arguments as comma-separated key=value pairs in `--model_args`:

| Argument | Description | Example |
|---|---|---|
| `pretrained` | Model ID or path | `meta-llama/Llama-3.1-8B-Instruct` |
| `max_model_len` | Maximum sequence length | `4096` |
| `tensor_parallel_size` | Number of GPUs | `4` |
| `dtype` | Model dtype | `auto`, `float16`, `bfloat16` |
| `gpu_memory_utilization` | GPU memory fraction | `0.9` |
| `quantization` | Quantization method | `fp8`, `awq`, `gptq` |
| `kv_cache_dtype` | KV cache dtype | `fp8`, `auto` |
| `add_bos_token` | Add BOS token | `True` |
| `speculative_config` | Speculative decoding config | `{"method":"ngram","num_speculative_tokens":5}` |
| `enable_expert_parallel` | Enable expert parallelism | `True` |
| `max_num_seqs` | Max concurrent sequences | `128` |

Example with multiple arguments:

```bash
lm_eval \
  --model vllm \
  --model_args "pretrained=meta-llama/Llama-3.1-70B-Instruct,max_model_len=4096,tensor_parallel_size=4,dtype=auto,gpu_memory_utilization=0.9" \
  --tasks gsm8k,mmlu \
  --batch_size auto
```

### local-completions Backend Arguments

| Argument | Description |
|---|---|
| `model` | Model name (must match server) |
| `base_url` | Server URL (e.g., `http://127.0.0.1:8000/v1/completions`) |
| `num_concurrent` | Number of concurrent requests |
| `tokenized_requests` | Whether to send pre-tokenized requests |
| `max_retries` | Number of retries on failure |

---

## Accuracy Regression Testing

vLLM uses `lm_eval` in its CI to detect accuracy regressions. Tests are located in `tests/entrypoints/llm/test_accuracy.py` and `tests/entrypoints/openai/correctness/test_lmeval.py`.

### Running Accuracy Tests

```bash
# Run accuracy tests with the V1 engine
pytest tests/entrypoints/llm/test_accuracy.py -v

# Run accuracy tests against the OpenAI-compatible server
pytest tests/entrypoints/openai/correctness/test_lmeval.py -v
```

### How Tests Work

Each test:
1. Runs `lm_eval.simple_evaluate()` with the `vllm` or `local-completions` backend
2. Compares the measured accuracy against an expected value
3. Asserts that the measured value is within a tolerance (`RTOL = 0.03`)

Example test structure:

```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="vllm",
    model_args="pretrained=Qwen/Qwen3-1.7B,max_model_len=4096",
    tasks="gsm8k",
    batch_size="auto",
)

measured_value = results["results"]["gsm8k"]["exact_match,strict-match"]
expected_value = 0.68
assert abs(measured_value - expected_value) < 0.03
```

---

## GSM8K Evaluation Suite

vLLM includes a dedicated GSM8K evaluation suite in `tests/evals/gsm8k/` that provides faster, more controlled evaluation than the full `lm_eval` harness.

### Running GSM8K Evaluations

```bash
# Run with pytest (like CI)
pytest -s -v tests/evals/gsm8k/test_gsm8k_correctness.py \
  --config-list-file=tests/evals/gsm8k/configs/models-small.txt

# Run standalone
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8000
python tests/evals/gsm8k/gsm8k_eval.py --port 8000
```

### GSM8K Config Format

Model configs in `tests/evals/gsm8k/configs/` use YAML format:

```yaml
model_name: "Qwen/Qwen3-0.6B-FP8"
accuracy_threshold: 0.375   # Minimum expected accuracy
num_questions: 1319         # Number of test questions (full set = 1319)
num_fewshot: 5              # Few-shot examples from train set
server_args: "--enforce-eager --max-model-len 4096"
```

For multi-GPU or complex configurations:

```yaml
model_name: "deepseek-ai/DeepSeek-V3.2"
accuracy_threshold: 0.95
num_questions: 1319
num_fewshot: 5
startup_max_wait_seconds: 1200
server_args: >-
  --enforce-eager
  --max-model-len 4096
  --tensor-parallel-size 8
  --enable-expert-parallel
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
```

With environment variables:

```yaml
model_name: "Qwen/Qwen3-1.7B"
accuracy_threshold: 0.68
num_questions: 1319
num_fewshot: 5
server_args: "--max-model-len 4096"
env:
  VLLM_USE_FLASHINFER_MOE_FP4: "1"
```

---

## Evaluating Quantized Models

### FP8 Quantization

```bash
lm_eval \
  --model vllm \
  --model_args "pretrained=neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8,max_model_len=4096,dtype=float16" \
  --tasks gsm8k \
  --batch_size auto
```

### FP8 KV Cache

```bash
lm_eval \
  --model vllm \
  --model_args "pretrained=Qwen/Qwen3-1.7B,max_model_len=4096,kv_cache_dtype=fp8" \
  --tasks gsm8k \
  --batch_size auto
```

### AWQ / GPTQ

```bash
lm_eval \
  --model vllm \
  --model_args "pretrained=TheBloke/Llama-2-7B-Chat-AWQ,max_model_len=4096,quantization=awq" \
  --tasks gsm8k \
  --batch_size auto
```

---

## Evaluating with Speculative Decoding

Speculative decoding should not change accuracy — use `lm_eval` to verify:

```bash
# Baseline (no speculative decoding)
lm_eval \
  --model vllm \
  --model_args "pretrained=meta-llama/Llama-3.1-8B-Instruct,max_model_len=4096" \
  --tasks gsm8k \
  --batch_size auto \
  --output_path results/baseline/

# With speculative decoding (ngram)
lm_eval \
  --model vllm \
  --model_args "pretrained=meta-llama/Llama-3.1-8B-Instruct,max_model_len=4096,speculative_config={\"method\":\"ngram\",\"num_speculative_tokens\":5,\"prompt_lookup_max\":5}" \
  --tasks gsm8k \
  --batch_size auto \
  --output_path results/speculative/
```

---

## Evaluating Multi-GPU Models

```bash
lm_eval \
  --model vllm \
  --model_args "pretrained=meta-llama/Llama-3.1-70B-Instruct,max_model_len=4096,tensor_parallel_size=4,dtype=auto" \
  --tasks gsm8k,mmlu \
  --batch_size auto \
  --output_path results/llama-70b/
```

---

## Evaluating Against a Remote Server

For evaluating a production server or a server on a different machine:

```bash
lm_eval \
  --model local-completions \
  --model_args "model=meta-llama/Llama-3.1-8B-Instruct,base_url=http://your-server:8000/v1/completions,num_concurrent=100,tokenized_requests=False" \
  --tasks gsm8k \
  --output_path results/remote/
```

---

## Chunked Prefill Accuracy Verification

Chunked prefill should not affect accuracy. Verify with:

```bash
# Start server with chunked prefill
vllm serve Qwen/Qwen2-1.5B-Instruct \
  --max-model-len 4096 \
  --enable-chunked-prefill

# Evaluate
lm_eval \
  --model local-completions \
  --model_args "model=Qwen/Qwen2-1.5B-Instruct,base_url=http://127.0.0.1:8000/v1/completions,num_concurrent=500,tokenized_requests=False" \
  --tasks gsm8k
```

---

## Interpreting Results

`lm_eval` returns a nested dictionary:

```python
results = lm_eval.simple_evaluate(...)

# Access task results
task_results = results["results"]["gsm8k"]
# {'exact_match,strict-match': 0.68, 'exact_match,flexible-extract': 0.71, ...}

# Access model configuration
model_config = results["config"]

# Access per-sample results (if --log_samples was set)
samples = results["samples"]
```

### Common Metrics

| Task | Metric Key | Description |
|---|---|---|
| `gsm8k` | `exact_match,strict-match` | Exact string match |
| `mmlu` | `acc` | Accuracy |
| `hellaswag` | `acc_norm` | Length-normalized accuracy |
| `arc_challenge` | `acc_norm` | Length-normalized accuracy |
| `truthfulqa_mc1` | `acc` | Accuracy |

---

## Saving and Comparing Results

```bash
# Save results to JSON
lm_eval \
  --model vllm \
  --model_args "pretrained=meta-llama/Llama-3.1-8B-Instruct,max_model_len=4096" \
  --tasks gsm8k,mmlu \
  --batch_size auto \
  --output_path results/llama-3.1-8b/ \
  --log_samples
```

Results are saved as `results.json` in the output directory. Compare two runs:

```python
import json

with open("results/baseline/results.json") as f:
    baseline = json.load(f)

with open("results/quantized/results.json") as f:
    quantized = json.load(f)

for task in ["gsm8k", "mmlu"]:
    metric = "exact_match,strict-match" if task == "gsm8k" else "acc"
    b = baseline["results"][task][metric]
    q = quantized["results"][task][metric]
    print(f"{task}: baseline={b:.3f}, quantized={q:.3f}, delta={q-b:+.3f}")
```

---

## Related Pages

- [Performance Benchmarks](performance_benchmarks.md) — Throughput and latency benchmarks
- [Profiling](profiling.md) — Deep-dive performance analysis
- [Benchmark CLI Reference](cli.md) — Full `vllm bench` CLI reference
