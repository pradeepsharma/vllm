---
description: >
  Complete reference for `vllm run-batch` — offline batch inference using
  the OpenAI Batch API format, with local and remote file support.
---

# `vllm run-batch` — Batch Inference Reference

`vllm run-batch` processes a JSONL batch file offline using the vLLM engine and writes results to an output JSONL file. It implements the [OpenAI Batch API](https://platform.openai.com/docs/guides/batch) format, making it a drop-in replacement for OpenAI's batch processing service.

```
vllm run-batch -i INPUT.jsonl -o OUTPUT.jsonl --model <model> [options]
```

Both `-i` (input) and `-o` (output) are required.

---

## :zap: Quick Examples

```bash
# Process a local batch file
vllm run-batch \
    -i requests.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct

# Fetch input from a URL, write output locally
vllm run-batch \
    -i https://raw.githubusercontent.com/vllm-project/vllm/main/examples/offline_inference/openai_batch/openai_example_batch.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct

# Upload output to a remote URL
vllm run-batch \
    -i requests.jsonl \
    -o https://my-storage.example.com/results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct

# Enable Prometheus metrics during processing
vllm run-batch \
    -i requests.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --enable-metrics \
    --port 9090

# Use tensor parallelism for large models
vllm run-batch \
    -i requests.jsonl \
    -o results.jsonl \
    --model meta-llama/Llama-3.2-70B-Instruct \
    --tensor-parallel-size 4
```

---

## :page_facing_up: Input File Format

The input file is a JSONL file (one JSON object per line) following the OpenAI Batch API format. Each line is a `BatchRequestInput` object:

```json
{
  "custom_id": "request-1",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": {
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is the capital of France?"}
    ],
    "max_tokens": 100
  }
}
```

### Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `custom_id` | `str` | ✅ | Developer-provided ID to match outputs to inputs. Must be unique within the batch. |
| `method` | `str` | ✅ | HTTP method. Currently only `POST` is supported. |
| `url` | `str` | ✅ | API endpoint path. See [Supported Endpoints](#supported-endpoints). |
| `body` | `object` | ✅ | Request body matching the endpoint's schema. |

### Supported Endpoints

| URL | Request Type | Description |
|---|---|---|
| `/v1/chat/completions` | `ChatCompletionRequest` | Chat completions (most common) |
| `/v1/embeddings` | `EmbeddingRequest` | Text embeddings |
| `/v1/audio/transcriptions` | `BatchTranscriptionRequest` | Audio transcription (use `file_url` instead of `file`) |
| `/v1/audio/translations` | `BatchTranslationRequest` | Audio translation (use `file_url` instead of `file`) |
| `/score` | `ScoreRequest` | Relevance scoring |
| `/rerank` | `RerankRequest` | Document reranking |

!!! note "Audio batch requests"
    For `/v1/audio/transcriptions` and `/v1/audio/translations`, use `file_url` (a URL or base64 data URL) instead of the `file` field, since batch requests cannot include multipart file uploads.

### Example: Mixed batch file

```jsonl
{"custom_id": "chat-1", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 50}}
{"custom_id": "embed-1", "method": "POST", "url": "/v1/embeddings", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "input": "The quick brown fox"}}
{"custom_id": "chat-2", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Meta-Llama-3-8B-Instruct", "messages": [{"role": "user", "content": "What is 2+2?"}], "max_tokens": 20}}
```

---

## :page_facing_up: Output File Format

The output file is a JSONL file with one `BatchRequestOutput` object per input line:

```json
{
  "id": "vllm-abc123",
  "custom_id": "request-1",
  "response": {
    "status_code": 200,
    "request_id": "req-xyz789",
    "body": {
      "id": "chatcmpl-...",
      "object": "chat.completion",
      "choices": [
        {
          "index": 0,
          "message": {"role": "assistant", "content": "Paris."},
          "finish_reason": "stop"
        }
      ],
      "usage": {"prompt_tokens": 25, "completion_tokens": 2, "total_tokens": 27}
    }
  },
  "error": null
}
```

### Output Fields

| Field | Type | Description |
|---|---|---|
| `id` | `str` | vLLM-generated unique ID for this output record. |
| `custom_id` | `str` | Echoes the `custom_id` from the input, for matching. |
| `response` | `object` or `null` | The HTTP response. `null` if a non-HTTP error occurred. |
| `response.status_code` | `int` | HTTP status code (200 for success). |
| `response.request_id` | `str` | Unique request ID. |
| `response.body` | `object` | The full API response body. |
| `error` | `object` or `null` | Error details for non-HTTP failures. `null` on success. |

---

## :clipboard: Arguments

### Batch-Specific Arguments

| Flag | Short | Type | Default | Required | Description |
|---|---|---|---|---|---|
| `--input-file` | `-i` | `str` | — | ✅ | Path or URL to the input JSONL file. Local paths and `http://`/`https://` URLs are supported. Remote files are fetched via HTTP GET. |
| `--output-file` | `-o` | `str` | — | ✅ | Path or URL for the output JSONL file. Local paths and `http://`/`https://` URLs are supported. Remote files are uploaded via HTTP PUT with automatic retry (up to 5 attempts). |
| `--output-tmp-dir` | — | `str` | — | — | Temporary directory for staging the output file before uploading to a remote URL. |
| `--enable-metrics` | — | flag | `false` | — | Start a Prometheus metrics HTTP server during batch processing. |
| `--host` | — | `str` | — | — | Hostname for the Prometheus metrics server (only used when `--enable-metrics` is set). |
| `--port` | — | `int` | `8000` | — | Port for the Prometheus metrics server (only used when `--enable-metrics` is set). |

### Frontend Arguments (inherited from `BaseFrontendArgs`)

These arguments are shared with `vllm serve` and control chat templates, tool calling, and logging behavior.

| Flag | Type | Default | Description |
|---|---|---|---|
| `--lora-modules` | `str` (repeatable) | — | LoRA module configurations (`name=path` or JSON format). |
| `--chat-template` | `str` | — | Path to a Jinja2 chat template file or inline template string. |
| `--chat-template-content-format` | `str` | `auto` | How to render message content: `string` or `openai`. |
| `--response-role` | `str` | `assistant` | Role name returned when `add_generation_prompt=true`. |
| `--return-tokens-as-token-ids` | flag | `false` | Represent tokens as `token_id:{id}` strings in logprobs. |
| `--enable-auto-tool-choice` | flag | `false` | Enable automatic tool choice. Requires `--tool-call-parser`. |
| `--tool-call-parser` | `str` | — | Tool call parser name. |
| `--tool-parser-plugin` | `str` | — | Import path of a custom tool parser plugin. |
| `--log-config-file` | `str` | `$VLLM_LOGGING_CONFIG_PATH` | Path to a JSON logging config file. |
| `--max-log-len` | `int` | `None` | Maximum prompt characters printed in log messages. |

### Engine Arguments

`vllm run-batch` accepts all the same engine arguments as `vllm serve`. The most commonly used ones are:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--model` | `str` | — | HuggingFace model ID or local path. **Required** (no default for batch mode). |
| `--tokenizer` | `str` | — | Tokenizer ID or path (defaults to model's tokenizer). |
| `--dtype` | `str` | `auto` | Model weight dtype: `auto`, `half`, `float16`, `bfloat16`, `float32`. |
| `--max-model-len` | `int` | — | Maximum sequence length. |
| `--tensor-parallel-size` | `int` | `1` | Number of GPUs for tensor parallelism. |
| `--pipeline-parallel-size` | `int` | `1` | Number of pipeline stages. |
| `--gpu-memory-utilization` | `float` | `0.90` | Fraction of GPU memory for the KV cache. |
| `--quantization` | `str` | — | Quantization method (e.g., `awq`, `gptq`, `fp8`). |
| `--enable-lora` | flag | `false` | Enable LoRA adapter support. |
| `--load-format` | `str` | `auto` | Weight loading format. |
| `--trust-remote-code` | flag | `false` | Allow remote code execution from HuggingFace Hub. |
| `--enforce-eager` | flag | `false` | Disable CUDA graph capture. |
| `--seed` | `int` | `0` | Random seed. |

For the full list of engine arguments, see the [`vllm serve` reference](vllm_serve.md).

---

## :arrows_counterclockwise: File Transport

### Input

- **Local file**: Opened directly with UTF-8 encoding.
- **HTTP/HTTPS URL**: Fetched via `aiohttp` HTTP GET. The server must return the file content directly.

### Output

- **Local file**: Written directly with UTF-8 encoding.
- **HTTP/HTTPS URL**: Uploaded via `aiohttp` HTTP PUT with a 1000-second timeout. Automatically retries up to 5 times with a 5-second delay between attempts.
- **`--output-tmp-dir`**: When writing to a remote URL, the output is first written to a temporary file in this directory, then uploaded. Useful for large batches where you want to avoid holding everything in memory.

---

## :bar_chart: Prometheus Metrics

When `--enable-metrics` is set, a Prometheus metrics HTTP server is started at `http://<host>:<port>/metrics`. This exposes the same engine metrics as `vllm serve`, including:

- Token throughput (prompt and generation tokens per second)
- Request latency percentiles
- KV cache utilization
- Queue depth

```bash
vllm run-batch \
    -i requests.jsonl \
    -o results.jsonl \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --enable-metrics \
    --host 0.0.0.0 \
    --port 9090
```

Then scrape metrics at `http://localhost:9090/metrics`.

---

## :bulb: Tips and Best Practices

### Batch file preparation

Ensure each line in your input JSONL is a valid, complete JSON object. Empty lines are skipped automatically. Use `custom_id` values that are meaningful to your application — they are echoed verbatim in the output for matching.

### Handling errors

Failed requests appear in the output with a non-200 `response.status_code` or a non-null `error` field. Process the output file and filter by `response.status_code != 200` or `error != null` to identify failures.

```python
import json

with open("results.jsonl") as f:
    for line in f:
        result = json.loads(line)
        if result["error"] or result["response"]["status_code"] != 200:
            print(f"Failed: {result['custom_id']}: {result['error']}")
```

### Large batches

For very large batches, use `--output-tmp-dir` when writing to a remote URL to avoid memory pressure. The output is staged locally first, then uploaded.

### Unsupported endpoints

If a request URL is not recognized, the output will contain an error message listing the supported endpoints. Check the `error` field in the output for details.

---

## :link: Related

- [CLI Overview](index.md) — All vLLM CLI commands
- [OpenAI Batch API documentation](https://platform.openai.com/docs/guides/batch) — OpenAI's batch format specification
- [`vllm serve` reference](vllm_serve.md) — Full engine argument reference
- [Offline inference examples](../examples/) — Python API examples for batch processing
