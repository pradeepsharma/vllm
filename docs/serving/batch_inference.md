# Batch Inference

vLLM's batch runner processes large collections of requests offline from a JSONL file, without running an HTTP server. This is ideal for bulk processing tasks where you have many requests to process and don't need real-time responses.

---

## Overview

The batch runner reads requests from a JSONL input file, processes them using the vLLM engine, and writes results to a JSONL output file. Each line in the input file is an independent request; each line in the output file is the corresponding response.

**Supported endpoints:**
- `/v1/chat/completions` — Chat completions
- `/v1/embeddings` — Text embeddings
- `/v1/audio/transcriptions` — Audio transcription
- `/v1/audio/translations` — Audio translation
- `/score` — Cross-encoder scoring
- `/rerank` — Document re-ranking

---

## Quick Start

### Prepare Input File

Create a JSONL file where each line is a batch request:

```jsonl
{"custom_id": "req-001", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "What is the capital of France?"}], "max_tokens": 100}}
{"custom_id": "req-002", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "What is the capital of Germany?"}], "max_tokens": 100}}
{"custom_id": "req-003", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "meta-llama/Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "What is the capital of Japan?"}], "max_tokens": 100}}
```

### Run the Batch

```bash
python -m vllm.entrypoints.openai.run_batch \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-file input.jsonl \
  --output-file output.jsonl
```

Or using the CLI:

```bash
vllm run-batch \
  --model meta-llama/Llama-3.1-8B-Instruct \
  -i input.jsonl \
  -o output.jsonl
```

### Read Output

```python
import json

with open("output.jsonl") as f:
    for line in f:
        result = json.loads(line)
        print(f"ID: {result['custom_id']}")
        if result["response"]:
            body = result["response"]["body"]
            print(f"Answer: {body['choices'][0]['message']['content']}")
        elif result["error"]:
            print(f"Error: {result['error']}")
        print()
```

---

## Input Format

Each line in the input JSONL file must be a `BatchRequestInput` object:

```json
{
  "custom_id": "unique-request-id",
  "method": "POST",
  "url": "/v1/chat/completions",
  "body": { ... }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `custom_id` | `string` | Developer-provided ID to match outputs to inputs |
| `method` | `string` | HTTP method (currently only `"POST"`) |
| `url` | `string` | API endpoint URL |
| `body` | `object` | Request body (same as the HTTP endpoint) |

### URL Routing

The `url` field determines which handler processes the request:

| URL | Handler |
|-----|---------|
| `/v1/chat/completions` | Chat completions |
| `/v1/embeddings` | Embeddings |
| `/v1/audio/transcriptions` | Audio transcription |
| `/v1/audio/translations` | Audio translation |
| `*/score` | Cross-encoder scoring |
| `*/rerank` | Document re-ranking |

---

## Output Format

Each line in the output JSONL file is a `BatchRequestOutput` object:

```json
{
  "id": "batch-output-abc123",
  "custom_id": "req-001",
  "response": {
    "status_code": 200,
    "request_id": "internal-req-id",
    "body": { ... }
  },
  "error": null
}
```

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Unique output identifier |
| `custom_id` | `string` | Matches the input `custom_id` |
| `response` | `object \| null` | Response data (null on error) |
| `response.status_code` | `int` | HTTP status code |
| `response.body` | `object` | Response body |
| `error` | `any \| null` | Error information (null on success) |

---

## Examples by Endpoint

### Chat Completions

```jsonl
{"custom_id": "chat-001", "method": "POST", "url": "/v1/chat/completions", "body": {
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Summarize: The quick brown fox jumps over the lazy dog."}
  ],
  "max_tokens": 50,
  "temperature": 0.0
}}
```

### Embeddings

```jsonl
{"custom_id": "emb-001", "method": "POST", "url": "/v1/embeddings", "body": {
  "model": "BAAI/bge-base-en-v1.5",
  "input": "The quick brown fox jumps over the lazy dog."
}}
{"custom_id": "emb-002", "method": "POST", "url": "/v1/embeddings", "body": {
  "model": "BAAI/bge-base-en-v1.5",
  "input": ["Text one", "Text two", "Text three"]
}}
```

### Audio Transcription

For audio, use `file_url` instead of `file` (which requires a file upload):

```jsonl
{"custom_id": "audio-001", "method": "POST", "url": "/v1/audio/transcriptions", "body": {
  "model": "openai/whisper-large-v3-turbo",
  "file_url": "https://example.com/audio1.mp3",
  "language": "en",
  "response_format": "json"
}}
{"custom_id": "audio-002", "method": "POST", "url": "/v1/audio/transcriptions", "body": {
  "model": "openai/whisper-large-v3-turbo",
  "file_url": "data:audio/wav;base64,<base64-encoded-audio>",
  "language": "fr"
}}
```

The `file_url` field accepts:
- HTTP/HTTPS URLs: `https://example.com/audio.mp3`
- Base64 data URLs: `data:audio/wav;base64,<data>`

### Audio Translation

```jsonl
{"custom_id": "trans-001", "method": "POST", "url": "/v1/audio/translations", "body": {
  "model": "openai/whisper-large-v3",
  "file_url": "https://example.com/german_audio.mp3"
}}
```

### Scoring

```jsonl
{"custom_id": "score-001", "method": "POST", "url": "/v1/score", "body": {
  "model": "BAAI/bge-reranker-v2-m3",
  "queries": "What is the capital of France?",
  "documents": ["Paris is the capital of France.", "Berlin is the capital of Germany."]
}}
```

### Re-ranking

```jsonl
{"custom_id": "rerank-001", "method": "POST", "url": "/v1/rerank", "body": {
  "model": "BAAI/bge-reranker-v2-m3",
  "query": "What is the capital of France?",
  "documents": [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Rome is the capital of Italy."
  ],
  "top_n": 2
}}
```

---

## CLI Arguments

```bash
python -m vllm.entrypoints.openai.run_batch [options]
```

### Required Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--input-file` | `-i` | Path or URL to input JSONL file |
| `--output-file` | `-o` | Path or URL to output JSONL file |
| `--model` | | Model to use for inference |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-tmp-dir` | `null` | Temp directory for output before upload |
| `--enable-metrics` | `false` | Enable Prometheus metrics server |
| `--host` | `null` | Metrics server host |
| `--port` | `8000` | Metrics server port |

All [engine arguments](../configuration/engine_args.md) are also supported (e.g., `--tensor-parallel-size`, `--dtype`, `--max-model-len`).

### Frontend Arguments

All [frontend arguments](openai_compatible_server.md#configuration) from `BaseFrontendArgs` are supported, including:

| Argument | Description |
|----------|-------------|
| `--chat-template` | Custom chat template |
| `--enable-auto-tool-choice` | Enable tool calling |
| `--tool-call-parser` | Tool call parser |
| `--max-log-len` | Max log length |

---

## Remote Files

The batch runner supports reading input from and writing output to remote URLs:

```bash
# Read from HTTP URL
python -m vllm.entrypoints.openai.run_batch \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-file https://example.com/batch_input.jsonl \
  --output-file https://example.com/batch_output.jsonl \
  --output-tmp-dir /tmp/vllm-batch
```

- **Input**: HTTP GET request to download the file
- **Output**: HTTP PUT request to upload the file
- `--output-tmp-dir`: Required when output is a URL (stores locally before uploading)

---

## Prometheus Metrics

Enable metrics to monitor batch progress:

```bash
python -m vllm.entrypoints.openai.run_batch \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --input-file input.jsonl \
  --output-file output.jsonl \
  --enable-metrics \
  --host 0.0.0.0 \
  --port 8001
```

Access metrics at `http://localhost:8001/metrics`.

---

## Progress Tracking

The batch runner displays a progress bar:

```
Processing: 67% Completed | 670/1000 [02:15<01:05, 4.97 req/s]
```

---

## Error Handling

Failed requests are recorded in the output file with `error` set and `response` as `null`:

```json
{
  "id": "batch-output-xyz",
  "custom_id": "req-failed",
  "response": null,
  "error": {
    "message": "URL /v1/unknown was used. Supported endpoints: /v1/chat/completions, ..."
  }
}
```

Process errors in your output reader:

```python
import json

successes = []
failures = []

with open("output.jsonl") as f:
    for line in f:
        result = json.loads(line)
        if result["error"]:
            failures.append(result)
        elif result["response"]["status_code"] == 200:
            successes.append(result)
        else:
            failures.append(result)

print(f"Successes: {len(successes)}, Failures: {len(failures)}")
```

---

## Performance Tips

1. **Large batches**: The batch runner processes all requests concurrently using asyncio — larger batches amortize model loading overhead.
2. **Tensor parallelism**: Use `--tensor-parallel-size` for large models to fit them in GPU memory.
3. **Max model length**: Set `--max-model-len` to limit context length and increase throughput.
4. **Disable logging**: Use `--disable-log-requests` to reduce I/O overhead for large batches.
5. **Chunked prefill**: Enable `--enable-chunked-prefill` for better memory efficiency with long prompts.

---

## Comparison: Batch vs. Online Serving

| Feature | Batch Runner | Online Server |
|---------|-------------|---------------|
| Interface | JSONL file | HTTP API |
| Latency | Higher (offline) | Lower (real-time) |
| Throughput | Maximum | Balanced |
| Streaming | Not supported | Supported |
| Concurrency | All requests at once | Continuous |
| Use case | Bulk processing | Interactive apps |
| Metrics | Optional Prometheus | Built-in |

---

## See Also

- [Chat Completions](chat_completions.md) — Chat request format
- [Embeddings](embeddings.md) — Embedding request format
- [Speech-to-Text](speech_to_text.md) — Audio transcription format
- [OpenAI-Compatible Server](openai_compatible_server.md) — Online serving
