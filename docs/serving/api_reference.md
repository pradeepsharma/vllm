# API Reference

Complete reference for all vLLM HTTP and WebSocket endpoints. vLLM implements the OpenAI API specification with extensions for additional capabilities.

---

## Base URL

All HTTP endpoints are served under the configured host and port (default `http://localhost:8000`). OpenAI-compatible endpoints use the `/v1` prefix.

---

## Authentication

When `--api-key` is set (or the `VLLM_API_KEY` environment variable is configured), all `/v1/*` endpoints require a Bearer token:

```
Authorization: Bearer <your-api-key>
```

Health and utility endpoints (`/health`, `/ping`) do not require authentication.

---

## Common Response Formats

### Error Response

All endpoints return a consistent error envelope on failure:

```json
{
  "object": "error",
  "message": "Human-readable error description",
  "type": "invalid_request_error",
  "param": null,
  "code": 400
}
```

| Field | Type | Description |
|-------|------|-------------|
| `object` | `string` | Always `"error"` |
| `message` | `string` | Human-readable description |
| `type` | `string` | Error category |
| `param` | `string \| null` | Parameter that caused the error |
| `code` | `integer` | HTTP status code |

### Usage Info

Most generation endpoints include token usage in responses:

```json
{
  "prompt_tokens": 42,
  "completion_tokens": 128,
  "total_tokens": 170,
  "prompt_tokens_details": null
}
```

---

## Generation Endpoints

### POST /v1/chat/completions

Create a chat completion. Compatible with the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat).

**Request Body**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `messages` | `array` | **required** | List of messages in the conversation |
| `model` | `string \| null` | `null` | Model identifier (uses server default if omitted) |
| `frequency_penalty` | `float \| null` | `0.0` | Penalize repeated tokens by frequency |
| `logit_bias` | `object \| null` | `null` | Token ID → bias value map |
| `logprobs` | `bool \| null` | `false` | Return log probabilities |
| `top_logprobs` | `int \| null` | `0` | Number of top log probs to return per token |
| `max_tokens` | `int \| null` | `null` | *(Deprecated)* Use `max_completion_tokens` |
| `max_completion_tokens` | `int \| null` | `null` | Maximum tokens to generate |
| `n` | `int \| null` | `1` | Number of completions to generate |
| `presence_penalty` | `float \| null` | `0.0` | Penalize tokens already present |
| `response_format` | `object \| null` | `null` | Output format (`text`, `json_object`, `json_schema`) |
| `seed` | `int \| null` | `null` | Random seed for reproducibility |
| `stop` | `string \| array \| null` | `[]` | Stop sequences |
| `stream` | `bool \| null` | `false` | Enable streaming via SSE |
| `stream_options` | `object \| null` | `null` | Streaming options (e.g., `include_usage`) |
| `temperature` | `float \| null` | `null` | Sampling temperature (0–2) |
| `top_p` | `float \| null` | `null` | Nucleus sampling probability |
| `tools` | `array \| null` | `null` | Tool definitions for function calling |
| `tool_choice` | `string \| object` | `"none"` | Tool selection strategy |
| `reasoning_effort` | `string \| null` | `null` | Reasoning effort: `"low"`, `"medium"`, `"high"` |
| `parallel_tool_calls` | `bool \| null` | `true` | Allow multiple tool calls per response |
| `user` | `string \| null` | `null` | User identifier (ignored by vLLM) |

**vLLM Extensions**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_beam_search` | `bool` | `false` | Use beam search instead of sampling |
| `top_k` | `int \| null` | `null` | Top-k sampling |
| `min_p` | `float \| null` | `null` | Minimum probability threshold |
| `repetition_penalty` | `float \| null` | `null` | Repetition penalty |
| `stop_token_ids` | `array \| null` | `[]` | Token IDs that stop generation |
| `min_tokens` | `int` | `0` | Minimum tokens to generate |
| `skip_special_tokens` | `bool` | `true` | Skip special tokens in output |
| `truncate_prompt_tokens` | `int \| null` | `null` | Truncate prompt to N tokens |
| `prompt_logprobs` | `int \| null` | `null` | Return N log probs per prompt token |
| `add_generation_prompt` | `bool` | `true` | Add generation prompt from chat template |
| `chat_template` | `string \| null` | `null` | Override chat template |
| `chat_template_kwargs` | `object \| null` | `null` | Extra kwargs for chat template |
| `structured_outputs` | `object \| null` | `null` | Structured output constraints |
| `priority` | `int` | `0` | Request priority (lower = higher priority) |
| `request_id` | `string` | auto | Custom request identifier |
| `cache_salt` | `string \| null` | `null` | Prefix cache salt for multi-tenant security |

**Response (non-streaming)**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1710000000,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop",
      "logprobs": null
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 9,
    "total_tokens": 19
  }
}
```

**Response (streaming)**

Each SSE chunk has `Content-Type: text/event-stream`:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1710000000,"model":"...","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}

data: [DONE]
```

---

### POST /v1/completions

Create a text completion. Compatible with the [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions).

**Request Body**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `string \| null` | `null` | Model identifier |
| `prompt` | `string \| array \| null` | `null` | Input prompt(s) |
| `echo` | `bool \| null` | `false` | Echo prompt in response |
| `frequency_penalty` | `float \| null` | `0.0` | Frequency penalty |
| `logit_bias` | `object \| null` | `null` | Token bias map |
| `logprobs` | `int \| null` | `null` | Number of log probs to return |
| `max_tokens` | `int \| null` | `16` | Maximum tokens to generate |
| `n` | `int` | `1` | Number of completions |
| `presence_penalty` | `float \| null` | `0.0` | Presence penalty |
| `seed` | `int \| null` | `null` | Random seed |
| `stop` | `string \| array \| null` | `[]` | Stop sequences |
| `stream` | `bool \| null` | `false` | Enable SSE streaming |
| `suffix` | `string \| null` | `null` | *(Not supported)* |
| `temperature` | `float \| null` | `null` | Sampling temperature |
| `top_p` | `float \| null` | `null` | Nucleus sampling |

**vLLM Extensions** — same sampling parameters as chat completions plus:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `add_special_tokens` | `bool` | `true` | Add BOS/EOS tokens |
| `response_format` | `object \| null` | `null` | Structured output format |
| `kv_transfer_params` | `object \| null` | `null` | Disaggregated serving params |

**Response**

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1710000000,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "text": " world!",
      "logprobs": null,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 2,
    "completion_tokens": 2,
    "total_tokens": 4
  }
}
```

---

### POST /v1/responses

Create a stateful response. Compatible with the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses).

**Request Body**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | `string \| array` | **required** | Input text or message list |
| `model` | `string \| null` | `null` | Model identifier |
| `instructions` | `string \| null` | `null` | System-level instructions |
| `max_output_tokens` | `int \| null` | `null` | Maximum output tokens |
| `previous_response_id` | `string \| null` | `null` | Chain from a previous response |
| `stream` | `bool \| null` | `false` | Enable SSE streaming |
| `temperature` | `float \| null` | `null` | Sampling temperature |
| `top_p` | `float \| null` | `null` | Nucleus sampling |
| `tools` | `array` | `[]` | Tool definitions |
| `tool_choice` | `string \| object` | `"auto"` | Tool selection strategy |
| `reasoning` | `object \| null` | `null` | Reasoning configuration |
| `store` | `bool \| null` | `true` | Store response for retrieval |
| `truncation` | `string \| null` | `"disabled"` | Context truncation: `"auto"` or `"disabled"` |

**Response**

```json
{
  "id": "resp_abc123",
  "object": "response",
  "created_at": 1710000000,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "status": "completed",
  "output": [
    {
      "type": "message",
      "role": "assistant",
      "content": [{"type": "output_text", "text": "Hello!"}]
    }
  ],
  "usage": {
    "input_tokens": 10,
    "output_tokens": 5,
    "total_tokens": 15
  }
}
```

---

### GET /v1/responses/{response_id}

Retrieve a previously stored response.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `response_id` | `string` | The response ID to retrieve |

**Query Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `starting_after` | `int \| null` | `null` | Return events after this sequence number |
| `stream` | `bool \| null` | `false` | Stream the response |

---

### POST /v1/responses/{response_id}/cancel

Cancel an in-flight response.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `response_id` | `string` | The response ID to cancel |

---

## Audio Endpoints

### POST /v1/audio/transcriptions

Transcribe audio to text. Compatible with the [OpenAI Audio Transcriptions API](https://platform.openai.com/docs/api-reference/audio/createTranscription).

**Request** — `multipart/form-data`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `file` | `file` | **required** | Audio file (FLAC, MP3, MP4, WAV, WEBM, etc.) |
| `model` | `string` | **required** | Model identifier |
| `language` | `string \| null` | `null` | ISO-639-1 language code |
| `prompt` | `string` | `""` | Optional style guide text |
| `response_format` | `string` | `"json"` | `json`, `text`, `srt`, `verbose_json`, `vtt` |
| `temperature` | `float` | `0.0` | Sampling temperature |
| `stream` | `bool \| null` | `false` | Enable SSE streaming |
| `timestamp_granularities[]` | `array` | `[]` | `word` and/or `segment` timestamps |

**Response (`json`)**

```json
{
  "text": "Hello, this is a transcription.",
  "usage": {"type": "duration", "seconds": 5}
}
```

**Response (`verbose_json`)**

```json
{
  "text": "Hello, this is a transcription.",
  "language": "en",
  "duration": "5.42",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 2.5,
      "text": "Hello, this is a transcription.",
      "tokens": [50364, 938, 428],
      "temperature": 0.0,
      "avg_logprob": -0.245,
      "compression_ratio": 1.235
    }
  ]
}
```

---

### POST /v1/audio/translations

Translate audio to English. Compatible with the [OpenAI Audio Translations API](https://platform.openai.com/docs/api-reference/audio/createTranslation).

**Request** — `multipart/form-data`

Same fields as `/v1/audio/transcriptions`. Note: `openai/whisper-large-v3-turbo` does not support translation.

**Response**

```json
{
  "text": "Hello, this is the translated text."
}
```

---

### WebSocket /v1/realtime

Real-time audio transcription over WebSocket.

**Connection**

```
ws://localhost:8000/v1/realtime
```

**Client → Server Events**

| Event Type | Description |
|------------|-------------|
| `input_audio_buffer.append` | Send base64-encoded PCM16 audio chunk |
| `input_audio_buffer.commit` | Trigger transcription (set `final: true` to end session) |
| `session.update` | Update session parameters (e.g., model) |

**Server → Client Events**

| Event Type | Description |
|------------|-------------|
| `session.created` | Connection established |
| `transcription.delta` | Incremental transcription text |
| `transcription.done` | Final transcription with usage |
| `error` | Error notification |

---

## Pooling / Embedding Endpoints

### POST /v1/embeddings

Generate text embeddings. Compatible with the [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings).

**Request Body**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `input` | `string \| array` | **required** | Text(s) to embed |
| `model` | `string` | **required** | Embedding model identifier |
| `encoding_format` | `string` | `"float"` | `"float"` or `"base64"` |
| `dimensions` | `int \| null` | `null` | Output dimension (if model supports it) |

**Response**

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0023, -0.0094, ...]
    }
  ],
  "model": "BAAI/bge-base-en-v1.5",
  "usage": {
    "prompt_tokens": 8,
    "total_tokens": 8
  }
}
```

---

### POST /pooling

Return raw pooling outputs (arbitrary nested tensors).

Same request format as `/v1/embeddings`. Output `data[].embedding` may be a nested list.

---

### POST /classify

Classify input text using a sequence classification model.

**Request Body**

| Field | Type | Description |
|-------|------|-------------|
| `input` | `string \| array` | Text(s) to classify |
| `model` | `string` | Classification model identifier |

**Response**

```json
{
  "id": "classify-abc123",
  "object": "list",
  "created": 1710000000,
  "model": "jason9693/Qwen2.5-1.5B-apeach",
  "data": [
    {
      "index": 0,
      "label": "positive",
      "probs": [0.87, 0.13],
      "num_classes": 2
    }
  ],
  "usage": {"prompt_tokens": 10, "total_tokens": 10, "completion_tokens": 0}
}
```

---

### POST /score

Score sentence pairs using a cross-encoder or embedding model.

**Request Body**

| Field | Type | Description |
|-------|------|-------------|
| `queries` | `string \| array` | Query text(s) |
| `documents` | `string \| array` | Document text(s) |
| `model` | `string` | Scoring model identifier |
| `encoding_format` | `string` | `"float"` (default) |

**Response**

```json
{
  "id": "score-abc123",
  "object": "list",
  "created": 1710000000,
  "model": "BAAI/bge-reranker-v2-m3",
  "data": [
    {"index": 0, "object": "score", "score": 0.97}
  ],
  "usage": {}
}
```

---

### POST /rerank, POST /v1/rerank, POST /v2/rerank

Re-rank documents by relevance to a query. Compatible with Jina AI v1 and Cohere v1/v2 re-rank APIs.

**Request Body**

| Field | Type | Description |
|-------|------|-------------|
| `query` | `string` | The search query |
| `documents` | `array` | Documents to re-rank |
| `model` | `string` | Re-ranking model |
| `top_n` | `int \| null` | Return top N results |

---

## Model Management

### GET /v1/models

List all available models.

**Response**

```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.1-8B-Instruct",
      "object": "model",
      "created": 1710000000,
      "owned_by": "vllm",
      "root": "meta-llama/Llama-3.1-8B-Instruct",
      "parent": null,
      "permission": []
    }
  ]
}
```

---

## Utility Endpoints

### GET /health

Server health check. Returns `200 OK` when the engine is healthy.

### POST /tokenize

Tokenize text without generating.

**Request Body**

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "prompt": "Hello, world!"
}
```

**Response**

```json
{
  "tokens": [9906, 11, 1917, 0],
  "count": 4
}
```

### POST /detokenize

Convert token IDs back to text.

**Request Body**

```json
{
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "tokens": [9906, 11, 1917, 0]
}
```

**Response**

```json
{
  "prompt": "Hello, world!"
}
```

---

## Anthropic-Compatible Endpoints

### POST /v1/messages

Create a message using the Anthropic Messages API format.

**Request Body**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `string` | **required** | Model identifier |
| `messages` | `array` | **required** | Conversation messages |
| `max_tokens` | `int` | **required** | Maximum tokens to generate |
| `system` | `string \| array \| null` | `null` | System prompt |
| `temperature` | `float \| null` | `null` | Sampling temperature |
| `top_p` | `float \| null` | `null` | Nucleus sampling |
| `top_k` | `int \| null` | `null` | Top-k sampling |
| `stream` | `bool \| null` | `false` | Enable SSE streaming |
| `tools` | `array \| null` | `null` | Tool definitions |
| `tool_choice` | `object \| null` | `null` | Tool selection |
| `stop_sequences` | `array \| null` | `null` | Stop sequences |

**Response**

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [{"type": "text", "text": "Hello!"}],
  "model": "claude-3-5-sonnet",
  "stop_reason": "end_turn",
  "usage": {"input_tokens": 10, "output_tokens": 5}
}
```

### POST /v1/messages/count_tokens

Count tokens for an Anthropic-format request without generating.

**Response**

```json
{
  "input_tokens": 42
}
```

---

## SageMaker Endpoints

### GET /ping, POST /ping

SageMaker health check endpoint. Returns `200 OK` when healthy.

### POST /invocations

SageMaker inference endpoint. Automatically routes to the appropriate handler based on request body schema (chat completion, completion, embedding, etc.).

---

## HTTP Status Codes

| Code | Meaning |
|------|---------|
| `200` | Success |
| `400` | Bad request / validation error |
| `401` | Unauthorized (invalid or missing API key) |
| `404` | Model or resource not found |
| `422` | Unprocessable entity |
| `500` | Internal server error |
| `501` | Not implemented (model doesn't support this endpoint) |

---

## Rate Limiting & Headers

### Request Headers

| Header | Description |
|--------|-------------|
| `Authorization: Bearer <key>` | API key authentication |
| `Content-Type: application/json` | Required for JSON endpoints |
| `X-Request-Id` | Optional custom request ID (requires `--enable-request-id-headers`) |

### Response Headers

| Header | Description |
|--------|-------------|
| `X-Request-Id` | Echo of request ID (when enabled) |
| `endpoint-load-metrics-format` | Load metrics format (when requested) |

---

## See Also

- [Chat Completions Guide](chat_completions.md)
- [Text Completions Guide](completions.md)
- [Embeddings Guide](embeddings.md)
- [Streaming Guide](streaming.md)
- [OpenAI-Compatible Server Setup](openai_compatible_server.md)
