# POST /v1/completions

The legacy text completions endpoint generates text from a raw prompt string or token list. It is compatible with the OpenAI Completions API (v1) and is suitable for base models that have not been instruction-tuned.

**Source:** `vllm/entrypoints/openai/completion/`

> **Note:** For instruction-tuned or chat models, prefer the [Chat Completions API](chat-completions.md) (`POST /v1/chat/completions`). The completions endpoint is considered legacy by OpenAI but remains fully supported in vLLM.

---

## Endpoint

```
POST /v1/completions
```

Content-Type: `application/json`

---

## Request Schema

The request body is defined in `vllm/entrypoints/openai/completion/protocol.py` as `CompletionRequest`.

### Core OpenAI-Compatible Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `string \| null` | `null` | Model name to use. If `null`, uses the server's default model. |
| `prompt` | `string \| array \| null` | `null` | The prompt(s) to complete. Can be a string, list of strings, list of token IDs, or list of token ID lists. |
| `echo` | `bool \| null` | `false` | Echo the prompt in the response alongside the completion. |
| `frequency_penalty` | `float \| null` | `0.0` | Penalizes tokens based on their frequency in the output. Range: `-2.0` to `2.0`. |
| `logit_bias` | `object \| null` | `null` | Map of token IDs to bias values (`-100` to `100`). |
| `logprobs` | `int \| null` | `null` | Number of top log-probability tokens to return per position. |
| `max_tokens` | `int \| null` | `16` | Maximum number of tokens to generate. |
| `n` | `int` | `1` | Number of completions to generate per prompt. |
| `presence_penalty` | `float \| null` | `0.0` | Penalizes tokens that have appeared at all in the output. |
| `seed` | `int \| null` | `null` | Random seed for reproducibility. |
| `stop` | `string \| array \| null` | `[]` | Stop sequences. |
| `stream` | `bool \| null` | `false` | Stream partial results as Server-Sent Events. |
| `stream_options` | `object \| null` | `null` | Options for streaming (see [Chat Completions](chat-completions.md#stream-options)). |
| `suffix` | `string \| null` | `null` | Suffix appended after the completion. |
| `temperature` | `float \| null` | `null` | Sampling temperature. |
| `top_p` | `float \| null` | `null` | Nucleus sampling probability threshold. |
| `user` | `string \| null` | `null` | User identifier (ignored by vLLM). |

### vLLM Sampling Extensions

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `use_beam_search` | `bool` | `false` | Use beam search instead of sampling. |
| `top_k` | `int \| null` | `null` | Top-K sampling. |
| `min_p` | `float \| null` | `null` | Minimum probability threshold. |
| `repetition_penalty` | `float \| null` | `null` | Penalizes repeated tokens. |
| `length_penalty` | `float` | `1.0` | Exponential length penalty (beam search only). |
| `stop_token_ids` | `array \| null` | `[]` | Token IDs that stop generation. |
| `include_stop_str_in_output` | `bool` | `false` | Include the stop string in the output. |
| `ignore_eos` | `bool` | `false` | Continue past the EOS token. |
| `min_tokens` | `int` | `0` | Minimum tokens to generate. |
| `skip_special_tokens` | `bool` | `true` | Skip special tokens in decoded output. |
| `spaces_between_special_tokens` | `bool` | `true` | Add spaces between special tokens. |
| `truncate_prompt_tokens` | `int \| null` | `null` | Truncate prompt to this many tokens. |
| `allowed_token_ids` | `array \| null` | `null` | Restrict generation to these token IDs. |
| `prompt_logprobs` | `int \| null` | `null` | Return log probabilities for prompt tokens. |

### vLLM Extra Parameters

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `prompt_embeds` | `bytes \| array \| null` | `null` | Raw prompt embeddings (bypasses tokenization). |
| `add_special_tokens` | `bool` | `true` | Add BOS/EOS special tokens to the prompt. |
| `response_format` | `object \| null` | `null` | Output format: `{"type": "text"}`, `{"type": "json_object"}`, `{"type": "json_schema", ...}`, or `{"type": "structural_tag", ...}`. |
| `structured_outputs` | `object \| null` | `null` | Additional structured output parameters. |
| `priority` | `int` | `0` | Request priority (lower = higher priority). |
| `request_id` | `string` | auto-generated | Custom request ID for tracing. |
| `return_tokens_as_token_ids` | `bool \| null` | `null` | Represent tokens as `"token_id:{id}"` strings in logprobs. |
| `return_token_ids` | `bool \| null` | `null` | Include token IDs alongside generated text. |
| `cache_salt` | `string \| null` | `null` | Salt for prefix cache isolation. |
| `kv_transfer_params` | `object \| null` | `null` | KV-cache transfer parameters for disaggregated serving. |
| `vllm_xargs` | `object \| null` | `null` | Custom extension parameters. |
| `repetition_detection` | `object \| null` | `null` | Parameters for detecting and stopping repetitive N-gram patterns. |

---

## Response Schema

### Non-Streaming Response

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1714000000,
  "model": "meta-llama/Llama-3.1-8B",
  "choices": [
    {
      "index": 0,
      "text": " is the capital of France.",
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null,
      "token_ids": null,
      "prompt_logprobs": null,
      "prompt_token_ids": null
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 7,
    "total_tokens": 13
  },
  "system_fingerprint": null,
  "kv_transfer_params": null
}
```

### Streaming Response (SSE)

```
data: {"id":"cmpl-abc123","object":"text_completion","created":1714000000,"model":"meta-llama/Llama-3.1-8B","choices":[{"index":0,"text":" is","logprobs":null,"finish_reason":null}]}

data: {"id":"cmpl-abc123","object":"text_completion","created":1714000000,"model":"meta-llama/Llama-3.1-8B","choices":[{"index":0,"text":" the capital","logprobs":null,"finish_reason":null}]}

data: {"id":"cmpl-abc123","object":"text_completion","created":1714000000,"model":"meta-llama/Llama-3.1-8B","choices":[{"index":0,"text":"","logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":6,"completion_tokens":7,"total_tokens":13}}

data: [DONE]
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | `string` | Unique completion ID (format: `cmpl-{uuid}`). |
| `object` | `string` | Always `"text_completion"`. |
| `created` | `int` | Unix timestamp. |
| `model` | `string` | Model name used. |
| `choices` | `array` | List of completion choices. |
| `choices[].index` | `int` | Choice index. |
| `choices[].text` | `string` | Generated text. |
| `choices[].logprobs` | `object \| null` | Log probabilities (if requested). |
| `choices[].finish_reason` | `string \| null` | Why generation stopped: `"stop"`, `"length"`. |
| `choices[].stop_reason` | `int \| string \| null` | The specific stop string or token ID. |
| `choices[].token_ids` | `array \| null` | Token IDs of generated text (if `return_token_ids=true`). |
| `choices[].prompt_token_ids` | `array \| null` | Token IDs of the prompt (if `return_token_ids=true`). |
| `usage` | `object` | Token usage statistics. |
| `kv_transfer_params` | `object \| null` | KV-cache transfer parameters. |

---

## Examples

### Basic Completion

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="token-abc123")

response = client.completions.create(
    model="meta-llama/Llama-3.1-8B",
    prompt="The capital of France",
    max_tokens=50,
    temperature=0.0
)
print(response.choices[0].text)
# Output: " is Paris, the City of Light..."
```

### Multiple Completions

```python
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B",
    prompt="Once upon a time",
    n=3,
    max_tokens=50,
    temperature=1.0
)
for i, choice in enumerate(response.choices):
    print(f"Choice {i}: {choice.text}")
```

### Echo Prompt with Logprobs

```python
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B",
    prompt="The sky is",
    max_tokens=5,
    echo=True,
    logprobs=5
)
# Response includes both prompt and completion text
print(response.choices[0].text)
print(response.choices[0].logprobs)
```

### Streaming Completion

```python
stream = client.completions.create(
    model="meta-llama/Llama-3.1-8B",
    prompt="Write a haiku about mountains:",
    max_tokens=50,
    stream=True
)
for chunk in stream:
    print(chunk.choices[0].text, end="", flush=True)
```

### Raw HTTP Request

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer token-abc123" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "prompt": "The capital of France is",
    "max_tokens": 20,
    "temperature": 0.0,
    "stop": ["\n", "."]
  }'
```

### Token ID Prompt

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "prompt": [1, 450, 7483, 310, 3444, 338],
    "max_tokens": 10
  }'
```

---

## Default Sampling Parameters

The completions endpoint uses these defaults when not specified:

| Parameter | Default |
|-----------|---------|
| `temperature` | `1.0` |
| `top_p` | `1.0` |
| `top_k` | `0` (disabled) |
| `min_p` | `0.0` |
| `repetition_penalty` | `1.0` |

---

## Related Pages

- [POST /v1/chat/completions](chat-completions.md) — Chat completions (recommended for instruction models)
- [Batch Inference](batch-inference.md) — Offline batch processing
- [Error Handling](error-handling.md) — HTTP status codes and error formats
