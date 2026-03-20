# Text Completions

The `POST /v1/completions` endpoint provides raw text completion — the model continues a given prompt without any chat formatting. It is compatible with the [OpenAI Completions API](https://platform.openai.com/docs/api-reference/completions).

!!! tip "Chat vs. Completions"
    For most use cases, prefer the [Chat Completions](chat_completions.md) endpoint. Use the Completions endpoint when you need precise control over the raw prompt format, or when working with base (non-instruction-tuned) models.

---

## Quick Start

### Start the Server

```bash
vllm serve meta-llama/Llama-3.1-8B \
  --host 0.0.0.0 \
  --port 8000
```

### Send a Request

=== "Python (OpenAI SDK)"

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    response = client.completions.create(
        model="meta-llama/Llama-3.1-8B",
        prompt="The capital of France is",
        max_tokens=10,
        temperature=0.0,
    )
    print(response.choices[0].text)
    # → " Paris."
    ```

=== "curl"

    ```bash
    curl http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "meta-llama/Llama-3.1-8B",
        "prompt": "The capital of France is",
        "max_tokens": 10,
        "temperature": 0.0
      }'
    ```

---

## Request Parameters

### Standard OpenAI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `string \| null` | `null` | Model identifier (uses server default if omitted) |
| `prompt` | `string \| array \| null` | `null` | Input prompt(s) — string, list of strings, or token ID arrays |
| `echo` | `bool \| null` | `false` | Include the prompt in the response |
| `frequency_penalty` | `float \| null` | `0.0` | Penalize repeated tokens by frequency |
| `logit_bias` | `object \| null` | `null` | Token ID → bias value map |
| `logprobs` | `int \| null` | `null` | Return top-N log probs per token |
| `max_tokens` | `int \| null` | `16` | Maximum tokens to generate |
| `n` | `int` | `1` | Number of completions to generate |
| `presence_penalty` | `float \| null` | `0.0` | Penalize tokens that have appeared at all |
| `seed` | `int \| null` | `null` | Random seed for reproducibility |
| `stop` | `string \| array \| null` | `[]` | Stop sequences |
| `stream` | `bool \| null` | `false` | Enable SSE streaming |
| `stream_options` | `object \| null` | `null` | Streaming options (e.g., `include_usage`) |
| `temperature` | `float \| null` | `null` | Sampling temperature |
| `top_p` | `float \| null` | `null` | Nucleus sampling probability |
| `user` | `string \| null` | `null` | User identifier (ignored) |

### vLLM Sampling Extensions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_beam_search` | `bool` | `false` | Use beam search (set `n` to beam width) |
| `top_k` | `int \| null` | `null` | Top-k sampling |
| `min_p` | `float \| null` | `null` | Minimum probability threshold |
| `repetition_penalty` | `float \| null` | `null` | Multiplicative repetition penalty |
| `length_penalty` | `float` | `1.0` | Length penalty for beam search |
| `stop_token_ids` | `array \| null` | `[]` | Token IDs that stop generation |
| `include_stop_str_in_output` | `bool` | `false` | Include stop string in output |
| `ignore_eos` | `bool` | `false` | Ignore EOS token |
| `min_tokens` | `int` | `0` | Minimum tokens before stop sequences apply |
| `skip_special_tokens` | `bool` | `true` | Skip special tokens in output |
| `spaces_between_special_tokens` | `bool` | `true` | Add spaces between special tokens |
| `truncate_prompt_tokens` | `int \| null` | `null` | Truncate prompt to N tokens from the right |
| `allowed_token_ids` | `array \| null` | `null` | Whitelist of allowed token IDs |
| `prompt_logprobs` | `int \| null` | `null` | Return N log probs per prompt token |

### vLLM Extra Extensions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `add_special_tokens` | `bool` | `true` | Add BOS/EOS tokens to the prompt |
| `response_format` | `object \| null` | `null` | Structured output format |
| `structured_outputs` | `object \| null` | `null` | Structured output constraints |
| `priority` | `int` | `0` | Request priority (lower = higher priority) |
| `request_id` | `string` | auto | Custom request identifier |
| `return_tokens_as_token_ids` | `bool \| null` | `null` | Return token IDs in logprobs |
| `return_token_ids` | `bool \| null` | `null` | Include token IDs in response |
| `cache_salt` | `string \| null` | `null` | Prefix cache salt for multi-tenant security |
| `kv_transfer_params` | `object \| null` | `null` | Disaggregated serving parameters |
| `repetition_detection` | `object \| null` | `null` | Early stopping on repetitive patterns |

---

## Response Format

### Non-Streaming Response

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1710000000,
  "model": "meta-llama/Llama-3.1-8B",
  "choices": [
    {
      "index": 0,
      "text": " Paris.",
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null
    }
  ],
  "usage": {
    "prompt_tokens": 6,
    "completion_tokens": 2,
    "total_tokens": 8
  }
}
```

### Streaming Response

Each chunk is a Server-Sent Event:

```
data: {"id":"cmpl-abc123","object":"text_completion","created":1710000000,"model":"...","choices":[{"index":0,"text":" Paris","finish_reason":null}]}

data: {"id":"cmpl-abc123","object":"text_completion","created":1710000000,"model":"...","choices":[{"index":0,"text":".","finish_reason":"stop"}]}

data: [DONE]
```

---

## Common Use Cases

### Batch Prompts

Send multiple prompts in a single request:

```python
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B",
    prompt=[
        "The capital of France is",
        "The capital of Germany is",
        "The capital of Japan is",
    ],
    max_tokens=5,
    temperature=0.0,
)

for choice in response.choices:
    print(f"[{choice.index}] {choice.text}")
```

### Echo Mode

Include the prompt in the response (useful for debugging):

```python
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B",
    prompt="The capital of France is",
    max_tokens=5,
    echo=True,
)
print(response.choices[0].text)
# → "The capital of France is Paris."
```

### Log Probabilities

Get token-level probabilities:

```python
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B",
    prompt="The capital of France is",
    max_tokens=3,
    logprobs=5,  # Return top 5 alternatives per token
)

for i, token_logprobs in enumerate(response.choices[0].logprobs.top_logprobs):
    print(f"Position {i}: {token_logprobs}")
```

### Beam Search

Use beam search for higher-quality outputs:

```python
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B",
    prompt="Write a haiku about autumn:",
    max_tokens=50,
    n=4,  # Beam width
    extra_body={"use_beam_search": True},
)

for choice in response.choices:
    print(f"Beam {choice.index}: {choice.text}")
```

### Structured Output

Force the model to produce valid JSON:

```python
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    prompt='Extract JSON: {"name": "',
    max_tokens=50,
    response_format={"type": "json_object"},
)
```

### Prompt Token IDs

Pass pre-tokenized input directly:

```python
# Using token IDs instead of text
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B",
    prompt=[128000, 791, 6864, 315, 9822, 374],  # Token IDs
    max_tokens=5,
)
```

---

## Streaming

```python
stream = client.completions.create(
    model="meta-llama/Llama-3.1-8B",
    prompt="Once upon a time,",
    max_tokens=100,
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].text, end="", flush=True)
```

### Stream with Usage

```python
stream = client.completions.create(
    model="meta-llama/Llama-3.1-8B",
    prompt="Hello",
    max_tokens=20,
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in stream:
    if chunk.usage:
        print(f"\nTotal tokens: {chunk.usage.total_tokens}")
    else:
        print(chunk.choices[0].text, end="", flush=True)
```

---

## Repetition Detection

Automatically stop generation when the model starts repeating itself:

```python
response = client.completions.create(
    model="meta-llama/Llama-3.1-8B",
    prompt="List items:",
    max_tokens=500,
    extra_body={
        "repetition_detection": {
            "ngram_size": 5,
            "max_repeats": 3,
        }
    },
)
```

This is useful for preventing runaway generation where the model gets stuck in a loop.

---

## Render Endpoint (Debug)

Preview the tokenized prompt without generating:

```bash
curl http://localhost:8000/v1/completions/render \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "prompt": "The capital of France is"
  }'
```

---

## Differences from Chat Completions

| Feature | Completions | Chat Completions |
|---------|-------------|-----------------|
| Input format | Raw text / token IDs | Structured messages |
| Chat template | Not applied | Applied automatically |
| Multi-turn | Manual (include history in prompt) | Built-in |
| Tool calling | Not supported | Supported |
| Multimodal | Not supported | Supported |
| `echo` parameter | Supported | Not supported |
| `suffix` parameter | Listed but not supported | N/A |

---

## Performance Tips

1. **Batch prompts**: Pass a list of prompts to process them in parallel.
2. **Prefix caching**: Common prompt prefixes are cached automatically — structure prompts to share prefixes.
3. **Token IDs**: Passing pre-tokenized input avoids tokenization overhead.
4. **`add_special_tokens=False`**: If you're manually managing BOS/EOS tokens, disable auto-insertion.
5. **`truncate_prompt_tokens`**: Use negative values to keep the last N tokens of a long prompt.

---

## See Also

- [Chat Completions](chat_completions.md) — Multi-turn conversations with tool calling
- [Streaming Guide](streaming.md) — SSE streaming deep dive
- [API Reference](api_reference.md) — Full schema documentation
- [Batch Inference](batch_inference.md) — Offline batch processing
