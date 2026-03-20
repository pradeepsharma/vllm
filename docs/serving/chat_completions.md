# Chat Completions

The `POST /v1/chat/completions` endpoint is vLLM's primary interface for multi-turn conversational AI. It is fully compatible with the [OpenAI Chat Completions API](https://platform.openai.com/docs/api-reference/chat), meaning any client that works with OpenAI will work with vLLM.

---

## Quick Start

### Start the Server

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key my-secret-key
```

### Send a Request

=== "Python (OpenAI SDK)"

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="my-secret-key",
    )

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
    )
    print(response.choices[0].message.content)
    # → "The capital of France is Paris."
    ```

=== "curl"

    ```bash
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer my-secret-key" \
      -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "What is the capital of France?"}
        ]
      }'
    ```

---

## Message Roles

| Role | Description |
|------|-------------|
| `system` | Sets the assistant's behavior and persona |
| `user` | Human turn in the conversation |
| `assistant` | Model's previous responses |
| `tool` | Tool call results (for function calling) |

### Multi-turn Conversation

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello, Alice! How can I help you?"},
        {"role": "user", "content": "What's my name?"},
    ],
)
```

---

## Streaming

Enable streaming to receive tokens as they are generated, rather than waiting for the full response.

```python
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
```

### Stream with Usage Statistics

```python
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,
    stream_options={"include_usage": True},
)

for chunk in stream:
    if chunk.usage:
        print(f"Tokens used: {chunk.usage.total_tokens}")
```

---

## Sampling Parameters

Control how the model generates text:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temperature` | `float` | model default | Randomness (0 = deterministic, 2 = very random) |
| `top_p` | `float` | `1.0` | Nucleus sampling: keep tokens with cumulative prob ≤ top_p |
| `top_k` | `int` | `null` | Keep only top-k most probable tokens |
| `min_p` | `float` | `null` | Minimum probability threshold |
| `frequency_penalty` | `float` | `0.0` | Penalize tokens proportional to their frequency |
| `presence_penalty` | `float` | `0.0` | Penalize tokens that have appeared at all |
| `repetition_penalty` | `float` | `null` | Multiplicative repetition penalty |
| `seed` | `int` | `null` | Random seed for reproducibility |
| `n` | `int` | `1` | Number of completions to generate |
| `max_completion_tokens` | `int` | `null` | Maximum tokens to generate |
| `stop` | `string \| array` | `[]` | Stop generation at these strings |
| `stop_token_ids` | `array` | `[]` | Stop generation at these token IDs |
| `min_tokens` | `int` | `0` | Minimum tokens before stop sequences apply |

### Example: Deterministic Output

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    temperature=0.0,
    seed=42,
)
```

---

## Structured Outputs

Force the model to produce output matching a specific schema.

### JSON Mode

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Extract: name and age from 'Alice is 30'."}],
    response_format={"type": "json_object"},
)
import json
data = json.loads(response.choices[0].message.content)
```

### JSON Schema

```python
from pydantic import BaseModel

class Person(BaseModel):
    name: str
    age: int

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Extract: name and age from 'Alice is 30'."}],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name": "Person",
            "schema": Person.model_json_schema(),
        },
    },
)
```

### Choice Constraint (vLLM Extension)

Force the model to output one of a fixed set of strings:

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Is this positive or negative? 'I love it!'"}],
    extra_body={
        "structured_outputs": {"choice": ["positive", "negative", "neutral"]},
    },
)
```

### Regex Constraint (vLLM Extension)

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Give me a US phone number."}],
    extra_body={
        "structured_outputs": {"regex": r"\d{3}-\d{3}-\d{4}"},
    },
)
```

---

## Tool Calling / Function Calling

vLLM supports OpenAI-compatible tool calling. Enable it with `--enable-auto-tool-choice` and `--tool-call-parser`.

### Server Setup

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json
```

### Define and Call Tools

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto",
)

# Check if the model wants to call a tool
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    print(f"Function: {tool_call.function.name}")
    print(f"Arguments: {tool_call.function.arguments}")
```

### Tool Choice Options

| Value | Description |
|-------|-------------|
| `"none"` | Never call tools (default) |
| `"auto"` | Model decides whether to call tools |
| `"required"` | Always call at least one tool |
| `{"type": "function", "function": {"name": "..."}}` | Force a specific tool |

### Available Tool Parsers

| Parser | Models |
|--------|--------|
| `llama3_json` | Llama 3.x |
| `hermes` | Hermes / Nous models |
| `mistral` | Mistral models |
| `qwen25` | Qwen 2.5 |
| `deepseekv3` | DeepSeek V3 |
| `internlm` | InternLM |
| `pythonic` | Models using Python-style tool calls |

---

## Log Probabilities

Get token-level log probabilities for analysis:

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Say hello."}],
    logprobs=True,
    top_logprobs=5,  # Return top 5 alternatives per token
)

for token_info in response.choices[0].logprobs.content:
    print(f"Token: {token_info.token!r}, logprob: {token_info.logprob:.3f}")
    for alt in token_info.top_logprobs:
        print(f"  Alt: {alt.token!r}, logprob: {alt.logprob:.3f}")
```

---

## Multimodal Inputs

For vision-capable models, include images in messages:

```python
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"},
                },
                {"type": "text", "text": "What is in this image?"},
            ],
        }
    ],
)
```

For audio-capable models:

```python
import base64

with open("audio.wav", "rb") as f:
    audio_b64 = base64.b64encode(f.read()).decode()

response = client.chat.completions.create(
    model="Qwen/Qwen2-Audio-7B-Instruct",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {"data": audio_b64, "format": "wav"},
                },
                {"type": "text", "text": "Transcribe this audio."},
            ],
        }
    ],
)
```

---

## Reasoning Models

For models with chain-of-thought reasoning (e.g., DeepSeek-R1, QwQ):

```python
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[{"role": "user", "content": "Solve: 2x + 5 = 13"}],
    reasoning_effort="high",  # "low", "medium", or "high"
)

# Access reasoning trace (vLLM extension)
print(response.choices[0].message.reasoning)
print(response.choices[0].message.content)
```

To disable reasoning output:

```python
response = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-R1",
    messages=[{"role": "user", "content": "What is 2+2?"}],
    extra_body={"include_reasoning": False},
)
```

---

## Chat Templates

vLLM uses the model's built-in chat template from its tokenizer configuration. You can override it:

```bash
# Use a custom template file
vllm serve <model> --chat-template ./my-template.jinja

# Or inline
vllm serve <model> --chat-template "{{ messages | tojson }}"
```

### Template Content Format

Some models expect content as a string; others expect the OpenAI list format. vLLM auto-detects this, but you can override:

```bash
vllm serve <model> --chat-template-content-format openai
# or
vllm serve <model> --chat-template-content-format string
```

### Per-Request Template Override

If `--trust-request-chat-template` is enabled on the server:

```python
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello!"}],
    extra_body={
        "chat_template": "{% for m in messages %}{{ m.content }}{% endfor %}",
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
```

---

## Render Endpoint (Debug)

Preview the rendered prompt without generating:

```bash
curl http://localhost:8000/v1/chat/completions/render \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Returns the tokenized prompt as it would be sent to the engine.

---

## vLLM-Specific Extensions

Pass these via `extra_body` in the OpenAI client:

| Parameter | Description |
|-----------|-------------|
| `use_beam_search` | Use beam search (set `n` to beam width) |
| `top_k` | Top-k sampling |
| `min_p` | Minimum probability filter |
| `repetition_penalty` | Multiplicative repetition penalty |
| `stop_token_ids` | Token IDs that stop generation |
| `include_stop_str_in_output` | Include stop string in output |
| `ignore_eos` | Ignore EOS token |
| `min_tokens` | Minimum tokens before stop sequences |
| `skip_special_tokens` | Skip special tokens in output |
| `spaces_between_special_tokens` | Add spaces between special tokens |
| `truncate_prompt_tokens` | Truncate prompt to N tokens |
| `prompt_logprobs` | Return N log probs per prompt token |
| `allowed_token_ids` | Whitelist of allowed token IDs |
| `bad_words` | Words to prevent in output |
| `add_generation_prompt` | Add generation prompt from template |
| `continue_final_message` | Continue the last message instead of starting new |
| `add_special_tokens` | Add BOS/EOS tokens |
| `documents` | RAG documents for template |
| `chat_template` | Override chat template |
| `chat_template_kwargs` | Extra template kwargs |
| `mm_processor_kwargs` | Multimodal processor kwargs |
| `structured_outputs` | Structured output constraints |
| `priority` | Request priority (lower = higher priority) |
| `request_id` | Custom request identifier |
| `cache_salt` | Prefix cache salt |
| `kv_transfer_params` | Disaggregated serving params |
| `echo` | Prepend last message if same role |
| `return_tokens_as_token_ids` | Return token IDs instead of strings in logprobs |
| `token_ids` | Return token IDs in response |

---

## Performance Tips

1. **Batch requests**: Send multiple messages in a single request using `n > 1` or send concurrent requests.
2. **Prefix caching**: Reuse common system prompts across requests — vLLM caches KV states automatically.
3. **Streaming**: Use `stream=True` for better perceived latency in interactive applications.
4. **Max tokens**: Set `max_completion_tokens` to avoid runaway generation.
5. **Temperature 0**: Use `temperature=0` for deterministic, reproducible outputs.

---

## See Also

- [API Reference](api_reference.md) — Full schema documentation
- [Streaming Guide](streaming.md) — Deep dive into SSE streaming
- [OpenAI-Compatible Server](openai_compatible_server.md) — Server configuration
- [Text Completions](completions.md) — Raw text completion endpoint
