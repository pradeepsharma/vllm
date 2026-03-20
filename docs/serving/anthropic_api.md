# Anthropic-Compatible API

vLLM provides an Anthropic-compatible API that allows you to use the [Anthropic Python SDK](https://github.com/anthropic-ai/anthropic-sdk-python) and any client built for the Claude API to interact with vLLM-served models.

The Anthropic API is served alongside the OpenAI-compatible API on the same server — no separate process is needed.

---

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/messages` | POST | Create a message (Anthropic Messages API) |
| `/v1/messages/count_tokens` | POST | Count tokens without generating |

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

=== "Python (Anthropic SDK)"

    ```python
    import anthropic

    client = anthropic.Anthropic(
        base_url="http://localhost:8000",
        api_key="my-secret-key",
    )

    message = client.messages.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ],
    )

    print(message.content[0].text)
    # → "The capital of France is Paris."
    ```

=== "curl"

    ```bash
    curl http://localhost:8000/v1/messages \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer my-secret-key" \
      -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "max_tokens": 1024,
        "messages": [
          {"role": "user", "content": "What is the capital of France?"}
        ]
      }'
    ```

---

## Messages API

### `POST /v1/messages`

Create a message using the Anthropic Messages API format.

#### Request Body

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | `string` | **required** | Model identifier |
| `messages` | `array` | **required** | Conversation messages |
| `max_tokens` | `int` | **required** | Maximum tokens to generate (must be > 0) |
| `system` | `string \| array \| null` | `null` | System prompt (string or content blocks) |
| `temperature` | `float \| null` | `null` | Sampling temperature |
| `top_p` | `float \| null` | `null` | Nucleus sampling |
| `top_k` | `int \| null` | `null` | Top-k sampling |
| `stream` | `bool \| null` | `false` | Enable SSE streaming |
| `tools` | `array \| null` | `null` | Tool definitions |
| `tool_choice` | `object \| null` | `null` | Tool selection strategy |
| `stop_sequences` | `array \| null` | `null` | Stop sequences |
| `metadata` | `object \| null` | `null` | Request metadata |

#### Message Format

Messages follow the Anthropic format with `role` and `content`:

```json
{
  "role": "user",
  "content": "Hello, Claude!"
}
```

Content can also be a list of content blocks:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "What's in this image?"},
    {
      "type": "image",
      "source": {
        "type": "url",
        "url": "https://example.com/image.jpg"
      }
    }
  ]
}
```

#### Content Block Types

| Type | Description |
|------|-------------|
| `text` | Plain text content |
| `image` | Image (URL or base64) |
| `tool_use` | Tool call from assistant |
| `tool_result` | Tool result from user |
| `thinking` | Extended thinking content |

---

#### Response Format

```json
{
  "id": "msg_abc123",
  "type": "message",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "The capital of France is Paris."
    }
  ],
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 14,
    "output_tokens": 9
  }
}
```

#### Stop Reasons

| Value | Description |
|-------|-------------|
| `end_turn` | Model reached a natural stopping point |
| `max_tokens` | Hit the `max_tokens` limit |
| `stop_sequence` | Hit a stop sequence |
| `tool_use` | Model wants to use a tool |

---

## Multi-Turn Conversations

```python
messages = [
    {"role": "user", "content": "My name is Alice."},
    {"role": "assistant", "content": "Hello, Alice! How can I help you?"},
    {"role": "user", "content": "What's my name?"},
]

message = client.messages.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_tokens=100,
    messages=messages,
)
print(message.content[0].text)
# → "Your name is Alice."
```

---

## System Prompts

```python
message = client.messages.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_tokens=500,
    system="You are a concise assistant. Answer in one sentence.",
    messages=[
        {"role": "user", "content": "What is machine learning?"}
    ],
)
```

System prompts can also use content blocks:

```python
message = client.messages.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_tokens=500,
    system=[
        {"type": "text", "text": "You are a helpful assistant."},
        {"type": "text", "text": "Always respond in English."},
    ],
    messages=[{"role": "user", "content": "Bonjour!"}],
)
```

---

## Streaming

```python
with client.messages.stream(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Tell me a story."}],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### Streaming Events

The streaming response uses Server-Sent Events with Anthropic-format events:

| Event Type | Description |
|------------|-------------|
| `message_start` | Message object created |
| `content_block_start` | New content block started |
| `content_block_delta` | Incremental content delta |
| `content_block_stop` | Content block complete |
| `message_delta` | Message-level delta (stop reason, usage) |
| `message_stop` | Message complete |
| `ping` | Keep-alive ping |
| `error` | Error notification |

---

## Tool Calling

```python
tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name",
                },
            },
            "required": ["location"],
        },
    }
]

message = client.messages.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
)

# Check for tool use
for block in message.content:
    if block.type == "tool_use":
        print(f"Tool: {block.name}")
        print(f"Input: {block.input}")
```

### Tool Choice Options

| Type | Description |
|------|-------------|
| `"auto"` | Model decides whether to use tools |
| `"any"` | Model must use at least one tool |
| `"tool"` | Force a specific tool (requires `name`) |
| `"none"` | Never use tools |

```python
# Force a specific tool
message = client.messages.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_tokens=1024,
    tools=tools,
    tool_choice={"type": "tool", "name": "get_weather"},
    messages=[{"role": "user", "content": "What's the weather?"}],
)
```

### Tool Result

After receiving a tool call, send the result back:

```python
# First turn: model calls a tool
response1 = client.messages.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_tokens=1024,
    tools=tools,
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
)

tool_call = response1.content[0]  # type: tool_use

# Second turn: provide tool result
response2 = client.messages.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    max_tokens=1024,
    tools=tools,
    messages=[
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "content": response1.content},
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_call.id,
                    "content": "The weather in Paris is 18°C and sunny.",
                }
            ],
        },
    ],
)
print(response2.content[0].text)
```

---

## Token Counting

Count tokens for a request without generating a response:

```python
count = client.messages.count_tokens(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "user", "content": "What is the capital of France?"}
    ],
)
print(f"Input tokens: {count.input_tokens}")
```

### `POST /v1/messages/count_tokens`

**Request Body**

| Field | Type | Description |
|-------|------|-------------|
| `model` | `string` | Model identifier |
| `messages` | `array` | Messages to count |
| `system` | `string \| array \| null` | System prompt |
| `tools` | `array \| null` | Tool definitions |
| `tool_choice` | `object \| null` | Tool choice |

**Response**

```json
{
  "input_tokens": 42
}
```

---

## Error Handling

The Anthropic API returns errors in Anthropic format (not OpenAI format):

```json
{
  "type": "error",
  "error": {
    "type": "invalid_request_error",
    "message": "max_tokens must be positive"
  }
}
```

```python
import anthropic

try:
    message = client.messages.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        max_tokens=-1,  # Invalid
        messages=[{"role": "user", "content": "Hello"}],
    )
except anthropic.BadRequestError as e:
    print(f"Error: {e.message}")
```

---

## Differences from OpenAI Chat Completions

| Feature | Anthropic `/v1/messages` | OpenAI `/v1/chat/completions` |
|---------|--------------------------|-------------------------------|
| System prompt | Top-level `system` field | `{"role": "system", ...}` message |
| Max tokens | Required `max_tokens` field | Optional `max_completion_tokens` |
| Tool schema | `input_schema` (JSON Schema) | `parameters` (JSON Schema) |
| Tool choice | `{"type": "auto/any/tool/none"}` | `"auto"/"none"/"required"` or named |
| Stop reason | `stop_reason` | `finish_reason` |
| Usage | `input_tokens`, `output_tokens` | `prompt_tokens`, `completion_tokens` |
| Content | List of typed content blocks | String or list |
| Streaming | Anthropic SSE events | OpenAI SSE chunks |

---

## Authentication

The Anthropic API uses the same authentication as the OpenAI API. When `--api-key` is set, include it as a Bearer token:

```python
client = anthropic.Anthropic(
    base_url="http://localhost:8000",
    api_key="my-secret-key",
)
```

Or via the `ANTHROPIC_API_KEY` environment variable:

```bash
export ANTHROPIC_API_KEY="my-secret-key"
```

---

## See Also

- [Chat Completions](chat_completions.md) — OpenAI-compatible chat endpoint
- [Streaming Guide](streaming.md) — SSE streaming patterns
- [API Reference](api_reference.md) — Full endpoint reference
- [OpenAI-Compatible Server](openai_compatible_server.md) — Server configuration
