# Responses API

The `POST /v1/responses` endpoint implements the [OpenAI Responses API](https://platform.openai.com/docs/api-reference/responses) — a stateful, multi-turn interface that stores conversation history server-side and supports chaining responses together.

!!! note "When to Use Responses vs. Chat Completions"
    - Use **Chat Completions** when you manage conversation history client-side (most common).
    - Use **Responses API** when you want the server to manage state, or when building agents that chain multiple tool calls together.

---

## Quick Start

### Start the Server

The Responses API is enabled automatically for generation models:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

### Create a Response

=== "Python (OpenAI SDK)"

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    response = client.responses.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        input="What is the capital of France?",
    )

    print(response.output_text)
    # → "The capital of France is Paris."
    print(f"Response ID: {response.id}")
    ```

=== "curl"

    ```bash
    curl http://localhost:8000/v1/responses \
      -H "Content-Type: application/json" \
      -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "input": "What is the capital of France?"
      }'
    ```

---

## Request Parameters

### Standard OpenAI Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `string \| array` | **required** | Input text or list of message items |
| `model` | `string \| null` | `null` | Model identifier |
| `instructions` | `string \| null` | `null` | System-level instructions (like a system message) |
| `max_output_tokens` | `int \| null` | `null` | Maximum tokens to generate |
| `max_tool_calls` | `int \| null` | `null` | Maximum tool calls per response |
| `previous_response_id` | `string \| null` | `null` | Chain from a previous response |
| `stream` | `bool \| null` | `false` | Enable SSE streaming |
| `temperature` | `float \| null` | `null` | Sampling temperature |
| `top_p` | `float \| null` | `null` | Nucleus sampling |
| `tools` | `array` | `[]` | Tool definitions |
| `tool_choice` | `string \| object` | `"auto"` | Tool selection strategy |
| `reasoning` | `object \| null` | `null` | Reasoning configuration |
| `store` | `bool \| null` | `true` | Store response for later retrieval |
| `truncation` | `string \| null` | `"disabled"` | Context truncation: `"auto"` or `"disabled"` |
| `parallel_tool_calls` | `bool \| null` | `true` | Allow multiple tool calls |
| `metadata` | `object \| null` | `null` | Arbitrary metadata |
| `background` | `bool \| null` | `false` | Run in background |
| `include` | `array \| null` | `null` | Additional fields to include in response |
| `service_tier` | `string` | `"auto"` | Service tier |
| `user` | `string \| null` | `null` | User identifier |

### vLLM Extensions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `request_id` | `string` | auto | Custom request identifier |
| `priority` | `int` | `0` | Request priority (lower = higher priority) |
| `cache_salt` | `string \| null` | `null` | Prefix cache salt for multi-tenant security |
| `top_k` | `int \| null` | `null` | Top-k sampling |
| `repetition_penalty` | `float \| null` | `null` | Repetition penalty |
| `seed` | `int \| null` | `null` | Random seed |
| `stop` | `string \| array \| null` | `[]` | Stop sequences |
| `ignore_eos` | `bool` | `false` | Ignore EOS token |
| `skip_special_tokens` | `bool` | `true` | Skip special tokens |
| `include_stop_str_in_output` | `bool` | `false` | Include stop string in output |
| `structured_outputs` | `object \| null` | `null` | Structured output constraints |
| `mm_processor_kwargs` | `object \| null` | `null` | Multimodal processor kwargs |
| `media_io_kwargs` | `object \| null` | `null` | Media IO connector kwargs |
| `enable_response_messages` | `bool` | `false` | Include messages in response object |
| `previous_input_messages` | `array \| null` | `null` | Previous messages in harmony format |
| `vllm_xargs` | `object \| null` | `null` | Custom extension parameters |

---

## Response Format

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
      "id": "msg_abc123",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "The capital of France is Paris."
        }
      ]
    }
  ],
  "usage": {
    "input_tokens": 10,
    "output_tokens": 8,
    "total_tokens": 18,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0}
  }
}
```

---

## Multi-Turn Conversations

Chain responses together using `previous_response_id`:

```python
# First turn
response1 = client.responses.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    input="My name is Alice.",
)
print(response1.output_text)

# Second turn — references the first
response2 = client.responses.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    input="What's my name?",
    previous_response_id=response1.id,
)
print(response2.output_text)
# → "Your name is Alice."
```

---

## Streaming

```python
stream = client.responses.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    input="Tell me a story about a robot.",
    stream=True,
)

for event in stream:
    if hasattr(event, "delta") and event.delta:
        print(event.delta, end="", flush=True)
```

### Streaming Events

The streaming response uses Server-Sent Events with typed events:

| Event Type | Description |
|------------|-------------|
| `response.created` | Response object created |
| `response.in_progress` | Generation in progress |
| `response.output_item.added` | New output item started |
| `response.content_part.added` | New content part started |
| `response.output_text.delta` | Incremental text delta |
| `response.output_text.done` | Text part complete |
| `response.content_part.done` | Content part complete |
| `response.output_item.done` | Output item complete |
| `response.completed` | Full response complete |
| `response.reasoning_text.delta` | Reasoning text delta |
| `response.reasoning_text.done` | Reasoning text complete |

---

## Tool Calling

```python
tools = [
    {
        "type": "function",
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
            },
            "required": ["location"],
        },
    }
]

response = client.responses.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    input="What's the weather in Paris?",
    tools=tools,
    tool_choice="auto",
)

# Check for tool calls
for item in response.output:
    if item.type == "function_call":
        print(f"Tool: {item.name}")
        print(f"Args: {item.arguments}")
```

---

## Retrieve a Stored Response

When `store=True` (the default), responses are stored and can be retrieved later:

```python
# Create and store a response
response = client.responses.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    input="Hello!",
    store=True,
)
response_id = response.id

# Retrieve it later
retrieved = client.responses.retrieve(response_id)
print(retrieved.output_text)
```

### Retrieve with Streaming

```python
stream = client.responses.retrieve(
    response_id,
    stream=True,
)
for event in stream:
    print(event)
```

---

## Cancel a Response

Cancel an in-flight response:

```python
# Start a long-running response
response = client.responses.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    input="Write a very long essay about the history of computing.",
    stream=True,
)

# Cancel it
client.responses.cancel(response.id)
```

---

## System Instructions

Use `instructions` as a system-level prompt:

```python
response = client.responses.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    instructions="You are a concise assistant. Answer in one sentence.",
    input="What is machine learning?",
)
```

---

## Reasoning Models

For models with chain-of-thought reasoning:

```python
response = client.responses.create(
    model="deepseek-ai/DeepSeek-R1",
    input="Solve: if x² + 5x + 6 = 0, find x.",
    reasoning={"effort": "high"},
)

# Access reasoning trace
for item in response.output:
    if item.type == "reasoning":
        print("Reasoning:", item.content[0].text)
    elif item.type == "message":
        print("Answer:", item.content[0].text)
```

---

## Context Truncation

When conversation history grows too long, enable automatic truncation:

```python
response = client.responses.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    input="Continue our conversation...",
    previous_response_id="resp_very_long_history",
    truncation="auto",  # Automatically truncate old context
)
```

---

## Structured Outputs

```python
from pydantic import BaseModel

class Answer(BaseModel):
    answer: str
    confidence: float

response = client.responses.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    input="What is 2+2?",
    text={"format": {"type": "json_schema", "schema": Answer.model_json_schema()}},
)
```

---

## Differences from Chat Completions

| Feature | Responses API | Chat Completions |
|---------|--------------|-----------------|
| State management | Server-side | Client-side |
| Response chaining | `previous_response_id` | Manual history |
| Response retrieval | `GET /v1/responses/{id}` | Not supported |
| Response cancellation | `POST /v1/responses/{id}/cancel` | Not supported |
| Streaming events | Typed SSE events | Generic SSE chunks |
| Tool calling | Supported | Supported |
| Background execution | `background=true` | Not supported |

---

## See Also

- [Chat Completions](chat_completions.md) — Stateless multi-turn conversations
- [Streaming Guide](streaming.md) — SSE streaming deep dive
- [API Reference](api_reference.md) — Full schema documentation
