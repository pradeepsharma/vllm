# Streaming Responses

vLLM supports streaming for all generation endpoints, allowing clients to receive tokens as they are generated rather than waiting for the complete response. This dramatically improves perceived latency for interactive applications.

---

## How Streaming Works

vLLM uses **Server-Sent Events (SSE)** for HTTP streaming — a standard web protocol where the server pushes data to the client over a persistent HTTP connection. Each event is a line starting with `data:` followed by a JSON payload.

```
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache

data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":"Hello"},"index":0}]}

data: {"id":"chatcmpl-abc","choices":[{"delta":{"content":" world"},"index":0}]}

data: {"id":"chatcmpl-abc","choices":[{"delta":{},"finish_reason":"stop","index":0}]}

data: [DONE]
```

The stream ends with the sentinel `data: [DONE]`.

---

## Chat Completions Streaming

### Basic Streaming

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Tell me a story."}],
    stream=True,
)

for chunk in stream:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()  # Final newline
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
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
    if chunk.usage:
        print(f"\nUsage: {chunk.usage.total_tokens} tokens")
```

### Streaming Chunk Format

Each chunk has this structure:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion.chunk",
  "created": 1710000000,
  "model": "meta-llama/Llama-3.1-8B-Instruct",
  "choices": [
    {
      "index": 0,
      "delta": {
        "role": "assistant",   // Only in first chunk
        "content": "Hello"     // Incremental text
      },
      "finish_reason": null    // "stop", "length", etc. in last chunk
    }
  ],
  "usage": null               // Only present if stream_options.include_usage=true
}
```

### First Chunk

The first chunk contains the `role` field:

```json
{
  "choices": [{"delta": {"role": "assistant", "content": ""}, "finish_reason": null}]
}
```

### Last Chunk

The last chunk has `finish_reason` set and empty `content`:

```json
{
  "choices": [{"delta": {}, "finish_reason": "stop"}]
}
```

### Finish Reasons

| Value | Description |
|-------|-------------|
| `stop` | Natural end of generation (EOS or stop sequence) |
| `length` | Hit `max_completion_tokens` limit |
| `tool_calls` | Model wants to call a tool |
| `content_filter` | Content filtered |

---

## Text Completions Streaming

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

### Completion Chunk Format

```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1710000000,
  "model": "meta-llama/Llama-3.1-8B",
  "choices": [
    {
      "index": 0,
      "text": " there",
      "logprobs": null,
      "finish_reason": null
    }
  ]
}
```

---

## Responses API Streaming

The Responses API uses typed SSE events rather than generic chunks:

```python
stream = client.responses.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    input="Tell me a story.",
    stream=True,
)

for event in stream:
    event_type = event.type
    if event_type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
    elif event_type == "response.completed":
        print(f"\nDone. Tokens: {event.response.usage.total_tokens}")
```

### Responses API Event Types

| Event | Description |
|-------|-------------|
| `response.created` | Response object initialized |
| `response.in_progress` | Generation started |
| `response.output_item.added` | New output item (e.g., message) started |
| `response.content_part.added` | New content part started |
| `response.output_text.delta` | Incremental text |
| `response.output_text.done` | Text part complete |
| `response.content_part.done` | Content part complete |
| `response.output_item.done` | Output item complete |
| `response.completed` | Full response complete |
| `response.reasoning_text.delta` | Reasoning text delta |
| `response.reasoning_text.done` | Reasoning text complete |

### SSE Event Format for Responses API

Each event includes the event type in the SSE `event:` field:

```
event: response.output_text.delta
data: {"type":"response.output_text.delta","delta":"Hello","output_index":0,"content_index":0,"sequence_number":3}

event: response.completed
data: {"type":"response.completed","response":{...},"sequence_number":10}
```

---

## Audio Transcription Streaming

```python
with open("audio.mp3", "rb") as f:
    stream = client.audio.transcriptions.create(
        model="openai/whisper-large-v3-turbo",
        file=f,
        stream=True,
    )

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

---

## Raw HTTP Streaming

For languages without an OpenAI SDK, use raw HTTP with SSE parsing:

=== "Python (httpx)"

    ```python
    import httpx
    import json

    with httpx.Client() as client:
        with client.stream(
            "POST",
            "http://localhost:8000/v1/chat/completions",
            headers={"Authorization": "Bearer EMPTY"},
            json={
                "model": "meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": "Hello!"}],
                "stream": True,
            },
        ) as response:
            for line in response.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    content = chunk["choices"][0]["delta"].get("content", "")
                    if content:
                        print(content, end="", flush=True)
    ```

=== "curl"

    ```bash
    curl http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer EMPTY" \
      -d '{
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": true
      }'
    ```

=== "JavaScript (fetch)"

    ```javascript
    const response = await fetch("http://localhost:8000/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",
      },
      body: JSON.stringify({
        model: "meta-llama/Llama-3.1-8B-Instruct",
        messages: [{ role: "user", content: "Hello!" }],
        stream: true,
      }),
    });

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value);
      for (const line of text.split("\n")) {
        if (line.startsWith("data: ") && line !== "data: [DONE]") {
          const chunk = JSON.parse(line.slice(6));
          const content = chunk.choices[0]?.delta?.content || "";
          process.stdout.write(content);
        }
      }
    }
    ```

---

## Streaming with Tool Calls

When the model calls a tool, the tool call is streamed incrementally:

```python
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=[...],
    tool_choice="auto",
    stream=True,
)

tool_calls = {}
for chunk in stream:
    delta = chunk.choices[0].delta

    # Accumulate tool call arguments
    if delta.tool_calls:
        for tc in delta.tool_calls:
            idx = tc.index
            if idx not in tool_calls:
                tool_calls[idx] = {"id": tc.id, "name": tc.function.name, "args": ""}
            if tc.function.arguments:
                tool_calls[idx]["args"] += tc.function.arguments

    # Regular content
    if delta.content:
        print(delta.content, end="", flush=True)

# Process accumulated tool calls
for tc in tool_calls.values():
    import json
    args = json.loads(tc["args"])
    print(f"\nTool call: {tc['name']}({args})")
```

---

## Streaming with Log Probabilities

```python
stream = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Say hello."}],
    stream=True,
    logprobs=True,
    top_logprobs=3,
)

for chunk in stream:
    choice = chunk.choices[0]
    if choice.delta.content:
        print(choice.delta.content, end="")
    if choice.logprobs and choice.logprobs.content:
        for token_info in choice.logprobs.content:
            print(f"\n  Token: {token_info.token!r}, logprob: {token_info.logprob:.3f}")
```

---

## Cancellation

Clients can cancel streaming requests by closing the connection. vLLM detects the disconnection and stops generation:

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

async def generate_with_timeout():
    stream = await client.chat.completions.create(
        model="meta-llama/Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": "Write a very long essay."}],
        stream=True,
    )

    try:
        async with asyncio.timeout(5.0):  # Cancel after 5 seconds
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
    except asyncio.TimeoutError:
        print("\n[Cancelled after timeout]")
        await stream.close()

asyncio.run(generate_with_timeout())
```

---

## Performance Considerations

### Time to First Token (TTFT)

Streaming reduces perceived latency by delivering the first token as soon as it is generated, rather than waiting for the complete response. For interactive applications, TTFT is often more important than total generation time.

### Continuous Batching

vLLM uses continuous batching — new requests are added to the batch as soon as a slot is available, without waiting for the current batch to complete. This maximizes GPU utilization while maintaining low latency for streaming clients.

### Keep-Alive

The HTTP server uses keep-alive connections by default. The timeout is controlled by `VLLM_HTTP_TIMEOUT_KEEP_ALIVE` (default: 5 seconds).

### Buffering

Avoid buffering the SSE stream in proxies or load balancers. Configure nginx with:

```nginx
proxy_buffering off;
proxy_cache off;
proxy_set_header Connection '';
proxy_http_version 1.1;
chunked_transfer_encoding on;
```

---

## WebSocket Streaming (Realtime API)

For audio transcription, vLLM also supports WebSocket-based streaming. See the [Realtime API](realtime_api.md) guide for details.

---

## See Also

- [Chat Completions](chat_completions.md) — Chat streaming details
- [Text Completions](completions.md) — Completion streaming
- [Responses API](responses_api.md) — Typed SSE events
- [Realtime API](realtime_api.md) — WebSocket streaming
- [API Reference](api_reference.md) — Full endpoint reference
