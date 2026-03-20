# gRPC Server

vLLM provides a gRPC server for high-performance, low-latency inference using the Protocol Buffers binary format. The gRPC server is ideal for internal microservices where you control both the client and server, and need maximum throughput with minimal overhead.

---

## Overview

The gRPC server implements the `VllmEngine` service with the following RPCs:

| RPC | Type | Description |
|-----|------|-------------|
| `Generate` | Server-streaming | Text generation (streaming) |
| `Embed` | Unary | Embeddings (planned) |
| `HealthCheck` | Unary | Server health probe |
| `Abort` | Unary | Cancel in-flight requests |
| `GetModelInfo` | Unary | Model metadata |
| `GetServerInfo` | Unary | Server state and uptime |

---

## Quick Start

### Start the gRPC Server

```bash
python -m vllm.entrypoints.grpc_server \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 50051
```

### Generate Text

```python
import grpc
import asyncio
from vllm.grpc import vllm_engine_pb2, vllm_engine_pb2_grpc

async def generate():
    async with grpc.aio.insecure_channel("localhost:50051") as channel:
        stub = vllm_engine_pb2_grpc.VllmEngineStub(channel)

        request = vllm_engine_pb2.GenerateRequest(
            request_id="req-001",
            text="The capital of France is",
            stream=True,
            sampling_params=vllm_engine_pb2.SamplingParams(
                temperature=0.0,
                max_tokens=20,
            ),
        )

        async for response in stub.Generate(request):
            if response.HasField("chunk"):
                print(response.chunk.text, end="", flush=True)
            if response.HasField("complete"):
                print()
                print(f"Finish reason: {response.complete.finish_reason}")

asyncio.run(generate())
```

---

## Server Configuration

### Command-Line Arguments

```bash
python -m vllm.entrypoints.grpc_server \
  --model <model_path> \
  --host 0.0.0.0 \
  --port 50051 \
  [engine args...]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | **required** | Model path or HuggingFace ID |
| `--host` | `0.0.0.0` | Host to bind the gRPC server |
| `--port` | `50051` | Port to bind the gRPC server |
| `--disable-log-stats-server` | `false` | Disable server-side stats logging |

All [engine arguments](../configuration/engine_args.md) (e.g., `--tensor-parallel-size`, `--dtype`, `--max-model-len`) are also supported.

### Example with Engine Args

```bash
python -m vllm.entrypoints.grpc_server \
  --model meta-llama/Llama-3.1-70B-Instruct \
  --host 0.0.0.0 \
  --port 50051 \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --max-model-len 8192
```

---

## Service Definition

The `VllmEngine` service is defined in `vllm/grpc/vllm_engine.proto`. Key message types:

### `GenerateRequest`

```protobuf
message GenerateRequest {
  string request_id = 1;
  oneof input {
    string text = 2;           // Text prompt
    TokenizedInput tokenized = 3;  // Pre-tokenized input
  }
  SamplingParams sampling_params = 4;
  bool stream = 5;
}
```

### `SamplingParams`

```protobuf
message SamplingParams {
  float temperature = 1;
  float top_p = 2;
  int32 top_k = 3;
  float min_p = 4;
  float frequency_penalty = 5;
  float presence_penalty = 6;
  int32 max_tokens = 7;
  repeated string stop = 8;
  repeated int32 stop_token_ids = 9;
  bool skip_special_tokens = 10;
  bool include_stop_str_in_output = 11;
  bool ignore_eos = 12;
  int32 min_tokens = 13;
  float repetition_penalty = 14;
  int64 seed = 15;

  // Structured outputs (oneof)
  oneof constraint {
    string json_schema = 20;
    string regex = 21;
    string grammar = 22;
    string structural_tag = 23;
    bool json_object = 24;
    ChoiceConstraint choice = 25;
  }
}
```

### `GenerateResponse`

```protobuf
message GenerateResponse {
  oneof response {
    GenerateChunk chunk = 1;      // Streaming chunk
    GenerateComplete complete = 2; // Final response
  }
}

message GenerateChunk {
  string text = 1;
  repeated int32 token_ids = 2;
}

message GenerateComplete {
  string text = 1;
  string finish_reason = 2;
  repeated int32 token_ids = 3;
  int32 prompt_tokens = 4;
  int32 completion_tokens = 5;
}
```

---

## RPCs in Detail

### Generate

Streaming text generation. The server yields `GenerateResponse` messages:

- **Streaming mode** (`stream=True`): Yields `chunk` responses as tokens are generated, followed by a `complete` response when finished.
- **Non-streaming mode** (`stream=False`): Yields only a single `complete` response.

```python
# Streaming generation
request = vllm_engine_pb2.GenerateRequest(
    request_id="req-001",
    text="Once upon a time,",
    stream=True,
    sampling_params=vllm_engine_pb2.SamplingParams(
        temperature=0.8,
        max_tokens=100,
        stop=[".", "!"],
    ),
)

async for response in stub.Generate(request):
    if response.HasField("chunk"):
        print(response.chunk.text, end="", flush=True)
    elif response.HasField("complete"):
        print(f"\nFinish: {response.complete.finish_reason}")
        print(f"Tokens: {response.complete.prompt_tokens} + {response.complete.completion_tokens}")
```

### Pre-tokenized Input

Pass token IDs directly to skip tokenization:

```python
request = vllm_engine_pb2.GenerateRequest(
    request_id="req-002",
    tokenized=vllm_engine_pb2.TokenizedInput(
        input_ids=[128000, 791, 6864, 315, 9822, 374],
        original_text="The capital of France is",  # Optional
    ),
    stream=False,
    sampling_params=vllm_engine_pb2.SamplingParams(
        temperature=0.0,
        max_tokens=5,
    ),
)

async for response in stub.Generate(request):
    if response.HasField("complete"):
        print(response.complete.text)
```

### HealthCheck

```python
response = await stub.HealthCheck(
    vllm_engine_pb2.HealthCheckRequest()
)
print(f"Healthy: {response.healthy}")
print(f"Message: {response.message}")
```

### Abort

Cancel one or more in-flight requests:

```python
await stub.Abort(
    vllm_engine_pb2.AbortRequest(
        request_ids=["req-001", "req-002"]
    )
)
```

### GetModelInfo

```python
info = await stub.GetModelInfo(
    vllm_engine_pb2.GetModelInfoRequest()
)
print(f"Model: {info.model_path}")
print(f"Max context: {info.max_context_length}")
print(f"Vocab size: {info.vocab_size}")
print(f"Vision: {info.supports_vision}")
print(f"Generation: {info.is_generation}")
```

### GetServerInfo

```python
info = await stub.GetServerInfo(
    vllm_engine_pb2.GetServerInfoRequest()
)
print(f"Active requests: {info.active_requests}")
print(f"Uptime: {info.uptime_seconds:.1f}s")
print(f"Server type: {info.server_type}")
```

---

## Structured Outputs

The gRPC server supports structured output constraints via `SamplingParams`:

```python
# JSON Schema constraint
request = vllm_engine_pb2.GenerateRequest(
    request_id="req-003",
    text='Extract: {"name": "',
    stream=False,
    sampling_params=vllm_engine_pb2.SamplingParams(
        max_tokens=50,
        json_schema='{"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}',
    ),
)

# Regex constraint
request = vllm_engine_pb2.GenerateRequest(
    request_id="req-004",
    text="Phone number: ",
    stream=False,
    sampling_params=vllm_engine_pb2.SamplingParams(
        max_tokens=20,
        regex=r"\d{3}-\d{3}-\d{4}",
    ),
)

# Choice constraint
request = vllm_engine_pb2.GenerateRequest(
    request_id="req-005",
    text="Sentiment: ",
    stream=False,
    sampling_params=vllm_engine_pb2.SamplingParams(
        max_tokens=10,
        choice=vllm_engine_pb2.ChoiceConstraint(
            choices=["positive", "negative", "neutral"]
        ),
    ),
)
```

---

## gRPC Reflection

The server enables [gRPC reflection](https://grpc.github.io/grpc/core/md_doc_server-reflection.html), allowing tools like `grpcurl` to discover and call services without the `.proto` file:

```bash
# List services
grpcurl -plaintext localhost:50051 list

# Describe the VllmEngine service
grpcurl -plaintext localhost:50051 describe vllm.VllmEngine

# Call HealthCheck
grpcurl -plaintext localhost:50051 vllm.VllmEngine/HealthCheck

# Call GetModelInfo
grpcurl -plaintext localhost:50051 vllm.VllmEngine/GetModelInfo

# Generate text
grpcurl -plaintext -d '{
  "request_id": "test-001",
  "text": "Hello, world!",
  "stream": false,
  "sampling_params": {"temperature": 0.0, "max_tokens": 10}
}' localhost:50051 vllm.VllmEngine/Generate
```

---

## Error Handling

The gRPC server maps errors to standard gRPC status codes:

| gRPC Status | Cause |
|-------------|-------|
| `INVALID_ARGUMENT` | Invalid request parameters (equivalent to HTTP 400) |
| `INTERNAL` | Server-side error (equivalent to HTTP 500) |
| `UNIMPLEMENTED` | RPC not yet implemented (e.g., `Embed`) |

```python
import grpc

try:
    async for response in stub.Generate(request):
        ...
except grpc.aio.AioRpcError as e:
    if e.code() == grpc.StatusCode.INVALID_ARGUMENT:
        print(f"Invalid request: {e.details()}")
    elif e.code() == grpc.StatusCode.INTERNAL:
        print(f"Server error: {e.details()}")
```

---

## Server Options

The gRPC server is configured with unlimited message sizes:

```python
options = [
    ("grpc.max_send_message_length", -1),    # Unlimited
    ("grpc.max_receive_message_length", -1), # Unlimited
]
```

---

## Graceful Shutdown

The server handles `SIGTERM` and `SIGINT` signals with a 5-second grace period:

```bash
# Send SIGTERM for graceful shutdown
kill -TERM <pid>

# Or use Ctrl+C
```

The shutdown sequence:
1. Stop accepting new connections
2. Wait up to 5 seconds for in-flight requests to complete
3. Shut down the AsyncLLM engine

---

## Comparison: gRPC vs. HTTP

| Feature | gRPC Server | HTTP Server |
|---------|-------------|-------------|
| Protocol | HTTP/2 + Protobuf | HTTP/1.1 or HTTP/2 + JSON |
| Serialization | Binary (Protobuf) | Text (JSON) |
| Streaming | Native bidirectional | SSE (server-to-client) |
| Schema | Strongly typed `.proto` | OpenAPI/JSON Schema |
| Tooling | `grpcurl`, generated clients | `curl`, any HTTP client |
| Overhead | Lower | Higher |
| Compatibility | Internal services | OpenAI-compatible clients |
| Reflection | Built-in | OpenAPI docs |

---

## When to Use gRPC

**Use gRPC when:**
- Building internal microservices where you control both client and server
- Latency and throughput are critical
- You want strongly-typed, schema-validated requests
- You need efficient binary serialization for large payloads (e.g., token arrays)

**Use HTTP when:**
- You need OpenAI API compatibility
- Clients are external or use the OpenAI SDK
- You want easy debugging with `curl`
- You need browser-based clients

---

## See Also

- [OpenAI-Compatible Server](openai_compatible_server.md) — HTTP server setup
- [API Reference](api_reference.md) — HTTP endpoint reference
- [Streaming Guide](streaming.md) — Streaming patterns
