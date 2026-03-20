# Serving Overview

vLLM provides multiple serving interfaces to fit different deployment scenarios — from a drop-in OpenAI-compatible HTTP server to a high-performance gRPC endpoint and offline batch processing. This section covers everything you need to deploy and operate vLLM in production.

---

## Serving Interfaces

| Interface | Protocol | Best For |
|-----------|----------|----------|
| [OpenAI-Compatible Server](openai_compatible_server.md) | HTTP/REST | Drop-in replacement for OpenAI API clients |
| [gRPC Server](grpc_server.md) | gRPC / Protobuf | Low-latency, high-throughput internal services |
| [Anthropic-Compatible API](anthropic_api.md) | HTTP/REST | Claude API clients and Anthropic SDK users |
| [Batch Inference](batch_inference.md) | Offline / JSONL | Large-scale offline processing |
| [AWS SageMaker](sagemaker.md) | HTTP (SageMaker protocol) | Managed ML deployment on AWS |

---

## API Endpoints at a Glance

### Generation APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /v1/chat/completions` | HTTP | Chat completions (OpenAI-compatible) |
| `POST /v1/completions` | HTTP | Text completions (OpenAI-compatible) |
| `POST /v1/responses` | HTTP | Stateful responses API |
| `GET /v1/responses/{id}` | HTTP | Retrieve a stored response |
| `POST /v1/responses/{id}/cancel` | HTTP | Cancel an in-flight response |
| `WS /v1/realtime` | WebSocket | Real-time audio transcription |

### Audio APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /v1/audio/transcriptions` | HTTP | Speech-to-text transcription |
| `POST /v1/audio/translations` | HTTP | Audio translation to English |

### Pooling / Embedding APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /v1/embeddings` | HTTP | Text embeddings |
| `POST /pooling` | HTTP | Raw pooling outputs |
| `POST /classify` | HTTP | Sequence classification |
| `POST /score` | HTTP | Cross-encoder scoring |
| `POST /rerank` | HTTP | Document re-ranking |

### Utility APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| `GET /v1/models` | HTTP | List available models |
| `GET /health` | HTTP | Server health check |
| `POST /tokenize` | HTTP | Tokenize text |
| `POST /detokenize` | HTTP | Detokenize token IDs |

### Anthropic API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `POST /v1/messages` | HTTP | Anthropic Messages API |
| `POST /v1/messages/count_tokens` | HTTP | Count tokens for a request |

---

## Quick Start

### Start the Server

```bash
# Install vLLM
pip install vllm

# Start the OpenAI-compatible server
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key my-secret-key
```

### Send a Request

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="my-secret-key",
)

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "Hello, world!"}],
)
print(response.choices[0].message.content)
```

---

## Endpoint Deep Dives

- **[Chat Completions](chat_completions.md)** — Multi-turn conversations, tool calling, vision, structured outputs
- **[Text Completions](completions.md)** — Raw text completion with logprobs, beam search, and more
- **[Embeddings](embeddings.md)** — Dense vector representations for semantic search and RAG
- **[Responses API](responses_api.md)** — Stateful, multi-turn responses with tool use and memory
- **[Realtime API](realtime_api.md)** — WebSocket streaming audio transcription
- **[Speech-to-Text](speech_to_text.md)** — Batch audio transcription and translation
- **[Anthropic API](anthropic_api.md)** — Claude-compatible messages endpoint
- **[gRPC Server](grpc_server.md)** — High-performance binary protocol server

---

## Cross-Cutting Topics

- **[Streaming Responses](streaming.md)** — Server-Sent Events (SSE) and WebSocket streaming
- **[Batch Inference](batch_inference.md)** — Offline JSONL batch processing
- **[API Reference](api_reference.md)** — Complete endpoint reference with schemas
- **[AWS SageMaker](sagemaker.md)** — Managed deployment on SageMaker

---

## Deployment Guides

For scaling and distributed deployments, see:

- [Data Parallel Deployment](data_parallel_deployment.md)
- [Expert Parallel Deployment](expert_parallel_deployment.md)
- [Context Parallel Deployment](context_parallel_deployment.md)
- [Parallelism & Scaling](parallelism_scaling.md)
