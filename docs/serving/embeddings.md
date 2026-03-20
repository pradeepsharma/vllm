# Embeddings

The `POST /v1/embeddings` endpoint generates dense vector representations of text (and optionally images or audio) using embedding models. It is compatible with the [OpenAI Embeddings API](https://platform.openai.com/docs/api-reference/embeddings).

---

## Quick Start

### Start the Server

```bash
vllm serve BAAI/bge-base-en-v1.5 \
  --host 0.0.0.0 \
  --port 8000
```

### Generate Embeddings

=== "Python (OpenAI SDK)"

    ```python
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    response = client.embeddings.create(
        model="BAAI/bge-base-en-v1.5",
        input="The quick brown fox jumps over the lazy dog.",
    )

    embedding = response.data[0].embedding
    print(f"Embedding dimension: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")
    ```

=== "curl"

    ```bash
    curl http://localhost:8000/v1/embeddings \
      -H "Content-Type: application/json" \
      -d '{
        "model": "BAAI/bge-base-en-v1.5",
        "input": "The quick brown fox jumps over the lazy dog."
      }'
    ```

---

## Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input` | `string \| array` | **required** | Text(s) to embed |
| `model` | `string` | **required** | Embedding model identifier |
| `encoding_format` | `string` | `"float"` | `"float"` or `"base64"` |
| `dimensions` | `int \| null` | `null` | Output dimension (if model supports Matryoshka) |

### vLLM Extensions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `add_special_tokens` | `bool` | `true` | Add BOS/EOS tokens |
| `truncate_prompt_tokens` | `int \| null` | `null` | Truncate to N tokens |
| `priority` | `int` | `0` | Request priority |
| `request_id` | `string` | auto | Custom request identifier |
| `use_activation` | `bool` | `false` | Use activation layer output |

---

## Response Format

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.0023, -0.0094, 0.0012, ...]
    }
  ],
  "model": "BAAI/bge-base-en-v1.5",
  "usage": {
    "prompt_tokens": 9,
    "total_tokens": 9,
    "completion_tokens": 0
  }
}
```

---

## Batch Embeddings

Embed multiple texts in a single request:

```python
texts = [
    "What is machine learning?",
    "How does neural network training work?",
    "Explain gradient descent.",
]

response = client.embeddings.create(
    model="BAAI/bge-base-en-v1.5",
    input=texts,
)

embeddings = [item.embedding for item in response.data]
print(f"Generated {len(embeddings)} embeddings")
```

---

## Semantic Similarity

Use embeddings to compute similarity between texts:

```python
import numpy as np
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        model="BAAI/bge-base-en-v1.5",
        input=text,
    )
    return response.data[0].embedding

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

query = "What is the capital of France?"
documents = [
    "Paris is the capital and largest city of France.",
    "The Eiffel Tower is located in Paris.",
    "Berlin is the capital of Germany.",
]

query_emb = get_embedding(query)
doc_embs = [get_embedding(doc) for doc in documents]

for doc, emb in zip(documents, doc_embs):
    sim = cosine_similarity(query_emb, emb)
    print(f"Similarity: {sim:.3f} | {doc[:50]}...")
```

---

## Base64 Encoding

For efficient transfer of large embedding batches:

```python
import base64
import numpy as np

response = client.embeddings.create(
    model="BAAI/bge-base-en-v1.5",
    input=["Hello, world!"],
    encoding_format="base64",
)

# Decode base64 to float32 array
b64_data = response.data[0].embedding
embedding = np.frombuffer(base64.b64decode(b64_data), dtype=np.float32)
print(f"Embedding shape: {embedding.shape}")
```

---

## Matryoshka Embeddings (Dimension Reduction)

Some models (e.g., `nomic-ai/nomic-embed-text-v1.5`) support variable output dimensions:

```python
# Full dimension
response_full = client.embeddings.create(
    model="nomic-ai/nomic-embed-text-v1.5",
    input="Hello, world!",
)
print(f"Full dim: {len(response_full.data[0].embedding)}")  # 768

# Reduced dimension
response_small = client.embeddings.create(
    model="nomic-ai/nomic-embed-text-v1.5",
    input="Hello, world!",
    dimensions=128,
)
print(f"Reduced dim: {len(response_small.data[0].embedding)}")  # 128
```

---

## Chat-Format Embeddings

If the embedding model has a chat template, you can pass messages instead of plain text. This is useful for models that expect instruction-formatted input:

```python
from openai import OpenAI
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessageParam
from openai.types.create_embedding_response import CreateEmbeddingResponse
from typing import Union, Literal

def create_chat_embeddings(
    client: OpenAI,
    *,
    messages: list[ChatCompletionMessageParam],
    model: str,
    encoding_format: Union[Literal["base64", "float"], NotGiven] = NOT_GIVEN,
) -> CreateEmbeddingResponse:
    return client.post(
        "/embeddings",
        cast_to=CreateEmbeddingResponse,
        body={
            "messages": messages,
            "model": model,
            "encoding_format": encoding_format,
        },
    )

# Use with instruction-following embedding models
response = create_chat_embeddings(
    client,
    model="intfloat/e5-mistral-7b-instruct",
    messages=[
        {
            "role": "user",
            "content": "Represent this sentence for searching: What is machine learning?",
        }
    ],
)
```

---

## Multimodal Embeddings

For vision-language embedding models, pass images alongside text.

### VLM2Vec

```bash
vllm serve TIGER-Lab/VLM2Vec-Full \
  --runner pooling \
  --trust-remote-code \
  --max-model-len 4096 \
  --chat-template examples/pooling/embed/template/vlm2vec_phi3v.jinja
```

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/embeddings",
    json={
        "model": "TIGER-Lab/VLM2Vec-Full",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/image.jpg"},
                    },
                    {"type": "text", "text": "Represent the given image."},
                ],
            }
        ],
        "encoding_format": "float",
    },
)
embedding = response.json()["data"][0]["embedding"]
```

---

## Pooling API

The `/pooling` endpoint returns raw pooling outputs, which may be arbitrary nested tensors (not just 1-D float arrays). Use this for models that produce structured outputs:

```python
import requests

response = requests.post(
    "http://localhost:8000/pooling",
    json={
        "model": "my-pooling-model",
        "input": "Hello, world!",
    },
)
data = response.json()["data"][0]["embedding"]
# data may be a nested list
```

---

## Classification API

For sequence classification models (e.g., sentiment analysis, topic classification):

```bash
vllm serve jason9693/Qwen2.5-1.5B-apeach
```

```python
import requests

response = requests.post(
    "http://localhost:8000/classify",
    json={
        "model": "jason9693/Qwen2.5-1.5B-apeach",
        "input": ["I love this product!", "This is terrible."],
    },
)

for item in response.json()["data"]:
    print(f"Label: {item['label']}, Probs: {item['probs']}")
```

---

## Score / Re-rank API

For cross-encoder models that score query-document pairs:

```python
import requests

# Score a single pair
response = requests.post(
    "http://localhost:8000/score",
    json={
        "model": "BAAI/bge-reranker-v2-m3",
        "queries": "What is the capital of France?",
        "documents": "Paris is the capital of France.",
    },
)
score = response.json()["data"][0]["score"]
print(f"Relevance score: {score:.3f}")

# Re-rank multiple documents
response = requests.post(
    "http://localhost:8000/rerank",
    json={
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "What is the capital of France?",
        "documents": [
            "Berlin is the capital of Germany.",
            "Paris is the capital of France.",
            "Rome is the capital of Italy.",
        ],
        "top_n": 2,
    },
)
for result in response.json()["results"]:
    print(f"Score: {result['relevance_score']:.3f} | {result['document']['text'][:50]}")
```

---

## Serving Embedding Models

### Basic Embedding Model

```bash
vllm serve BAAI/bge-base-en-v1.5
```

### Instruction-Following Embedding Model

```bash
vllm serve intfloat/e5-mistral-7b-instruct \
  --max-model-len 4096
```

### Cross-Encoder Re-ranker

```bash
vllm serve BAAI/bge-reranker-v2-m3
```

### Vision-Language Embedding Model

```bash
vllm serve TIGER-Lab/VLM2Vec-Full \
  --runner pooling \
  --trust-remote-code \
  --max-model-len 4096 \
  --chat-template examples/pooling/embed/template/vlm2vec_phi3v.jinja
```

---

## Performance Tips

1. **Batch inputs**: Pass multiple texts in a single request — vLLM processes them in parallel.
2. **Base64 encoding**: Use `encoding_format="base64"` for large batches to reduce JSON serialization overhead.
3. **Install orjson**: `pip install orjson` speeds up embedding response serialization significantly.
4. **Truncation**: Set `truncate_prompt_tokens` to avoid errors on long inputs.
5. **Dimension reduction**: Use Matryoshka models with `dimensions` parameter to trade accuracy for speed.

---

## See Also

- [API Reference](api_reference.md) — Full schema documentation
- [OpenAI-Compatible Server](openai_compatible_server.md) — Server configuration
- [Batch Inference](batch_inference.md) — Offline batch embedding
