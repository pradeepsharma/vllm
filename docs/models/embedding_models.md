# Embedding Models

vLLM supports a wide range of embedding and pooling models for semantic search, retrieval-augmented generation (RAG), reranking, and reward modeling. This guide covers how to use embedding models, configure pooling strategies, and work with different embedding types.

---

## Overview

Embedding models in vLLM produce dense vector representations of text (and optionally images) rather than generating new tokens. They are used for:

- **Semantic search** — finding similar documents by embedding similarity
- **Retrieval-Augmented Generation (RAG)** — retrieving relevant context for LLM generation
- **Reranking** — scoring query-document pairs for relevance
- **Reward modeling** — scoring model outputs for RLHF
- **Late interaction retrieval** — ColBERT-style token-level matching

---

## Quick Start

### Text Embeddings

```python
from vllm import LLM

llm = LLM(model="BAAI/bge-base-en-v1.5", task="embed")

outputs = llm.embed(["Hello, world!", "How are you?"])
for output in outputs:
    print(output.outputs.embedding)  # list of floats
    print(len(output.outputs.embedding))  # embedding dimension
```

### Batch Embeddings

```python
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning is a subset of artificial intelligence.",
    "vLLM provides high-throughput LLM serving.",
]

outputs = llm.embed(texts)
embeddings = [o.outputs.embedding for o in outputs]
```

### Using the OpenAI-Compatible API

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000/v1", api_key="token")

response = client.embeddings.create(
    model="BAAI/bge-base-en-v1.5",
    input=["Hello, world!", "How are you?"],
)

for item in response.data:
    print(item.embedding[:5])  # First 5 dimensions
    print(item.index)
```

### Serving an Embedding Model

```bash
vllm serve BAAI/bge-base-en-v1.5 --task embed
```

---

## Pooling Strategies

Embedding models aggregate token-level hidden states into a single vector using a **pooling strategy**. vLLM supports three sequence-level pooling types:

| Pooling Type | Description | Common Use |
|---|---|---|
| `CLS` | Use the `[CLS]` token representation | BERT-style models |
| `LAST` | Use the last non-padding token | Decoder-only models (Llama, Qwen2) |
| `MEAN` | Average all token representations | Some sentence transformers |

And two token-level pooling types:

| Pooling Type | Description | Common Use |
|---|---|---|
| `ALL` | Return all token embeddings | ColBERT late interaction |
| `STEP` | Return embeddings at step tokens | Process reward models |

### Configuring Pooling

Most models have a default pooling strategy defined in their implementation. You can override it:

```python
from vllm import LLM
from vllm.config import PoolerConfig

llm = LLM(
    model="meta-llama/Llama-3.1-8B",
    task="embed",
    override_pooler_config=PoolerConfig(
        pooling_type="MEAN",  # Override to mean pooling
        normalize=True,       # L2-normalize embeddings
    ),
)
```

Or via CLI:

```bash
vllm serve meta-llama/Llama-3.1-8B \
    --task embed \
    --override-pooler-config '{"pooling_type": "MEAN"}'
```

### Matryoshka Embeddings

Some models support **Matryoshka Representation Learning** — you can truncate embeddings to a smaller dimension while retaining most of the semantic information:

```python
llm = LLM(
    model="Snowflake/snowflake-arctic-embed-l-v2.0",
    task="embed",
    override_pooler_config=PoolerConfig(dimensions=256),  # Truncate to 256 dims
)
```

### Chunked Processing for Long Inputs

For inputs longer than the model's maximum position embeddings, enable chunked processing:

```python
llm = LLM(
    model="BAAI/bge-base-en-v1.5",
    task="embed",
    override_pooler_config=PoolerConfig(enable_chunked_processing=True),
)
```

This splits long inputs into chunks, processes them separately, and aggregates using weighted averaging.

---

## Supported Embedding Models

### Text-Only Embedding Models

| Model | Architecture | Pooling | Dimensions |
|---|---|---|---|
| `bert-base-uncased` | `BertModel` | CLS | 768 |
| `BAAI/bge-base-en-v1.5` | `BertModel` | CLS | 768 |
| `BAAI/bge-large-en-v1.5` | `BertModel` | CLS | 1024 |
| `BAAI/bge-m3` | `BgeM3EmbeddingModel` | CLS | 1024 |
| `FacebookAI/roberta-large` | `RobertaModel` | CLS | 1024 |
| `FacebookAI/xlm-roberta-large` | `XLMRobertaModel` | CLS | 1024 |
| `answerdotai/ModernBERT-base` | `ModernBertModel` | CLS | 768 |
| `nomic-ai/nomic-embed-text-v1.5` | `NomicBertModel` | MEAN | 768 |
| `Snowflake/snowflake-arctic-embed-l-v2.0` | `GteModel` | CLS | 1024 |
| `Alibaba-NLP/gte-Qwen2-7B-instruct` | `Qwen2Model` | LAST | 3584 |
| `GritLM/GritLM-7B` | `GritLM` | MEAN | 4096 |
| `Qwen/Qwen2.5-7B` | `Qwen2ForCausalLM` | LAST | 3584 |
| `meta-llama/Llama-3.1-8B` | `LlamaModel` | LAST | 4096 |
| `google/gemma-2-9b` | `Gemma2Model` | LAST | 3584 |
| `internlm/internlm2-7b-reward` | `InternLM2ForRewardModel` | LAST | 4096 |
| `Qwen/Qwen2.5-Math-RM-72B` | `Qwen2ForRewardModel` | LAST | 7168 |

### Multimodal Embedding Models

| Model | Architecture | Modalities | Dimensions |
|---|---|---|---|
| `openai/clip-vit-large-patch14` | `CLIPModel` | Image | 768 |
| `google/siglip-so400m-patch14-384` | `SiglipModel` | Image | 1152 |
| `llava-hf/llava-v1.6-mistral-7b-hf` | `LlavaNextForConditionalGeneration` | Image+Text | 4096 |
| `Qwen/Qwen2-VL-7B-Instruct` | `Qwen2VLForConditionalGeneration` | Image+Text | 3584 |
| `microsoft/Phi-3-vision-128k-instruct` | `Phi3VForCausalLM` | Image+Text | 3072 |

---

## Cross-Encoders and Rerankers

Cross-encoders score query-document pairs jointly, producing a relevance score rather than separate embeddings. They are used for reranking retrieved documents.

### Using Cross-Encoders

```python
from vllm import LLM

llm = LLM(model="cross-encoder/ms-marco-MiniLM-L-6-v2", task="score")

# Score query-document pairs
pairs = [
    ("What is machine learning?", "Machine learning is a type of AI."),
    ("What is machine learning?", "The weather is sunny today."),
]

outputs = llm.score(pairs)
for pair, output in zip(pairs, outputs):
    print(f"Score: {output.outputs.score:.4f} | {pair[1][:50]}")
```

### Supported Reranker Models

| Model | Architecture |
|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | `BertForSequenceClassification` |
| `cross-encoder/stsb-roberta-large` | `RobertaForSequenceClassification` |
| `Alibaba-NLP/gte-reranker-modernbert-8b` | `GteNewForSequenceClassification` |
| `answerdotai/ModernBERT-large` | `ModernBertForSequenceClassification` |
| `jinaai/jina-reranker-m0` | `JinaVLForRanking` |

---

## Late Interaction Models (ColBERT)

ColBERT-style models return token-level embeddings for both queries and documents, enabling fine-grained late interaction scoring.

### Using ColBERT Models

```python
from vllm import LLM

llm = LLM(model="colbert-ir/colbertv2.0", task="embed")

# Embed query (returns token-level embeddings)
query_outputs = llm.embed(["What is machine learning?"])
query_embeddings = query_outputs[0].outputs.embedding  # [num_tokens, dim]

# Embed documents
doc_outputs = llm.embed(["Machine learning is a type of AI."])
doc_embeddings = doc_outputs[0].outputs.embedding  # [num_tokens, dim]
```

### Supported ColBERT Models

| Model | Architecture |
|---|---|
| `colbert-ir/colbertv2.0` | `HF_ColBERT` |
| `jinaai/jina-colbert-v2` | `ColBERTJinaRobertaModel` |
| `Qwen/Qwen3-VL` (ColQwen3) | `ColQwen3` |

---

## SPLADE Sparse Embeddings

SPLADE models produce sparse embeddings (bag-of-words style) that are efficient for lexical retrieval.

```python
from vllm import LLM

llm = LLM(model="naver/splade-v3", task="embed")

outputs = llm.embed(["Machine learning is a type of AI."])
# Sparse embedding: most dimensions are 0
sparse_embedding = outputs[0].outputs.embedding
```

---

## Reward Models

Reward models score text sequences for quality, used in RLHF training and inference-time scaling.

### Process Reward Models (PRMs)

PRMs score each step in a chain-of-thought reasoning process:

```python
from vllm import LLM

llm = LLM(model="Qwen/Qwen2.5-Math-PRM-7B", task="reward")

# Score reasoning steps
outputs = llm.score([
    ("Solve: 2+2=?", "Step 1: Add 2 and 2. Step 2: The answer is 4."),
])
print(outputs[0].outputs.score)
```

### Outcome Reward Models (ORMs)

ORMs score the final output of a model:

```python
llm = LLM(model="Qwen/Qwen2.5-Math-RM-72B", task="reward")

outputs = llm.score([
    ("What is 2+2?", "The answer is 4."),
    ("What is 2+2?", "The answer is 5."),
])
for output in outputs:
    print(output.outputs.score)
```

---

## Using Decoder-Only Models as Embedding Models

Many decoder-only LLMs (Llama, Qwen2, Mistral, etc.) can be used as embedding models by pooling their hidden states. vLLM supports this through the `task="embed"` parameter:

```python
from vllm import LLM

# Use Llama-3.1-8B as an embedding model
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    task="embed",
)

outputs = llm.embed(["Hello, world!"])
print(len(outputs[0].outputs.embedding))  # 4096
```

!!! note "Pooling for decoder-only models"
    Decoder-only models default to `LAST` token pooling, which uses the last non-padding token's hidden state as the embedding. This is the most common approach for LLM-based embeddings.

---

## GritLM: Unified Generative and Embedding Model

GritLM is a unique model that supports both text generation and embedding in a single model:

```python
from vllm import LLM

llm = LLM(model="GritLM/GritLM-7B", task="embed")

# Embedding mode
embed_outputs = llm.embed(["Hello, world!"])

# Generation mode (switch task)
llm_gen = LLM(model="GritLM/GritLM-7B", task="generate")
gen_outputs = llm_gen.generate(["Hello, world!"])
```

---

## Performance Tips

1. **Use `task="embed"` explicitly** — this disables the LM head and saves memory.
2. **Batch requests** — embedding models benefit greatly from large batch sizes.
3. **Use FP16/BF16** — embedding quality is rarely affected by reduced precision.
4. **Enable tensor parallelism** — for large embedding models (7B+), use `--tensor-parallel-size`.
5. **Set `max_model_len`** — embedding models rarely need long contexts; set a lower limit to save memory.

```bash
vllm serve BAAI/bge-large-en-v1.5 \
    --task embed \
    --max-model-len 512 \
    --dtype bfloat16
```

---

## Related

- [Supported Models](supported_models.md) — full list of embedding architectures
- [Adding a New Model](adding_model.md) — implementing a new embedding model
- [Engine Arguments](../configuration/engine_args.md) — `--task`, `--override-pooler-config`, and related flags
