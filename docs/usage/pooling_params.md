# PoolingParams Reference

[`PoolingParams`][vllm.PoolingParams] controls how vLLM's pooling models produce their outputs — including embedding dimensions, activation functions, and task-specific behavior.

Pooling params are used with the following `LLM` methods:

- `llm.embed()` — dense embedding vectors
- `llm.classify()` — classification logits / probabilities
- `llm.score()` — relevance / similarity scores
- `llm.reward()` — reward model outputs
- `llm.encode()` — generic pooling (requires explicit `pooling_task`)

```python
from vllm.pooling_params import PoolingParams

params = PoolingParams(dimensions=512)
outputs = llm.embed(texts, pooling_params=params)
```

---

## Overview

Pooling models differ from generative models in that they produce a fixed-size representation of the input rather than generating new tokens. The `PoolingParams` class configures how that representation is computed and returned.

### Supported Tasks

| Task | Method | Description |
|---|---|---|
| `"embed"` | `llm.embed()` | Dense embedding vector (1-D float array) |
| `"classify"` | `llm.classify()` | Class probability distribution |
| `"score"` | `llm.score()` | Scalar relevance score |
| `"token_classify"` | `llm.reward()` | Per-token classification (reward models) |
| `"token_embed"` | — | Per-token embeddings (late interaction, e.g., ColBERT) |

---

## Parameter Reference

### `use_activation`

**Type:** `bool | None` · **Default:** `None`

Whether to apply the pooler's activation function to the output.

- `None` → use the pooler's default (typically `True`)
- `True` → apply activation (e.g., sigmoid for classification, tanh for some embeddings)
- `False` → return raw logits / hidden states without activation

```python
# Get raw logits instead of probabilities
params = PoolingParams(use_activation=False)
outputs = llm.classify(texts, pooling_params=params)
```

**Valid for tasks:** `embed`, `classify`, `score`, `token_embed`, `token_classify`

---

### `dimensions`

**Type:** `int | None` · **Default:** `None`

Truncate the output embedding to this many dimensions. Only supported by models that implement **Matryoshka Representation Learning (MRL)**, which allows embeddings to be truncated to smaller sizes while retaining most of their semantic quality.

- `None` → use the model's full embedding dimension
- `int` → truncate to this many dimensions (must be a supported size for the model)

```python
from vllm import LLM
from vllm.pooling_params import PoolingParams

llm = LLM(model="nomic-ai/nomic-embed-text-v1.5", runner="pooling")

# Full dimension (e.g., 768)
full_outputs = llm.embed(texts)
print(f"Full dim: {len(full_outputs[0].outputs.embedding)}")

# Reduced to 256 dimensions
small_params = PoolingParams(dimensions=256)
small_outputs = llm.embed(texts, pooling_params=small_params)
print(f"Reduced dim: {len(small_outputs[0].outputs.embedding)}")
```

!!! warning "Model support required"
    If you set `dimensions` on a model that does not support matryoshka representation, vLLM raises a `ValueError`. Check the model card to confirm MRL support.

!!! note "Supported dimension values"
    Many MRL models only support specific dimension values (e.g., 64, 128, 256, 512, 768). vLLM validates the requested dimension against the model's supported list and raises an error if it is not supported.

**Valid for tasks:** `embed`, `token_embed`

---

### `step_tag_id`

**Type:** `int | None` · **Default:** `None`

For **step pooling models** (process reward models that score individual reasoning steps): the token ID of the step tag. The pooler extracts hidden states at positions where this token appears.

```python
# For a process reward model that uses a special step token
params = PoolingParams(step_tag_id=12345)
outputs = llm.reward(prompts, pooling_params=params)
```

!!! note
    Only valid when the model's pooler is configured with `tok_pooling_type="STEP"`. Setting this on other models raises a `ValueError`.

---

### `returned_token_ids`

**Type:** `list[int] | None` · **Default:** `None`

For step pooling models: a list of token IDs whose hidden states should be returned. Allows selective extraction of hidden states at specific token positions.

```python
params = PoolingParams(
    step_tag_id=12345,
    returned_token_ids=[100, 200, 300],
)
```

!!! note
    Only valid when the model's pooler is configured with `tok_pooling_type="STEP"`.

---

## Internal Parameters

The following parameters are used internally by vLLM and are not intended for direct use in application code:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `task` | `PoolingTask \| None` | `None` | Set automatically by `llm.embed()`, `llm.classify()`, etc. |
| `requires_token_ids` | `bool` | `False` | Whether the output should include token IDs |
| `skip_reading_prefix_cache` | `bool \| None` | `None` | Disable prefix cache reading for this request |
| `extra_kwargs` | `dict[str, Any] \| None` | `None` | Extra arguments for custom pooling implementations |
| `output_kind` | `RequestOutputKind` | `FINAL_ONLY` | Always `FINAL_ONLY` for pooling (streaming not supported) |

---

## Valid Parameters by Task

Not all parameters are valid for all tasks. vLLM validates the combination and raises a `ValueError` if an unsupported parameter is set:

| Parameter | `embed` | `classify` | `score` | `token_embed` | `token_classify` |
|---|:---:|:---:|:---:|:---:|:---:|
| `use_activation` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `dimensions` | ✅ | ❌ | ❌ | ✅ | ❌ |

---

## Usage Examples

### Basic Embedding

```python
from vllm import LLM
from vllm.pooling_params import PoolingParams

llm = LLM(model="intfloat/e5-small", runner="pooling")

texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A fast auburn fox leaps above a sleepy canine.",
]

# Default parameters
outputs = llm.embed(texts)
for text, output in zip(texts, outputs):
    print(f"Text: {text!r}")
    print(f"Embedding dim: {len(output.outputs.embedding)}")
```

### Reduced-Dimension Embedding (Matryoshka)

```python
llm = LLM(model="nomic-ai/nomic-embed-text-v1.5", runner="pooling")

params_256 = PoolingParams(dimensions=256)
params_512 = PoolingParams(dimensions=512)

outputs_256 = llm.embed(texts, pooling_params=params_256)
outputs_512 = llm.embed(texts, pooling_params=params_512)

print(f"256-dim: {len(outputs_256[0].outputs.embedding)}")
print(f"512-dim: {len(outputs_512[0].outputs.embedding)}")
```

### Classification with Raw Logits

```python
llm = LLM(model="jason9693/Qwen2.5-1.5B-apeach", runner="pooling")

# Get raw logits (before softmax)
params = PoolingParams(use_activation=False)
outputs = llm.classify(texts, pooling_params=params)

for text, output in zip(texts, outputs):
    logits = output.outputs.probs  # raw logits when use_activation=False
    print(f"Logits: {logits}")
```

### Scoring / Reranking

```python
llm = LLM(model="BAAI/bge-reranker-v2-m3", runner="pooling")

query = "What is the capital of France?"
documents = [
    "The capital of Brazil is Brasilia.",
    "The capital of France is Paris.",
]

# Default pooling params (use_activation=True for score task)
outputs = llm.score(query, documents)
for doc, output in zip(documents, outputs):
    print(f"Score: {output.outputs.score:.4f} | {doc}")
```

### Per-Request Pooling Params

Pass a list of `PoolingParams` — one per prompt — for fine-grained control:

```python
texts = ["short text", "another text", "third text"]
params_list = [
    PoolingParams(dimensions=256),
    PoolingParams(dimensions=512),
    PoolingParams(),  # full dimension
]

outputs = llm.embed(texts, pooling_params=params_list)
for text, output in zip(texts, outputs):
    print(f"{text!r}: dim={len(output.outputs.embedding)}")
```

---

## Configuring Default Pooling Parameters

You can set default pooling parameters at the model level using `PoolerConfig` when constructing the `LLM`:

```python
from vllm import LLM
from vllm.config import PoolerConfig

llm = LLM(
    model="nomic-ai/nomic-embed-text-v1.5",
    runner="pooling",
    pooler_config=PoolerConfig(
        dimensions=256,        # default to 256-dim embeddings
        use_activation=True,
    ),
)

# These requests will use 256-dim by default
outputs = llm.embed(texts)
```

Per-request `PoolingParams` override the model-level defaults.

---

## See Also

- [Offline Inference Guide](offline_inference.md) — using `embed()`, `classify()`, `score()`, and `reward()`
- [Output Types](output_types.md) — `EmbeddingRequestOutput`, `ClassificationRequestOutput`, `ScoringRequestOutput`
- [Pooling Models](../models/pooling_models.md) — supported pooling model architectures
