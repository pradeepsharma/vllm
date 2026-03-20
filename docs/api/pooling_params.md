# PoolingParams

```python
from vllm import PoolingParams
```

`PoolingParams` controls how hidden states are pooled for embedding, classification, scoring, and reward models. It is used with the pooling-oriented methods of [`LLM`](llm.md) (`embed`, `classify`, `score`, `reward`, `encode`) and with [`AsyncLLMEngine.encode`](async_llm_engine.md#encode).

---

## Constructor

```python
PoolingParams(
    use_activation: bool | None = None,
    dimensions: int | None = None,
    step_tag_id: int | None = None,
    returned_token_ids: list[int] | None = None,
)
```

---

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_activation` | `bool \| None` | `None` | Whether to apply the pooler's activation function to the output. `None` uses the pooler's default (typically `True`). |
| `dimensions` | `int \| None` | `None` | Reduce embedding dimensions for models that support Matryoshka Representation Learning (MRL). `None` returns the full embedding. Only valid for `embed` and `token_embed` tasks. |
| `step_tag_id` | `int \| None` | `None` | Token ID of the step tag for step-level pooling models. Only valid when the model uses `STEP` pooling. |
| `returned_token_ids` | `list[int] \| None` | `None` | Specific token IDs whose hidden states should be returned. Only valid for step pooling models. |

---

## Internal / Advanced Parameters

These fields are set internally by the engine and should not normally be set by users:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `task` | `PoolingTask \| None` | `None` | The pooling task. Set automatically by the engine based on the method called (`"embed"`, `"classify"`, `"score"`, `"token_embed"`, `"token_classify"`, `"plugin"`). |
| `requires_token_ids` | `bool` | `False` | Whether the pooler requires token IDs in the output. |
| `skip_reading_prefix_cache` | `bool \| None` | `None` | Skip reading from the prefix cache. Set automatically for tasks that require full token sequences. |
| `extra_kwargs` | `dict \| None` | `None` | Extra keyword arguments for custom pooling implementations. |
| `output_kind` | `RequestOutputKind` | `FINAL_ONLY` | Always `FINAL_ONLY` for pooling requests. |

---

## Valid Parameters by Task

Different pooling tasks support different parameters:

| Task | Supported Parameters |
|------|---------------------|
| `embed` | `dimensions`, `use_activation` |
| `classify` | `use_activation` |
| `score` | `use_activation` |
| `token_embed` | `dimensions`, `use_activation` |
| `token_classify` | `use_activation` |

Passing unsupported parameters for a task raises a `ValueError`.

---

## Methods

### `clone`

```python
def clone() -> PoolingParams
```

Return a deep copy of this `PoolingParams` instance.

---

### `verify`

```python
def verify(model_config: ModelConfig) -> None
```

Validate the parameters against the model configuration. Called automatically by the engine before processing a request.

---

## Matryoshka Embeddings

Models that support Matryoshka Representation Learning (MRL) allow you to reduce the embedding dimensionality without retraining:

```python
from vllm import LLM, PoolingParams

llm = LLM(model="nomic-ai/nomic-embed-text-v1.5")

# Full-dimensional embeddings
full = llm.embed(["Hello world"])
print(len(full[0].outputs.embedding))  # e.g. 768

# Reduced-dimensional embeddings
reduced = llm.embed(
    ["Hello world"],
    pooling_params=PoolingParams(dimensions=256),
)
print(len(reduced[0].outputs.embedding))  # 256
```

!!! warning
    Setting `dimensions` on a model that does not support MRL raises a `ValueError`. Check `model_config.is_matryoshka` before using this parameter.

---

## Examples

### Default embeddings

```python
from vllm import LLM, PoolingParams

llm = LLM(model="BAAI/bge-base-en-v1.5")

# Use all defaults
results = llm.embed(["Hello world"])
print(results[0].outputs.embedding[:5])
```

### Embeddings without activation

```python
results = llm.embed(
    ["Hello world"],
    pooling_params=PoolingParams(use_activation=False),
)
```

### Classification

```python
from vllm import LLM, PoolingParams

llm = LLM(model="cross-encoder/ms-marco-MiniLM-L-6-v2", convert="classify")

results = llm.classify(
    ["This is a positive review."],
    pooling_params=PoolingParams(use_activation=True),
)
print(results[0].outputs.probs)
```

### Scoring / reranking

```python
from vllm import LLM, PoolingParams

llm = LLM(model="cross-encoder/ms-marco-MiniLM-L-6-v2")

scores = llm.score(
    "What is the capital of France?",
    ["Paris is the capital of France.", "London is in England."],
    pooling_params=PoolingParams(use_activation=True),
)
for s in scores:
    print(s.outputs.score)
```

---

## Relationship to PoolingTask

The `task` field is set automatically by the engine based on which method you call:

| Method | `task` value |
|--------|-------------|
| `LLM.embed()` | `"embed"` |
| `LLM.classify()` | `"classify"` |
| `LLM.score()` | `"embed"` or `"classify"` (model-dependent) |
| `LLM.reward()` | `"token_classify"` |
| `LLM.encode(pooling_task="token_embed")` | `"token_embed"` |

You can also set `task` explicitly when calling `LLM.encode()`:

```python
from vllm.tasks import PoolingTask

results = llm.encode(
    prompts,
    pooling_params=PoolingParams(),
    pooling_task="embed",
)
```
