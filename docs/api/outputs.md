# Output Types

vLLM defines a hierarchy of output types returned by its inference APIs. Understanding these types is essential for processing results from `LLM.generate`, `LLM.embed`, `LLM.classify`, `LLM.score`, and their async counterparts.

---

## Overview

```
RequestOutput                    ← text generation output
  └── outputs: list[CompletionOutput]

PoolingRequestOutput[T]          ← base pooling output (generic)
  ├── EmbeddingRequestOutput     ← embed() output
  │     └── outputs: EmbeddingOutput
  ├── ClassificationRequestOutput ← classify() output
  │     └── outputs: ClassificationOutput
  └── ScoringRequestOutput       ← score() output
        └── outputs: ScoringOutput
```

---

## RequestOutput

```python
from vllm import RequestOutput
```

The output of a text generation request. Returned by `LLM.generate`, `LLM.chat`, `LLM.beam_search`, and `AsyncLLMEngine.generate`.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `request_id` | `str` | Unique identifier for the request. |
| `prompt` | `str \| None` | The input prompt string. `None` if `skip_tokenizer_init=True`. For encoder-decoder models, this is the decoder input prompt. |
| `prompt_token_ids` | `list[int] \| None` | Token IDs of the input prompt. |
| `prompt_logprobs` | `PromptLogprobs \| None` | Log probabilities for each prompt token. Only populated when `SamplingParams.prompt_logprobs` is set. |
| `outputs` | `list[CompletionOutput]` | The generated output sequences. Contains `n` elements when `SamplingParams.n > 1`. |
| `finished` | `bool` | `True` when the request is fully complete (all `n` outputs have finished). |
| `metrics` | `RequestStateStats \| None` | Timing and performance metrics for this request. |
| `lora_request` | `LoRARequest \| None` | The LoRA adapter used for this request. |
| `encoder_prompt` | `str \| None` | The encoder input prompt (encoder-decoder models only). |
| `encoder_prompt_token_ids` | `list[int] \| None` | Token IDs of the encoder input (encoder-decoder models only). |
| `num_cached_tokens` | `int \| None` | Number of prompt tokens served from the prefix cache. |
| `kv_transfer_params` | `dict \| None` | KV transfer parameters for disaggregated serving. |

### Example

```python
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
outputs = llm.generate(
    ["Hello, world!"],
    SamplingParams(max_tokens=64, logprobs=2),
)

for req_output in outputs:
    print(f"Request ID: {req_output.request_id}")
    print(f"Prompt: {req_output.prompt!r}")
    print(f"Finished: {req_output.finished}")
    print(f"Cached tokens: {req_output.num_cached_tokens}")

    for completion in req_output.outputs:
        print(f"  [{completion.index}] {completion.text!r}")
        print(f"  Finish reason: {completion.finish_reason}")
```

---

## CompletionOutput

```python
from vllm import CompletionOutput
```

A single generated output sequence within a `RequestOutput`. When `SamplingParams.n > 1`, a `RequestOutput` contains multiple `CompletionOutput` objects.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `index` | `int` | Index of this output within the request (0-based). |
| `text` | `str` | The generated text. |
| `token_ids` | `Sequence[int]` | Token IDs of the generated text. |
| `cumulative_logprob` | `float \| None` | Sum of log probabilities of all generated tokens. `None` when `logprobs` is not requested. |
| `logprobs` | `SampleLogprobs \| None` | Per-token log probability distributions. `None` when `SamplingParams.logprobs` is not set. |
| `routed_experts` | `np.ndarray \| None` | Routed expert indices for MoE models. Shape: `[seq_len, layer_num, topk]`. Only populated when `enable_return_routed_experts=True`. |
| `finish_reason` | `str \| None` | Why generation stopped. Values: `"stop"` (EOS or stop string), `"length"` (max tokens reached), `"abort"` (request aborted). `None` if not yet finished. |
| `stop_reason` | `int \| str \| None` | The specific stop string or token ID that triggered stopping. `None` if stopped for another reason. |
| `lora_request` | `LoRARequest \| None` | The LoRA adapter used for this output. |

### Methods

#### `finished`

```python
def finished() -> bool
```

Return `True` if this output sequence has a `finish_reason`.

### Example

```python
for req_output in outputs:
    for completion in req_output.outputs:
        print(f"Text: {completion.text!r}")
        print(f"Tokens: {list(completion.token_ids)}")
        print(f"Finish reason: {completion.finish_reason}")
        print(f"Stop reason: {completion.stop_reason}")

        if completion.logprobs:
            # logprobs is a list of dicts: [{token_id: Logprob}, ...]
            for step, token_logprobs in enumerate(completion.logprobs):
                print(f"  Step {step}: {token_logprobs}")
```

---

## PoolingRequestOutput

```python
from vllm import PoolingRequestOutput
```

The base output type for all pooling requests. Generic over the output type `T`.

```python
class PoolingRequestOutput(Generic[T]):
    request_id: str
    outputs: T
    prompt_token_ids: list[int]
    num_cached_tokens: int
    finished: bool
```

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `request_id` | `str` | Unique identifier for the request. |
| `outputs` | `T` | The pooling result. Type depends on the subclass. |
| `prompt_token_ids` | `list[int]` | Token IDs of the input prompt. |
| `num_cached_tokens` | `int` | Number of prompt tokens served from the prefix cache. |
| `finished` | `bool` | Always `True` for pooling requests (pooling is non-streaming). |

---

## PoolingOutput

```python
from vllm import PoolingOutput
```

The raw pooling output containing a tensor of hidden states. This is the `outputs` field of a `PoolingRequestOutput` before it is converted to a task-specific type.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | `torch.Tensor` | The pooled hidden states tensor. Shape depends on the pooling task and model. |

---

## EmbeddingRequestOutput

```python
from vllm import EmbeddingRequestOutput
```

Output of `LLM.embed()`. Subclass of `PoolingRequestOutput[EmbeddingOutput]`.

### Attributes

Inherits all attributes from `PoolingRequestOutput`, with `outputs` typed as `EmbeddingOutput`.

---

## EmbeddingOutput

```python
from vllm import EmbeddingOutput
```

The embedding vector for a single input.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `embedding` | `list[float]` | The dense embedding vector. Length equals the model's hidden dimension (or the requested `dimensions` for Matryoshka models). |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `hidden_size` | `int` | Length of the embedding vector (`len(self.embedding)`). |

### Example

```python
from vllm import LLM

llm = LLM(model="BAAI/bge-base-en-v1.5")
results = llm.embed(["Hello world", "vLLM is fast"])

for result in results:
    emb = result.outputs.embedding
    print(f"Embedding dim: {result.outputs.hidden_size}")
    print(f"First 5 values: {emb[:5]}")
```

---

## ClassificationRequestOutput

```python
from vllm import ClassificationRequestOutput
```

Output of `LLM.classify()`. Subclass of `PoolingRequestOutput[ClassificationOutput]`.

### Attributes

Inherits all attributes from `PoolingRequestOutput`, with `outputs` typed as `ClassificationOutput`.

---

## ClassificationOutput

```python
from vllm import ClassificationOutput
```

The class probability distribution for a single input.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `probs` | `list[float]` | Probability vector. Length equals the number of classes. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `num_classes` | `int` | Number of classes (`len(self.probs)`). |

### Example

```python
from vllm import LLM

llm = LLM(model="cross-encoder/ms-marco-MiniLM-L-6-v2", convert="classify")
results = llm.classify(["This is a great product!"])

for result in results:
    print(f"Classes: {result.outputs.num_classes}")
    print(f"Probs: {result.outputs.probs}")
    predicted_class = result.outputs.probs.index(max(result.outputs.probs))
    print(f"Predicted class: {predicted_class}")
```

---

## ScoringRequestOutput

```python
from vllm import ScoringRequestOutput
```

Output of `LLM.score()`. Subclass of `PoolingRequestOutput[ScoringOutput]`.

### Attributes

Inherits all attributes from `PoolingRequestOutput`, with `outputs` typed as `ScoringOutput`.

---

## ScoringOutput

```python
from vllm import ScoringOutput
```

The similarity score for a single input pair.

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `score` | `float` | The similarity score. For cross-encoders and embedding cosine similarity, this is typically in `[-1, 1]` or `[0, 1]`. For late-interaction models (ColBERT), the range depends on the model. |

### Example

```python
from vllm import LLM

llm = LLM(model="cross-encoder/ms-marco-MiniLM-L-6-v2")

scores = llm.score(
    "What is the capital of France?",
    ["Paris is the capital.", "London is in England."],
)

for i, result in enumerate(scores):
    print(f"Pair {i}: score = {result.outputs.score:.4f}")
```

---

## Log Probability Types

### `Logprob`

```python
from vllm.logprobs import Logprob
```

A single log probability entry.

| Attribute | Type | Description |
|-----------|------|-------------|
| `logprob` | `float` | The log probability value. |
| `rank` | `int \| None` | Rank of this token among all vocabulary tokens (1 = most likely). |
| `decoded_token` | `str \| None` | The decoded string for this token. |

### `SampleLogprobs`

```python
SampleLogprobs = list[dict[int, Logprob] | None]
```

Per-step log probabilities for generated tokens. Each element is a dict mapping token ID to `Logprob`, or `None` if log probs were not requested for that step.

### `PromptLogprobs`

```python
PromptLogprobs = list[dict[int, Logprob] | None]
```

Per-token log probabilities for prompt tokens. Same structure as `SampleLogprobs`.

---

## Working with Outputs

### Accessing generated text

```python
outputs = llm.generate(prompts, sampling_params)

for req in outputs:
    # First (and usually only) completion
    text = req.outputs[0].text
    finish = req.outputs[0].finish_reason
```

### Multiple outputs (n > 1)

```python
params = SamplingParams(n=3, temperature=1.0, max_tokens=64)
outputs = llm.generate(["Tell me a joke."], sampling_params=params)

for req in outputs:
    for i, completion in enumerate(req.outputs):
        print(f"Output {i}: {completion.text!r}")
```

### Checking finish reason

```python
for req in outputs:
    completion = req.outputs[0]
    if completion.finish_reason == "length":
        print("Warning: output was truncated by max_tokens")
    elif completion.finish_reason == "stop":
        print(f"Stopped by: {completion.stop_reason!r}")
```

### Processing log probabilities

```python
params = SamplingParams(logprobs=5, max_tokens=32)
outputs = llm.generate(["Hello"], sampling_params=params)

for req in outputs:
    completion = req.outputs[0]
    if completion.logprobs:
        for step, token_logprobs in enumerate(completion.logprobs):
            if token_logprobs:
                # Sort by log probability (descending)
                top_tokens = sorted(
                    token_logprobs.items(),
                    key=lambda x: x[1].logprob,
                    reverse=True,
                )
                print(f"Step {step} top tokens:")
                for token_id, lp in top_tokens[:3]:
                    print(f"  {lp.decoded_token!r}: {lp.logprob:.3f}")
```

### Embedding similarity

```python
import numpy as np
from vllm import LLM

llm = LLM(model="BAAI/bge-base-en-v1.5")
results = llm.embed(["Hello world", "Hi there", "Goodbye"])

embeddings = np.array([r.outputs.embedding for r in results])

# Cosine similarity between first and second
cos_sim = np.dot(embeddings[0], embeddings[1]) / (
    np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
)
print(f"Similarity: {cos_sim:.4f}")
```
