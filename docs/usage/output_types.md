# Output Types Reference

vLLM returns structured Python objects from every inference call. Understanding these types lets you extract exactly the information you need — generated text, token IDs, log probabilities, finish reasons, embeddings, scores, and more.

---

## Overview

| Method | Returns | Output object inside |
|---|---|---|
| `llm.generate()` | `list[RequestOutput]` | `list[CompletionOutput]` |
| `llm.chat()` | `list[RequestOutput]` | `list[CompletionOutput]` |
| `llm.beam_search()` | `list[BeamSearchOutput]` | `list[BeamSearchSequence]` |
| `llm.embed()` | `list[EmbeddingRequestOutput]` | `EmbeddingOutput` |
| `llm.classify()` | `list[ClassificationRequestOutput]` | `ClassificationOutput` |
| `llm.score()` | `list[ScoringRequestOutput]` | `ScoringOutput` |
| `llm.reward()` | `list[PoolingRequestOutput]` | `PoolingOutput` |
| `llm.encode()` | `list[PoolingRequestOutput]` | `PoolingOutput` |

---

## Generative Model Outputs

### `RequestOutput`

The top-level output object for text generation requests (`generate()` and `chat()`).

```python
from vllm.outputs import RequestOutput
```

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `request_id` | `str` | Unique identifier for this request |
| `prompt` | `str \| None` | The original prompt string (decoder prompt for encoder-decoder models) |
| `prompt_token_ids` | `list[int] \| None` | Token IDs of the prompt |
| `prompt_logprobs` | `PromptLogprobs \| None` | Log probabilities for prompt tokens (if `prompt_logprobs` was set in `SamplingParams`) |
| `outputs` | `list[CompletionOutput]` | One `CompletionOutput` per requested completion (`n` completions total) |
| `finished` | `bool` | Whether all completions for this request are complete |
| `metrics` | `RequestStateStats \| None` | Timing and performance metrics |
| `lora_request` | `LoRARequest \| None` | The LoRA adapter used, if any |
| `encoder_prompt` | `str \| None` | Encoder prompt string (encoder-decoder models only) |
| `encoder_prompt_token_ids` | `list[int] \| None` | Encoder prompt token IDs (encoder-decoder models only) |
| `num_cached_tokens` | `int \| None` | Number of prompt tokens served from the prefix cache |
| `kv_transfer_params` | `dict[str, Any] \| None` | Parameters for remote KV cache transfer (disaggregated prefill) |

**Example:**

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(
    ["The capital of France is", "The future of AI is"],
    SamplingParams(temperature=0.8, max_tokens=50),
)

for output in outputs:
    print(f"Request ID:      {output.request_id}")
    print(f"Prompt:          {output.prompt!r}")
    print(f"Prompt tokens:   {len(output.prompt_token_ids)} tokens")
    print(f"Cached tokens:   {output.num_cached_tokens}")
    print(f"Finished:        {output.finished}")
    print(f"Num completions: {len(output.outputs)}")
    print()
```

---

### `CompletionOutput`

Represents a single generated completion within a `RequestOutput`. When `n=1` (the default), there is exactly one `CompletionOutput` per request, accessed as `output.outputs[0]`.

```python
from vllm.outputs import CompletionOutput
```

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `index` | `int` | Index of this completion (0 to n-1) |
| `text` | `str` | The generated text (empty if `detokenize=False`) |
| `token_ids` | `Sequence[int]` | Token IDs of the generated text |
| `cumulative_logprob` | `float \| None` | Sum of log probabilities of all generated tokens |
| `logprobs` | `SampleLogprobs \| None` | Per-token log probabilities (if `logprobs` was set) |
| `finish_reason` | `str \| None` | Why generation stopped: `"stop"`, `"length"`, or `"abort"` |
| `stop_reason` | `int \| str \| None` | The specific stop string or token ID that triggered stopping |
| `lora_request` | `LoRARequest \| None` | The LoRA adapter used for this completion |
| `routed_experts` | `np.ndarray \| None` | Routed expert indices for MoE models (shape: `[seq_len, layer_num, topk]`) |

**`finish_reason` values:**

| Value | Meaning |
|---|---|
| `"stop"` | Generation stopped due to EOS token or a `stop` string/token |
| `"length"` | Generation stopped because `max_tokens` was reached |
| `"abort"` | Request was aborted |
| `None` | Generation is still in progress (streaming) |

**Example:**

```python
outputs = llm.generate(
    "Tell me a story",
    SamplingParams(n=3, temperature=1.0, max_tokens=100),
)

for completion in outputs[0].outputs:
    print(f"Completion [{completion.index}]:")
    print(f"  Text:            {completion.text!r}")
    print(f"  Token count:     {len(completion.token_ids)}")
    print(f"  Cumulative logp: {completion.cumulative_logprob:.4f}")
    print(f"  Finish reason:   {completion.finish_reason}")
    print(f"  Stop reason:     {completion.stop_reason}")
```

**Checking if a completion is finished:**

```python
completion = outputs[0].outputs[0]
if completion.finished():
    print("Done:", completion.text)
```

---

### Accessing Log Probabilities

When `logprobs` is set in `SamplingParams`, `completion.logprobs` is a list of dictionaries — one per generated token. Each dictionary maps token IDs to `Logprob` objects.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(
    "The capital of France is",
    SamplingParams(temperature=0.0, logprobs=3, max_tokens=5),
)

completion = outputs[0].outputs[0]
print(f"Generated: {completion.text!r}")
print()

for step, token_logprobs in enumerate(completion.logprobs or []):
    print(f"Step {step}:")
    for token_id, logprob in sorted(
        token_logprobs.items(), key=lambda x: x[1].logprob, reverse=True
    ):
        print(f"  token_id={token_id:6d}  "
              f"logprob={logprob.logprob:8.4f}  "
              f"rank={logprob.rank}  "
              f"token={logprob.decoded_token!r}")
```

See the [Log Probabilities guide](logprobs.md) for more.

---

### Accessing Prompt Log Probabilities

When `prompt_logprobs` is set, `output.prompt_logprobs` is a list of dictionaries — one per prompt token. The first entry is always `None` (no log prob for the first token).

```python
outputs = llm.generate(
    "Hello world",
    SamplingParams(prompt_logprobs=2, max_tokens=10),
)

for i, token_logprobs in enumerate(outputs[0].prompt_logprobs or []):
    if token_logprobs is None:
        print(f"Prompt token {i}: (no logprob for first token)")
    else:
        top = max(token_logprobs.items(), key=lambda x: x[1].logprob)
        print(f"Prompt token {i}: top = {top[1].decoded_token!r} "
              f"({top[1].logprob:.4f})")
```

---

## Pooling Model Outputs

### `PoolingRequestOutput`

The base output class for all pooling model requests. Specialized subclasses (`EmbeddingRequestOutput`, `ClassificationRequestOutput`, `ScoringRequestOutput`) are returned by the specific methods.

```python
from vllm.outputs import PoolingRequestOutput
```

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `request_id` | `str` | Unique identifier for this request |
| `outputs` | `PoolingOutput` | The pooling result (a `torch.Tensor`) |
| `prompt_token_ids` | `list[int]` | Token IDs of the input prompt |
| `num_cached_tokens` | `int` | Number of tokens served from the prefix cache |
| `finished` | `bool` | Whether the request is complete (always `True` for pooling) |

---

### `PoolingOutput`

The raw output of a pooling operation — a `torch.Tensor` containing the pooled hidden states.

```python
from vllm.outputs import PoolingOutput
```

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `data` | `torch.Tensor` | The pooled hidden states tensor |

---

### `EmbeddingOutput`

The output of an embedding request. Contains the embedding vector as a Python list of floats.

```python
from vllm.outputs import EmbeddingOutput
```

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `embedding` | `list[float]` | The embedding vector |
| `hidden_size` | `int` | Length of the embedding vector (property) |

---

### `EmbeddingRequestOutput`

Returned by `llm.embed()`. A specialization of `PoolingRequestOutput` where `outputs` is an `EmbeddingOutput`.

```python
from vllm.outputs import EmbeddingRequestOutput
```

**Example:**

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
outputs = llm.embed(["Hello world", "Goodbye world"])

for output in outputs:
    embedding = output.outputs.embedding  # list[float]
    print(f"Request ID:    {output.request_id}")
    print(f"Embedding dim: {output.outputs.hidden_size}")
    print(f"First 4 vals:  {embedding[:4]}")
    print(f"Cached tokens: {output.num_cached_tokens}")
    print()
```

**Computing cosine similarity:**

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

outputs = llm.embed(["cat", "dog", "automobile"])
embeddings = [o.outputs.embedding for o in outputs]

print(f"cat vs dog:        {cosine_similarity(embeddings[0], embeddings[1]):.4f}")
print(f"cat vs automobile: {cosine_similarity(embeddings[0], embeddings[2]):.4f}")
```

---

### `ClassificationOutput`

The output of a classification request. Contains the class probability distribution.

```python
from vllm.outputs import ClassificationOutput
```

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `probs` | `list[float]` | Probability for each class |
| `num_classes` | `int` | Number of classes (property) |

---

### `ClassificationRequestOutput`

Returned by `llm.classify()`. A specialization of `PoolingRequestOutput` where `outputs` is a `ClassificationOutput`.

```python
from vllm.outputs import ClassificationRequestOutput
```

**Example:**

```python
from vllm import LLM

llm = LLM(model="jason9693/Qwen2.5-1.5B-apeach", runner="pooling")
texts = [
    "I love this product!",
    "This is terrible.",
    "The weather is nice today.",
]

outputs = llm.classify(texts)

for text, output in zip(texts, outputs):
    probs = output.outputs.probs
    predicted = probs.index(max(probs))
    print(f"Text:      {text!r}")
    print(f"Probs:     {[f'{p:.3f}' for p in probs]}")
    print(f"Predicted: class {predicted}")
    print()
```

---

### `ScoringOutput`

The output of a scoring request. Contains a single scalar relevance score.

```python
from vllm.outputs import ScoringOutput
```

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `score` | `float` | The relevance / similarity score |

---

### `ScoringRequestOutput`

Returned by `llm.score()`. A specialization of `PoolingRequestOutput` where `outputs` is a `ScoringOutput`.

```python
from vllm.outputs import ScoringRequestOutput
```

**Example:**

```python
from vllm import LLM

llm = LLM(model="BAAI/bge-reranker-v2-m3", runner="pooling")

query = "What is the capital of France?"
documents = [
    "The capital of Brazil is Brasilia.",
    "The capital of France is Paris.",
    "Paris is known for the Eiffel Tower.",
]

outputs = llm.score(query, documents)

# Sort documents by relevance score
ranked = sorted(
    zip(documents, outputs),
    key=lambda x: x[1].outputs.score,
    reverse=True,
)

print("Ranked results:")
for rank, (doc, output) in enumerate(ranked, 1):
    print(f"  [{rank}] score={output.outputs.score:.4f} | {doc}")
```

---

## Beam Search Outputs

### `BeamSearchOutput`

Returned by `llm.beam_search()`. Contains the top-`beam_width` sequences.

```python
from vllm.beam_search import BeamSearchOutput
```

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `sequences` | `list[BeamSearchSequence]` | The best sequences, sorted by score (best first) |

---

### `BeamSearchSequence`

A single candidate sequence from beam search.

```python
from vllm.beam_search import BeamSearchSequence
```

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `tokens` | `list[int]` | Token IDs of the generated sequence |
| `text` | `str \| None` | Decoded text (populated when returned to the user) |
| `cum_logprob` | `float` | Cumulative log probability of the sequence |
| `logprobs` | `list[dict[int, Logprob]]` | Per-step log probabilities |
| `finish_reason` | `str \| None` | Why this sequence ended |
| `stop_reason` | `int \| str \| None` | The stop token or string that triggered stopping |
| `lora_request` | `LoRARequest \| None` | LoRA adapter used |

**Example:**

```python
from vllm import LLM
from vllm.sampling_params import BeamSearchParams

llm = LLM(model="facebook/opt-125m")

outputs = llm.beam_search(
    [{"prompt": "The capital of France is"}],
    BeamSearchParams(beam_width=4, max_tokens=20),
)

for i, seq in enumerate(outputs[0].sequences):
    print(f"Beam [{i}]: score={seq.cum_logprob:.4f} | {seq.text!r}")
```

---

## Log Probability Types

### `Logprob`

A single log probability entry for one token at one position.

```python
from vllm.logprobs import Logprob
```

**Attributes:**

| Attribute | Type | Description |
|---|---|---|
| `logprob` | `float` | The log probability of this token |
| `rank` | `int \| None` | Vocabulary rank of this token (1 = most likely) |
| `decoded_token` | `str \| None` | The decoded string for this token |

**Type aliases:**

```python
# One position: maps token_id → Logprob
LogprobsOnePosition = dict[int, Logprob]

# All positions for prompt tokens
PromptLogprobs = list[LogprobsOnePosition | None]

# All positions for generated tokens
SampleLogprobs = list[LogprobsOnePosition]
```

---

### `FlatLogprobs`

A memory-efficient alternative to `list[dict[int, Logprob]]` that stores all logprob data in flat primitive lists, significantly reducing Python garbage collection overhead.

Enabled by setting `flat_logprobs=True` in `SamplingParams`.

```python
from vllm.logprobs import FlatLogprobs

# Access like a list
for position_logprobs in flat_logprobs:
    if position_logprobs:
        for token_id, logprob in position_logprobs.items():
            print(token_id, logprob.logprob)
```

---

## Complete Example: Extracting All Information

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")

outputs = llm.generate(
    ["The capital of France is", "The future of AI is"],
    SamplingParams(
        n=2,
        temperature=0.8,
        max_tokens=30,
        logprobs=3,
        prompt_logprobs=1,
    ),
)

for req_output in outputs:
    print("=" * 60)
    print(f"Prompt: {req_output.prompt!r}")
    print(f"Prompt tokens: {len(req_output.prompt_token_ids)}")
    print(f"Cached tokens: {req_output.num_cached_tokens}")

    # Prompt log probabilities
    if req_output.prompt_logprobs:
        print("\nPrompt logprobs (top-1 per token):")
        for i, pos_lp in enumerate(req_output.prompt_logprobs):
            if pos_lp:
                top = max(pos_lp.items(), key=lambda x: x[1].logprob)
                print(f"  [{i}] {top[1].decoded_token!r}: {top[1].logprob:.4f}")

    # Completions
    for completion in req_output.outputs:
        print(f"\nCompletion [{completion.index}]:")
        print(f"  Text:          {completion.text!r}")
        print(f"  Tokens:        {list(completion.token_ids)}")
        print(f"  Cumul logprob: {completion.cumulative_logprob:.4f}")
        print(f"  Finish reason: {completion.finish_reason}")

        # Per-token logprobs
        if completion.logprobs:
            print("  Token logprobs:")
            for step, pos_lp in enumerate(completion.logprobs):
                top3 = sorted(pos_lp.items(), key=lambda x: x[1].logprob, reverse=True)[:3]
                tokens_str = ", ".join(
                    f"{lp.decoded_token!r}({lp.logprob:.2f})"
                    for _, lp in top3
                )
                print(f"    step {step}: {tokens_str}")
```

---

## See Also

- [Log Probabilities Guide](logprobs.md) — detailed logprobs usage and interpretation
- [Beam Search Guide](beam_search.md) — `BeamSearchOutput` and `BeamSearchSequence`
- [SamplingParams Reference](sampling_params.md) — controlling what gets returned
- [PoolingParams Reference](pooling_params.md) — pooling model parameters
