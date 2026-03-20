# Usage Guide

This section covers the core usage patterns for vLLM — from running your first offline inference job to understanding every parameter that controls generation quality and behavior.

---

## What's in This Section

<div class="grid cards" markdown>

-   :material-server-off: **Offline Inference**

    ---

    Run batch inference locally using the `LLM` class — no server required. Covers text generation, chat, embeddings, scoring, and classification.

    [:octicons-arrow-right-24: Offline Inference Guide](offline_inference.md)

-   :material-tune: **SamplingParams Reference**

    ---

    Complete reference for every field in `SamplingParams`: temperature, top-p, top-k, penalties, stop conditions, structured outputs, and more.

    [:octicons-arrow-right-24: SamplingParams Reference](sampling_params.md)

-   :material-vector-combine: **PoolingParams Reference**

    ---

    Parameters for pooling models (embeddings, classification, scoring). Covers dimensions, activation, task selection, and step pooling.

    [:octicons-arrow-right-24: PoolingParams Reference](pooling_params.md)

-   :material-file-document-outline: **Output Types**

    ---

    Understand every output object returned by vLLM: `RequestOutput`, `CompletionOutput`, `EmbeddingRequestOutput`, `ScoringRequestOutput`, and more.

    [:octicons-arrow-right-24: Output Types Reference](output_types.md)

-   :material-magnify-scan: **Beam Search**

    ---

    Use beam search decoding for higher-quality, deterministic outputs. Covers `BeamSearchParams`, length penalty, and working with beam results.

    [:octicons-arrow-right-24: Beam Search Guide](beam_search.md)

-   :material-chart-bar: **Log Probabilities**

    ---

    Request and interpret token-level log probabilities for both generated tokens and prompt tokens. Covers `logprobs`, `prompt_logprobs`, and the `Logprob` data structure.

    [:octicons-arrow-right-24: Log Probabilities Guide](logprobs.md)

</div>

---

## Choosing the Right Inference Mode

vLLM supports two primary inference modes:

| Mode | When to Use | Entry Point |
|---|---|---|
| **Offline (batch)** | Processing a fixed dataset, research, evaluation | `LLM` class |
| **Online (server)** | Production API serving, real-time applications | `vllm serve` |

For online serving, see the [OpenAI-Compatible Server](../serving/openai_compatible_server.md) guide.

---

## Quick Reference: LLM Class Methods

| Method | Task | Model Type |
|---|---|---|
| `llm.generate()` | Text completion from raw prompts | Generative |
| `llm.chat()` | Chat completion with message history | Generative |
| `llm.beam_search()` | Beam search decoding | Generative |
| `llm.embed()` | Dense embedding vectors | Pooling |
| `llm.classify()` | Classification logits | Pooling |
| `llm.score()` | Similarity / relevance scores | Pooling |
| `llm.reward()` | Reward model outputs | Pooling |
| `llm.encode()` | Generic pooling (requires `pooling_task`) | Pooling |

---

## Common Patterns at a Glance

### Greedy decoding (deterministic)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate("The capital of France is", SamplingParams(temperature=0.0))
print(outputs[0].outputs[0].text)
```

### Sampling with nucleus (top-p)

```python
params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)
outputs = llm.generate(prompts, params)
```

### Chat with a system prompt

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain quantum entanglement simply."},
]
outputs = llm.chat(messages)
```

### Embeddings

```python
from vllm import LLM

llm = LLM(model="intfloat/e5-small", runner="pooling")
outputs = llm.embed(["Hello world", "Goodbye world"])
vector = outputs[0].outputs.embedding  # list[float]
```

---

## Next Steps

- New to vLLM? Start with the [Quickstart](../getting_started/quickstart.md).
- Need to configure the engine? See [Engine Arguments](../configuration/engine_args.md).
- Deploying to production? See the [Serving](../serving/openai_compatible_server.md) section.
