---
description: >
  How to use n-gram (prompt lookup) speculative decoding in vLLM — CPU and GPU
  implementations, configuration, and when to use it.
---

# N-gram speculation (prompt lookup decoding)

N-gram speculation — also called *prompt lookup decoding* — generates draft
tokens by searching the current context for a suffix that matches a prefix
elsewhere in the same sequence, then proposing the tokens that follow that
match. No neural network is involved, so there is no extra model to load or
train.

For background, see the original
[prompt lookup decoding thread](https://x.com/joao_gante/status/1747322413006643259).

---

## How n-gram speculation works

At each decoding step, vLLM:

1. Takes the last `n` tokens of the current context as a *query n-gram*, where
   `n` ranges from `prompt_lookup_min` to `prompt_lookup_max`.
2. Searches the context for the earliest occurrence of that n-gram.
3. Proposes the `num_speculative_tokens` tokens that follow the match as draft
   tokens.
4. The target model verifies the draft tokens in a single forward pass.

The algorithm tries the longest n-gram first and falls back to shorter ones if
no match is found. If no match is found for any n-gram length, no draft tokens
are proposed for that step.

```
Context: [… A B C D E F G … A B C]
                                ↑
                         Query n-gram (n=3): [A B C]
                                │
                         Match found at position i
                                │
                         Draft tokens: [D E F G …]
```

---

## CPU implementation (`ngram`)

The CPU implementation uses Numba-JIT compiled code for fast lookup. It runs
on the CPU in parallel with GPU computation, so it does not block the GPU.

### Offline inference

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    },
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

### Online serving

```bash
vllm serve Qwen/Qwen3-8B \
    --speculative_config '{
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4
    }'
```

---

## GPU implementation (`ngram_gpu`)

The GPU implementation uses fully vectorized PyTorch tensor operations compiled
with `torch.compile`. It runs entirely on the GPU and is designed for
asynchronous execution alongside the target model.

The GPU implementation uses an `unfold`-based sliding window approach to find
matches across all sequences in the batch simultaneously, making it more
efficient for large batches.

### Offline inference

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "method": "ngram_gpu",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    },
)

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(f"Prompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")
```

### Online serving

```bash
vllm serve Qwen/Qwen3-8B \
    --speculative_config '{
        "method": "ngram_gpu",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4
    }'
```

---

## Configuration reference

| Parameter | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | required | `"ngram"` for CPU or `"ngram_gpu"` for GPU. |
| `num_speculative_tokens` | `int` | required | Number of draft tokens to propose per step. |
| `prompt_lookup_max` | `int` | `5` | Maximum n-gram length to search for. |
| `prompt_lookup_min` | `int` | same as `prompt_lookup_max` | Minimum n-gram length to search for. |

!!! note
    If neither `prompt_lookup_min` nor `prompt_lookup_max` is specified, both
    default to `5`. If only one is specified, the other defaults to the same
    value. `prompt_lookup_min` must be less than or equal to `prompt_lookup_max`.

### Choosing n-gram length

- **Larger `prompt_lookup_max`** — finds longer, more specific matches. Fewer
  false positives, but may miss matches when the context is short.
- **Smaller `prompt_lookup_min`** — allows shorter matches as a fallback.
  More proposals but potentially lower acceptance rate.
- **`prompt_lookup_min == prompt_lookup_max`** — searches for exactly one
  n-gram length. Simplest to reason about.

A value of `4` or `5` for `prompt_lookup_max` is a good starting point for
most tasks.

---

## CPU vs. GPU implementation

| Aspect | `ngram` (CPU) | `ngram_gpu` (GPU) |
|---|---|---|
| Execution device | CPU (Numba JIT) | GPU (PyTorch + `torch.compile`) |
| Parallelism | Multi-threaded (up to 1 thread per TP rank) | Fully vectorized across batch |
| Compilation overhead | ~1 s at startup (Numba JIT) | First-run `torch.compile` overhead |
| Best for | Small batches, CPU-heavy workloads | Large batches, GPU-heavy workloads |
| Async with GPU | Yes — runs on CPU while GPU computes | Yes — runs on GPU stream |

---

## When to use n-gram speculation

N-gram speculation works best when the generated text is likely to repeat
phrases from the prompt or from earlier in the generation. High-gain scenarios
include:

- **Document summarization** — the model often echoes phrases from the source.
- **Code completion** — boilerplate patterns repeat frequently.
- **RAG (retrieval-augmented generation)** — the model quotes retrieved
  passages verbatim.
- **Translation** — proper nouns and technical terms are copied unchanged.
- **Structured output** — JSON keys, XML tags, and other schema elements repeat.

N-gram speculation provides little benefit for:

- **Creative writing** — low repetition, so few matches are found.
- **Short prompts** — not enough context to find matches.
- **Highly diverse outputs** — acceptance rate will be low.

---

## Combining with other settings

N-gram speculation is compatible with all standard vLLM settings including
tensor parallelism, prefix caching, and chunked prefill. No extra model weights
are required.

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    enable_prefix_caching=True,
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 5,
        "prompt_lookup_min": 2,
    },
)
```

---

## See also

- [Speculative decoding overview](README.md)
- [Suffix decoding](suffix.md) — a more powerful pattern-matching method that
  also caches past responses.
- [Internal design document](../../design/speculative_decoding_design.md)
