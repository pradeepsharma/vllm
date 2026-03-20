# Beam Search Decoding

Beam search is a deterministic decoding strategy that explores multiple candidate sequences simultaneously, keeping the top-`k` (beam width) candidates at each step. Unlike greedy decoding (which always picks the single most likely next token), beam search considers a broader search space and often produces higher-quality, more coherent outputs.

---

## When to Use Beam Search

| Scenario | Recommended Strategy |
|---|---|
| Machine translation | Beam search (width 4–8) |
| Summarization | Beam search (width 4–6) |
| Code generation (exact) | Greedy or beam search |
| Creative writing | Sampling (`temperature > 0`) |
| Chatbots / dialogue | Sampling |
| Data augmentation | Sampling with `n > 1` |

Beam search is best when you want **deterministic, high-quality outputs** and the task has a relatively constrained output space. For open-ended generation, sampling usually produces more natural and diverse results.

---

## Quick Start

```python
from vllm import LLM
from vllm.sampling_params import BeamSearchParams

llm = LLM(model="facebook/opt-125m")

outputs = llm.beam_search(
    prompts=[{"prompt": "The capital of France is"}],
    params=BeamSearchParams(
        beam_width=4,
        max_tokens=20,
    ),
)

# outputs[0].sequences is sorted best-first
best = outputs[0].sequences[0]
print(f"Best sequence: {best.text!r}")
print(f"Score:         {best.cum_logprob:.4f}")
```

---

## `BeamSearchParams` Reference

```python
from vllm.sampling_params import BeamSearchParams
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `beam_width` | `int` | *(required)* | Number of beams to maintain at each step |
| `max_tokens` | `int` | *(required)* | Maximum number of tokens to generate |
| `temperature` | `float` | `0.0` | Sampling temperature applied before beam selection |
| `length_penalty` | `float` | `1.0` | Exponent for length normalization |
| `ignore_eos` | `bool` | `False` | Continue generating past the EOS token |
| `include_stop_str_in_output` | `bool` | `False` | Include stop strings in the output text |

### `beam_width`

The number of candidate sequences to maintain at each decoding step. A larger beam width explores more of the search space but requires more memory and computation.

- `beam_width=1` → equivalent to greedy decoding
- `beam_width=4` → standard for many NLP tasks
- `beam_width=8` → higher quality, more compute

```python
params = BeamSearchParams(beam_width=4, max_tokens=50)
```

### `max_tokens`

The maximum number of tokens to generate. Beam search stops when all active beams have either hit `max_tokens` or produced an EOS token.

### `temperature`

When `temperature > 0`, the logits are scaled before computing beam scores. This introduces some randomness into the beam selection process, producing more diverse beams. The default `0.0` gives fully deterministic beam search.

```python
# Slightly stochastic beam search
params = BeamSearchParams(beam_width=4, max_tokens=50, temperature=0.3)
```

### `length_penalty`

Controls the preference for longer or shorter sequences. Applied as an exponent to the sequence length when computing the normalized beam score:

```
score = cumulative_logprob / (length ^ length_penalty)
```

- `length_penalty=1.0` → normalize by length (default, balanced)
- `length_penalty > 1.0` → prefer longer sequences
- `length_penalty < 1.0` → prefer shorter sequences
- `length_penalty=0.0` → no length normalization (raw cumulative logprob)

```python
# Prefer longer, more complete responses
params = BeamSearchParams(beam_width=4, max_tokens=100, length_penalty=1.2)

# Prefer shorter, more concise responses
params = BeamSearchParams(beam_width=4, max_tokens=100, length_penalty=0.8)
```

### `ignore_eos`

When `True`, the model continues generating even after producing the EOS token. Useful when you want to generate a fixed number of tokens regardless of the model's natural stopping point.

```python
params = BeamSearchParams(beam_width=4, max_tokens=50, ignore_eos=True)
```

---

## `llm.beam_search()` Method

```python
llm.beam_search(
    prompts: list[TokensPrompt | TextPrompt],
    params: BeamSearchParams,
    lora_request: list[LoRARequest] | LoRARequest | None = None,
    use_tqdm: bool = False,
    concurrency_limit: int | None = None,
) -> list[BeamSearchOutput]
```

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `prompts` | `list[TextPrompt \| TokensPrompt]` | Input prompts (text strings or token ID lists) |
| `params` | `BeamSearchParams` | Beam search configuration |
| `lora_request` | `LoRARequest \| list[LoRARequest] \| None` | LoRA adapter(s) to use |
| `use_tqdm` | `bool` | Show a progress bar (disabled when `concurrency_limit` is set) |
| `concurrency_limit` | `int \| None` | Maximum number of prompts processed concurrently |

**Prompt formats:**

```python
# Text prompt
{"prompt": "The capital of France is"}

# Token IDs prompt
{"prompt_token_ids": [1, 2, 3, 4, 5]}
```

---

## Output: `BeamSearchOutput`

`llm.beam_search()` returns a `list[BeamSearchOutput]`, one per input prompt.

```python
from vllm.beam_search import BeamSearchOutput, BeamSearchSequence
```

### `BeamSearchOutput`

| Attribute | Type | Description |
|---|---|---|
| `sequences` | `list[BeamSearchSequence]` | Top-`beam_width` sequences, sorted best-first |

### `BeamSearchSequence`

| Attribute | Type | Description |
|---|---|---|
| `text` | `str \| None` | Decoded text of the sequence |
| `tokens` | `list[int]` | Token IDs of the generated sequence |
| `cum_logprob` | `float` | Cumulative log probability (unnormalized) |
| `logprobs` | `list[dict[int, Logprob]]` | Per-step log probabilities |
| `finish_reason` | `str \| None` | Why this beam ended |
| `stop_reason` | `int \| str \| None` | Stop token or string |

---

## Examples

### Basic Beam Search

```python
from vllm import LLM
from vllm.sampling_params import BeamSearchParams

llm = LLM(model="facebook/opt-125m")

outputs = llm.beam_search(
    prompts=[{"prompt": "Translate to French: The cat sat on the mat."}],
    params=BeamSearchParams(beam_width=5, max_tokens=30),
)

print("Top 5 beams:")
for i, seq in enumerate(outputs[0].sequences):
    print(f"  [{i+1}] score={seq.cum_logprob:.4f} | {seq.text!r}")
```

### Batch Beam Search

```python
prompts = [
    {"prompt": "Summarize: The quick brown fox jumps over the lazy dog."},
    {"prompt": "Summarize: Artificial intelligence is transforming industries."},
    {"prompt": "Summarize: The stock market rose sharply on positive earnings."},
]

outputs = llm.beam_search(
    prompts=prompts,
    params=BeamSearchParams(beam_width=4, max_tokens=50, length_penalty=1.0),
)

for prompt_dict, output in zip(prompts, outputs):
    best = output.sequences[0]
    print(f"Input:  {prompt_dict['prompt']!r}")
    print(f"Output: {best.text!r}")
    print(f"Score:  {best.cum_logprob:.4f}")
    print()
```

### Comparing All Beams

```python
outputs = llm.beam_search(
    prompts=[{"prompt": "The best programming language is"}],
    params=BeamSearchParams(beam_width=6, max_tokens=15),
)

print("All beams (best to worst):")
for i, seq in enumerate(outputs[0].sequences):
    print(f"  [{i+1}] {seq.cum_logprob:8.4f} | {seq.text!r}")
```

### With Token ID Input

```python
tokenizer = llm.get_tokenizer()
token_ids = tokenizer.encode("The future of AI is", add_special_tokens=True)

outputs = llm.beam_search(
    prompts=[{"prompt_token_ids": token_ids}],
    params=BeamSearchParams(beam_width=4, max_tokens=25),
)

print(outputs[0].sequences[0].text)
```

### With LoRA

```python
from vllm.lora.request import LoRARequest

lora = LoRARequest("my-adapter", 1, "/path/to/lora")

outputs = llm.beam_search(
    prompts=[{"prompt": "Translate to Spanish: Hello, how are you?"}],
    params=BeamSearchParams(beam_width=4, max_tokens=30),
    lora_request=lora,
)
```

### Concurrency Limiting

For large batches, limit how many prompts are processed simultaneously to control memory usage:

```python
outputs = llm.beam_search(
    prompts=large_prompt_list,
    params=BeamSearchParams(beam_width=4, max_tokens=50),
    concurrency_limit=8,  # process 8 prompts at a time
)
```

---

## How Beam Search Works in vLLM

vLLM implements beam search on top of its standard sampling engine:

1. **Initialization:** Each prompt starts with a single beam containing the prompt tokens.

2. **Expansion:** At each step, vLLM generates `2 × beam_width` candidate next tokens for each active beam (following the Hugging Face Transformers convention). This over-generation ensures the best candidates are not missed.

3. **Pruning:** The candidates are scored and the top `beam_width` sequences are kept.

4. **Completion:** A beam is marked complete when it produces the EOS token (unless `ignore_eos=True`). Completed beams are collected and no longer expanded.

5. **Scoring:** The final score is computed as:
   ```
   score = cumulative_logprob / (sequence_length ^ length_penalty)
   ```
   where `sequence_length` excludes the EOS token.

6. **Output:** The top `beam_width` completed sequences are returned, sorted by score (best first).

---

## Beam Search vs. Sampling

| Aspect | Beam Search | Sampling |
|---|---|---|
| Determinism | Fully deterministic (temperature=0) | Stochastic |
| Diversity | Low (all beams converge) | High |
| Quality | Often higher for structured tasks | Better for open-ended tasks |
| Speed | Slower (multiple beams per step) | Faster |
| Memory | Higher (beam_width × prompt memory) | Lower |
| Multiple outputs | Top-k beams | n independent samples |

---

## Performance Considerations

- **Memory:** Beam search requires `beam_width` times more KV cache memory per prompt than greedy decoding.
- **Speed:** Each decoding step runs `beam_width` forward passes (batched), so throughput is lower than greedy decoding.
- **Batch size:** For large batches, use `concurrency_limit` to avoid OOM errors.
- **`max_tokens`:** The progress bar shows the upper bound on token steps; beam search may finish earlier when all beams hit EOS.

---

## See Also

- [SamplingParams Reference](sampling_params.md) — for sampling-based generation
- [Output Types](output_types.md) — `BeamSearchOutput` and `BeamSearchSequence` details
- [Offline Inference Guide](offline_inference.md) — other generation methods
